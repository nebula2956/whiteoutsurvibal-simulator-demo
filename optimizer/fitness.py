from __future__ import annotations

import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor, Future
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

from simulator.models import ArmyConfig, BattleResult
from simulator import run_simulation
from .genome import HeroPool, decode_genome
from .cache_store import ExpectedCacheStore


def _sim_expected_worker(config: ArmyConfig, enemy: ArmyConfig) -> BattleResult:
    """プロセスワーカー用トップレベル関数。ArmyConfigはpickle可能。"""
    from simulator import run_simulation
    return run_simulation(config, enemy, mode="expected", validate=False)


@dataclass
class FitnessWeights:
    """各指標の重み。0にすると無視。全て「高いほど良い」に正規化される。"""
    win_rate: float = 1.0
    survival_ratio: float = 0.0
    avg_kills: float = 0.0
    std_penalty: float = 0.0
    downside_risk: float = 0.0
    downside_percentile: int = 10

    def is_all_zero(self) -> bool:
        return (self.win_rate == 0 and self.survival_ratio == 0
                and self.avg_kills == 0 and self.std_penalty == 0
                and self.downside_risk == 0)


class FitnessEvaluator:
    """
    モンテカルロで多目的フィットネスを評価。CMA-ES用（最小化）。
    同一編成のシミュレーション結果をキャッシュし、再評価時に再利用する。
    """

    def __init__(
        self,
        enemy_config: ArmyConfig,
        own_template: ArmyConfig,
        pool: HeroPool,
        n_simulations: int = 30,
        weights: FitnessWeights | None = None,
        cache_max_keys: int = 20000,
        max_workers: int = 0,
    ):
        self.enemy_config = enemy_config
        self.own_template = own_template
        self.pool = pool
        self.n_simulations = n_simulations
        self.weights = weights or FitnessWeights()
        self.cache_max_keys = cache_max_keys
        self.max_workers = max_workers or min(4, os.cpu_count() or 1)
        # キャッシュ: (編成キー) → シミュレーション結果リスト
        self._cache: Dict[Tuple, List[BattleResult]] = {}
        # expected mode用キャッシュ: SQLite永続化（決定論的なので1回で十分）
        self._expected_cache = ExpectedCacheStore()
        self.cache_hits: int = 0
        self.cache_misses: int = 0
        self._pool: Optional[ProcessPoolExecutor] = None

    def _run_cached_simulations(self, config: ArmyConfig, n_needed: int) -> Optional[List[BattleResult]]:
        """キャッシュから既存結果を補完しつつ n_needed 件の BattleResult を返す。
        シミュレーション失敗時は None を返す。"""
        key = self._cache_key(config)
        cached = self._cache.get(key, [])
        n_more = max(0, n_needed - len(cached))

        if n_more == 0:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
            for _ in range(n_more):
                try:
                    cached.append(run_simulation(config, self.enemy_config, mode="random", validate=False))
                except Exception as e:
                    import traceback, sys
                    print(f"[FitnessEvaluator] シミュレーション失敗: {type(e).__name__}: {e}", file=sys.stderr)
                    traceback.print_exc(file=sys.stderr)
                    return None
            if len(self._cache) >= self.cache_max_keys and key not in self._cache:
                oldest = next(iter(self._cache))
                del self._cache[oldest]
            self._cache[key] = cached

        return cached[:n_needed]

    def __call__(self, x: np.ndarray, n_override: int | None = None) -> float:
        """CMA-ESは最小化なので負値を返す。n_overrideで再評価時のシミュレーション数を指定可能。"""
        config = decode_genome(x, self.pool, self.own_template)
        results = self._run_cached_simulations(config, n_override or self.n_simulations)
        if results is None:
            return 0.0
        return -self._composite_score(results)

    def evaluate_config(self, config: ArmyConfig, n_sim: int | None = None) -> float:
        """ArmyConfigを直接評価。正のスコアを返す（高いほど良い）。"""
        results = self._run_cached_simulations(config, n_sim or self.n_simulations)
        if results is None:
            return 0.0
        return self._composite_score(results)

    def evaluate_config_for_search(self, config: ArmyConfig, n_sim: int = 3) -> float:
        """MC少数バッチで探索用スコアを返す。GA/SHA用。正のスコア（高いほど良い）。"""
        results = self._run_cached_simulations(config, n_sim)
        if results is None:
            return 0.0
        return self._composite_score_for_search(results)

    def evaluate_config_expected(self, config: ArmyConfig) -> float:
        """expected mode（決定論的）で1回シミュレーション。Beam Search用の高速評価。"""
        key = ("expected", self._cache_key(config))

        cached = self._expected_cache.get(key)
        if cached is not None:
            self.cache_hits += 1
            return cached

        self.cache_misses += 1
        try:
            result = run_simulation(config, self.enemy_config, mode="expected", validate=False)
        except Exception as e:
            import traceback, sys
            print(f"[FitnessEvaluator] expected sim失敗: {type(e).__name__}: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            return 0.0

        score = self._composite_score_for_search([result])
        self._expected_cache.put(key, score)

        return score

    def evaluate_batch_expected(self, configs: List[ArmyConfig]) -> List[float]:
        """複数ArmyConfigをexpected modeで並列評価。キャッシュ済みはスキップ。"""
        results: List[Optional[float]] = [None] * len(configs)
        to_compute: List[Tuple[int, ArmyConfig, Tuple]] = []

        # Check cache first
        for i, config in enumerate(configs):
            key = ("expected", self._cache_key(config))
            cached = self._expected_cache.get(key)
            if cached is not None:
                results[i] = cached
                self.cache_hits += 1
            else:
                to_compute.append((i, config, key))
                self.cache_misses += 1

        if not to_compute:
            return results  # type: ignore

        # Parallel execution for cache misses
        new_entries: list[Tuple[Tuple, float]] = []

        if self.max_workers > 1 and len(to_compute) >= 4:
            if self._pool is None:
                self._pool = ProcessPoolExecutor(max_workers=self.max_workers)
            futures: List[Tuple[int, Tuple, Future]] = []
            for idx, config, key in to_compute:
                f = self._pool.submit(_sim_expected_worker, config, self.enemy_config)
                futures.append((idx, key, f))
            for idx, key, f in futures:
                try:
                    result = f.result()
                    score = self._composite_score_for_search([result])
                except Exception:
                    score = 0.0
                results[idx] = score
                new_entries.append((key, score))
        else:
            for idx, config, key in to_compute:
                results[idx] = self.evaluate_config_expected(config)

        # Batch write new entries to persistent cache
        if new_entries:
            self._expected_cache.put_batch(new_entries)

        return results  # type: ignore

    def _cache_key(self, config: ArmyConfig) -> Tuple:
        """ArmyConfigからキャッシュキーを生成。比率は小数4桁で丸める。敵編成も含む。"""
        ratios = (round(config.troop_ratio["infantry"], 4),
                  round(config.troop_ratio["lancer"], 4))
        heroes = tuple(
            (h.hero_id, h.gear_level, h.position)
            for h in sorted(config.heroes, key=lambda h: h.position)
        )
        enemy_ratios = (round(self.enemy_config.troop_ratio.get("infantry", 0), 4),
                        round(self.enemy_config.troop_ratio.get("lancer", 0), 4))
        enemy_heroes = tuple(
            (h.hero_id, h.gear_level, h.position)
            for h in sorted(self.enemy_config.heroes, key=lambda h: h.position)
        )
        return (ratios, heroes, enemy_ratios, enemy_heroes)

    def _composite_score_for_search(self, results: List[BattleResult]) -> float:
        """探索フェーズ用の連続スコア。MC少数バッチでも差別化可能。
        ユーザー設定の重みをベースに、生存率と撃破率を最低限加味する。"""
        w = self.weights
        if not results:
            return 0.0

        total_cap = max(self.own_template.rally_capacity, 1)
        enemy_cap = max(self.enemy_config.rally_capacity, 1)

        wins, survivals, kill_ratios = [], [], []
        for r in results:
            wins.append(1.0 if r.winner == "a" else 0.0)
            survived = sum(r.final_counts_a.values()) if r.final_counts_a else 0
            survivals.append(survived / total_cap)
            own_kills = sum(r.total_kills_by_side.get("a", {}).values())
            kill_ratios.append(own_kills / enemy_cap)

        score = 0.0
        if w.win_rate != 0:
            score += w.win_rate * np.mean(wins)
        if w.survival_ratio != 0:
            score += w.survival_ratio * np.mean(survivals)
        else:
            score += 0.3 * np.mean(survivals)
        if w.avg_kills != 0:
            score += w.avg_kills * np.mean(kill_ratios)
        else:
            score += 0.2 * np.mean(kill_ratios)
        return score

    def _composite_score(self, results: List[BattleResult]) -> float:
        w = self.weights
        if not results:
            return 0.0

        total_cap = max(self.own_template.rally_capacity, 1)
        enemy_cap = max(self.enemy_config.rally_capacity, 1)

        wins = []
        survival_ratios = []
        kill_ratios = []

        for r in results:
            wins.append(1.0 if r.winner == "a" else 0.0)

            survived = sum(r.final_counts_a.values()) if r.final_counts_a else 0
            survival_ratios.append(survived / total_cap)

            own_kills = sum(r.total_kills_by_side.get("a", {}).values())  # a = 自軍が敵を倒した数
            kill_ratios.append(own_kills / enemy_cap)

        sr = np.array(survival_ratios)

        score = 0.0

        if w.win_rate != 0:
            score += w.win_rate * np.mean(wins)

        if w.survival_ratio != 0:
            score += w.survival_ratio * np.mean(sr)

        if w.avg_kills != 0:
            score += w.avg_kills * np.mean(kill_ratios)

        if w.std_penalty != 0:
            score -= w.std_penalty * np.std(sr)

        if w.downside_risk != 0:
            cutoff = max(1, int(len(sr) * w.downside_percentile / 100))
            worst = np.sort(sr)[:cutoff]
            score += w.downside_risk * np.mean(worst)

        return score
