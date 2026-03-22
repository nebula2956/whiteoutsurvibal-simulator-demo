"""GA (Genetic Algorithm) + CMA-ES Ratio Refinement optimizer.

Phase 1: リーダーSHA（MC少数バッチ評価）
Phase 2: メンバー4人の組み合わせをGAで最適化
Phase 3: CMA-ES比率精査（RatioRefiner再利用）
Phase 4: 最終再評価
"""
from __future__ import annotations

import itertools
import math
import random
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple, Dict

from simulator.models import ArmyConfig, HeroConfig
from .genome import HeroPool, OwnedHero, build_army_config
from .fitness import FitnessEvaluator
from .beam_runner import (
    simplex_grid, build_ratio_grid, build_coarse_grid,
    _leaders_to_heroes, _full_heroes, candidate_label,
    RatioRefiner, PhaseInfo,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class GAConfig:
    # Phase 1: Leader screening
    leader_top_k: int = 20
    n_member_samples: int = 5
    ratio_grid_resolution: int = 5
    ratio_dirichlet_extra: int = 15
    coarse_grid_resolution: int = 4
    # Phase 2: GA
    population_size: int = 40
    generations: int = 20
    n_sim_per_eval: int = 3
    elite_ratio: float = 0.2
    crossover_rate: float = 0.7
    mutation_rate: float = 0.2
    ga_top_k: int = 5              # GA からPhase 3へ渡す上位数
    # Phase 3: CMA-ES refinement
    refine_top_k: int = 5
    refine_cma_popsize: int = 14
    refine_cma_generations: int = 25
    refine_n_sim: int = 50
    refine_cma_sigma: float = 0.7
    refine_cma_restarts: int = 2
    # Final
    n_reeval_simulations: int = 200


# ---------------------------------------------------------------------------
# Individual
# ---------------------------------------------------------------------------

@dataclass
class Individual:
    members: List[str]
    ratio: Tuple[float, float, float]
    score: float = 0.0


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class GAResult:
    best_config: ArmyConfig
    best_score: float
    best_score_reeval: float
    confidence_interval: Tuple[float, float]
    label: str
    top_candidates: List[Tuple[ArmyConfig, float, str]]
    cache_hits: int
    cache_misses: int


# ---------------------------------------------------------------------------
# GA operators
# ---------------------------------------------------------------------------

def _crossover(
    parent1: Individual, parent2: Individual, pool_members: List[str],
) -> Individual:
    """一様交叉: 両親のメンバーを混ぜて4人を選出。"""
    combined = list(set(parent1.members + parent2.members))
    random.shuffle(combined)
    child = combined[:4]
    while len(child) < 4:
        candidate = random.choice(pool_members)
        if candidate not in child:
            child.append(candidate)
    # 比率はベストスコアの親から継承
    ratio = parent1.ratio if parent1.score >= parent2.score else parent2.ratio
    return Individual(members=sorted(child), ratio=ratio)


def _mutate(
    individual: Individual, pool_members: List[str], mutation_rate: float = 0.2,
) -> Individual:
    """突然変異: 各メンバーを確率的にランダム入れ替え。"""
    members = list(individual.members)
    for i in range(len(members)):
        if random.random() < mutation_rate:
            candidates = [m for m in pool_members if m not in members]
            if candidates:
                members[i] = random.choice(candidates)
    return Individual(members=sorted(members), ratio=individual.ratio)


def _tournament_select(
    population: List[Individual], tournament_size: int = 3,
) -> Individual:
    """トーナメント選択。"""
    participants = random.sample(population, min(tournament_size, len(population)))
    return max(participants, key=lambda ind: ind.score)


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _evaluate_on_grid_mc(
    heroes: List[HeroConfig],
    ratio_grid: List[Tuple[float, float, float]],
    evaluator: FitnessEvaluator,
    n_sim: int = 3,
) -> Tuple[float, Tuple[float, float, float]]:
    """比率グリッド上でMC少数バッチ評価し、最良の(score, ratio)を返す。"""
    best_score = float("-inf")
    best_ratio = ratio_grid[0]
    template = evaluator.own_template

    for ratio in ratio_grid:
        config = build_army_config(heroes, ratio, template)
        score = evaluator.evaluate_config_for_search(config, n_sim=n_sim)
        if score > best_score:
            best_score = score
            best_ratio = ratio

    return best_score, best_ratio


# ---------------------------------------------------------------------------
# Main Optimizer
# ---------------------------------------------------------------------------

def estimate_ga_cost(cfg: GAConfig, pool: HeroPool) -> dict:
    """推定シミュレーション回数を計算。UIの表示用。"""
    n_i, n_l, n_a = pool.leader_dims
    total_leaders = n_i * n_l * n_a

    full_grid_size = len(simplex_grid(cfg.ratio_grid_resolution, min_val=0.05)) + cfg.ratio_dirichlet_extra
    coarse_grid_size = len(simplex_grid(cfg.coarse_grid_resolution, min_val=0.1))

    # Phase 1: SHA (MC版)
    sha_r1 = total_leaders * 2 * coarse_grid_size * cfg.n_sim_per_eval
    sha_r2 = int(total_leaders * 0.5) * 3 * coarse_grid_size * cfg.n_sim_per_eval
    sha_r3 = max(cfg.leader_top_k, int(total_leaders * 0.25)) * cfg.n_member_samples * full_grid_size * cfg.n_sim_per_eval
    phase1 = sha_r1 + sha_r2 + sha_r3

    # Phase 2: GA
    # 最大 population_size * generations 個体を評価（キャッシュで削減）
    phase2_per_leader = cfg.population_size * cfg.generations * full_grid_size * cfg.n_sim_per_eval
    phase2 = cfg.leader_top_k * phase2_per_leader
    # キャッシュで~50%削減と見積もり
    phase2_with_cache = int(phase2 * 0.5)

    # Phase 3: CMA-ES (× restarts for multi-start)
    phase3 = cfg.refine_top_k * cfg.refine_cma_popsize * cfg.refine_cma_generations * cfg.refine_n_sim * cfg.refine_cma_restarts

    # Phase 4
    phase4 = cfg.refine_top_k * cfg.n_reeval_simulations

    return {
        "phase1_mc": phase1,
        "phase2_mc_raw": phase2,
        "phase2_mc_cached": phase2_with_cache,
        "phase3_mc": phase3,
        "phase4_mc": phase4,
        "total_mc": phase1 + phase2_with_cache + phase3 + phase4,
    }


class GAOptimizer:
    """GA + CMA-ES Ratio Refinement optimizer."""

    def __init__(self, evaluator: FitnessEvaluator, pool: HeroPool,
                 config: GAConfig):
        self.evaluator = evaluator
        self.pool = pool
        self.config = config

        self.full_grid = build_ratio_grid(config)
        self.coarse_grid = build_coarse_grid(config)

    def run(self, callback: Optional[Callable[[PhaseInfo], None]] = None
            ) -> GAResult:
        cfg = self.config

        # --- Phase 1: Leader Screening (MC版) ---
        leader_results = self._phase1_leader_screen_mc(callback)

        # --- Phase 2: GA ---
        ga_results = self._phase2_ga(leader_results, callback)

        # --- Phase 3: CMA-ES Refinement ---
        refined = self._phase3_cma_refine(ga_results, callback)

        # --- Phase 4: Final Re-evaluation ---
        best_config, best_score, best_label = refined[0]
        reeval_score = self.evaluator.evaluate_config(best_config, n_sim=cfg.n_reeval_simulations)

        ci = self._confidence_interval(reeval_score, cfg.n_reeval_simulations)

        if callback:
            callback(PhaseInfo(
                phase="reeval", progress=1.0,
                message=f"再評価完了: {reeval_score:.4f} (95%CI: {ci[0]:.4f}-{ci[1]:.4f})",
                best_label=best_label, best_score=reeval_score,
            ))

        return GAResult(
            best_config=best_config,
            best_score=best_score,
            best_score_reeval=reeval_score,
            confidence_interval=ci,
            label=best_label,
            top_candidates=refined[:cfg.refine_top_k],
            cache_hits=self.evaluator.cache_hits,
            cache_misses=self.evaluator.cache_misses,
        )

    # -----------------------------------------------------------------------
    # Phase 1: Leader SHA (MC版)
    # -----------------------------------------------------------------------

    def _phase1_leader_screen_mc(
        self, callback: Optional[Callable[[PhaseInfo], None]],
    ) -> List[Tuple[Tuple[int, int, int], float, Tuple[float, float, float]]]:
        """Successive Halving でリーダーをスクリーニング（MC少数バッチ評価）。"""
        cfg = self.config
        pool = self.pool

        leader_combos = list(itertools.product(
            range(len(pool.infantry_leaders)),
            range(len(pool.lancer_leaders)),
            range(len(pool.archer_leaders)),
        ))

        sha_rounds = [
            (2, self.coarse_grid, 0.5),
            (3, self.coarse_grid, 0.5),
            (cfg.n_member_samples, self.full_grid, None),
        ]

        active: List[Tuple[int, int, int]] = list(leader_combos)
        leader_data: Dict[Tuple[int, int, int], Tuple[float, int, float, Tuple[float, float, float]]] = {}

        total_work = sum(
            len(active) if r == 0 else int(len(leader_combos) * 0.5 ** r)
            for r in range(len(sha_rounds))
        )
        work_done = 0

        for round_i, (n_samples, grid, keep_ratio) in enumerate(sha_rounds):
            for combo in active:
                i_idx, l_idx, a_idx = combo
                leader_heroes = _leaders_to_heroes(
                    pool.infantry_leaders[i_idx],
                    pool.lancer_leaders[l_idx],
                    pool.archer_leaders[a_idx],
                )

                prev_sum, prev_count, prev_best_score, prev_best_ratio = leader_data.get(
                    combo, (0.0, 0, float("-inf"), grid[0])
                )

                round_scores = []
                best_score_this = prev_best_score
                best_ratio_this = prev_best_ratio

                for _ in range(n_samples):
                    if len(pool.members) >= 4:
                        members = sorted(random.sample(pool.members, 4))
                    else:
                        members = sorted(random.choices(pool.members, k=4))
                    heroes = _full_heroes(leader_heroes, members)
                    score, ratio = _evaluate_on_grid_mc(
                        heroes, grid, self.evaluator, n_sim=cfg.n_sim_per_eval,
                    )
                    round_scores.append(score)
                    if score > best_score_this:
                        best_score_this = score
                        best_ratio_this = ratio

                new_sum = prev_sum + sum(round_scores)
                new_count = prev_count + len(round_scores)
                leader_data[combo] = (new_sum, new_count, best_score_this, best_ratio_this)

                work_done += 1
                if callback and work_done % max(1, total_work // 20) == 0:
                    best_combo = max(active, key=lambda c: leader_data.get(c, (0, 1, 0, grid[0]))[0] / max(leader_data.get(c, (0, 1, 0, grid[0]))[1], 1))
                    bd = leader_data[best_combo]
                    best_leaders = _leaders_to_heroes(
                        pool.infantry_leaders[best_combo[0]],
                        pool.lancer_leaders[best_combo[1]],
                        pool.archer_leaders[best_combo[2]],
                    )
                    label = candidate_label(best_leaders, ["?"] * 4, bd[3])
                    top_leaders = sorted(
                        [(c, leader_data.get(c, (0.0, 1, 0.0, grid[0]))) for c in active],
                        key=lambda x: x[1][0] / max(x[1][1], 1), reverse=True
                    )[:5]
                    top_labels = [
                        {
                            "label": candidate_label(
                                _leaders_to_heroes(
                                    pool.infantry_leaders[c[0]],
                                    pool.lancer_leaders[c[1]],
                                    pool.archer_leaders[c[2]],
                                ),
                                ["?"] * 4, d[3],
                            ),
                            "score": d[0] / max(d[1], 1),
                        }
                        for c, d in top_leaders
                    ]
                    callback(PhaseInfo(
                        phase="leader_screen",
                        progress=work_done / total_work,
                        message=f"SHA Round {round_i+1}/{len(sha_rounds)}: "
                                f"{len(active)} 候補評価中 (MC {cfg.n_sim_per_eval}回)",
                        best_label=label, best_score=bd[0] / max(bd[1], 1),
                        detail={"round": round_i + 1, "active": len(active),
                                "top_leaders": top_labels},
                    ))

            # Halve
            if keep_ratio is not None:
                scored = [
                    (combo, leader_data[combo][0] / max(leader_data[combo][1], 1))
                    for combo in active
                ]
                scored.sort(key=lambda x: x[1], reverse=True)
                n_keep = max(cfg.leader_top_k, int(len(scored) * keep_ratio))
                active = [combo for combo, _ in scored[:n_keep]]
            else:
                scored = [
                    (combo, leader_data[combo][0] / max(leader_data[combo][1], 1))
                    for combo in active
                ]
                scored.sort(key=lambda x: x[1], reverse=True)
                active = [combo for combo, _ in scored[:cfg.leader_top_k]]

        results = []
        for combo in active:
            data = leader_data[combo]
            avg_score = data[0] / max(data[1], 1)
            best_ratio = data[3]
            results.append((combo, avg_score, best_ratio))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:cfg.leader_top_k]

    # -----------------------------------------------------------------------
    # Phase 2: GA
    # -----------------------------------------------------------------------

    def _phase2_ga(
        self,
        leader_results: List[Tuple[Tuple[int, int, int], float, Tuple[float, float, float]]],
        callback: Optional[Callable[[PhaseInfo], None]],
    ) -> List[Tuple[List[HeroConfig], List[str], float, Tuple[float, float, float]]]:
        """各リーダーセットに対してGAでメンバー最適化。"""
        cfg = self.config
        pool = self.pool
        all_candidates = []
        total_leaders = len(leader_results)

        for li, (leader_idx, _, init_ratio) in enumerate(leader_results):
            i_idx, l_idx, a_idx = leader_idx
            leader_heroes = _leaders_to_heroes(
                pool.infantry_leaders[i_idx],
                pool.lancer_leaders[l_idx],
                pool.archer_leaders[a_idx],
            )

            # 初期集団を生成
            population: List[Individual] = []
            for _ in range(cfg.population_size):
                if len(pool.members) >= 4:
                    members = sorted(random.sample(pool.members, 4))
                else:
                    members = sorted(random.choices(pool.members, k=4))
                population.append(Individual(members=members, ratio=init_ratio))

            # 評価済みキャッシュ（メンバーセット → (score, ratio)）
            eval_cache: Dict[tuple, Tuple[float, Tuple[float, float, float]]] = {}

            for gen in range(cfg.generations):
                # 評価
                for ind in population:
                    cache_key = tuple(ind.members)
                    if cache_key in eval_cache:
                        ind.score, ind.ratio = eval_cache[cache_key]
                    else:
                        heroes = _full_heroes(leader_heroes, ind.members)
                        score, ratio = _evaluate_on_grid_mc(
                            heroes, self.full_grid, self.evaluator,
                            n_sim=cfg.n_sim_per_eval,
                        )
                        ind.score = score
                        ind.ratio = ratio
                        eval_cache[cache_key] = (score, ratio)

                # ソート
                population.sort(key=lambda x: x.score, reverse=True)

                # コールバック
                if callback:
                    best = population[0]
                    best_label = candidate_label(leader_heroes, best.members, best.ratio)
                    # 集団の多様性: ユニークな編成数
                    unique_sets = len(set(tuple(ind.members) for ind in population))
                    top5 = [
                        {"label": candidate_label(leader_heroes, ind.members, ind.ratio),
                         "score": ind.score}
                        for ind in population[:5]
                    ]
                    callback(PhaseInfo(
                        phase="ga_search",
                        progress=(li + (gen + 1) / cfg.generations) / total_leaders,
                        message=f"GA: リーダー{li+1}/{total_leaders} 世代{gen+1}/{cfg.generations} "
                                f"best={best.score:.3f} 多様性={unique_sets}/{cfg.population_size}",
                        best_label=best_label, best_score=best.score,
                        detail={"leader_idx": li + 1, "total_leaders": total_leaders,
                                "generation": gen + 1, "total_generations": cfg.generations,
                                "diversity": unique_sets, "eval_cache_size": len(eval_cache),
                                "top_candidates": top5},
                    ))

                # 最終世代は次世代を作らない
                if gen == cfg.generations - 1:
                    break

                # 次世代生成
                n_elite = max(1, int(cfg.population_size * cfg.elite_ratio))
                next_gen = [Individual(members=list(ind.members), ratio=ind.ratio, score=ind.score)
                            for ind in population[:n_elite]]

                while len(next_gen) < cfg.population_size:
                    if random.random() < cfg.crossover_rate:
                        p1 = _tournament_select(population)
                        p2 = _tournament_select(population)
                        child = _crossover(p1, p2, pool.members)
                    else:
                        parent = _tournament_select(population)
                        child = Individual(
                            members=list(parent.members),
                            ratio=parent.ratio,
                        )
                    child = _mutate(child, pool.members, cfg.mutation_rate)
                    next_gen.append(child)

                population = next_gen

            # このリーダーセットの上位候補を収集
            seen_sets: set = set()
            for ind in population:
                member_key = tuple(ind.members)
                if member_key not in seen_sets:
                    seen_sets.add(member_key)
                    all_candidates.append((leader_heroes, ind.members, ind.score, ind.ratio))
                if len(seen_sets) >= cfg.ga_top_k:
                    break

        # 全リーダーの候補をまとめてソート
        all_candidates.sort(key=lambda x: x[2], reverse=True)

        # 重複除去
        seen: set = set()
        unique: List[Tuple[List[HeroConfig], List[str], float, Tuple[float, float, float]]] = []
        for leader_h, members, score, ratio in all_candidates:
            hero_key = (tuple(h.hero_id for h in leader_h), tuple(members))
            if hero_key not in seen:
                seen.add(hero_key)
                unique.append((leader_h, members, score, ratio))

        return unique

    # -----------------------------------------------------------------------
    # Phase 3: CMA-ES Refinement (RatioRefiner再利用)
    # -----------------------------------------------------------------------

    def _phase3_cma_refine(
        self,
        ga_results: List[Tuple[List[HeroConfig], List[str], float, Tuple[float, float, float]]],
        callback: Optional[Callable[[PhaseInfo], None]],
    ) -> List[Tuple[ArmyConfig, float, str]]:
        cfg = self.config
        top_k = ga_results[:cfg.refine_top_k]
        refined: List[Tuple[ArmyConfig, float, str]] = []

        for i, (leader_heroes, member_ids, ga_score, ga_ratio) in enumerate(top_k):
            heroes = _full_heroes(leader_heroes, member_ids)
            refiner = RatioRefiner(self.evaluator, heroes, cfg)
            best_ratio, best_score = refiner.run(
                ga_ratio,
                callback=callback,
                candidate_idx=i,
                total_candidates=len(top_k),
                beam_score=ga_score,
            )

            config = build_army_config(heroes, best_ratio, self.evaluator.own_template)
            label = candidate_label(leader_heroes, member_ids, best_ratio)
            refined.append((config, best_score, label))

        refined.sort(key=lambda x: x[1], reverse=True)
        return refined

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _confidence_interval(win_rate: float, n: int) -> Tuple[float, float]:
        se = math.sqrt(max(win_rate * (1 - win_rate), 0) / max(n, 1))
        margin = 1.96 * se
        return (max(0.0, win_rate - margin), min(1.0, win_rate + margin))
