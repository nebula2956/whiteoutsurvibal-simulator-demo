"""Phase 3: Adaptive Monte Carlo 評価。

多忠実度評価 + 共通乱数（Common Random Numbers）により、
候補間の比較分散を削減しつつ評価コストを抑える。

アルゴリズム:
    1. 低忠実度スクリーニング (n_reeval_low 回 MC)
       → 下位 (1 - prune_keep_ratio) を打ち切り
    2. 高忠実度再評価 (n_reeval_high 回 MC, 共通乱数)
       → J_LCB / J_robust でランキング

共通乱数 (Common Random Numbers):
    全候補に同じシード列を使ってバトル乱数を固定する。
    候補間の比較分散が大幅に減り、少ない試行数でも順位が安定する。
    ただし候補の「絶対スコア」には使わず、あくまで比較・ランキング用。
"""
from __future__ import annotations

import random as _random
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np

from optimizer.fitness import FitnessEvaluator
from simulator import run_simulation
from simulator.models import ArmyConfig, BattleResult
from .config import OptimizerV2Config
from .hero_selector import PhaseInfo
from .scoring import compute_score, compute_score_from_wins, battle_win_score


@dataclass
class ReevalResult:
    """Phase 3 の再評価結果。"""
    config: ArmyConfig
    score: float                          # J_LCB / J_robust / mean
    mu: float                             # 平均勝率（draw=0.5）
    sigma: float                          # 標準偏差
    ci_lo: float                          # 95%CI 下限
    ci_hi: float                          # 95%CI 上限
    n_sims: int                           # 評価に使った試合数
    label: str = ""


class AdaptiveMC:
    """多忠実度 Monte Carlo 評価（Phase 3）。

    低忠実度スクリーニング → 打ち切り → 高忠実度評価の 2 段階で
    評価コストを削減しながら信頼性の高いランキングを得る。

    Args:
        evaluator:  FitnessEvaluator（enemy_config 取得のため）
        cfg:        OptimizerV2Config
    """

    def __init__(self, evaluator: FitnessEvaluator, cfg: OptimizerV2Config):
        self.evaluator = evaluator
        self.cfg = cfg
        # 共通乱数: セッション開始時に n_reeval_high 個のシードを生成
        rng = np.random.default_rng(cfg.seed)
        self._shared_seeds = rng.integers(0, 2**31, size=cfg.n_reeval_high).tolist()

    def run(
        self,
        candidates: List[Tuple[ArmyConfig, float, str]],
        callback: Optional[Callable[[PhaseInfo], None]] = None,
    ) -> List[ReevalResult]:
        """candidates を多忠実度評価してランキングを返す。

        Args:
            candidates: [(config, prior_score, label), ...]
                        prior_score は Phase 2 の CMA-ES スコア（参考値）
            callback:   進捗コールバック

        Returns:
            ReevalResult のリスト（score 降順）
        """
        # ── Step 1: 低忠実度スクリーニング ────────────────────────
        low_results = self._screen_low(candidates, callback)

        # ── Step 2: 打ち切り ──────────────────────────────────────
        survivors = self._prune(low_results)

        # ── Step 3: 高忠実度再評価（共通乱数）────────────────────
        final = self._eval_high(survivors, callback)

        return sorted(final, key=lambda r: r.score, reverse=True)

    def _screen_low(
        self,
        candidates: List[Tuple[ArmyConfig, float, str]],
        callback: Optional[Callable],
    ) -> List[Tuple[ArmyConfig, float, List[float], str]]:
        """低忠実度 MC でスコアをつける。"""
        results = []
        n = len(candidates)
        for i, (config, prior_score, label) in enumerate(candidates):
            wins = self._run_mc(config, self.cfg.n_reeval_low, use_shared=False)
            score = compute_score_from_wins(
                wins,
                mode=self.cfg.scoring_mode,
                lcb_beta=self.cfg.lcb_beta,
                lambda_var=self.cfg.lambda_var,
            )
            results.append((config, score, wins, label))
            if callback:
                callback(PhaseInfo(
                    phase="reeval_low",
                    progress=0.9 + 0.04 * (i + 1) / max(n, 1),
                    message=f"低忠実度スクリーニング {i+1}/{n}: {label} score={score:.4f}",
                    best_label=label,
                    best_score=score,
                    detail={"step": "low_fidelity", "candidate": i + 1, "total": n},
                ))
        return results

    def _prune(
        self,
        results: List[Tuple[ArmyConfig, float, List[float], str]],
    ) -> List[Tuple[ArmyConfig, float, str]]:
        """スコア下位を除外する。"""
        sorted_res = sorted(results, key=lambda x: x[1], reverse=True)
        keep_n = max(1, round(len(sorted_res) * self.cfg.prune_keep_ratio))
        return [(r[0], r[1], r[3]) for r in sorted_res[:keep_n]]

    def _eval_high(
        self,
        candidates: List[Tuple[ArmyConfig, float, str]],
        callback: Optional[Callable],
    ) -> List[ReevalResult]:
        """高忠実度 MC（共通乱数）で最終評価する。"""
        results = []
        n = len(candidates)
        for i, (config, prior_score, label) in enumerate(candidates):
            wins = self._run_mc(config, self.cfg.n_reeval_high, use_shared=True)
            result = self._compute_result(config, wins, label)
            results.append(result)
            if callback:
                callback(PhaseInfo(
                    phase="reeval",
                    progress=0.94 + 0.06 * (i + 1) / max(n, 1),
                    message=(
                        f"高忠実度評価 {i+1}/{n}: {label} "
                        f"μ={result.mu:.3f}±{result.sigma:.3f} "
                        f"J={result.score:.4f}"
                    ),
                    best_label=label,
                    best_score=result.score,
                    detail={
                        "step":  "high_fidelity",
                        "candidate": i + 1,
                        "total": n,
                        "mu":    result.mu,
                        "sigma": result.sigma,
                        "ci_lo": result.ci_lo,
                        "ci_hi": result.ci_hi,
                    },
                ))
        return results

    def _run_mc(
        self,
        config: ArmyConfig,
        n_sims: int,
        use_shared: bool,
    ) -> List[float]:
        """n_sims 回の MC を実行し、勝敗スコアリスト [0, 0.5, 1.0] を返す。

        Args:
            use_shared: True → 共通乱数シードを使用（比較ランキング用）
                        False → 独立した乱数（スクリーニング用）
        """
        wins = []
        seeds = self._shared_seeds[:n_sims] if use_shared else [None] * n_sims

        for seed in seeds:
            if seed is not None:
                _random.seed(int(seed))
                np.random.seed(int(seed))
            try:
                result = run_simulation(
                    config, self.evaluator.enemy_config,
                    mode="random", validate=False,
                )
                wins.append(battle_win_score(result))
            except Exception:
                wins.append(0.0)

        return wins

    def _compute_result(
        self,
        config: ArmyConfig,
        wins: List[float],
        label: str,
    ) -> ReevalResult:
        """wins リストから ReevalResult を計算する。"""
        arr = np.array(wins)
        mu = float(np.mean(arr))
        sigma = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        se = sigma / np.sqrt(len(arr)) if len(arr) > 0 else 0.0
        ci_lo = mu - 1.96 * se
        ci_hi = mu + 1.96 * se

        score = compute_score_from_wins(
            wins,
            mode=self.cfg.scoring_mode,
            lcb_beta=self.cfg.lcb_beta,
            lambda_var=self.cfg.lambda_var,
        )
        return ReevalResult(
            config=config,
            score=score,
            mu=mu,
            sigma=sigma,
            ci_lo=ci_lo,
            ci_hi=ci_hi,
            n_sims=len(wins),
            label=label,
        )
