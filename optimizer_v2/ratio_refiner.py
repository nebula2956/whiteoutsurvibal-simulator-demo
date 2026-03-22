"""Phase 2: CMA-ES による兵種比率最適化。

固定した英雄セットに対して、兵種比率 r ∈ Δ²（2-simplex）を CMA-ES で最適化する。

比率のパラメータ化:
    z ∈ ℝ²（制約なし）→ x = (z[0], z[1], 0) → softmax → r
    （第3成分の log を 0 に固定することで自由度を 2 に削減）

beam_runner.py の RatioRefiner を optimizer_v2 向けに移植。
コールバックシグネチャを PhaseInfo に統一している。
"""
from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import cma
import numpy as np

from optimizer.fitness import FitnessEvaluator
from optimizer.genome import build_army_config
from simulator.models import ArmyConfig, HeroConfig
from .config import OptimizerV2Config
from .hero_selector import PhaseInfo
from .scoring import compute_score_from_wins
from simulator.models import BattleResult


class RatioRefiner:
    """CMA-ES on 2D log-ratio space for a fixed hero set (Phase 2).

    マルチスタートで複数の初期比率から探索し、最良を返す。
    評価は evaluate_config_expected（SQLiteキャッシュ付き）を使用。
    """

    def __init__(
        self,
        evaluator: FitnessEvaluator,
        heroes: List[HeroConfig],
        cfg: OptimizerV2Config,
    ):
        self.evaluator = evaluator
        self.heroes = heroes
        self.cfg = cfg

    def run(
        self,
        initial_ratio: Tuple[float, float, float] = (1/3, 1/3, 1/3),
        callback: Optional[Callable[[PhaseInfo], None]] = None,
        candidate_idx: int = 0,
        total_candidates: int = 1,
        prior_score: float = 0.0,
    ) -> Tuple[Tuple[float, float, float], float]:
        """マルチスタート CMA-ES。restarts 回の実行から最良を選ぶ。

        Returns:
            (best_ratio, best_score) — best_score は expected mode スコア
        """
        n_restarts = max(1, self.cfg.cma_restarts)
        total_gens = self.cfg.cma_generations * n_restarts

        # スタートポイント: 指定比率 + Dirichlet ランダム (n_restarts - 1) 個
        rng = np.random.default_rng(self.cfg.seed + candidate_idx)
        start_points: List[Tuple[float, float, float]] = [initial_ratio]
        for _ in range(n_restarts - 1):
            d = rng.dirichlet([2, 2, 2])
            start_points.append((float(d[0]), float(d[1]), float(d[2])))

        best_ratio = initial_ratio
        best_score = float("-inf")
        gen_offset = 0

        for restart_idx, start_ratio in enumerate(start_points):
            ratio, score = self._run_single(
                start_ratio,
                callback=callback,
                candidate_idx=candidate_idx,
                total_candidates=total_candidates,
                prior_score=prior_score,
                gen_offset=gen_offset,
                total_gens=total_gens,
                restart_idx=restart_idx,
                n_restarts=n_restarts,
            )
            if score > best_score:
                best_ratio = ratio
                best_score = score
            gen_offset += self.cfg.cma_generations

        return best_ratio, best_score

    def _run_single(
        self,
        initial_ratio: Tuple[float, float, float],
        callback: Optional[Callable],
        candidate_idx: int,
        total_candidates: int,
        prior_score: float,
        gen_offset: int,
        total_gens: int,
        restart_idx: int,
        n_restarts: int,
    ) -> Tuple[Tuple[float, float, float], float]:
        # log-ratio パラメータ化（第3成分は 0 に固定）
        r = np.clip(np.array(initial_ratio, dtype=float), 0.01, None)
        r = r / r.sum()
        x0 = np.clip(np.log(r)[:2], -4.9, 4.9)

        opts = {
            "popsize":         self.cfg.cma_popsize,
            "maxiter":         self.cfg.cma_generations,
            "verbose":         -9,
            "bounds":          [[-5.0, -5.0], [5.0, 5.0]],
            "tolfun":          0,
            "tolx":            0,
            "tolflatfitness":  self.cfg.cma_generations + 1,
            "tolfunhist":      0,
            "tolstagnation":   self.cfg.cma_generations + 1,
        }

        es = cma.CMAEvolutionStrategy(x0.tolist(), self.cfg.cma_sigma, opts)
        template = self.evaluator.own_template
        generation = 0

        while not es.stop():
            solutions = es.ask()
            fitnesses = []
            for s in solutions:
                ratio = self._decode_ratio(np.array(s))
                config = build_army_config(self.heroes, ratio, template)
                score = self.evaluator.evaluate_config_expected(config)
                fitnesses.append(-score)  # CMA-ES は最小化
            es.tell(solutions, fitnesses)
            generation += 1

            if callback:
                global_gen = gen_offset + generation
                restart_label = f" (restart {restart_idx+1}/{n_restarts})" if n_restarts > 1 else ""
                callback(PhaseInfo(
                    phase="ratio_refine",
                    progress=(candidate_idx + global_gen / total_gens) / total_candidates,
                    message=(
                        f"CMA-ES {candidate_idx+1}/{total_candidates} "
                        f"世代{global_gen}/{total_gens}: "
                        f"best={-es.result.fbest:.4f} σ={es.sigma:.4f}{restart_label}"
                    ),
                    best_label="",
                    best_score=-es.result.fbest,
                    detail={
                        "candidate": candidate_idx + 1,
                        "total":     total_candidates,
                        "generation": global_gen,
                        "sigma":     es.sigma,
                        "prior_score": prior_score,
                        "refined_score": -es.result.fbest,
                        "restart": restart_idx + 1,
                        "restarts": n_restarts,
                    },
                ))

        best_ratio = self._decode_ratio(np.array(es.result.xbest))
        best_score = float(-es.result.fbest)
        return best_ratio, best_score

    @staticmethod
    def _decode_ratio(x2d: np.ndarray) -> Tuple[float, float, float]:
        """2D log-ratio → 3成分の simplex 上の比率。"""
        log3 = np.array([x2d[0], x2d[1], 0.0])
        exp3 = np.exp(log3)
        r = exp3 / exp3.sum()
        return (float(r[0]), float(r[1]), float(r[2]))
