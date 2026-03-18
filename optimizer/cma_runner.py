
from __future__ import annotations

import numpy as np
import cma
from dataclasses import dataclass, field
from typing import Callable, Optional, List

from .genome import HeroPool, genome_label
from .fitness import FitnessEvaluator


@dataclass
class CMAConfig:
    population_size: int = 20
    max_iterations: int = 50
    sigma0: float = 0.5
    n_reeval_simulations: int = 200


@dataclass
class GenerationInfo:
    """1世代分の詳細情報。"""
    gen: int
    total: int
    best_fitness: float
    best_label: str
    sigma: float
    pop_labels: List[str]
    pop_fitnesses: List[float]


@dataclass
class OptimizationResult:
    best_x: np.ndarray
    best_fitness: float
    best_fitness_reeval: float
    history: List[GenerationInfo] = field(default_factory=list)
    label: str = ""
    cache_hits: int = 0
    cache_misses: int = 0


class CMAOptimizer:
    """CMA-ES による兵種比率 + 英雄組み合わせ最適化。"""

    def __init__(self, evaluator: FitnessEvaluator, pool: HeroPool, config: CMAConfig):
        self.evaluator = evaluator
        self.pool = pool
        self.config = config

    def run(self, callback: Optional[Callable[[GenerationInfo], None]] = None) -> OptimizationResult:
        dim = self.pool.genome_dim
        x0 = self._initial_point(dim)

        opts = {
            "popsize": self.config.population_size,
            "maxiter": self.config.max_iterations,
            "verbose": -9,
            "bounds": [self._lower_bounds(), self._upper_bounds()],
            # ノイズ対策: maxiter以外の収束判定を全て無効化
            # （モンテカルロノイズで早期停止すると探索不足になるため）
            "tolfun": 0,
            "tolx": 0,
            "tolflatfitness": self.config.max_iterations + 1,
            "tolfunhist": 0,
            "tolstagnation": self.config.max_iterations + 1,
        }

        es = cma.CMAEvolutionStrategy(x0.tolist(), self.config.sigma0, opts)

        history: List[GenerationInfo] = []
        gen = 0

        while not es.stop():
            gen += 1
            solutions = es.ask()
            fitnesses = [self.evaluator(np.array(s)) for s in solutions]
            es.tell(solutions, fitnesses)

            pop_scores = [-f for f in fitnesses]
            pop_labels = [genome_label(np.array(s), self.pool) for s in solutions]

            best_idx = int(np.argmax(pop_scores))
            gen_best_label = pop_labels[best_idx]

            overall_best_fit = -es.result.fbest

            info = GenerationInfo(
                gen=gen,
                total=self.config.max_iterations,
                best_fitness=overall_best_fit,
                best_label=gen_best_label,
                sigma=es.sigma,
                pop_labels=pop_labels,
                pop_fitnesses=pop_scores,
            )
            history.append(info)

            if callback:
                callback(info)

        best_x = np.array(es.result.xbest)
        noisy_best = -es.result.fbest

        # 最終解を多数回シミュレーションで再評価（キャッシュから既存結果を再利用）
        reeval_score = -self.evaluator(best_x, n_override=self.config.n_reeval_simulations)

        return OptimizationResult(
            best_x=best_x,
            best_fitness=noisy_best,
            best_fitness_reeval=reeval_score,
            history=history,
            label=genome_label(best_x, self.pool),
            cache_hits=self.evaluator.cache_hits,
            cache_misses=self.evaluator.cache_misses,
        )

    def _initial_point(self, dim: int) -> np.ndarray:
        x0 = np.zeros(dim)
        # softmax([0,0,0]) = [1/3, 1/3, 1/3] → 均等比率
        idx = 3
        # リーダー: 均等
        for n in self.pool.leader_dims:
            x0[idx:idx + n] = 0.5
            idx += n
        # メンバー: 各スロット n_m 次元、均等初期化
        n_m = len(self.pool.members)
        for i in range(4):
            x0[idx:idx + n_m] = 0.5
            idx += n_m
        return x0

    def _lower_bounds(self) -> list:
        n_i, n_l, n_a = self.pool.leader_dims
        n_m = len(self.pool.members)
        # 比率3 + リーダー + メンバー4スロット×n_m
        return [-5.0] * 3 + [0.0] * (n_i + n_l + n_a) + [0.0] * (4 * n_m)

    def _upper_bounds(self) -> list:
        n_i, n_l, n_a = self.pool.leader_dims
        n_m = len(self.pool.members)
        return [5.0] * 3 + [1.0] * (n_i + n_l + n_a) + [1.0] * (4 * n_m)
