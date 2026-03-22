"""OptimizerV2 の設定 dataclass。"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class OptimizerV2Config:
    # ── Phase 1: Optuna TPE による離散英雄探索 ──────────────────
    n_trials: int = 500          # TPE 試行数
    hero_top_k: int = 20         # Phase 2 に渡す英雄セット候補数
    seed: int = 42               # 再現性用シード

    # ── Phase 2: CMA-ES による兵種比率最適化 ─────────────────────
    cma_sigma: float = 0.7       # CMA-ES 初期 σ
    cma_popsize: int = 14        # CMA-ES 集団サイズ
    cma_generations: int = 25    # CMA-ES 最大世代数
    cma_restarts: int = 2        # マルチスタート回数
    cma_n_sim: int = 50          # Phase 2 評価の MC 回数（expected mode 以外が必要な場合）

    # ── Phase 3: Adaptive Monte Carlo ────────────────────────────
    n_reeval_low: int = 20       # 低忠実度スクリーニングの MC 回数
    n_reeval_high: int = 100     # 高忠実度再評価の MC 回数
    prune_keep_ratio: float = 0.5  # 低忠実度後に残す割合

    # ── スコアリング ──────────────────────────────────────────────
    scoring_mode: str = "lcb"    # "mean" | "robust" | "lcb"
    lcb_beta: float = 1.96       # J_LCB の β（1.96 = 95% CI 下限）
    lambda_var: float = 0.5      # J_robust の λ（μ - λσ）
