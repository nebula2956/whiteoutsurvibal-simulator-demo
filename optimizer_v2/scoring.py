"""リスク感度付きスコアリング。

J_mean(x)   = μ(x)
J_robust(x) = μ(x) - λ·σ(x)          ← 分散ペナルティ付き
J_LCB(x)    = μ(x) - β·SE(x)          ← 下側信頼境界（推奨デフォルト）

引き分けは 0.5 点として扱う（0.0 = 負け、0.5 = 引き分け、1.0 = 勝ち）。
"""
from __future__ import annotations

from typing import List

import numpy as np

from simulator.models import BattleResult
from optimizer.fitness import FitnessWeights


def battle_win_score(result: BattleResult) -> float:
    """1試合の勝敗を [0, 0.5, 1.0] に変換する。"""
    if result.winner == "a":
        return 1.0
    if result.winner == "draw":
        return 0.5
    return 0.0


def compute_score(
    results: List[BattleResult],
    own_cap: int,
    enemy_cap: int,
    weights: FitnessWeights,
    mode: str = "lcb",
    lcb_beta: float = 1.96,
    lambda_var: float = 0.5,
) -> float:
    """複数試合の BattleResult からリスク感度付きスコアを計算する。

    Args:
        results:     バトル結果リスト
        own_cap:     自軍ラリー容量（生存率の分母）
        enemy_cap:   敵軍ラリー容量（キル率の分母）
        weights:     各指標の重み
        mode:        "mean" | "robust" | "lcb"
        lcb_beta:    J_LCB の β（デフォルト 1.96 = 95%CI 下限）
        lambda_var:  J_robust の λ（デフォルト 0.5）

    Returns:
        スコア（高いほど良い）
    """
    if not results:
        return 0.0

    wins = [battle_win_score(r) for r in results]
    survivals = [
        sum(r.final_counts_a.values()) / max(own_cap, 1)
        if r.final_counts_a else 0.0
        for r in results
    ]
    kills = [
        sum(r.total_kills_by_side.get("a", {}).values()) / max(enemy_cap, 1)
        for r in results
    ]

    mu_w = float(np.mean(wins))
    mu_s = float(np.mean(survivals))
    mu_k = float(np.mean(kills))

    # 基本スコア（重み付き和）
    base = (weights.win_rate * mu_w
            + weights.survival_ratio * mu_s
            + weights.avg_kills * mu_k)

    if mode == "mean":
        return base

    # 分散・標準誤差
    sigma = float(np.std(wins, ddof=1)) if len(wins) > 1 else 0.0
    se = sigma / np.sqrt(len(wins)) if len(wins) > 0 else 0.0

    if mode == "robust":
        # J_robust = μ - λσ  （分散が大きいほどペナルティ）
        return base - lambda_var * sigma
    else:
        # J_LCB = μ - β·SE  （サンプル数が少ないほど保守的）
        return base - lcb_beta * se


def compute_score_from_wins(
    wins: List[float],
    mode: str = "lcb",
    lcb_beta: float = 1.96,
    lambda_var: float = 0.5,
) -> float:
    """win スコアリストのみから J_LCB / J_robust を計算する（簡易版）。"""
    if not wins:
        return 0.0
    mu = float(np.mean(wins))
    sigma = float(np.std(wins, ddof=1)) if len(wins) > 1 else 0.0
    se = sigma / np.sqrt(len(wins))
    if mode == "robust":
        return mu - lambda_var * sigma
    elif mode == "lcb":
        return mu - lcb_beta * se
    return mu
