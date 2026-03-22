"""OptimizerV2: Bilevel Mixed-Integer Bayesian Black-Box Optimization。

問題の定式化:
    x = (H, r)
        H : 英雄集合（離散）  — 7スロット × N候補
        r : 兵種比率（連続）  — r ∈ Δ²

    上位階層: H* = argmax_H [ max_r J(H, r) ]  ← 英雄の離散BO (TPE)
    下位階層: r* = argmax_r  J(H*, r)           ← 比率の連続最適化 (CMA-ES)

    最終スコア:
        J_mean(x)   = μ(x)
        J_robust(x) = μ(x) - λ·σ(x)
        J_LCB(x)    = μ(x) - β·SE(x)

実行フロー:
    Phase 1  HeroSelector   — Optuna TPE で英雄を探索 (expected mode)
    Phase 2  RatioRefiner   — CMA-ES で比率を精緻化 (expected mode)
    Phase 3  AdaptiveMC     — 多忠実度MC + 共通乱数で最終評価 (random mode)
"""
from __future__ import annotations

import queue
import threading
import traceback
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

from optimizer.fitness import FitnessEvaluator, FitnessWeights
from optimizer.genome import HeroPool, build_army_config
from simulator.models import ArmyConfig, HeroConfig
from .config import OptimizerV2Config
from .hero_selector import HeroSelector, HeroSet, PhaseInfo
from .ratio_refiner import RatioRefiner
from .adaptive_mc import AdaptiveMC, ReevalResult


@dataclass
class V2Result:
    """OptimizerV2 の最終結果。"""
    best_config: ArmyConfig
    best_score: float                    # J_LCB / J_robust / mean
    best_mu: float                       # 平均勝率（draw=0.5）
    best_sigma: float                    # 標準偏差
    best_ci: Tuple[float, float]         # 95% CI (lo, hi)
    top_results: List[ReevalResult]      # 上位候補（score 降順）
    phase1_hero_sets: List[HeroSet]      # Phase 1 の英雄セット候補
    cache_hits: int = 0
    cache_misses: int = 0


class OptimizerV2:
    """Bilevel 最適化の 3 フェーズを統括するメインクラス。

    Args:
        evaluator:  FitnessEvaluator（シミュレーターとキャッシュを内包）
        pool:       HeroPool（所持英雄プール）
        cfg:        OptimizerV2Config
    """

    def __init__(
        self,
        evaluator: FitnessEvaluator,
        pool: HeroPool,
        cfg: OptimizerV2Config,
    ):
        self.evaluator = evaluator
        self.pool = pool
        self.cfg = cfg

    def run(
        self,
        callback: Optional[Callable[[PhaseInfo], None]] = None,
    ) -> V2Result:
        """3フェーズを順に実行して V2Result を返す。"""

        # ── Phase 1: Optuna TPE で英雄選択 ────────────────────────
        selector = HeroSelector(self.pool, self.evaluator, self.cfg)
        hero_sets = selector.run(progress_cb=callback)

        if not hero_sets:
            raise RuntimeError("Phase 1: 英雄セットが見つかりませんでした。")

        if callback:
            callback(PhaseInfo(
                phase="hero_select",
                progress=1.0,
                message=f"Phase 1 完了: 上位 {len(hero_sets)} 英雄セットを選択",
                best_label=hero_sets[0].label,
                best_score=hero_sets[0].expected_score,
                detail={"phase1_top_k": len(hero_sets)},
            ))

        # ── Phase 2: CMA-ES で比率精緻化 ──────────────────────────
        refined: List[Tuple[ArmyConfig, float, str]] = []
        total = len(hero_sets)

        for i, hero_set in enumerate(hero_sets):
            refiner = RatioRefiner(self.evaluator, hero_set.heroes, self.cfg)
            best_ratio, best_score = refiner.run(
                initial_ratio=(1/3, 1/3, 1/3),
                callback=callback,
                candidate_idx=i,
                total_candidates=total,
                prior_score=hero_set.expected_score,
            )
            config = build_army_config(hero_set.heroes, best_ratio, self.evaluator.own_template)
            label = (f"{hero_set.label} "
                     f"盾{best_ratio[0]*100:.0f}%槍{best_ratio[1]*100:.0f}%弓{best_ratio[2]*100:.0f}%")
            refined.append((config, best_score, label))

        if callback:
            best_ref = max(refined, key=lambda x: x[1])
            callback(PhaseInfo(
                phase="ratio_refine",
                progress=1.0,
                message=f"Phase 2 完了: best expected score = {best_ref[1]:.4f}",
                best_label=best_ref[2],
                best_score=best_ref[1],
                detail={"phase2_candidates": len(refined)},
            ))

        # ── Phase 3: Adaptive MC で最終評価 ───────────────────────
        amc = AdaptiveMC(self.evaluator, self.cfg)
        final_results = amc.run(refined, callback=callback)

        if not final_results:
            raise RuntimeError("Phase 3: 再評価結果が空です。")

        best = final_results[0]

        if callback:
            callback(PhaseInfo(
                phase="reeval",
                progress=1.0,
                message=(
                    f"最適化完了: J={best.score:.4f} "
                    f"μ={best.mu:.3f}±{best.sigma:.3f} "
                    f"CI=[{best.ci_lo:.3f}, {best.ci_hi:.3f}]"
                ),
                best_label=best.label,
                best_score=best.score,
                detail={
                    "mu":    best.mu,
                    "sigma": best.sigma,
                    "ci_lo": best.ci_lo,
                    "ci_hi": best.ci_hi,
                    "n_sims": best.n_sims,
                },
            ))

        return V2Result(
            best_config=best.config,
            best_score=best.score,
            best_mu=best.mu,
            best_sigma=best.sigma,
            best_ci=(best.ci_lo, best.ci_hi),
            top_results=final_results,
            phase1_hero_sets=hero_sets,
            cache_hits=self.evaluator.cache_hits,
            cache_misses=self.evaluator.cache_misses,
        )


# ── Streamlit セッション管理 ────────────────────────────────────────────────

class V2Session:
    """Streamlit rerun 耐性のある V2 セッション管理。

    OptimizerSession（既存）と同じスレッドモデルを採用。
    """

    # フェーズ → 全体進捗の重み
    _PHASE_WEIGHTS = {
        "hero_select":  (0.0, 0.6),   # Phase 1: TPE は試行数多い
        "ratio_refine": (0.6, 0.85),  # Phase 2: CMA-ES
        "reeval_low":   (0.85, 0.94), # Phase 3a: 低忠実度
        "reeval":       (0.94, 1.0),  # Phase 3b: 高忠実度
    }

    def __init__(self):
        self.running = False
        self.finished = False
        self.error: Optional[str] = None
        self.result: Optional[V2Result] = None
        self._info_queue: queue.Queue[PhaseInfo] = queue.Queue()
        self._latest_info: Optional[PhaseInfo] = None
        self._thread: Optional[threading.Thread] = None
        self.score_history: list = []

    def start(
        self,
        evaluator: FitnessEvaluator,
        pool: HeroPool,
        cfg: OptimizerV2Config,
    ) -> None:
        if self.running:
            return
        self.running = True
        self.finished = False
        self.error = None
        self.result = None
        self.score_history = []
        self._thread = threading.Thread(
            target=self._run,
            args=(evaluator, pool, cfg),
            daemon=True,
        )
        self._thread.start()

    def _run(self, evaluator: FitnessEvaluator, pool: HeroPool, cfg: OptimizerV2Config) -> None:
        try:
            opt = OptimizerV2(evaluator, pool, cfg)
            self.result = opt.run(callback=lambda info: self._info_queue.put(info))
            self.finished = True
        except Exception as e:
            self.error = f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}"
            self.finished = True
        finally:
            self.running = False

    def get_latest_info(self) -> Optional[PhaseInfo]:
        """非ブロッキングでキューを消費し、スコア履歴を更新する。"""
        info = None
        while not self._info_queue.empty():
            try:
                info = self._info_queue.get_nowait()
                lo, hi = self._PHASE_WEIGHTS.get(info.phase, (0.0, 1.0))
                overall = lo + (hi - lo) * min(info.progress, 1.0)
                if info.best_score > 0:
                    self.score_history.append((round(overall, 4), round(info.best_score, 4)))
            except queue.Empty:
                break
        if info is not None:
            self._latest_info = info
        return self._latest_info
