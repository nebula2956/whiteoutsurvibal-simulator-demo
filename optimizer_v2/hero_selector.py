"""Phase 1: Optuna TPE による離散英雄選択。

英雄選択（離散）と兵種比率（連続）を分離するアーキテクチャの第一段階。
TPE はカテゴリカル変数を自然に扱えるため、argmax + softmax でエンコードする
旧来の CMA-ES より離散探索に適している。

評価は expected mode 1回（決定論的・SQLiteキャッシュ）で高速に行い、
上位 hero_top_k 件の英雄セットを Phase 2 に渡す。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import optuna

from optimizer.fitness import FitnessEvaluator
from optimizer.genome import HeroPool, OwnedHero, build_army_config
from simulator.models import ArmyConfig, HeroConfig
from .config import OptimizerV2Config

optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.getLogger("optuna").setLevel(logging.WARNING)


@dataclass
class PhaseInfo:
    """進捗コールバックで渡す情報。optimizer_session との互換性のため同名フィールドを持つ。"""
    phase: str
    progress: float          # 0.0 〜 1.0
    message: str
    best_label: str
    best_score: float
    detail: dict


@dataclass
class HeroSet:
    """Phase 1 が確定した英雄セット。Phase 2 で比率を最適化する。"""
    heroes: List[HeroConfig]
    expected_score: float    # expected mode での評価スコア
    label: str               # 可読ラベル


class HeroSelector:
    """Optuna TPE で英雄の組み合わせを探索する（Phase 1）。

    離散変数（英雄選択）を TPE に任せ、
    比率は均等固定で Phase 2 に引き渡す設計。

    探索空間:
        inf_leader  : infantry_leaders からインデックス選択
        lan_leader  : lancer_leaders からインデックス選択
        arc_leader  : archer_leaders からインデックス選択
        member_{0-3}: members からインデックス選択（重複可）
    """

    def __init__(
        self,
        pool: HeroPool,
        evaluator: FitnessEvaluator,
        cfg: OptimizerV2Config,
    ):
        self.pool = pool
        self.evaluator = evaluator
        self.cfg = cfg
        self._template = evaluator.own_template

    def run(
        self,
        progress_cb: Optional[Callable[[PhaseInfo], None]] = None,
    ) -> List[HeroSet]:
        """TPE で n_trials 試行し、上位 hero_top_k 件の HeroSet を返す。"""
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.cfg.seed),
        )

        n_trials = self.cfg.n_trials
        best_score = float("-inf")
        best_label = ""

        def _callback(study: optuna.Study, trial: optuna.Trial) -> None:
            nonlocal best_score, best_label
            if study.best_value > best_score:
                best_score = study.best_value
                best_label = _trial_label(trial, self.pool)
            if progress_cb:
                progress_cb(PhaseInfo(
                    phase="hero_select",
                    progress=trial.number / n_trials,
                    message=f"TPE 試行 {trial.number + 1}/{n_trials}: best={best_score:.4f}",
                    best_label=best_label,
                    best_score=best_score,
                    detail={"trial": trial.number + 1, "n_trials": n_trials,
                            "cache_hits": self.evaluator.cache_hits,
                            "cache_misses": self.evaluator.cache_misses},
                ))

        study.optimize(self._objective, n_trials=n_trials, callbacks=[_callback], show_progress_bar=False)

        return self._extract_top_k(study)

    def _objective(self, trial: optuna.Trial) -> float:
        config = self._suggest_config(trial)
        return self.evaluator.evaluate_config_expected(config)

    def _suggest_config(self, trial: optuna.Trial) -> ArmyConfig:
        """Optuna に各スロットを categorical として提案させ、ArmyConfig を構築する。"""
        inf_idx = trial.suggest_categorical("inf_leader", list(range(len(self.pool.infantry_leaders))))
        lan_idx = trial.suggest_categorical("lan_leader", list(range(len(self.pool.lancer_leaders))))
        arc_idx = trial.suggest_categorical("arc_leader", list(range(len(self.pool.archer_leaders))))
        m_idxs = [
            trial.suggest_categorical(f"member_{i}", list(range(len(self.pool.members))))
            for i in range(4)
        ]

        heroes = _build_heroes(
            self.pool.infantry_leaders[inf_idx],
            self.pool.lancer_leaders[lan_idx],
            self.pool.archer_leaders[arc_idx],
            [self.pool.members[i] for i in m_idxs],
        )
        # 比率は均等固定（Phase 2 で CMA-ES が最適化する）
        ratio = (1 / 3, 1 / 3, 1 / 3)
        return build_army_config(heroes, ratio, self._template)

    def _extract_top_k(self, study: optuna.Study) -> List[HeroSet]:
        """試行結果から上位 hero_top_k 件の重複なし HeroSet を返す。"""
        seen: set = set()
        result: List[HeroSet] = []

        # スコア降順でソート
        trials = sorted(
            [t for t in study.trials if t.value is not None],
            key=lambda t: t.value,
            reverse=True,
        )

        for trial in trials:
            if len(result) >= self.cfg.hero_top_k:
                break
            heroes = self._heroes_from_trial(trial)
            key = _heroes_key(heroes)
            if key in seen:
                continue
            seen.add(key)
            result.append(HeroSet(
                heroes=heroes,
                expected_score=trial.value,
                label=_trial_label(trial, self.pool),
            ))

        return result

    def _heroes_from_trial(self, trial: optuna.Trial) -> List[HeroConfig]:
        inf_idx = trial.params["inf_leader"]
        lan_idx = trial.params["lan_leader"]
        arc_idx = trial.params["arc_leader"]
        m_idxs = [trial.params[f"member_{i}"] for i in range(4)]
        return _build_heroes(
            self.pool.infantry_leaders[inf_idx],
            self.pool.lancer_leaders[lan_idx],
            self.pool.archer_leaders[arc_idx],
            [self.pool.members[i] for i in m_idxs],
        )


# ── ヘルパー ──────────────────────────────────────────────────────────────

def _build_heroes(
    inf: OwnedHero,
    lan: OwnedHero,
    arc: OwnedHero,
    members: List[str],
) -> List[HeroConfig]:
    return [
        HeroConfig(inf.hero_id, inf.gear_level, True,  0),
        HeroConfig(lan.hero_id, lan.gear_level, False, 1),
        HeroConfig(arc.hero_id, arc.gear_level, False, 2),
        *[HeroConfig(m, 0, False, 3 + i) for i, m in enumerate(members)],
    ]


def _heroes_key(heroes: List[HeroConfig]) -> Tuple:
    return tuple((h.hero_id, h.gear_level, h.position) for h in heroes)


def _trial_label(trial: optuna.Trial, pool: HeroPool) -> str:
    """試行パラメータを可読ラベルに変換する。"""
    try:
        inf = pool.infantry_leaders[trial.params["inf_leader"]].hero_id
        lan = pool.lancer_leaders[trial.params["lan_leader"]].hero_id
        arc = pool.archer_leaders[trial.params["arc_leader"]].hero_id
        ms = "/".join(pool.members[trial.params[f"member_{i}"]] for i in range(4))
        return f"L:{inf}/{lan}/{arc} M:{ms}"
    except Exception:
        return ""
