from __future__ import annotations
from collections import defaultdict
from typing import Dict, List

from .models import TurnLog, BattleResult, SkillActivation


class BattleLogger:

    def __init__(self):
        self._turn_logs: List[TurnLog] = []
        self._all_activations: List[SkillActivation] = []
        self._skill_kills_total: Dict[str, Dict[str, int]] = {
            "a": defaultdict(int),
            "b": defaultdict(int),
        }
        self._total_kills: Dict[str, Dict[str, int]] = {
            "a": defaultdict(int),
            "b": defaultdict(int),
        }
        # ターン中のA/B別スキルキル
        self._skill_kills_turn: Dict[str, Dict[str, int]] = {
            "a": defaultdict(int),
            "b": defaultdict(int),
        }
        self._current_side: str = "a"

    def set_current_side(self, side: str) -> None:
        self._current_side = side

    def log_turn(
        self,
        turn: int,
        army_a_counts_before: Dict[str, int],
        army_b_counts_before: Dict[str, int],
        a_kills_b: Dict[str, int],
        b_kills_a: Dict[str, int],
        a_kills_by_attacker: Dict[str, Dict[str, int]],
        b_kills_by_attacker: Dict[str, Dict[str, int]],
        activations: List[SkillActivation],
    ) -> None:
        self._turn_logs.append(TurnLog(
            turn=turn,
            army_a_counts=dict(army_a_counts_before),
            army_b_counts=dict(army_b_counts_before),
            a_kills_b=dict(a_kills_b),
            b_kills_a=dict(b_kills_a),
            a_kills_by_attacker=a_kills_by_attacker,
            b_kills_by_attacker=b_kills_by_attacker,
            skill_activations=list(activations),
            a_skill_kills_turn=dict(self._skill_kills_turn["a"]),
            b_skill_kills_turn=dict(self._skill_kills_turn["b"]),
        ))
        self._all_activations.extend(activations)
        for k, v in a_kills_b.items():
            self._total_kills["a"][k] += v
        for k, v in b_kills_a.items():
            self._total_kills["b"][k] += v
        self._skill_kills_turn = {"a": defaultdict(int), "b": defaultdict(int)}

    def record_skill_kills(self, skill_id: str, kills: int) -> None:
        if not skill_id or kills <= 0:
            return
        self._skill_kills_total[self._current_side][skill_id] += kills
        self._skill_kills_turn[self._current_side][skill_id] += kills

    def finalize(self, winner: str, total_turns: int,
                 final_counts_a: dict = None, final_counts_b: dict = None) -> BattleResult:
        return BattleResult(
            turn_logs=self._turn_logs,
            total_kills_by_side={
                "a": dict(self._total_kills["a"]),
                "b": dict(self._total_kills["b"]),
            },
            skill_kill_contributions={
                "a": dict(self._skill_kills_total["a"]),
                "b": dict(self._skill_kills_total["b"]),
            },
            skill_activation_log=self._all_activations,
            winner=winner,
            total_turns=total_turns,
            final_counts_a=final_counts_a,
            final_counts_b=final_counts_b,
        )
