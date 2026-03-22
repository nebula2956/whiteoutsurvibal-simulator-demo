from __future__ import annotations
import random
from typing import Dict, List, Optional

from .models import Army, SkillState, UNIT_TYPES


class TargetSelector:
    """
    ターゲット選択ルール:
    - 全兵種共通の優先順: 盾（infantry）→ 槍（lancer）→ 弓（archer）
    - 槍兵のみ 20% で弓兵へ貫通上書き（Lancer Ambush）
    """

    # 全兵種共通のターゲット優先順
    PRIORITY = UNIT_TYPES

    def __init__(self, troop_data: Dict):
        pass  # troops.json の attack_priority は使用しない

    def select(
        self,
        atk_type: str,
        enemy: Army,
        troop_skills: list,
        skill_state: SkillState,
        mode: str = "random",
    ) -> str:
        # 槍兵の貫通スキルチェック（20%で弓兵に直接攻撃）
        bypass = self._check_bypass(atk_type, enemy, troop_skills, mode)
        if bypass and enemy.units[bypass].is_alive():
            return bypass

        # 共通優先順: 盾 → 槍 → 弓
        for t in self.PRIORITY:
            if enemy.units[t].is_alive():
                return t

        return self.PRIORITY[0]  # 全滅時フォールバック

    def _check_bypass(
        self,
        atk_type: str,
        enemy: Army,
        troop_skills: list,
        mode: str,
    ) -> Optional[str]:
        """target_override スキルを確認し、バイパスターゲットを返す。"""
        for skill in troop_skills:
            if skill.get("type") != "target_override":
                continue
            if skill.get("trigger") != "on_attack":
                continue
            chance = skill.get("chance", 0.0)
            if mode == "expected":
                # 期待値モード: バイパスは確率が低い場合は不発とする
                fired = chance >= 0.5
            else:
                fired = random.random() < chance
            if fired:
                for eff in skill.get("effects", []):
                    if eff["type"] == "bypass_to":
                        return str(eff["value"])
        return None
