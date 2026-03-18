from __future__ import annotations
import json
import os
from typing import Dict, Any

import numpy as np

from .models import (
    TroopStats, UnitGroup, StatBuff, DamageMod,
    ArmyConfig, Army,
)
from .buff_aggregator import BuffAggregator


def ratios_to_counts(ratios, total: int) -> np.ndarray:
    """比率(float)を合計totalの整数配分に変換する（最大剰余法）。"""
    r = np.array(ratios, dtype=float)
    if r.sum() <= 0:
        r = np.zeros_like(r) + 1e-12
    r = np.clip(r, 0.0, None)
    r = r / r.sum()
    raw = r * total
    floored = np.floor(raw).astype(int)
    deficit = total - floored.sum()
    if deficit > 0:
        fracs = raw - floored
        idx = np.argsort(-fracs)
        for i in range(deficit):
            floored[idx[i]] += 1
    elif deficit < 0:
        surplus = -deficit
        fracs = raw - floored
        idx = np.argsort(fracs)
        for i in range(surplus):
            j = idx[i]
            if floored[j] > 0:
                floored[j] -= 1
    return floored


class TroopLoader:
    """troops.json をロードし、ティア・FCレベルに応じたスタットを提供する。"""

    def load(self, path: str) -> Dict[str, Any]:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return {k: v for k, v in data.items() if not k.startswith("_")}

    def get_stats(self, troop_data: Dict, unit_type: str, tier: str, fc_level: int) -> TroopStats:
        raw = troop_data[unit_type]["tiers"][tier]["fc_levels"][str(fc_level)]
        return TroopStats(
            atk=raw["atk"],
            def_=raw["def"],
            hp=raw["hp"],
            lethality=raw["lethality"],
        )

    def get_attack_priority(self, troop_data: Dict, unit_type: str) -> list:
        return troop_data[unit_type]["attack_priority"]

    def get_active_skills(
        self, troop_data: Dict, unit_type: str, tier: str, fc_level: int
    ) -> list:
        """FCレベル・ティアに応じてアクティブなスキルを解決（overrideチェーン適用）。"""
        raw_skills = troop_data[unit_type].get("skills", {})
        tier_num = int(tier[1:])

        candidates: Dict[str, dict] = {}  # skill_type -> skill_def（後勝ち）

        thresholds = [
            ("T1",   tier_num >= 1),
            ("T7",   tier_num >= 7),
            ("FC3",  fc_level >= 3),
            ("FC5",  fc_level >= 5),
            ("FC8",  fc_level >= 8),
            ("FC10", fc_level >= 10),
        ]

        for key, active in thresholds:
            if active and key in raw_skills:
                for skill in raw_skills[key]:
                    skill_type = skill.get("type", skill["id"])
                    candidates[skill_type] = skill

        return list(candidates.values())


class HeroLoader:
    """heroes.json をロードする。"""

    def load_all(self, heroes_path: str, epic_path: str | None = None) -> Dict[str, Any]:
        with open(heroes_path, encoding="utf-8") as f:
            data = json.load(f)

        result = {k: v for k, v in data.items() if k and not k.startswith("_")}

        # 後方互換: epic_path が指定されていれば追加読み込み
        if epic_path and os.path.exists(epic_path):
            with open(epic_path, encoding="utf-8") as f:
                epic = json.load(f)
            for k, v in epic.items():
                if k and not k.startswith("_"):
                    result[k] = v
        return result

    def get_gear_buff(self, hero_def: Dict, gear_level: int, role: str) -> StatBuff:
        """専用装備レベルに応じたステータスバフを返す（層3用）。"""
        if gear_level == 0 or "exclusive_gear" not in hero_def:
            return StatBuff()

        gear = hero_def["exclusive_gear"]
        buf = StatBuff()

        # base_stats_max: 線形スケール
        base_max = gear.get("base_stats_max", {})
        scale = gear_level / 10.0
        buf.lethality += base_max.get("lethality", 0.0) * scale
        buf.hp += base_max.get("hp", 0.0) * scale

        # skill_effects: step関数でスケール
        # 新形式: "values" [5要素] + 固定式 step=(level-1)//2
        # 旧形式: "values_by_step" + "level_to_step_map"（後方互換）
        for eff in gear.get("skill_effects", []):
            if eff.get("side") and eff["side"] != role:
                continue
            if "values" in eff:
                step_idx = min((gear_level - 1) // 2, 4)
                value = eff["values"][step_idx]
            else:
                step_map = eff.get("level_to_step_map", {})
                step_idx = step_map.get(str(gear_level), 0)
                value = eff["values_by_step"][step_idx]
            etype = eff["effect_type"]
            if etype == "atk_up":
                buf.atk += value
            elif etype == "def_up":
                buf.def_ += value
            elif etype == "hp_up":
                buf.hp += value
            elif etype == "lethality_up":
                buf.lethality += value

        return buf


class ArmyBuilder:
    """ArmyConfig から Army オブジェクトを構築する。"""

    def __init__(self, troop_data: Dict, hero_defs: Dict, aggregator: BuffAggregator):
        self.troop_data = troop_data
        self.hero_defs = hero_defs
        self.aggregator = aggregator
        self.troop_loader = TroopLoader()

    def build(self, config: ArmyConfig, validate: bool = True) -> Army:
        if validate:
            from .validator import validate_army_config
            result = validate_army_config(config, self.hero_defs)
            if not result.valid:
                raise ValueError(
                    "英雄編成エラー:\n" + "\n".join(f"  - {e}" for e in result.errors)
                )

        # 兵士グループ生成
        units: Dict[str, UnitGroup] = {}
        type_list = ["infantry", "lancer", "archer"]
        ratios = [config.troop_ratio.get(ut, 0.0) for ut in type_list]
        counts = ratios_to_counts(ratios, config.rally_capacity)
        for i, unit_type in enumerate(type_list):
            count = int(counts[i])
            stats = self.troop_loader.get_stats(
                self.troop_data, unit_type, config.troop_tier, config.fc_level
            )
            units[unit_type] = UnitGroup(
                unit_type=unit_type,
                tier=config.troop_tier,
                fc_level=config.fc_level,
                count=count,
                base_stats=stats,
                current_hp=float(stats.hp * count),
            )

        # バフ集約（3層 + ダメージ倍率）
        stat_layers, damage_mod, passive_stat_debuff, passive_dealt_debuff = \
            self.aggregator.compute(config, self.hero_defs, self.troop_data)

        return Army(
            units=units,
            stat_layers=stat_layers,
            damage_mod=damage_mod,
            active_damage_mod=DamageMod(),
            heroes=config.heroes,
            role=config.role,
            passive_stat_debuff_to_enemy=passive_stat_debuff,
            passive_dealt_debuff_to_enemy=passive_dealt_debuff,
        )
