from __future__ import annotations
from typing import Dict, Any, Tuple, List

from .models import StatBuff, ArmyStatLayers, DamageMod, ArmyConfig

UNIT_TYPES = ["infantry", "lancer", "archer"]


class BuffAggregator:
    """全バフソースを3層 + ダメージ倍率に分類して集約する。"""

    def compute(
        self,
        config: ArmyConfig,
        hero_defs: Dict[str, Any],
        troop_data: Dict[str, Any],
    ) -> Tuple[ArmyStatLayers, DamageMod, StatBuff, List[float]]:
        from .loaders import HeroLoader, TroopLoader

        hero_loader = HeroLoader()
        troop_loader = TroopLoader()

        # ---------- 層1: base（兵種別） ----------
        # 全部隊共通バフ + 兵種別バフを合算して各兵種の base に設定
        per_type = config.base_buff_per_type
        base: Dict[str, StatBuff] = {
            ut: StatBuff(
                atk=config.base_buff.atk + per_type[ut].atk,
                def_=config.base_buff.def_ + per_type[ut].def_,
                hp=config.base_buff.hp + per_type[ut].hp,
                lethality=config.base_buff.lethality + per_type[ut].lethality,
            )
            for ut in UNIT_TYPES
        }

        # 兵種パッシブスキル（body_of_light の def_up など）→ 該当兵種の base に加算
        for unit_type in UNIT_TYPES:
            skills = troop_loader.get_active_skills(
                troop_data, unit_type, config.troop_tier, config.fc_level
            )
            for skill in skills:
                if skill.get("trigger") != "passive":
                    continue
                for eff in skill.get("effects", []):
                    et = eff["type"]
                    if et == "atk_up":
                        base[unit_type].atk += eff["value"]
                    elif et == "def_up":
                        base[unit_type].def_ += eff["value"]
                    elif et == "hp_up":
                        base[unit_type].hp += eff["value"]
                    elif et == "lethality_up":
                        base[unit_type].lethality += eff["value"]

        # 英雄 base_stats → 該当兵種タイプの base のみに加算（ラリーリーダーのみ）
        for hc in config.heroes:
            if hc.hero_id not in hero_defs:
                continue
            if hc.position > 2:
                continue  # メンバーは base_stats 無効
            hdef = hero_defs[hc.hero_id]
            if hdef.get("rarity") == "Epic":
                continue  # Epic はリーダー不可
            troop_type = hdef.get("troop_type")  # "infantry" | "lancer" | "archer"
            if troop_type not in base:
                continue
            bs = hdef.get("base_stats", {})
            base[troop_type].atk += bs.get("atk", 1.0) - 1.0
            base[troop_type].def_ += bs.get("def", 1.0) - 1.0

        # ---------- 層2: hero（全兵種共通・英雄スキルバフ） ----------
        hero = StatBuff()
        damage_mod = DamageMod()
        passive_stat_debuff_to_enemy = StatBuff()
        passive_dealt_debuff_to_enemy: List[float] = []

        # パッシブスキル（ally_all → hero層, enemy_all → 敵デバフ）
        for hc in config.heroes:
            if hc.hero_id not in hero_defs:
                continue
            hdef = hero_defs[hc.hero_id]
            all_skills = hdef.get("skills", [])
            skills_to_use = all_skills if hc.position <= 2 else all_skills[:1]

            for skill in skills_to_use:
                if skill.get("trigger") != "passive":
                    continue
                target = skill.get("target", "")
                for eff in skill.get("effects", []):
                    if target == "ally_all":
                        _add_to_stat_or_dmg(hero, damage_mod, eff["type"], eff["value"], eff.get("unit_type"))
                    elif target == "enemy_all":
                        _add_to_enemy_debuff(
                            passive_stat_debuff_to_enemy,
                            passive_dealt_debuff_to_enemy,
                            eff["type"], eff["value"],
                        )

        # タレントスキル（ラリーリーダーのみ・層2に追加）
        for hc in config.heroes:
            if hc.hero_id not in hero_defs or hc.position > 2:
                continue
            talent = hero_defs[hc.hero_id].get("talent_skill")
            if not talent:
                continue
            for eff in talent.get("effects", []):
                _add_to_stat_or_dmg(hero, damage_mod, eff["type"], eff["value"])

        # ---------- 層3: gear（専用効果・ラリーリーダーのみ・全兵種共通） ----------
        gear = StatBuff()
        for hc in config.heroes:
            if hc.hero_id not in hero_defs or hc.position > 2:
                continue
            hdef = hero_defs[hc.hero_id]
            # base_stats_max（殺傷力/HP）→ 英雄の兵種の base 層に加算
            troop_type = hdef.get("troop_type")
            if hc.gear_level > 0 and troop_type in base:
                bsm = hdef.get("exclusive_gear", {}).get("base_stats_max", {})
                scale = hc.gear_level / 10.0
                base[troop_type].lethality += bsm.get("lethality", 0.0) * scale
                base[troop_type].hp += bsm.get("hp", 0.0) * scale
            # skill_effects → gear（全兵種共通）
            gear_buf = hero_loader.get_gear_buff(hdef, hc.gear_level, config.role)
            # get_gear_buff に含まれる base_stats_max 分を除去
            if hc.gear_level > 0:
                bsm = hdef.get("exclusive_gear", {}).get("base_stats_max", {})
                scale = hc.gear_level / 10.0
                gear_buf.lethality -= bsm.get("lethality", 0.0) * scale
                gear_buf.hp -= bsm.get("hp", 0.0) * scale
            _add_to_stat_only(gear, gear_buf)

        stat_layers = ArmyStatLayers(base=base, hero=hero, gear=gear)
        return stat_layers, damage_mod, passive_stat_debuff_to_enemy, passive_dealt_debuff_to_enemy


# ================================================================
# ヘルパー関数
# ================================================================

def _add_to_stat_or_dmg(stat: StatBuff, dmg: DamageMod, etype: str, value: float, unit_type: str = None) -> None:
    if etype == "atk_up":
        stat.atk += value
    elif etype == "def_up":
        stat.def_ += value
    elif etype == "hp_up":
        stat.hp += value
    elif etype == "lethality_up":
        stat.lethality += value
    elif etype == "damage_dealt_up":
        if unit_type:
            dmg.dealt_up_per_type[unit_type] = dmg.dealt_up_per_type.get(unit_type, 0.0) + value
        else:
            dmg.dealt_up += value
    elif etype == "damage_taken_down":
        dmg.taken_down.append(value)
    elif etype == "damage_taken_up":
        dmg.taken_up += value


def _add_to_enemy_debuff(
    stat_debuff: StatBuff,
    dealt_debuff_list: List[float],
    etype: str,
    value: float,
) -> None:
    if etype == "atk_down":
        stat_debuff.atk += value
    elif etype == "def_down":
        stat_debuff.def_ += value
    elif etype == "hp_down":
        stat_debuff.hp += value
    elif etype == "lethality_down":
        stat_debuff.lethality += value
    elif etype == "damage_dealt_down":
        dealt_debuff_list.append(value)


def _add_to_stat_only(target: StatBuff, src: StatBuff) -> None:
    target.atk += src.atk
    target.def_ += src.def_
    target.hp += src.hp
    target.lethality += src.lethality
