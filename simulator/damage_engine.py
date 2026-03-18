from __future__ import annotations
from math import sqrt
from typing import List, Dict

_SCALE_THRESHOLD = 10_000  # この人数以下は線形、超えたら√スケール


def troop_scale(count: int) -> float:
    """
    兵士スケーリング関数。
    - count ≤ 10,000: 正比例（count をそのまま返す）
    - count > 10,000: √比例（√count × √10000 = √count × 100）
    N=10,000 で連続になる設計。
    """
    if count <= _SCALE_THRESHOLD:
        return float(count)
    return sqrt(count) * sqrt(_SCALE_THRESHOLD)  # = sqrt(count) × 100

from .models import (
    UnitGroup, ArmyStatLayers, DamageMod, SkillModifiers,
    StatBuff, DamageResult,
)

# 相性マトリクス（troops.json の attack_priority に基づく）
COUNTER: Dict[tuple, float] = {
    ("infantry", "lancer"):  1.05,
    ("infantry", "archer"):  0.95,
    ("lancer",   "archer"):  1.05,
    ("lancer",   "infantry"):0.95,
    ("archer",   "infantry"):1.05,
    ("archer",   "lancer"):  0.95,
}


class DamageEngine:
    """
    ユーザー定義ダメージ計算式の完全実装。

    ダメージ =
      N × 兵種相性
      × (eff_atk × eff_leth) / (eff_def × eff_hp)       ← ← Crystal Lance なら ratio × 2
      + 追加ダメージ枠（Crystal Gunpowder, Gwen s2 等）
      × (1 + Σ自軍与ダメ上昇)
      × Π(1 - 各敵与ダメ減少)                            ← Bokan 等
      × (1 + Σ自軍被ダメ上昇デバフ)                       ← Gwen s1
      × Π(1 - 各敵被ダメ減少)                             ← Molly, Sergey, Crystal Shield 等
    """

    def compute(
        self,
        atk_group: UnitGroup,
        def_group: UnitGroup,
        # 攻撃側のバフ
        atk_stat: ArmyStatLayers,
        atk_dmg: DamageMod,          # パッシブ + ターン中の与ダメバフ
        # 防御側のバフ
        def_stat: ArmyStatLayers,
        def_dmg: DamageMod,          # パッシブ + ターン中の被ダメバフ
        # 敵から受けるクロスアーミーデバフ
        enemy_stat_debuff: StatBuff,          # Ling Xue 等（攻撃側ステータスを削減）
        enemy_dealt_debuffs: List[float],     # Bokan 等（Π(1-each)）
        # 攻撃スキル修飾子
        skill_mods: SkillModifiers,
        # 兵種相性ボーナス
        bonus_vs: float = 0.0,
    ) -> DamageResult:

        count = atk_group.effective_count()
        if count <= 0:
            return DamageResult(
                raw_damage=0.0, final_damage=0.0,
                kill_count=0, skill_kill_count=0, skill_procs={},
            )

        # 兵士スケーリング（≤10000: 線形, >10000: √比例）
        scale = troop_scale(count)

        # ============================================================
        # Step 1: 実効ステータス（3層乗算）
        #   層1 base[unit_type]: 研究・島・装備 + hero_base_stats(type-specific) + 兵種パッシブ
        #   層2 hero（全兵種共通）: 英雄スキルバフ。敵デバフ（Ling Xue等）をここで差し引く
        #   層3 gear（全兵種共通）: 専用装備
        # ============================================================
        atk_type = atk_group.unit_type
        def_type = def_group.unit_type

        atk_base = atk_stat.base[atk_type]
        def_base = def_stat.base[def_type]

        eff_atk = atk_group.base_stats.atk \
            * (1.0 + atk_base.atk) \
            * (1.0 + atk_stat.hero.atk - enemy_stat_debuff.atk + atk_dmg.extra_atk_up) \
            * (1.0 + atk_stat.gear.atk)

        eff_leth = atk_group.base_stats.lethality \
            * (1.0 + atk_base.lethality) \
            * (1.0 + atk_stat.hero.lethality - enemy_stat_debuff.lethality + atk_dmg.lethality_up) \
            * (1.0 + atk_stat.gear.lethality)

        eff_def = def_group.base_stats.def_ \
            * (1.0 + def_base.def_) \
            * (1.0 + def_stat.hero.def_) \
            * (1.0 + def_stat.gear.def_)

        eff_hp = def_group.base_stats.hp \
            * (1.0 + def_base.hp) \
            * (1.0 + def_stat.hero.hp) \
            * (1.0 + def_stat.gear.hp)

        # ============================================================
        # Step 2: base_dmg
        #   = N × 兵種相性 × (eff_atk × eff_leth) / (eff_def × eff_hp)
        #   Crystal Lance: ratio × 2（槍兵スキルはBase部分を2倍）
        # ============================================================
        ratio = (eff_atk * eff_leth) / max(eff_def * eff_hp, 1e-9)
        if skill_mods.crystal_lance:
            ratio *= 2.0

        counter = COUNTER.get((atk_group.unit_type, def_group.unit_type), 1.0)
        counter *= (1.0 + bonus_vs)  # T1/T7 兵種相性ボーナス

        base_dmg = scale * counter * ratio

        # ============================================================
        # Step 3: 追加ダメージ枠（Crystal Gunpowder, Gwen s2, Flame Charge）
        #   additional_dmg = base_dmg × Σfrac（各スキル分を加算）
        # ============================================================
        total_additional_frac = sum(skill_mods.additional_fracs)
        total_additional_frac += sum(atk_dmg.additional_fracs)  # ターン中発動分
        additional_dmg = base_dmg * total_additional_frac

        interim_dmg = base_dmg + additional_dmg

        # ============================================================
        # Step 4: ダメージ倍率スキル
        #   × (1 + Σ自軍与ダメ上昇) × Π(1 - 各敵与ダメ減少)
        #   × (1 + Σ被ダメ上昇デバフ) × Π(1 - 各敵被ダメ減少)
        # ============================================================

        # 与ダメ上昇（自軍パッシブ + 兵種別 + ターン中バフ + スキルロール追加分）
        dealt_up = atk_dmg.dealt_up + atk_dmg.dealt_up_per_type.get(atk_type, 0.0) + skill_mods.extra_dealt_up
        interim_dmg *= (1.0 + dealt_up)

        # 与ダメ減少（敵パッシブ Bokan 等 + ターン中デバフ + スキルロール追加分）
        all_dealt_down = list(enemy_dealt_debuffs) + def_dmg.dealt_down + skill_mods.extra_dealt_down
        for v in all_dealt_down:
            interim_dmg *= (1.0 - v)

        # 被ダメ上昇デバフ（Gwen S1, Mia S1 等 + スキルロール追加分）
        taken_up = atk_dmg.taken_up + skill_mods.extra_taken_up
        interim_dmg *= (1.0 + taken_up)

        # 被ダメ減少（敵防御スキル: Molly, Sergey, Crystal Shield等 + スキルロール追加分）
        all_taken_down = list(def_dmg.taken_down) + skill_mods.extra_taken_down
        for v in all_taken_down:
            interim_dmg *= (1.0 - v)

        final_dmg = max(0.0, interim_dmg)

        # キル数計算（防御側HP1ユニット分で割る）
        def_hp_per_unit = def_group.base_stats.hp * (1.0 + def_base.hp) \
                          * (1.0 + def_stat.hero.hp) * (1.0 + def_stat.gear.hp)
        kills = int(final_dmg / max(def_hp_per_unit, 1e-9))
        kills = min(kills, def_group.effective_count())

        # スキル別キル数の按分計算
        # total_frac = 1.0（ベース）+ Σスキル寄与率
        skill_fracs = skill_mods.skill_damage_fracs
        total_frac = 1.0 + sum(skill_fracs.values())
        skill_kills_by_id: Dict[str, int] = {}
        total_skill_kills = 0
        for skill_id, frac in skill_fracs.items():
            sk = int(kills * frac / total_frac)
            skill_kills_by_id[skill_id] = sk
            total_skill_kills += sk

        return DamageResult(
            raw_damage=base_dmg,
            final_damage=final_dmg,
            kill_count=kills,
            skill_kill_count=total_skill_kills,
            skill_procs=skill_kills_by_id,  # {skill_id: kills} として活用
        )

    def get_bonus_vs(self, atk_type: str, def_type: str, troop_skills: list) -> float:
        """T1/T7 の bonus_vs スキルが該当ターゲットに対するボーナス値を返す。"""
        bonus = 0.0
        for skill in troop_skills:
            if skill.get("type") != "bonus_vs":
                continue
            if skill.get("target_unit") == def_type:
                for eff in skill.get("effects", []):
                    if eff["type"] == "damage_up":
                        bonus += eff["value"]
        return bonus
