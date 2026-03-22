from __future__ import annotations
import random
from typing import Dict, List, Tuple, Literal

from .models import (
    Army, UnitGroup, DamageMod, SkillModifiers, SkillState,
    ActiveEffect, SkillActivation, BattleState, DecayingDamageBuff,
)
from typing import Optional


def _get_hero_skills(hc, hdef: dict) -> list:
    """
    ラリーリーダー（position <= 2）: 全スキル使用
    参加メンバー（position >= 3）: スキル1（索引0）のみ
    """
    all_skills = hdef.get("skills", [])
    if hc.position <= 2:
        return all_skills
    return all_skills[:1]


def _get_armies(state: "BattleState", side: str):
    """side から (自軍, 敵軍, 敵side) を返す。"""
    if side == "a":
        return state.army_a, state.army_b, "b"
    return state.army_b, state.army_a, "a"


def _add_skill_frac(mods: "SkillModifiers", skill_id: str, frac: float) -> None:
    """additional_fracs と skill_damage_fracs を同時に更新する。"""
    mods.additional_fracs.append(frac)
    mods.skill_damage_fracs[skill_id] = mods.skill_damage_fracs.get(skill_id, 0) + frac


class SkillEngine:
    """スキルトリガーを管理し、バフ/ダメージ修飾子を生成するエンジン。"""

    def __init__(self, mode: Literal["random", "expected"] = "random"):
        self.mode = mode

    def _roll(self, chance: float) -> bool:
        """効果適用の可否。期待値モードでは常にTrue（値はscaleで調整）。"""
        if self.mode == "random":
            return random.random() < chance
        return True  # 期待値モード: 常に発動

    def _should_log(self, chance: float) -> bool:
        """発動ログを記録するかどうか。
        - ランダムモード: _roll が既に通過済みなので常に True
        - 期待値モード: 確率ロールでログ件数を正規化
        """
        if self.mode == "random":
            return True  # _roll(chance) 通過後なので2重チェック不要
        return random.random() < chance

    def _scale(self, value: float, chance: float) -> float:
        if self.mode == "expected":
            return value * chance
        return value

    def _scale_with_duration(self, value: float, chance: float, duration: int) -> float:
        """期待値モード: duration考慮のスケーリング。
        duration > 1 なら定常状態確率 1-(1-p)^d を使用。"""
        if self.mode != "expected":
            return value
        if chance is None or chance >= 1.0:
            return value
        if duration <= 1:
            return value * chance
        uptime = 1.0 - (1.0 - chance) ** duration
        return value * uptime

    # ================================================================
    # on_battle_start: バトル開始時（減衰バフ初期化）
    # ================================================================
    def on_battle_start(
        self,
        state: BattleState,
        hero_defs: Dict,
    ) -> None:
        """passive_decaying スキルを SkillState に登録する。"""
        for side, army in [("a", state.army_a), ("b", state.army_b)]:
            buf_list = (state.skill_state.decaying_buffs_a
                        if side == "a" else state.skill_state.decaying_buffs_b)
            for hc in army.heroes:
                if hc.hero_id not in hero_defs:
                    continue
                if hc.position > 2:
                    continue  # メンバーはスキル1のみ（passive_decayingはS2以降が多い）
                hdef = hero_defs[hc.hero_id]
                for skill in _get_hero_skills(hc, hdef):
                    if skill.get("trigger") != "passive_decaying":
                        continue
                    for eff in skill.get("effects", []):
                        if eff["type"] != "decaying_damage_dealt_up":
                            continue
                        buf_list.append(DecayingDamageBuff(
                            skill_id=skill["id"],
                            unit_type=eff["unit_type"],
                            current_value=eff["initial_value"],
                            decay_rate=eff["decay_rate"],
                            remaining_attacks=eff["max_attacks"],
                        ))

    def consume_decaying_buff(
        self,
        state: BattleState,
        side: str,
        atk_unit_type: str,
    ) -> float:
        """
        該当兵種の減衰バフを消費して current_value を返す。
        攻撃後に decay_rate 倍して remaining_attacks を -1 する。
        """
        buf_list = (state.skill_state.decaying_buffs_a
                    if side == "a" else state.skill_state.decaying_buffs_b)
        total_bonus = 0.0
        expired = []
        for buf in buf_list:
            if buf.unit_type != atk_unit_type:
                continue
            if buf.remaining_attacks <= 0:
                expired.append(buf)
                continue
            total_bonus += buf.current_value
            buf.current_value *= buf.decay_rate
            buf.remaining_attacks -= 1
            if buf.remaining_attacks <= 0:
                expired.append(buf)
        for e in expired:
            if e in buf_list:
                buf_list.remove(e)
        return total_bonus

    # ================================================================
    # on_turn_start: 毎ターン開始時の確率スキル
    # ================================================================
    def on_turn_start(
        self,
        state: BattleState,
        hero_defs: Dict,
        side: str,
    ) -> List[SkillActivation]:
        """on_turn_start / every_n_turns スキルをロールして active_damage_mod を更新。"""
        activations: List[SkillActivation] = []
        army, _, _ = _get_armies(state, side)

        # 前ターンの一時バフをデクリメント
        self._tick_turn_effects(state, side)

        for hc in army.heroes:
            if hc.hero_id not in hero_defs:
                continue
            hdef = hero_defs[hc.hero_id]
            skills_to_use = _get_hero_skills(hc, hdef)

            for skill in skills_to_use:
                trigger = skill.get("trigger")

                # on_turn_start
                if trigger == "on_turn_start":
                    chance = skill.get("chance", 1.0)
                    if not self._roll(chance):
                        continue
                    duration = skill.get("duration", 1)
                    for eff in skill.get("effects", []):
                        val = self._scale_with_duration(eff["value"], chance, duration)
                        self._apply_to_active_dmg(army, eff["type"], val, duration, skill["id"], state, side)
                    # 期待値モードでも確率ロールでログ件数を正規化
                    if self._should_log(chance):
                        activations.append(SkillActivation(
                            turn=state.turn, skill_id=skill["id"],
                            unit_type="hero", triggered_by="on_turn_start",
                            effect_summary=skill.get("description_ja", ""),
                            side=side,
                        ))

                # every_n_turns
                elif trigger == "every_n_turns":
                    every_n = skill.get("every_n", 4)
                    counter_key = f"{side}_{skill['id']}_turns"
                    cnt = state.skill_state.turn_counters.get(counter_key, 0) + 1
                    state.skill_state.turn_counters[counter_key] = cnt
                    if cnt % every_n != 0:
                        continue
                    duration = skill.get("duration", 1)
                    for eff in skill.get("effects", []):
                        self._apply_to_active_dmg(army, eff["type"], eff["value"], duration, skill["id"], state, side)
                    activations.append(SkillActivation(
                        turn=state.turn, skill_id=skill["id"],
                        unit_type="hero", triggered_by="every_n_turns",
                        effect_summary=skill.get("description_ja", ""),
                        side=side,
                    ))

        return activations

    # ================================================================
    # on_every_attack: 攻撃毎確実発動（Gwen s1: 被ダメ上昇デバフ）
    # ================================================================
    def on_every_attack(
        self,
        atk_group: UnitGroup,
        def_group: UnitGroup,
        state: BattleState,
        side: str,
        hero_defs: Dict,
    ) -> Tuple[SkillModifiers, List[SkillActivation]]:
        """on_every_attack スキルを処理。SkillModifiers と発動ログを返す。"""
        mods = SkillModifiers()
        activations: List[SkillActivation] = []
        army, _, _ = _get_armies(state, side)

        for hc in army.heroes:
            if hc.hero_id not in hero_defs:
                continue
            hdef = hero_defs[hc.hero_id]
            skills_to_use = _get_hero_skills(hc, hdef)

            for skill in skills_to_use:
                if skill.get("trigger") != "on_every_attack":
                    continue
                # unit_restriction チェック（Akmos S3: infantry のみ）
                unit_restriction = skill.get("unit_restriction")
                if unit_restriction and unit_restriction != atk_group.unit_type:
                    continue
                for eff in skill.get("effects", []):
                    etype = eff["type"]
                    if etype == "damage_taken_up":
                        duration_turns = eff.get("duration_turns")
                        if duration_turns:
                            # duration_turns指定: active_dmgで持続管理（Akmos S3）
                            _, enemy_army, enemy_side = _get_armies(state, side)
                            self._apply_to_active_dmg(
                                enemy_army, "damage_taken_up", eff["value"],
                                duration_turns, skill["id"], state, enemy_side,
                            )
                        else:
                            # duration_hits=1: この攻撃の被ダメ上昇（Gwen S1）
                            mods.extra_taken_up += eff["value"]
                    elif etype == "extra_damage":
                        # Akmos S3: 追加ダメージ
                        _add_skill_frac(mods, skill["id"], eff["value"])
                activations.append(SkillActivation(
                    turn=state.turn, skill_id=skill["id"],
                    unit_type="hero", triggered_by="on_every_attack",
                    effect_summary=skill.get("description_ja", ""),
                    side=side,
                ))

        return mods, activations

    # ================================================================
    # on_attack: 攻撃時スキル
    # Crystal Lance, Crystal Gunpowder, Archer Volley, every_n_attacks
    # ================================================================
    def on_attack(
        self,
        atk_group: UnitGroup,
        def_group: UnitGroup,
        state: BattleState,
        side: str,
        hero_defs: Dict,
        troop_skills: list,
    ) -> Tuple[SkillModifiers, List[SkillActivation]]:
        mods = SkillModifiers()
        activations: List[SkillActivation] = []

        # ---------- 兵種スキル ----------
        gunpowder_fired = False
        for skill in troop_skills:
            if skill.get("trigger") != "on_attack":
                continue
            chance = skill.get("chance", 1.0)
            if not self._roll(chance):
                continue

            for eff in skill.get("effects", []):
                etype = eff["type"]
                if etype == "damage_multiply":
                    if self.mode == "expected":
                        # 期待値: chance*2x + (1-chance)*1x → 追加分はchance
                        _add_skill_frac(mods, skill["id"], chance)
                    else:
                        mods.crystal_lance = True
                        # Crystal Lance: base_dmgを1倍分追加（ratio×2 → +ratio相当）
                        mods.skill_damage_fracs[skill["id"]] = \
                            mods.skill_damage_fracs.get(skill["id"], 0) + 1.0
                elif etype == "extra_damage":
                    val = self._scale(eff["value"], chance)
                    _add_skill_frac(mods, skill["id"], val)
                    gunpowder_fired = True
                elif etype == "double_attack":
                    if self.mode == "expected":
                        # 期待値: chance*2回 + (1-chance)*1回 → 追加分はchance
                        _add_skill_frac(mods, skill["id"], chance)
                    else:
                        mods.double_attack = True
                        # 二回攻撃: 1回分の追加ダメージ相当（frac=1.0）
                        mods.skill_damage_fracs[skill["id"]] = \
                            mods.skill_damage_fracs.get(skill["id"], 0) + 1.0
                elif etype == "bypass_to":
                    mods.bypass_target = str(eff["value"])
                    # 迂回: ターゲット変更なのでダメージ寄与として記録
                    mods.skill_damage_fracs[skill["id"]] = \
                        mods.skill_damage_fracs.get(skill["id"], 0) + 0.5

            if self._should_log(chance):
                activations.append(SkillActivation(
                    turn=state.turn, skill_id=skill["id"],
                    unit_type=atk_group.unit_type, triggered_by="on_attack",
                    effect_summary=skill.get("description_ja", ""),
                    side=side,
                ))

        # Flame Charge: Gunpowder 発動時の追加ダメージ
        if gunpowder_fired:
            for skill in troop_skills:
                if skill.get("type") != "flame_charge":
                    continue
                for eff in skill.get("effects", []):
                    if eff["type"] == "extra_damage_when_gunpowder_active":
                        _add_skill_frac(mods, skill["id"], eff["value"])

        # ---------- 英雄スキル: on_attack（Norah S2, Hector S3 等）----------
        atk_type_for_hero = atk_group.unit_type
        hero_army, _, _ = _get_armies(state, side)
        for hc in hero_army.heroes:
            if hc.hero_id not in hero_defs:
                continue
            hdef = hero_defs[hc.hero_id]
            for skill in _get_hero_skills(hc, hdef):
                if skill.get("trigger") != "on_attack":
                    continue
                unit_restriction = skill.get("unit_restriction")
                if unit_restriction and unit_restriction != atk_type_for_hero:
                    continue
                chance = skill.get("chance", 1.0)
                if not self._roll(chance):
                    continue
                target = skill.get("target", "primary_target")
                for eff in skill.get("effects", []):
                    etype = eff["type"]
                    if etype == "extra_damage":
                        if target == "enemy_all":
                            # Norah S2等: 全敵ユニット種にAOEダメージ
                            mods.aoe_fracs.append(self._scale(eff["value"], chance))
                        else:
                            _add_skill_frac(mods, skill["id"], self._scale(eff["value"], chance))
                    elif etype == "extra_damage_multiplicative":
                        # Mia S2: 乗算での追加ダメージ（damage_dealt_up 枠）
                        mods.extra_dealt_up += self._scale(eff["value"], chance)
                    elif etype == "damage_multiply":
                        if self.mode == "expected":
                            _add_skill_frac(mods, skill["id"], chance)
                        else:
                            mods.crystal_lance = True
                            mods.skill_damage_fracs[skill["id"]] = \
                                mods.skill_damage_fracs.get(skill["id"], 0) + 1.0
                    elif etype == "damage_dealt_down":
                        # 敵軍への与ダメ減少デバフ（Alonzo S2, Greg S2 等）
                        _, enemy_army, enemy_side = _get_armies(state, side)
                        val = self._scale(eff["value"], chance)
                        duration = skill.get("duration", 1)
                        self._apply_to_active_dmg(
                            enemy_army, "damage_dealt_down", val,
                            duration, skill["id"], state, enemy_side,
                        )
                    elif etype == "damage_taken_up":
                        # 対象への被ダメ上昇デバフ（Mia S1 等）→ この攻撃の被ダメ上昇枠
                        val = self._scale(eff["value"], chance)
                        mods.extra_taken_up += val
                if self._should_log(chance):
                    activations.append(SkillActivation(
                        turn=state.turn, skill_id=skill["id"],
                        unit_type="hero", triggered_by="on_attack",
                        effect_summary=skill.get("description_ja", ""),
                        side=side,
                    ))

        # ---------- 英雄スキル: every_n_attacks ----------
        atk_type = atk_group.unit_type
        counter_key = f"{side}_{atk_type}_attacks"
        cnt = state.skill_state.unit_attack_counters.get(counter_key, 0) + 1
        state.skill_state.unit_attack_counters[counter_key] = cnt

        army, _, _ = _get_armies(state, side)
        for hc in army.heroes:
            if hc.hero_id not in hero_defs:
                continue
            hdef = hero_defs[hc.hero_id]
            skills_to_use = _get_hero_skills(hc, hdef)

            for skill in skills_to_use:
                if skill.get("trigger") != "every_n_attacks":
                    continue
                # unit_restriction チェック（Gwen s3: archer のみ）
                unit_restriction = skill.get("unit_restriction")
                if unit_restriction and unit_restriction != atk_type:
                    continue
                every_n = skill.get("every_n", 5)
                if cnt % every_n != 0:
                    continue

                # attack_skip フラグ（Akmos S1: 攻撃を休止）
                if skill.get("attack_skip"):
                    mods.attack_skip = True

                for eff in skill.get("effects", []):
                    etype = eff["type"]
                    if etype == "attack_count_extra_damage":
                        _add_skill_frac(mods, skill["id"], eff["value"])
                    elif etype == "target_vulnerability":
                        # Gwen S2: 次の攻撃の追加ダメージ枠に加算（サイド別）
                        state.skill_state.vulnerability[side][def_group.unit_type] = eff["value"]
                    elif etype == "aoe_extra_damage":
                        # Gwen s3: base_dmg × frac を全敵ユニット種に独立加算（AOE）
                        mods.aoe_fracs.append(eff["value"])
                    elif etype == "damage_taken_down":
                        # Akmos S1: 被ダメ軽減をactive_dmgに追加
                        duration = skill.get("duration", 1)
                        self._apply_to_active_dmg(
                            army, "damage_taken_down", eff["value"],
                            duration, skill["id"], state, side,
                        )
                    elif etype == "atk_up" and eff.get("stackable") and eff.get("permanent"):
                        # Rion S3: 永続スタック攻撃力バフ
                        target = skill.get("target", "")
                        if target == "self_unit":
                            # 該当兵種の active_damage_mod.extra_atk_up に永続加算
                            army.active_damage_mod.extra_atk_up += eff["value"]

                activations.append(SkillActivation(
                    turn=state.turn, skill_id=skill["id"],
                    unit_type="hero", triggered_by="every_n_attacks",
                    effect_summary=skill.get("description_ja", ""),
                    side=side,
                ))

        return mods, activations

    # ================================================================
    # on_receive_damage: Crystal Shield / White Heat Field
    # 結果を SkillModifiers の extra_taken_down に追加して返す
    # ================================================================
    def on_receive_damage(
        self,
        def_group: UnitGroup,
        state: BattleState,
        side: str,
        troop_skills: list,
        hero_defs: Dict = None,
    ) -> Tuple[List[float], bool, List[SkillActivation]]:
        """発動した防御スキルの被ダメ減少値リスト・回避フラグ・発動ログを返す。"""
        taken_down_values: List[float] = []
        dodge = False
        activations: List[SkillActivation] = []
        shield_fired = False

        for skill in troop_skills:
            if skill.get("trigger") != "on_receive_damage":
                continue
            chance = skill.get("chance", 1.0)
            if not self._roll(chance):
                continue

            for eff in skill.get("effects", []):
                etype = eff["type"]
                if etype == "damage_offset":
                    taken_down_values.append(self._scale(eff["value"], chance))
                    shield_fired = True
                elif etype == "damage_halve":
                    taken_down_values.append(self._scale(eff["value"], chance))

            # on_receive_damage は防御側のスキル（side は攻撃側の逆）
            _, _, def_side = _get_armies(state, side)
            if self._should_log(chance):
                activations.append(SkillActivation(
                    turn=state.turn, skill_id=skill["id"],
                    unit_type=def_group.unit_type, triggered_by="on_receive_damage",
                    effect_summary=skill.get("description_ja", ""),
                    side=def_side,
                ))

        # Body of Light: Crystal Shield 発動時の追加軽減
        if shield_fired:
            for skill in troop_skills:
                if skill.get("type") != "body_of_light":
                    continue
                for eff in skill.get("effects", []):
                    if eff["type"] == "extra_reduce_when_shield_active":
                        taken_down_values.append(eff["value"])

        # ---------- 英雄スキル: on_receive_damage（Reina S2: 回避） ----------
        if hero_defs:
            _, def_army, def_side = _get_armies(state, side)
            for hc in def_army.heroes:
                if hc.hero_id not in hero_defs:
                    continue
                hdef = hero_defs[hc.hero_id]
                for skill in _get_hero_skills(hc, hdef):
                    if skill.get("trigger") != "on_receive_damage":
                        continue
                    unit_restriction = skill.get("unit_restriction")
                    if unit_restriction and unit_restriction != def_group.unit_type:
                        continue
                    chance = skill.get("chance", 1.0)
                    if not self._roll(chance):
                        continue
                    for eff in skill.get("effects", []):
                        if eff["type"] == "dodge":
                            if self.mode == "expected":
                                # 回避確率をダメージ軽減として近似
                                taken_down_values.append(chance)
                            else:
                                dodge = True
                    if self._should_log(chance):
                        activations.append(SkillActivation(
                            turn=state.turn, skill_id=skill["id"],
                            unit_type=def_group.unit_type,
                            triggered_by="on_receive_damage",
                            effect_summary=skill.get("description_ja", ""),
                            side=def_side,
                        ))

        return taken_down_values, dodge, activations

    # ================================================================
    # ターン終了: active_damage_mod の duration をデクリメント
    # ================================================================
    def _tick_turn_effects(self, state: BattleState, side: str) -> None:
        eff_list = (
            state.skill_state.active_effects_a if side == "a"
            else state.skill_state.active_effects_b
        )
        army, _, _ = _get_armies(state, side)

        expired = []
        for eff in eff_list:
            if eff.remaining_turns is None:
                continue
            eff.remaining_turns -= 1
            if eff.remaining_turns <= 0:
                expired.append(eff)
                # active_damage_mod から該当値を取り消す
                _remove_from_active_dmg(army.active_damage_mod, eff.effect_type, eff.value)

        for e in expired:
            eff_list.remove(e)

    # ================================================================
    # active_damage_mod への追加（on_turn_start / every_n_turns 用）
    # ================================================================
    def _apply_to_active_dmg(
        self,
        army: Army,
        effect_type: str,
        value: float,
        duration: int,
        skill_id: str,
        state: BattleState,
        side: str,
    ) -> None:
        eff_list = (
            state.skill_state.active_effects_a if side == "a"
            else state.skill_state.active_effects_b
        )

        # 同一スキル+効果が発動中なら duration リフレッシュのみ（効果は重複しない）
        for existing in eff_list:
            if existing.skill_id == skill_id and existing.effect_type == effect_type:
                existing.remaining_turns = max(existing.remaining_turns or 0, duration)
                return

        dmg = army.active_damage_mod
        if effect_type == "damage_dealt_up":
            dmg.dealt_up += value
        elif effect_type == "damage_taken_down":
            dmg.taken_down.append(value)
        elif effect_type == "damage_dealt_down":
            dmg.dealt_down.append(value)
        elif effect_type == "damage_taken_up":
            dmg.taken_up += value
        elif effect_type == "lethality_up":
            dmg.lethality_up += value

        # 持続管理
        eff_list.append(ActiveEffect(
            skill_id=skill_id,
            effect_type=effect_type,
            value=value,
            remaining_turns=duration,
            remaining_hits=None,
            target="ally_all",
        ))


def _remove_from_active_dmg(dmg: DamageMod, effect_type: str, value: float) -> None:
    if effect_type == "damage_dealt_up":
        dmg.dealt_up = max(0.0, dmg.dealt_up - value)
    elif effect_type == "damage_taken_down":
        if value in dmg.taken_down:
            dmg.taken_down.remove(value)
    elif effect_type == "damage_dealt_down":
        if value in dmg.dealt_down:
            dmg.dealt_down.remove(value)
    elif effect_type == "damage_taken_up":
        dmg.taken_up = max(0.0, dmg.taken_up - value)
    elif effect_type == "lethality_up":
        dmg.lethality_up = max(0.0, dmg.lethality_up - value)
