from __future__ import annotations
from typing import Dict, List, Tuple, Any

from .models import (
    Army, BattleState, SkillState, DamageMod,
    SkillActivation, BattleResult, UNIT_TYPES,
)
from .damage_engine import DamageEngine
from .target_selector import TargetSelector
from .skill_engine import SkillEngine
from .logger import BattleLogger


def _merge_dmg(a: DamageMod, b: DamageMod) -> DamageMod:
    """パッシブ + アクティブのDamageModを合算して返す。"""
    return DamageMod(
        dealt_up=a.dealt_up + b.dealt_up,
        dealt_down=list(a.dealt_down) + list(b.dealt_down),
        taken_up=a.taken_up + b.taken_up,
        taken_down=list(a.taken_down) + list(b.taken_down),
        additional_fracs=list(a.additional_fracs) + list(b.additional_fracs),
        dealt_up_per_type={
            k: a.dealt_up_per_type.get(k, 0.0) + b.dealt_up_per_type.get(k, 0.0)
            for k in set(list(a.dealt_up_per_type.keys()) + list(b.dealt_up_per_type.keys()))
        },
        lethality_up=a.lethality_up + b.lethality_up,
        extra_atk_up=a.extra_atk_up + b.extra_atk_up,
    )


class BattleSimulator:
    """バトルループを制御するオーケストレーター。"""

    def __init__(
        self,
        army_a: Army,
        army_b: Army,
        skill_engine: SkillEngine,
        damage_engine: DamageEngine,
        target_selector: TargetSelector,
        hero_defs: Dict = None,
        troop_data: Dict = None,
        max_turns: int = 500,
    ):
        self.army_a = army_a
        self.army_b = army_b
        self.skill_engine = skill_engine
        self.damage_engine = damage_engine
        self.target_selector = target_selector
        self.hero_defs = hero_defs or {}
        self.troop_data = troop_data or {}
        self.max_turns = max_turns

    def run(self) -> BattleResult:
        state = BattleState(
            turn=0,
            army_a=self.army_a,
            army_b=self.army_b,
            skill_state=SkillState(),
        )
        logger = BattleLogger()

        # バトル開始時: passive_decaying スキルを SkillState に登録
        self.skill_engine.on_battle_start(state, self.hero_defs)

        for turn in range(1, self.max_turns + 1):
            state.turn = turn

            # ターン開始前の兵数記録
            a_counts = {t: u.effective_count() for t, u in state.army_a.units.items()}
            b_counts = {t: u.effective_count() for t, u in state.army_b.units.items()}

            # ターン開始スキル
            acts_a = self.skill_engine.on_turn_start(state, self.hero_defs, "a")
            acts_b = self.skill_engine.on_turn_start(state, self.hero_defs, "b")
            turn_activations = acts_a + acts_b

            # Phase 1: 両軍のスキルロール（同時）
            skills_a, acts = self._resolve_skills(state, "a")
            turn_activations += acts
            skills_b, acts = self._resolve_skills(state, "b")
            turn_activations += acts

            # Phase 2: 両軍のダメージ計算（スキル結果を使って同時計算）
            logger.set_current_side("a")
            dmg_a_to_b, dmg_a_by_atk = self._compute_damage(state, "a", skills_a, logger)
            logger.set_current_side("b")
            dmg_b_to_a, dmg_b_by_atk = self._compute_damage(state, "b", skills_b, logger)

            # ダメージ適用
            a_kills_b, a_kills_by_atk = self._apply_damage_detailed(
                state.army_b, dmg_a_to_b, dmg_a_by_atk, logger)
            b_kills_a, b_kills_by_atk = self._apply_damage_detailed(
                state.army_a, dmg_b_to_a, dmg_b_by_atk, logger)

            # 同一ターン内の同一スキルIDの重複ログを除去（on_every_attack等が兵種数分記録される問題）
            seen = set()
            deduped_activations = []
            for act in turn_activations:
                key = (act.skill_id, act.side)
                if key not in seen:
                    seen.add(key)
                    deduped_activations.append(act)

            logger.log_turn(turn, a_counts, b_counts,
                            a_kills_b, b_kills_a,
                            a_kills_by_atk, b_kills_by_atk,
                            deduped_activations)

            if self._is_over(state):
                break

        winner = self._determine_winner(state)
        final_a = {t: u.effective_count() for t, u in state.army_a.units.items()}
        final_b = {t: u.effective_count() for t, u in state.army_b.units.items()}
        return logger.finalize(winner, state.turn, final_a, final_b)

    def _resolve_skills(
        self,
        state: BattleState,
        side: str,
    ) -> Tuple[Dict[str, Tuple], List[SkillActivation]]:
        """
        Phase 1: 各兵種のスキルをロールし、結果を返す（stateへの副作用は最小限）。
        Returns:
            unit_skills: {unit_type: (target_type, skill_mods, troop_skills, dodged)}
            activations
        """
        atk_army = state.army_a if side == "a" else state.army_b
        def_army = state.army_b if side == "a" else state.army_a

        unit_skills: Dict[str, Tuple] = {}
        all_activations: List[SkillActivation] = []

        from .loaders import TroopLoader
        tl = TroopLoader()

        for unit_type, atk_group in atk_army.units.items():
            if not atk_group.is_alive():
                continue

            troop_skills = tl.get_active_skills(
                self.troop_data, unit_type,
                atk_group.tier, atk_group.fc_level,
            ) if self.troop_data else []

            # on_every_attack（Gwen S1: 被ダメ上昇, Akmos S3: 追加ダメージ等）
            tmp_target_type = self.target_selector.select(
                unit_type, def_army, troop_skills,
                state.skill_state, self.skill_engine.mode,
            )
            every_atk_mods, acts = self.skill_engine.on_every_attack(
                atk_group, def_army.units[tmp_target_type], state, side, self.hero_defs
            )
            all_activations.extend(acts)

            # 最終ターゲット選択（bypass 考慮）
            target_type = self.target_selector.select(
                unit_type, def_army, troop_skills,
                state.skill_state, self.skill_engine.mode,
            )
            def_group = def_army.units[target_type]

            # on_attack スキル修飾子
            skill_mods, acts = self.skill_engine.on_attack(
                atk_group, def_group, state, side, self.hero_defs, troop_skills
            )
            all_activations.extend(acts)

            # on_every_attack の修飾子をマージ
            skill_mods.additional_fracs.extend(every_atk_mods.additional_fracs)
            for k, v in every_atk_mods.skill_damage_fracs.items():
                skill_mods.skill_damage_fracs[k] = skill_mods.skill_damage_fracs.get(k, 0) + v

            # on_receive_damage（Crystal Shield 等）
            def_troop_skills = tl.get_active_skills(
                self.troop_data, target_type,
                def_group.tier, def_group.fc_level,
            ) if self.troop_data else []
            taken_down_vals, dodged, acts = self.skill_engine.on_receive_damage(
                def_group, state, side, def_troop_skills, self.hero_defs
            )
            skill_mods.extra_taken_down.extend(taken_down_vals)
            all_activations.extend(acts)

            # Gwen S2 脆弱: 自軍が付与した追加ダメージ枠（次の攻撃に適用）
            vuln = state.skill_state.vulnerability[side].get(target_type, 0.0)
            if vuln > 0:
                skill_mods.additional_fracs.append(vuln)

            # 減衰バフ（Hector S2 等）
            decay_bonus = self.skill_engine.consume_decaying_buff(state, side, unit_type)
            if decay_bonus > 0:
                skill_mods.extra_dealt_up += decay_bonus

            unit_skills[unit_type] = (target_type, skill_mods, troop_skills, dodged)

            # 脆弱デバフはヒット後リセット
            if target_type in state.skill_state.vulnerability[side]:
                del state.skill_state.vulnerability[side][target_type]

        return unit_skills, all_activations

    def _compute_damage(
        self,
        state: BattleState,
        side: str,
        unit_skills: Dict[str, Tuple],
        logger: "BattleLogger",
    ) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
        """Phase 2: スキル結果を使ってダメージ計算。"""
        atk_army = state.army_a if side == "a" else state.army_b
        def_army = state.army_b if side == "a" else state.army_a

        dmg_map: Dict[str, float] = {ut: 0.0 for ut in UNIT_TYPES}
        dmg_by_attacker: Dict[str, Dict[str, float]] = {
            ut: {t: 0.0 for t in UNIT_TYPES}
            for ut in UNIT_TYPES
        }

        # AOE: unit_type → 発生した AOE ダメージ量（base_dmg × Σaoe_fracs）
        aoe_dmg_by_source: Dict[str, float] = {}

        for unit_type, (target_type, skill_mods, troop_skills, dodged) in unit_skills.items():
            atk_group = atk_army.units[unit_type]
            def_group = def_army.units[target_type]

            if dodged or skill_mods.attack_skip:
                continue

            bonus_vs = self.damage_engine.get_bonus_vs(unit_type, target_type, troop_skills)
            atk_dmg = _merge_dmg(atk_army.damage_mod, atk_army.active_damage_mod)
            def_dmg = _merge_dmg(def_army.damage_mod, def_army.active_damage_mod)

            attack_count = 2 if skill_mods.double_attack else 1
            for _ in range(attack_count):
                result = self.damage_engine.compute(
                    atk_group=atk_group,
                    def_group=def_group,
                    atk_stat=atk_army.stat_layers,
                    atk_dmg=atk_dmg,
                    def_stat=def_army.stat_layers,
                    def_dmg=def_dmg,
                    enemy_stat_debuff=def_army.passive_stat_debuff_to_enemy,
                    enemy_dealt_debuffs=def_army.passive_dealt_debuff_to_enemy,
                    skill_mods=skill_mods,
                    bonus_vs=bonus_vs,
                )
                dmg_map[target_type] += result.final_damage
                dmg_by_attacker[unit_type][target_type] += result.final_damage
                for skill_id, sk in result.skill_procs.items():
                    if sk > 0:
                        logger.record_skill_kills(skill_id, sk)

                # AOE: base_dmg × Σaoe_fracs を記録（Gwen S3等）
                if skill_mods.aoe_fracs:
                    aoe_dmg_by_source[unit_type] = (
                        aoe_dmg_by_source.get(unit_type, 0.0)
                        + result.raw_damage * sum(skill_mods.aoe_fracs)
                    )

        # AOE ダメージを全生存ユニット種に独立加算
        for src_type, aoe_dmg in aoe_dmg_by_source.items():
            for t in UNIT_TYPES:
                if def_army.units[t].is_alive():
                    dmg_map[t] += aoe_dmg
                    dmg_by_attacker[src_type][t] += aoe_dmg

        return dmg_map, dmg_by_attacker

    def _apply_damage_detailed(
        self,
        defender: Army,
        dmg_map: Dict[str, float],
        dmg_by_attacker: Dict[str, Dict[str, float]],
        logger: "BattleLogger",
    ) -> Tuple[Dict[str, int], Dict[str, Dict[str, int]]]:
        """ダメージを適用し、ターゲット別キルとアタッカー別キルを返す。"""
        kills: Dict[str, int] = {}
        for t, dmg in dmg_map.items():
            group = defender.units[t]
            if not group.is_alive() or dmg <= 0:
                kills[t] = 0
                continue
            count_before = group.effective_count()
            group.current_hp = max(0.0, group.current_hp - dmg)
            group.count = group.effective_count()
            kills[t] = max(0, count_before - group.count)

        # アタッカー別キル数（ダメージ比例で按分）
        kills_by_attacker: Dict[str, Dict[str, int]] = {}
        for atk_type, atk_dmg_map in dmg_by_attacker.items():
            kills_by_attacker[atk_type] = {}
            for t, dmg in atk_dmg_map.items():
                total_dmg = dmg_map.get(t, 0.0)
                total_kills = kills.get(t, 0)
                if total_dmg > 0 and total_kills > 0:
                    kills_by_attacker[atk_type][t] = int(total_kills * dmg / total_dmg)
                else:
                    kills_by_attacker[atk_type][t] = 0

        return kills, kills_by_attacker

    def _is_over(self, state: BattleState) -> bool:
        a_alive = any(u.is_alive() for u in state.army_a.units.values())
        b_alive = any(u.is_alive() for u in state.army_b.units.values())
        return not (a_alive and b_alive)

    def _determine_winner(self, state: BattleState) -> str:
        a = sum(u.effective_count() for u in state.army_a.units.values())
        b = sum(u.effective_count() for u in state.army_b.units.values())
        if a > b:
            return "a"
        elif b > a:
            return "b"
        else:
            return "draw"
