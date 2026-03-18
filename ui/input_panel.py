from __future__ import annotations
import streamlit as st
from typing import List
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from simulator.models import ArmyConfig, HeroConfig, StatBuff

LEGEND_HEROES = [
    "Zinman", "Molly", "Natalia", "Jeronimo", "Gwen",
    "Hector", "Norah", "Flint", "Frender", "Alonzo",
    "Logan", "Greg", "Mia", "Akmos", "Reina", "Rion",
]
EPIC_HEROES   = ["Jessie", "Seo-yoon", "Sergey", "Patrick", "Bokan", "Ling Xue"]
ALL_HEROES    = LEGEND_HEROES + EPIC_HEROES
TIERS         = ["T10", "T11"]


def render_army_input(label: str, key_prefix: str) -> ArmyConfig:
    st.subheader(label)

    col1, col2, col3 = st.columns(3)
    with col1:
        tier = st.selectbox("ティア", TIERS, key=f"{key_prefix}_tier")
    with col2:
        fc = st.slider("FC レベル", 0, 10, 5, key=f"{key_prefix}_fc")
    with col3:
        capacity = st.number_input("集結容量（総兵数）", min_value=1000, max_value=5_000_000,
                                   value=100_000, step=10_000, key=f"{key_prefix}_cap")

    role = st.radio("ロール", ["attack", "defense"], horizontal=True, key=f"{key_prefix}_role")

    # ---- 兵種比 ----
    st.markdown("**兵種比（%）** ※合計が100%になるよう自動正規化")
    c1, c2, c3 = st.columns(3)
    with c1:
        inf_r = st.number_input("盾兵 (%)", min_value=0, max_value=100, value=40, step=1, key=f"{key_prefix}_inf")
    with c2:
        lan_r = st.number_input("槍兵 (%)", min_value=0, max_value=100, value=30, step=1, key=f"{key_prefix}_lan")
    with c3:
        arc_r = st.number_input("弓兵 (%)", min_value=0, max_value=100, value=30, step=1, key=f"{key_prefix}_arc")
    total = inf_r + lan_r + arc_r or 1
    troop_ratio = {
        "infantry": inf_r / total,
        "lancer":   lan_r / total,
        "archer":   arc_r / total,
    }
    st.caption(f"実比率: 盾{inf_r/total*100:.1f}% 槍{lan_r/total*100:.1f}% 弓{arc_r/total*100:.1f}%")

    # ---- ベースバフ ----
    with st.expander("基礎バフ（%）- 研究・島・装備等"):
        st.markdown("**全部隊共通**")
        b1, b2, b3, b4 = st.columns(4)
        with b1:
            b_atk  = st.number_input("攻撃力",  0.0, 2000.0, 0.0, step=5.0, key=f"{key_prefix}_batk") / 100
        with b2:
            b_def  = st.number_input("防御力",  0.0, 2000.0, 0.0, step=5.0, key=f"{key_prefix}_bdef") / 100
        with b3:
            b_hp   = st.number_input("HP",      0.0, 2000.0, 0.0, step=5.0, key=f"{key_prefix}_bhp")  / 100
        with b4:
            b_leth = st.number_input("殺傷力",  0.0, 2000.0, 0.0, step=5.0, key=f"{key_prefix}_bleth") / 100
        base_buff = StatBuff(atk=b_atk, def_=b_def, hp=b_hp, lethality=b_leth)

        st.markdown("**兵種別**")
        per_type_buffs = {}
        for ut, ut_label in [("infantry", "盾兵"), ("lancer", "槍兵"), ("archer", "弓兵")]:
            st.caption(ut_label)
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                pt_atk  = st.number_input("攻撃力",  0.0, 2000.0, 0.0, step=5.0, key=f"{key_prefix}_{ut}_atk")  / 100
            with c2:
                pt_def  = st.number_input("防御力",  0.0, 2000.0, 0.0, step=5.0, key=f"{key_prefix}_{ut}_def")  / 100
            with c3:
                pt_hp   = st.number_input("HP",      0.0, 2000.0, 0.0, step=5.0, key=f"{key_prefix}_{ut}_hp")   / 100
            with c4:
                pt_leth = st.number_input("殺傷力",  0.0, 2000.0, 0.0, step=5.0, key=f"{key_prefix}_{ut}_leth") / 100
            per_type_buffs[ut] = StatBuff(atk=pt_atk, def_=pt_def, hp=pt_hp, lethality=pt_leth)

    # ---- 英雄（ラリーリーダー3名 + メンバー4名）----
    st.markdown("**英雄編成**")
    heroes: List[HeroConfig] = []

    with st.expander("ラリーリーダー（各兵種1名・Legend のみ・全スキル使用）"):
        LEADER_SLOTS = [
            ("盾兵リーダー",  ["Natalia", "Jeronimo", "Hector", "Flint", "Logan", "Akmos"], 0),
            ("槍兵リーダー",  ["Molly", "Norah", "Frender", "Mia", "Reina"],              1),
            ("弓兵リーダー",  ["Zinman", "Gwen", "Alonzo", "Greg", "Rion"],               2),
        ]
        for label, candidates, pos in LEADER_SLOTS:
            cols = st.columns([3, 2])
            with cols[0]:
                hero_id = st.selectbox(label, ["（なし）"] + candidates,
                                       key=f"{key_prefix}_lead{pos}")
            with cols[1]:
                gear = st.slider("専用装備レベル", 0, 10, 0, key=f"{key_prefix}_lead_gear{pos}")
            if hero_id != "（なし）":
                heroes.append(HeroConfig(
                    hero_id=hero_id, gear_level=gear,
                    is_rally_leader=(pos == 0), position=pos,
                ))

    with st.expander("参加メンバー（4名・スキル1のみ使用・専用装備効果なし）"):
        for i in range(4):
            hero_id = st.selectbox(f"メンバー {i+1}", ["（なし）"] + ALL_HEROES,
                                   key=f"{key_prefix}_member{i}")
            if hero_id != "（なし）":
                heroes.append(HeroConfig(
                    hero_id=hero_id, gear_level=0,
                    is_rally_leader=False, position=3 + i,
                ))

    return ArmyConfig(
        rally_capacity=int(capacity),
        troop_ratio=troop_ratio,
        troop_tier=tier,
        fc_level=fc,
        heroes=heroes,
        base_buff=base_buff,
        base_buff_per_type=per_type_buffs,
        role=role,
    )


def render_army_base_input(label: str, key_prefix: str) -> ArmyConfig:
    """最適化用: バフ・ティア・FC・容量のみ入力（比率と英雄はCMA-ESが決定）。"""
    st.subheader(label)

    col1, col2, col3 = st.columns(3)
    with col1:
        tier = st.selectbox("ティア", TIERS, key=f"{key_prefix}_tier")
    with col2:
        fc = st.slider("FC レベル", 0, 10, 5, key=f"{key_prefix}_fc")
    with col3:
        capacity = st.number_input("集結容量（総兵数）", min_value=1000, max_value=5_000_000,
                                   value=100_000, step=10_000, key=f"{key_prefix}_cap")

    role = st.radio("ロール", ["attack", "defense"], horizontal=True, key=f"{key_prefix}_role")

    with st.expander("基礎バフ（%）- 研究・島・装備等"):
        st.markdown("**全部隊共通**")
        b1, b2, b3, b4 = st.columns(4)
        with b1:
            b_atk  = st.number_input("攻撃力",  0.0, 2000.0, 0.0, step=5.0, key=f"{key_prefix}_batk") / 100
        with b2:
            b_def  = st.number_input("防御力",  0.0, 2000.0, 0.0, step=5.0, key=f"{key_prefix}_bdef") / 100
        with b3:
            b_hp   = st.number_input("HP",      0.0, 2000.0, 0.0, step=5.0, key=f"{key_prefix}_bhp")  / 100
        with b4:
            b_leth = st.number_input("殺傷力",  0.0, 2000.0, 0.0, step=5.0, key=f"{key_prefix}_bleth") / 100
        base_buff = StatBuff(atk=b_atk, def_=b_def, hp=b_hp, lethality=b_leth)

        st.markdown("**兵種別**")
        per_type_buffs = {}
        for ut, ut_label in [("infantry", "盾兵"), ("lancer", "槍兵"), ("archer", "弓兵")]:
            st.caption(ut_label)
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                pt_atk  = st.number_input("攻撃力",  0.0, 2000.0, 0.0, step=5.0, key=f"{key_prefix}_{ut}_atk")  / 100
            with c2:
                pt_def  = st.number_input("防御力",  0.0, 2000.0, 0.0, step=5.0, key=f"{key_prefix}_{ut}_def")  / 100
            with c3:
                pt_hp   = st.number_input("HP",      0.0, 2000.0, 0.0, step=5.0, key=f"{key_prefix}_{ut}_hp")   / 100
            with c4:
                pt_leth = st.number_input("殺傷力",  0.0, 2000.0, 0.0, step=5.0, key=f"{key_prefix}_{ut}_leth") / 100
            per_type_buffs[ut] = StatBuff(atk=pt_atk, def_=pt_def, hp=pt_hp, lethality=pt_leth)

    # ダミーの比率と空英雄リスト（CMA-ESがデコード時に上書き）
    return ArmyConfig(
        rally_capacity=int(capacity),
        troop_ratio={"infantry": 1/3, "lancer": 1/3, "archer": 1/3},
        troop_tier=tier,
        fc_level=fc,
        heroes=[],
        base_buff=base_buff,
        base_buff_per_type=per_type_buffs,
        role=role,
    )
