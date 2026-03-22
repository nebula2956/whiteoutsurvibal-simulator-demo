from __future__ import annotations
import streamlit as st
import pandas as pd
from collections import defaultdict
from typing import Dict

from simulator.models import BattleResult, UNIT_TYPES

UNIT_LABELS = {"infantry": "盾兵", "lancer": "槍兵", "archer": "弓兵"}


def render_battle_result(result: BattleResult) -> None:
    winner_label = {"a": "自軍（A）の勝利！", "b": "敵軍（B）の勝利！", "draw": "引き分け"}.get(result.winner, "不明")
    st.success(f"**結果: {winner_label}**（{result.total_turns} ターン）")

    # ---- キルデスサマリー（画像1スタイル）----
    st.subheader("戦闘一覧")
    _render_battle_summary(result)

    # ---- スキル詳細（画像2スタイル）----
    st.subheader("スキル詳細")
    _render_skill_detail(result)

    # ---- ターン別ログ ----
    st.subheader("ターン別ログ")
    _render_turn_log(result)

    # ---- 兵数推移チャート ----
    st.subheader("兵数推移")
    _render_troop_chart(result)


# ================================================================
# 戦闘一覧（画像1スタイル）
# ================================================================
def _render_battle_summary(result: BattleResult) -> None:
    """A vs B の初期部隊・損傷・生存を表形式で表示。"""
    if not result.turn_logs:
        return

    first_log = result.turn_logs[0]
    last_log = result.turn_logs[-1]

    def side_stats(initial_counts, kills, final_counts):
        total_initial = sum(initial_counts.values())
        total_killed  = sum(kills.values())
        total_survived = sum(final_counts.values())
        return total_initial, total_killed, total_survived

    a_initial = first_log.army_a_counts
    b_initial = first_log.army_b_counts

    # 全ターンのキル集計
    a_total_killed = {ut: result.total_kills_by_side.get("a", {}).get(ut, 0)
                      for ut in UNIT_TYPES}
    b_total_killed = {ut: result.total_kills_by_side.get("b", {}).get(ut, 0)
                      for ut in UNIT_TYPES}

    a_final = result.final_counts_a or last_log.army_a_counts
    b_final = result.final_counts_b or last_log.army_b_counts

    rows = []
    for ut, lbl in UNIT_LABELS.items():
        rows.append({
            "兵種": lbl,
            "A 部隊": a_initial.get(ut, 0),
            "A 損傷": a_total_killed.get(ut, 0),
            "A 生存": a_final.get(ut, 0),
            "B 部隊": b_initial.get(ut, 0),
            "B 損傷": b_total_killed.get(ut, 0),
            "B 生存": b_final.get(ut, 0),
        })

    # 合計行
    rows.append({
        "兵種": "合計",
        "A 部隊": sum(a_initial.values()),
        "A 損傷": sum(a_total_killed.values()),
        "A 生存": sum(a_final.values()),
        "B 部隊": sum(b_initial.values()),
        "B 損傷": sum(b_total_killed.values()),
        "B 生存": sum(b_final.values()),
    })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)


# ================================================================
# スキル詳細（画像2スタイル）
# ================================================================
def _render_skill_detail(result: BattleResult) -> None:
    """A/B軍それぞれのスキルの発動回数とキル数を表示。"""

    # side別に activation_counts・skill_desc・skill_hero を集計
    activation_counts = {"a": defaultdict(int), "b": defaultdict(int)}
    skill_desc: Dict[str, str] = {}
    skill_hero: Dict[str, str] = {}
    for act in result.skill_activation_log:
        s = act.side
        activation_counts[s][act.skill_id] += 1
        if act.skill_id not in skill_desc:
            skill_desc[act.skill_id] = act.effect_summary
            skill_hero[act.skill_id] = act.unit_type

    # A/B軍別スキルキル
    skill_kills_a = result.skill_kill_contributions.get("a", {})
    skill_kills_b = result.skill_kill_contributions.get("b", {})

    col_a, col_b = st.columns(2)

    for col, side_label, acts_dict, kills_dict in [
        (col_a, "自軍（A）スキル", activation_counts["a"], skill_kills_a),
        (col_b, "敵軍（B）スキル", activation_counts["b"], skill_kills_b),
    ]:
        with col:
            st.markdown(f"**{side_label}**")
            if not acts_dict:
                st.info("スキル発動なし")
                continue
            rows = []
            for sid, act_cnt in sorted(acts_dict.items()):
                kills = kills_dict.get(sid, 0)
                rows.append({
                    "英雄/兵種": skill_hero.get(sid, ""),
                    "スキルID": sid,
                    "説明": skill_desc.get(sid, ""),
                    "発動": act_cnt,
                    "キル": kills if kills > 0 else "-",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ================================================================
# ターン別ログ
# ================================================================
def _render_turn_log(result: BattleResult) -> None:
    if not result.turn_logs:
        return

    turn_idx = st.slider("ターン", 1, len(result.turn_logs), 1, key="turn_scrubber") - 1
    log = result.turn_logs[turn_idx]

    st.markdown(f"**Turn {log.turn}**")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**A軍**")
        rows = []
        for ut, lbl in UNIT_LABELS.items():
            before = log.army_a_counts.get(ut, 0)
            loss   = log.b_kills_a.get(ut, 0)
            kill   = sum(log.a_kills_by_attacker.get(ut, {}).values()) if log.a_kills_by_attacker else 0
            rows.append({"兵種": lbl, "開始兵数": before, "損傷": loss, "キル": kill})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with col_b:
        st.markdown("**B軍**")
        rows = []
        for ut, lbl in UNIT_LABELS.items():
            before = log.army_b_counts.get(ut, 0)
            loss   = log.a_kills_b.get(ut, 0)
            kill   = sum(log.b_kills_by_attacker.get(ut, {}).values()) if log.b_kills_by_attacker else 0
            rows.append({"兵種": lbl, "開始兵数": before, "損傷": loss, "キル": kill})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # スキル発動（このターン）- A/B軍別
    if log.skill_activations:
        a_acts = [a for a in log.skill_activations if a.side == "a"]
        b_acts = [a for a in log.skill_activations if a.side == "b"]
        with st.expander(f"スキル発動（A軍:{len(a_acts)}件 / B軍:{len(b_acts)}件）"):
            sc1, sc2 = st.columns(2)
            with sc1:
                st.markdown("**自軍（A）**")
                if a_acts:
                    st.dataframe(pd.DataFrame([
                        {"スキルID": a.skill_id, "兵種/英雄": a.unit_type,
                         "トリガー": a.triggered_by, "説明": a.effect_summary}
                        for a in a_acts
                    ]), use_container_width=True, hide_index=True)
                else:
                    st.info("発動なし")
            with sc2:
                st.markdown("**敵軍（B）**")
                if b_acts:
                    st.dataframe(pd.DataFrame([
                        {"スキルID": a.skill_id, "兵種/英雄": a.unit_type,
                         "トリガー": a.triggered_by, "説明": a.effect_summary}
                        for a in b_acts
                    ]), use_container_width=True, hide_index=True)
                else:
                    st.info("発動なし")

    # スキル別キル（このターン）A/B別
    if log.a_skill_kills_turn or log.b_skill_kills_turn:
        with st.expander("スキル別キル（このターン）"):
            sk1, sk2 = st.columns(2)
            with sk1:
                st.markdown("**自軍（A）**")
                if log.a_skill_kills_turn:
                    rows = [{"スキルID": sid, "キル数": k}
                            for sid, k in sorted(log.a_skill_kills_turn.items(), key=lambda x: -x[1])]
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                else:
                    st.info("スキルキルなし")
            with sk2:
                st.markdown("**敵軍（B）**")
                if log.b_skill_kills_turn:
                    rows = [{"スキルID": sid, "キル数": k}
                            for sid, k in sorted(log.b_skill_kills_turn.items(), key=lambda x: -x[1])]
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                else:
                    st.info("スキルキルなし")


# ================================================================
# 兵数推移チャート
# ================================================================
def _render_troop_chart(result: BattleResult) -> None:
    if not result.turn_logs:
        return
    chart_rows = []
    for log in result.turn_logs:
        row = {"ターン": log.turn}
        for ut, lbl in UNIT_LABELS.items():
            row[f"A_{lbl}"] = log.army_a_counts.get(ut, 0)
            row[f"B_{lbl}"] = log.army_b_counts.get(ut, 0)
        chart_rows.append(row)
    st.line_chart(pd.DataFrame(chart_rows).set_index("ターン"))
