from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
from ui.input_panel import render_army_input, render_army_base_input
from ui.result_panel import render_battle_result
from ui.optimizer_panel import render_optimizer_panel
from simulator import run_simulation

st.set_page_config(page_title="WoS Battle Simulator", layout="wide")
st.title("WoS Battle Simulator")

tab_sim, tab_opt = st.tabs(["シミュレーション", "編成最適化（CMA-ES）"])

# ================================
# Tab 1: シミュレーション
# ================================
with tab_sim:
    sim_mode = st.radio("スキル確率モード", ["random（確率ロール）", "expected（期待値）"],
                        horizontal=True, key="sim_mode")
    mode_str = "random" if "random" in sim_mode else "expected"

    col_a, col_b = st.columns(2)
    with col_a:
        config_a = render_army_input("自軍（A）", "a")
    with col_b:
        config_b = render_army_input("敵軍（B）", "b")

    if st.button("シミュレーション実行", type="primary"):
        try:
            with st.spinner("計算中..."):
                st.session_state["sim_result"] = run_simulation(config_a, config_b, mode=mode_str)
        except ValueError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"実行エラー: {type(e).__name__}: {e}")

    if "sim_result" in st.session_state:
        render_battle_result(st.session_state["sim_result"])

# ================================
# Tab 2: GA 最適化
# ================================
with tab_opt:
    st.info("CMA-ESで兵種比率と英雄の組み合わせを同時最適化します。")
    col_a2, col_b2 = st.columns(2)
    with col_a2:
        config_own = render_army_base_input("自軍（バフ・基本設定）", "opt_a")
    with col_b2:
        config_enemy = render_army_input("敵軍（固定）", "opt_b")

    render_optimizer_panel(config_own, config_enemy)
