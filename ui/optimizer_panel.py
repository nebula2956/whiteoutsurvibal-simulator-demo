from __future__ import annotations

import json
import os
import streamlit as st
import pandas as pd
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from simulator.models import ArmyConfig
from optimizer.genome import OwnedHero, HeroPool
from optimizer.cma_runner import CMAOptimizer, CMAConfig, GenerationInfo
from optimizer.beam_runner import BeamSearchOptimizer, BeamSearchConfig, PhaseInfo, estimate_cost
from optimizer.ga_runner import GAConfig, estimate_ga_cost
from optimizer.optimizer_session import OptimizerSession
from optimizer.fitness import FitnessEvaluator, FitnessWeights
from optimizer_v2 import OptimizerV2Config, V2Session

# --- 英雄定義を読み込み ---
_BASE = os.path.dirname(os.path.dirname(__file__))


def _load_hero_defs():
    with open(os.path.join(_BASE, "data", "heroes.json"), encoding="utf-8") as f:
        data = json.load(f)
    return {k: v for k, v in data.items() if k and not k.startswith("_")}


HERO_DEFS = _load_hero_defs()

INFANTRY_LEGENDS = [k for k, v in HERO_DEFS.items()
                    if v.get("rarity") == "legend" and v.get("troop_type") == "infantry"]
LANCER_LEGENDS = [k for k, v in HERO_DEFS.items()
                  if v.get("rarity") == "legend" and v.get("troop_type") == "lancer"]
ARCHER_LEGENDS = [k for k, v in HERO_DEFS.items()
                  if v.get("rarity") == "legend" and v.get("troop_type") == "archer"]
ALL_HERO_IDS = list(HERO_DEFS.keys())


def _build_hero_pool(key_prefix: str) -> HeroPool:
    infantry_leaders = []
    lancer_leaders = []
    archer_leaders = []
    members = []

    st.markdown("##### 所持レジェンド英雄（リーダー候補）")

    # リーダー一括操作ボタン
    all_leader_ids = INFANTRY_LEGENDS + LANCER_LEGENDS + ARCHER_LEGENDS
    lcol1, lcol2, lcol3, lcol4 = st.columns(4)
    with lcol1:
        if st.button("全選択", key=f"{key_prefix}_leader_all"):
            for hero_id in all_leader_ids:
                st.session_state[f"{key_prefix}_own_{hero_id}"] = True
    with lcol2:
        if st.button("全解除", key=f"{key_prefix}_leader_none"):
            for hero_id in all_leader_ids:
                st.session_state[f"{key_prefix}_own_{hero_id}"] = False
    with lcol3:
        gear_all_val = st.number_input("専用Lv一括", 0, 10, 10, step=1, key=f"{key_prefix}_gear_all_val", label_visibility="collapsed")
    with lcol4:
        if st.button("専用Lv一括設定", key=f"{key_prefix}_gear_all"):
            for hero_id in all_leader_ids:
                st.session_state[f"{key_prefix}_gear_{hero_id}"] = int(gear_all_val)

    for group_label, candidates, dest in [
        ("盾兵", INFANTRY_LEGENDS, infantry_leaders),
        ("槍兵", LANCER_LEGENDS, lancer_leaders),
        ("弓兵", ARCHER_LEGENDS, archer_leaders),
    ]:
        st.caption(group_label)
        for hero_id in candidates:
            cols = st.columns([3, 2, 1])
            with cols[0]:
                owned = st.checkbox(hero_id, value=True, key=f"{key_prefix}_own_{hero_id}")
            with cols[1]:
                gear = st.slider("専用Lv", 0, 10, 0, key=f"{key_prefix}_gear_{hero_id}")
            if owned:
                dest.append(OwnedHero(hero_id=hero_id, gear_level=gear))

    st.markdown("##### 所持メンバー英雄（参加候補）")

    # メンバー一括選択ボタン
    mcol1, mcol2 = st.columns(2)
    with mcol1:
        if st.button("全選択", key=f"{key_prefix}_memb_all"):
            for hero_id in ALL_HERO_IDS:
                st.session_state[f"{key_prefix}_memb_{hero_id}"] = True
    with mcol2:
        if st.button("全解除", key=f"{key_prefix}_memb_none"):
            for hero_id in ALL_HERO_IDS:
                st.session_state[f"{key_prefix}_memb_{hero_id}"] = False

    for hero_id in ALL_HERO_IDS:
        owned = st.checkbox(hero_id, value=True, key=f"{key_prefix}_memb_{hero_id}")
        if owned:
            members.append(hero_id)

    return HeroPool(
        infantry_leaders=infantry_leaders,
        lancer_leaders=lancer_leaders,
        archer_leaders=archer_leaders,
        members=members,
    )


def _build_weights(key_prefix: str) -> FitnessWeights:
    st.markdown("##### 評価指標の重み")
    st.caption("0にすると無視。値が大きいほど重視。")

    c1, c2 = st.columns(2)
    with c1:
        w_win = st.number_input("勝率", 0.0, 10.0, 1.0, step=0.1,
                                key=f"{key_prefix}_w_win")
        w_surv = st.number_input("平均残兵率", 0.0, 10.0, 0.0, step=0.1,
                                 key=f"{key_prefix}_w_surv")
        w_kills = st.number_input("平均与キル率", 0.0, 10.0, 0.0, step=0.1,
                                  key=f"{key_prefix}_w_kills")
    with c2:
        w_std = st.number_input("標準偏差ペナルティ", 0.0, 10.0, 0.0, step=0.1,
                                key=f"{key_prefix}_w_std",
                                help="高いほど結果のブレを嫌う")
        w_down = st.number_input("下側期待損失", 0.0, 10.0, 0.0, step=0.1,
                                 key=f"{key_prefix}_w_down",
                                 help="最悪ケースの残兵率を重視")
        down_pct = st.slider("下側パーセンタイル (%)", 5, 30, 10, step=5,
                             key=f"{key_prefix}_down_pct",
                             help="下位何%を下側損失とするか")

    return FitnessWeights(
        win_rate=w_win,
        survival_ratio=w_surv,
        avg_kills=w_kills,
        std_penalty=w_std,
        downside_risk=w_down,
        downside_percentile=down_pct,
    )


def render_optimizer_panel(own_config: ArmyConfig, enemy_config: ArmyConfig) -> None:
    st.subheader("編成最適化")

    mode = st.radio("最適化モード", ["V2 ★NEW", "GA + CMA-ES", "Beam Search + CMA-ES", "CMA-ES (従来)"],
                    key="opt_mode", horizontal=True)

    with st.expander("所持英雄設定", expanded=True):
        pool = _build_hero_pool("opt")

    n_i, n_l, n_a = pool.leader_dims
    if n_i == 0 or n_l == 0 or n_a == 0:
        st.warning("各兵種のリーダー候補を最低1体ずつ選択してください。")
        return
    if len(pool.members) == 0:
        st.warning("メンバー候補を最低1体選択してください。")
        return

    if mode == "V2 ★NEW":
        _render_v2(own_config, enemy_config, pool)
        return

    with st.expander("評価指標の重み設定"):
        weights = _build_weights("opt")

    if weights.is_all_zero():
        st.error("評価指標の重みを最低1つは0より大きくしてください。")
        return

    if mode == "GA + CMA-ES":
        _render_ga_search(own_config, enemy_config, pool, weights)
    elif mode == "Beam Search + CMA-ES":
        _render_beam_search(own_config, enemy_config, pool, weights)
    else:
        _render_cma_es(own_config, enemy_config, pool, weights)


# ---------------------------------------------------------------------------
# V2: Bilevel Mixed-Integer Bayesian Optimization
# ---------------------------------------------------------------------------

def _render_v2(
    own_config: ArmyConfig, enemy_config: ArmyConfig,
    pool: HeroPool,
) -> None:
    st.markdown(
        "**Bilevel 最適化 (TPE → CMA-ES → Adaptive MC)**\n\n"
        "Phase 1: Optuna TPE で英雄を探索 → Phase 2: CMA-ES で比率を精緻化 → "
        "Phase 3: 多忠実度 MC + 共通乱数で最終ランキング"
    )

    with st.expander("探索パラメータ", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            n_trials = st.slider("TPE 試行数 (Phase 1)", 100, 2000, 500, step=100, key="v2_n_trials")
            hero_top_k = st.slider("上位英雄セット数 (Phase 2へ渡す)", 5, 50, 20, step=5, key="v2_top_k")
            scoring_mode = st.selectbox("スコアリングモード", ["lcb", "robust", "mean"],
                                        key="v2_scoring_mode",
                                        help="lcb=下側信頼境界(推奨), robust=分散ペナルティ, mean=単純平均")
        with col2:
            n_reeval_low = st.slider("低忠実度 MC 回数 (Phase 3 スクリーニング)", 10, 50, 20, step=5,
                                     key="v2_reeval_low")
            n_reeval_high = st.slider("高忠実度 MC 回数 (Phase 3 最終評価)", 50, 300, 100, step=50,
                                      key="v2_reeval_high")
            lcb_beta = st.slider("LCB β (保守度)", 0.5, 3.0, 1.96, step=0.1, key="v2_lcb_beta",
                                 help="大きいほど不確実な候補を保守的に評価。1.96=95%CI下限")

        st.markdown("**CMA-ES 設定 (Phase 2)**")
        col3, col4 = st.columns(2)
        with col3:
            cma_sigma = st.slider("初期σ", 0.3, 1.5, 0.7, step=0.1, key="v2_cma_sigma")
            cma_generations = st.slider("CMA 世代数", 10, 100, 25, step=5, key="v2_cma_gens")
        with col4:
            cma_restarts = st.slider("マルチスタート数", 1, 5, 2, key="v2_cma_restarts")
            cma_popsize = st.slider("CMA 集団サイズ", 8, 48, 14, step=2, key="v2_cma_pop")

    # --- Session state management ---
    session_key = "v2_session"
    session: V2Session | None = st.session_state.get(session_key)

    # Running state
    if session is not None and session.running:
        progress_bar = st.progress(0)
        status_text = st.empty()
        score_chart = st.empty()
        live_detail = st.empty()

        info = session.get_latest_info()
        if info is not None:
            _display_v2_progress(info, progress_bar, status_text, live_detail)

        if session.score_history:
            df_hist = pd.DataFrame(session.score_history, columns=["進捗", "ベストスコア"])
            score_chart.line_chart(df_hist.set_index("進捗"))

        if st.button("中止", key="v2_stop"):
            st.session_state[session_key] = None
            st.rerun()

        import time
        time.sleep(0.5)
        st.rerun()

    # Finished state
    if session is not None and session.finished:
        if session.error:
            st.error(f"最適化エラー: {session.error}")
            st.session_state[session_key] = None
            return

        result = session.result
        if result is None:
            st.warning("結果が取得できませんでした。")
            st.session_state[session_key] = None
            return

        _display_v2_results(result, session.score_history)

        if st.button("結果をクリア", key="v2_clear"):
            st.session_state[session_key] = None
            st.rerun()
        return

    # Idle state
    if not st.button("最適化実行", key="v2_run"):
        return

    cfg = OptimizerV2Config(
        n_trials=n_trials,
        hero_top_k=hero_top_k,
        scoring_mode=scoring_mode,
        n_reeval_low=n_reeval_low,
        n_reeval_high=n_reeval_high,
        lcb_beta=lcb_beta,
        cma_sigma=cma_sigma,
        cma_generations=cma_generations,
        cma_restarts=cma_restarts,
        cma_popsize=cma_popsize,
    )
    evaluator = FitnessEvaluator(
        enemy_config=enemy_config,
        own_template=own_config,
        pool=pool,
        weights=FitnessWeights(),
    )
    session = V2Session()
    session.start(evaluator, pool, cfg)
    st.session_state[session_key] = session
    st.rerun()


def _display_v2_progress(info, progress_bar, status_text, live_detail) -> None:
    phase_names = {
        "hero_select":  "Phase 1: TPE 英雄探索",
        "ratio_refine": "Phase 2: CMA-ES 比率最適化",
        "reeval_low":   "Phase 3a: 低忠実度スクリーニング",
        "reeval":       "Phase 3b: 高忠実度最終評価",
    }
    phase_weights = {
        "hero_select":  (0.0, 0.6),
        "ratio_refine": (0.6, 0.85),
        "reeval_low":   (0.85, 0.94),
        "reeval":       (0.94, 1.0),
    }
    phase_label = phase_names.get(info.phase, info.phase)
    lo, hi = phase_weights.get(info.phase, (0.0, 1.0))
    overall = lo + (hi - lo) * min(info.progress, 1.0)
    progress_bar.progress(min(overall, 1.0))

    status_text.markdown(f"**{phase_label}** | ベスト: `{info.best_score:.4f}`")

    detail_lines = [info.message]
    if info.phase == "hero_select":
        trial = info.detail.get("trial", "?")
        n_trials = info.detail.get("n_trials", "?")
        ch = info.detail.get("cache_hits", 0)
        cm = info.detail.get("cache_misses", 1)
        hit_rate = ch / max(ch + cm, 1) * 100
        detail_lines.insert(1, f"試行 {trial}/{n_trials} | キャッシュヒット率: {hit_rate:.0f}%")
    elif info.phase == "ratio_refine":
        cand = info.detail.get("candidate", "?")
        total = info.detail.get("total", "?")
        sigma = info.detail.get("sigma", 0.0)
        prior = info.detail.get("prior_score", 0.0)
        refined = info.detail.get("refined_score", 0.0)
        detail_lines.insert(1, f"候補 {cand}/{total} | σ={sigma:.4f} | Phase1スコア: {prior:.4f} → {refined:.4f}")
    elif info.phase in ("reeval_low", "reeval"):
        step = info.detail.get("candidate", "?")
        total = info.detail.get("total", "?")
        mu = info.detail.get("mu")
        sigma = info.detail.get("sigma")
        if mu is not None:
            detail_lines.insert(1, f"候補 {step}/{total} | μ={mu:.3f} σ={sigma:.3f}")
        else:
            detail_lines.insert(1, f"候補 {step}/{total}")

    live_detail.markdown(f"```\n{chr(10).join(detail_lines)}\n```")


def _display_v2_results(result, score_history: list = None) -> None:
    st.markdown("**最適化完了!**")

    if score_history:
        df_hist = pd.DataFrame(score_history, columns=["進捗", "ベストスコア"])
        st.line_chart(df_hist.set_index("進捗"))

    ci_lo, ci_hi = result.best_ci
    st.success(
        f"最適解: {result.top_results[0].label}\n\n"
        f"J スコア: **{result.best_score:.4f}** | "
        f"平均勝率: **{result.best_mu:.3f}** ± {result.best_sigma:.3f}\n\n"
        f"95%信頼区間: [{ci_lo:.3f}, {ci_hi:.3f}]"
    )

    st.caption(
        f"キャッシュ: ヒット {result.cache_hits} / ミス {result.cache_misses} "
        f"(ヒット率 {result.cache_hits / max(result.cache_hits + result.cache_misses, 1) * 100:.0f}%)"
    )

    if result.top_results:
        st.markdown("**上位候補 (Phase 3 評価結果)**")
        rows = []
        for r in result.top_results:
            rows.append({
                "編成": r.label,
                "J スコア": f"{r.score:.4f}",
                "平均勝率 μ": f"{r.mu:.3f}",
                "標準偏差 σ": f"{r.sigma:.3f}",
                "95%CI": f"[{r.ci_lo:.3f}, {r.ci_hi:.3f}]",
                "試合数": r.n_sims,
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    if result.phase1_hero_sets:
        with st.expander(f"Phase 1 英雄セット候補 ({len(result.phase1_hero_sets)}件)"):
            rows2 = []
            for hs in result.phase1_hero_sets:
                rows2.append({"英雄セット": hs.label, "期待スコア": f"{hs.expected_score:.4f}"})
            st.dataframe(pd.DataFrame(rows2), use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# GA + CMA-ES mode
# ---------------------------------------------------------------------------

def _render_ga_search(
    own_config: ArmyConfig, enemy_config: ArmyConfig,
    pool: HeroPool, weights: FitnessWeights,
) -> None:
    n_i, n_l, n_a = pool.leader_dims
    n_m = len(pool.members)
    total_leaders = n_i * n_l * n_a
    from math import comb
    total_members = comb(n_m, min(4, n_m))
    st.caption(f"リーダー候補: {total_leaders} 組 | メンバー組合せ: C({n_m},4) = {total_members:,}")

    with st.expander("GA探索パラメータ"):
        col1, col2 = st.columns(2)
        with col1:
            leader_top_k = st.slider("リーダー上位K", 5, 50, 20, step=5, key="ga_leader_k")
            pop_size = st.slider("集団サイズ", 20, 100, 40, step=10, key="ga_pop")
            generations = st.slider("世代数", 10, 50, 20, step=5, key="ga_gens")
        with col2:
            n_sim = st.slider("MC回数/評価", 1, 10, 3, step=1, key="ga_nsim")
            refine_top_k = st.slider("CMA-ES精査数", 2, 10, 5, key="ga_refine_k")
            n_reeval = st.slider("最終再評価回数", 50, 500, 200, step=50, key="ga_reeval")

        st.markdown("**CMA-ES 比率最適化設定**")
        col3, col4 = st.columns(2)
        with col3:
            cma_sigma = st.slider("初期σ (探索幅)", 0.3, 1.5, 0.7, step=0.1, key="ga_cma_sigma")
            cma_popsize = st.slider("CMA集団サイズ", 8, 48, 14, step=2, key="ga_cma_pop")
        with col4:
            cma_generations = st.slider("CMA世代数", 10, 200, 25, step=5, key="ga_cma_gens")
            cma_restarts = st.slider("マルチスタート数", 1, 5, 2, key="ga_cma_restarts")

    cfg_preview = GAConfig(
        leader_top_k=leader_top_k, population_size=pop_size,
        generations=generations, n_sim_per_eval=n_sim,
        refine_top_k=refine_top_k, n_reeval_simulations=n_reeval,
        refine_cma_sigma=cma_sigma, refine_cma_popsize=cma_popsize,
        refine_cma_generations=cma_generations, refine_cma_restarts=cma_restarts,
    )
    cost = estimate_ga_cost(cfg_preview, pool)
    st.caption(f"推定: ~{cost['total_mc']:,} MC sims")
    if cost['total_mc'] > 500_000:
        st.warning("計算量が大きいです。集団サイズまたは世代数を減らすことを推奨します。")

    # --- Session state management ---
    session_key = "ga_session"
    session: OptimizerSession | None = st.session_state.get(session_key)

    # Running state
    if session is not None and session.is_running():
        progress_bar = st.progress(0)
        status_text = st.empty()
        score_chart = st.empty()
        live_detail = st.empty()

        info = session.get_latest_info()
        if info is not None:
            _display_ga_progress(info, progress_bar, status_text, live_detail)

        if session.info_history:
            with st.expander(f"ログ履歴 ({len(session.info_history)}件)", expanded=False):
                log_lines = []
                for h in session.info_history:
                    top = h.detail.get("top_candidates") or h.detail.get("top_leaders")
                    top_str = ""
                    if top:
                        top_str = " | " + ", ".join(f"{c['score']:.3f}:{c['label']}" for c in top[:3])
                    log_lines.append(f"[{h.phase}] {h.message}{top_str}")
                st.text("\n".join(log_lines))

        if session.score_history:
            import pandas as pd
            df_hist = pd.DataFrame(session.score_history, columns=["進捗", "ベストスコア"])
            score_chart.line_chart(df_hist.set_index("進捗"))

        if st.button("中止", key="ga_stop"):
            st.session_state[session_key] = None
            st.rerun()

        import time
        time.sleep(0.5)
        st.rerun()

    # Finished state
    if session is not None and session.is_finished():
        if session.status.error:
            st.error(f"最適化エラー: {session.status.error}")
            st.session_state[session_key] = None
            return

        result = session.result
        if result is None:
            st.warning("結果が取得できませんでした。")
            st.session_state[session_key] = None
            return

        _display_ga_results(result, n_reeval, session.score_history)

        if st.button("結果をクリア", key="ga_clear"):
            st.session_state[session_key] = None
            st.rerun()
        return

    # Idle state
    if not st.button("最適化実行", key="ga_run"):
        return

    cfg = GAConfig(
        leader_top_k=leader_top_k,
        population_size=pop_size,
        generations=generations,
        n_sim_per_eval=n_sim,
        refine_top_k=refine_top_k,
        n_reeval_simulations=n_reeval,
        refine_cma_sigma=cma_sigma,
        refine_cma_popsize=cma_popsize,
        refine_cma_generations=cma_generations,
        refine_cma_restarts=cma_restarts,
    )
    evaluator = FitnessEvaluator(
        enemy_config=enemy_config,
        own_template=own_config,
        pool=pool,
        weights=weights,
    )
    session = OptimizerSession()
    session.start(evaluator, pool, cfg)
    st.session_state[session_key] = session
    st.rerun()


def _display_ga_progress(
    info: PhaseInfo,
    progress_bar, status_text, live_detail,
) -> None:
    """GA PhaseInfoからプログレス表示を更新。"""
    phase_names = {
        "leader_screen": "Phase 1: リーダー探索 (MC)",
        "ga_search": "Phase 2: GA メンバー最適化",
        "cma_refine": "Phase 3: CMA-ES比率最適化",
        "reeval": "Phase 4: 最終再評価",
    }
    phase_label = phase_names.get(info.phase, info.phase)

    phase_weights = {"leader_screen": (0.0, 0.2), "ga_search": (0.2, 0.7),
                     "cma_refine": (0.7, 0.9), "reeval": (0.9, 1.0)}
    lo, hi = phase_weights.get(info.phase, (0.0, 1.0))
    overall = lo + (hi - lo) * info.progress
    progress_bar.progress(min(overall, 1.0))

    status_text.markdown(
        f"**{phase_label}** | ベスト: `{info.best_score:.4f}`"
    )

    detail_lines = [info.message]

    top_candidates = info.detail.get("top_candidates") or info.detail.get("top_leaders")
    if top_candidates:
        detail_lines.append("─── 上位候補 ───")
        for rank, item in enumerate(top_candidates, 1):
            detail_lines.append(f"{rank}. [{item['score']:.4f}] {item['label']}")

    if info.phase == "leader_screen":
        active = info.detail.get("active", "?")
        rnd = info.detail.get("round", "?")
        detail_lines.insert(1, f"Round {rnd} | 残候補: {active}")
    elif info.phase == "ga_search":
        li = info.detail.get("leader_idx", "?")
        total = info.detail.get("total_leaders", "?")
        gen = info.detail.get("generation", "?")
        total_gen = info.detail.get("total_generations", "?")
        diversity = info.detail.get("diversity", "?")
        cache_size = info.detail.get("eval_cache_size", "?")
        detail_lines.insert(1, f"リーダー {li}/{total} | 世代 {gen}/{total_gen} | "
                               f"多様性: {diversity} | キャッシュ: {cache_size}")
    elif info.phase == "cma_refine":
        cand = info.detail.get("candidate", "?")
        total = info.detail.get("total", "?")
        beam_sc = info.detail.get("beam_score", 0.0)
        ref_sc = info.detail.get("refined_score", 0.0)
        detail_lines.insert(1, f"候補 {cand}/{total} | GAスコア: {beam_sc:.4f} → 精査後: {ref_sc:.4f}")

    live_detail.markdown(f"```\n{chr(10).join(detail_lines)}\n```")


def _display_ga_results(result, n_reeval: int, score_history: list = None) -> None:
    """GAResultを表示。"""
    ci_lo, ci_hi = result.confidence_interval
    st.markdown("**最適化完了!**")

    if score_history:
        df_hist = pd.DataFrame(score_history, columns=["進捗", "ベストスコア"])
        st.line_chart(df_hist.set_index("進捗"))

    st.success(
        f"最適解: {result.label}\n\n"
        f"探索中スコア: {result.best_score:.4f} | "
        f"再評価スコア ({n_reeval}回): **{result.best_score_reeval:.4f}**\n\n"
        f"95%信頼区間: [{ci_lo:.4f}, {ci_hi:.4f}]"
    )

    st.caption(
        f"キャッシュ: ヒット {result.cache_hits} / ミス {result.cache_misses} "
        f"(ヒット率 {result.cache_hits / max(result.cache_hits + result.cache_misses, 1) * 100:.0f}%)"
    )

    if result.top_candidates:
        st.markdown("**上位候補**")
        rows = []
        for config, score, label in result.top_candidates:
            rows.append({"編成": label, "スコア": f"{score:.4f}"})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Beam Search + CMA-ES mode
# ---------------------------------------------------------------------------

def _render_beam_search(
    own_config: ArmyConfig, enemy_config: ArmyConfig,
    pool: HeroPool, weights: FitnessWeights,
) -> None:
    n_i, n_l, n_a = pool.leader_dims
    n_m = len(pool.members)
    total_leaders = n_i * n_l * n_a
    from math import comb
    total_members = comb(n_m, min(4, n_m))
    st.caption(f"リーダー候補: {total_leaders} 組 | メンバー組合せ: C({n_m},4) = {total_members:,}")

    with st.expander("探索パラメータ"):
        col1, col2 = st.columns(2)
        with col1:
            leader_top_k = st.slider("リーダー上位K", 5, 50, 20, step=5, key="bs_leader_k")
            beam_w = st.slider("ビーム幅 (最大)", 4, 30, 12, step=2, key="bs_beam_w")
        with col2:
            refine_top_k = st.slider("CMA-ES精査数", 2, 10, 5, key="bs_refine_k")
            n_reeval = st.slider("最終再評価回数", 50, 500, 200, step=50, key="bs_reeval")

        st.markdown("**CMA-ES 比率最適化設定**")
        col3, col4 = st.columns(2)
        with col3:
            bs_cma_sigma = st.slider("初期σ (探索幅)", 0.3, 1.5, 0.7, step=0.1, key="bs_cma_sigma")
            bs_cma_popsize = st.slider("CMA集団サイズ", 8, 48, 14, step=2, key="bs_cma_pop")
        with col4:
            bs_cma_generations = st.slider("CMA世代数", 10, 200, 25, step=5, key="bs_cma_gens")
            bs_cma_restarts = st.slider("マルチスタート数", 1, 5, 2, key="bs_cma_restarts")

    beam_widths = (beam_w, max(beam_w - 4, 4), max(beam_w - 6, 4), max(beam_w - 8, 4))
    cfg_preview = BeamSearchConfig(
        leader_top_k=leader_top_k, beam_widths=beam_widths,
        refine_top_k=refine_top_k, n_reeval_simulations=n_reeval,
        refine_cma_sigma=bs_cma_sigma, refine_cma_popsize=bs_cma_popsize,
        refine_cma_generations=bs_cma_generations, refine_cma_restarts=bs_cma_restarts,
    )
    cost = estimate_cost(cfg_preview, pool)
    st.caption(
        f"推定: ~{cost['total_expected']:,} expected-mode sims "
        f"(キャッシュ前: {cost['phase2_expected_raw']:,}) + "
        f"~{cost['total_mc']:,} MC sims"
    )
    if cost['total_expected'] > 500_000:
        st.warning("計算量が大きいです。ビーム幅またはリーダー上位Kを減らすことを推奨します。")

    # --- Session state management ---
    session_key = "bs_session"
    session: OptimizerSession | None = st.session_state.get(session_key)

    # Running state: poll for progress
    if session is not None and session.is_running():
        progress_bar = st.progress(0)
        status_text = st.empty()
        score_chart = st.empty()
        live_detail = st.empty()

        info = session.get_latest_info()
        if info is not None:
            _display_beam_progress(info, progress_bar, status_text, live_detail)

        # ログ履歴
        if session.info_history:
            with st.expander(f"ログ履歴 ({len(session.info_history)}件)", expanded=False):
                log_lines = []
                for h in session.info_history:
                    top = h.detail.get("top_candidates") or h.detail.get("top_leaders")
                    top_str = ""
                    if top:
                        top_str = " | " + ", ".join(f"{c['score']:.3f}:{c['label']}" for c in top[:3])
                    log_lines.append(f"[{h.phase}] {h.message}{top_str}")
                st.text("\n".join(log_lines))

        # スコア推移グラフ
        if session.score_history:
            import pandas as pd
            df_hist = pd.DataFrame(session.score_history, columns=["進捗", "ベストスコア"])
            score_chart.line_chart(df_hist.set_index("進捗"))

        if st.button("中止", key="bs_stop"):
            # スレッドはdaemonなので参照を切るだけで良い
            st.session_state[session_key] = None
            st.rerun()

        # Auto-refresh while running
        import time
        time.sleep(0.5)
        st.rerun()

    # Finished state: display results
    if session is not None and session.is_finished():
        if session.status.error:
            st.error(f"最適化エラー: {session.status.error}")
            st.session_state[session_key] = None
            return

        result = session.result
        if result is None:
            st.warning("結果が取得できませんでした。")
            st.session_state[session_key] = None
            return

        _display_beam_results(result, n_reeval, session.score_history)

        if st.button("結果をクリア", key="bs_clear"):
            st.session_state[session_key] = None
            st.rerun()
        return

    # Idle state: show run button
    if not st.button("最適化実行", key="bs_run"):
        return

    cfg = BeamSearchConfig(
        leader_top_k=leader_top_k,
        beam_widths=beam_widths,
        refine_top_k=refine_top_k,
        n_reeval_simulations=n_reeval,
        refine_cma_sigma=bs_cma_sigma,
        refine_cma_popsize=bs_cma_popsize,
        refine_cma_generations=bs_cma_generations,
        refine_cma_restarts=bs_cma_restarts,
    )
    evaluator = FitnessEvaluator(
        enemy_config=enemy_config,
        own_template=own_config,
        pool=pool,
        weights=weights,
    )
    session = OptimizerSession()
    session.start(evaluator, pool, cfg)
    st.session_state[session_key] = session
    st.rerun()


def _display_beam_progress(
    info: PhaseInfo,
    progress_bar, status_text, live_detail,
) -> None:
    """PhaseInfoからプログレス表示を更新。"""
    phase_names = {
        "leader_screen": "Phase 1: リーダー探索",
        "beam_search": "Phase 2: メンバービームサーチ",
        "cma_refine": "Phase 3: CMA-ES比率最適化",
        "reeval": "Phase 4: 最終再評価",
    }
    phase_label = phase_names.get(info.phase, info.phase)

    phase_weights = {"leader_screen": (0.0, 0.3), "beam_search": (0.3, 0.7),
                     "cma_refine": (0.7, 0.9), "reeval": (0.9, 1.0)}
    lo, hi = phase_weights.get(info.phase, (0.0, 1.0))
    overall = lo + (hi - lo) * info.progress
    progress_bar.progress(min(overall, 1.0))

    status_text.markdown(
        f"**{phase_label}** | ベスト: `{info.best_score:.4f}`"
    )

    # 詳細テキスト: 上位候補リスト表示
    detail_lines = [info.message]

    top_candidates = info.detail.get("top_candidates") or info.detail.get("top_leaders")
    if top_candidates:
        detail_lines.append("─── 上位候補 ───")
        for rank, item in enumerate(top_candidates, 1):
            detail_lines.append(f"{rank}. [{item['score']:.4f}] {item['label']}")

    # Phase別の追加情報
    if info.phase == "leader_screen":
        active = info.detail.get("active", "?")
        rnd = info.detail.get("round", "?")
        detail_lines.insert(1, f"Round {rnd} | 残候補: {active}")
    elif info.phase == "beam_search":
        li = info.detail.get("leader_idx", "?")
        total = info.detail.get("total_leaders", "?")
        cache = info.detail.get("cache_size", "?")
        detail_lines.insert(1, f"リーダー {li}/{total} | キャッシュサイズ: {cache}")
    elif info.phase == "cma_refine":
        cand = info.detail.get("candidate", "?")
        total = info.detail.get("total", "?")
        beam_sc = info.detail.get("beam_score", 0.0)
        ref_sc = info.detail.get("refined_score", 0.0)
        detail_lines.insert(1, f"候補 {cand}/{total} | ビームスコア: {beam_sc:.4f} → 精査後: {ref_sc:.4f}")

    live_detail.markdown(f"```\n{chr(10).join(detail_lines)}\n```")


def _display_beam_results(result, n_reeval: int, score_history: list = None) -> None:
    """BeamSearchResultを表示。"""
    ci_lo, ci_hi = result.confidence_interval
    st.markdown("**最適化完了!**")

    # スコア推移グラフ
    if score_history:
        df_hist = pd.DataFrame(score_history, columns=["進捗", "ベストスコア"])
        st.line_chart(df_hist.set_index("進捗"))

    st.success(
        f"最適解: {result.label}\n\n"
        f"探索中スコア: {result.best_score:.4f} | "
        f"再評価スコア ({n_reeval}回): **{result.best_score_reeval:.4f}**\n\n"
        f"95%信頼区間: [{ci_lo:.4f}, {ci_hi:.4f}]"
    )

    st.caption(
        f"キャッシュ: ヒット {result.cache_hits} / ミス {result.cache_misses} "
        f"(ヒット率 {result.cache_hits / max(result.cache_hits + result.cache_misses, 1) * 100:.0f}%)"
    )

    if result.top_candidates:
        st.markdown("**上位候補**")
        rows = []
        for config, score, label in result.top_candidates:
            rows.append({"編成": label, "スコア": f"{score:.4f}"})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# CMA-ES mode (従来)
# ---------------------------------------------------------------------------

def _render_cma_es(
    own_config: ArmyConfig, enemy_config: ArmyConfig,
    pool: HeroPool, weights: FitnessWeights,
) -> None:
    n_i, n_l, n_a = pool.leader_dims
    n_m = len(pool.members)
    st.caption(f"探索次元: {pool.genome_dim} "
               f"(比率3 + 盾L{n_i} + 槍L{n_l} + 弓L{n_a} + メンバー4×{n_m})")

    col1, col2 = st.columns(2)
    with col1:
        n_sim = st.slider("モンテカルロ回数/個体", 10, 200, 30, step=10, key="opt_nsim")
    with col2:
        pop = st.slider("個体数", 10, 100, 20, step=10, key="opt_pop")
        gens = st.slider("最大世代数", 10, 200, 50, step=10, key="opt_gens")

    total_sims = pop * gens * n_sim
    st.caption(f"推定シミュレーション回数: 最大 {total_sims:,} 回 "
               f"(キャッシュにより実際は少なくなります)")
    if total_sims > 500_000:
        st.warning("計算量が非常に大きいです。個体数か世代数を減らすことを推奨します。")

    if not st.button("最適化実行", key="opt_run"):
        return

    progress_bar = st.progress(0)
    status_text = st.empty()
    live_detail = st.empty()
    live_pop_table = st.empty()

    all_history: list[GenerationInfo] = []

    def on_generation(info: GenerationInfo) -> None:
        all_history.append(info)
        progress_bar.progress(min(info.gen / info.total, 1.0))

        status_text.markdown(
            f"**世代 {info.gen}/{info.total}** | "
            f"ベスト: `{info.best_fitness:.4f}` | "
            f"sigma: `{info.sigma:.4f}`"
        )

        live_detail.markdown(
            f"```\n"
            f"世代{info.gen}ベスト: {info.best_label}\n"
            f"```"
        )

        label_counts = Counter(info.pop_labels)
        top5 = label_counts.most_common(5)
        rows = []
        for label, count in top5:
            scores = [s for l, s in zip(info.pop_labels, info.pop_fitnesses) if l == label]
            avg = sum(scores) / len(scores)
            rows.append({
                "編成": label,
                "個体数": count,
                "平均スコア": f"{avg:.4f}",
            })
        live_pop_table.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    config = CMAConfig(
        population_size=pop,
        max_iterations=gens,
    )
    evaluator = FitnessEvaluator(
        enemy_config=enemy_config,
        own_template=own_config,
        pool=pool,
        n_simulations=n_sim,
        weights=weights,
    )

    optimizer = CMAOptimizer(evaluator, pool, config)
    opt_result = optimizer.run(callback=on_generation)

    progress_bar.progress(1.0)
    live_detail.empty()
    live_pop_table.empty()

    if not all_history:
        st.warning("最適化が収束条件により即座に終了しました。パラメータを調整してください。")
        return

    status_text.markdown(f"**最適化完了!** 最終sigma: `{all_history[-1].sigma:.4f}`")

    st.success(
        f"最適解: {opt_result.label}\n\n"
        f"探索中スコア: {opt_result.best_fitness:.4f} | "
        f"再評価スコア ({config.n_reeval_simulations}回): **{opt_result.best_fitness_reeval:.4f}**"
    )

    st.caption(
        f"キャッシュ: ヒット {opt_result.cache_hits} / ミス {opt_result.cache_misses} "
        f"(ヒット率 {opt_result.cache_hits / max(opt_result.cache_hits + opt_result.cache_misses, 1) * 100:.0f}%)"
    )

    chart_data = pd.DataFrame([
        {"世代": h.gen, "ベストスコア": h.best_fitness, "sigma": h.sigma}
        for h in all_history
    ])

    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.markdown("**スコア推移**")
        st.line_chart(chart_data.set_index("世代")[["ベストスコア"]])
    with col_c2:
        st.markdown("**sigma (探索幅) 推移**")
        st.line_chart(chart_data.set_index("世代")[["sigma"]])

    last = all_history[-1]
    st.markdown("**最終世代の集団分布**")
    label_counts = Counter(last.pop_labels)
    rows = []
    for label, count in label_counts.most_common(10):
        scores = [s for l, s in zip(last.pop_labels, last.pop_fitnesses) if l == label]
        avg = sum(scores) / len(scores)
        rows.append({
            "編成": label,
            "個体数": count,
            "平均スコア": f"{avg:.4f}",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
