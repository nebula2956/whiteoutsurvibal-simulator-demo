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
from optimizer.optimizer_session import OptimizerSession
from optimizer.fitness import FitnessEvaluator, FitnessWeights

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

    mode = st.radio("最適化モード", ["Beam Search + CMA-ES", "CMA-ES (従来)"],
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

    with st.expander("評価指標の重み設定"):
        weights = _build_weights("opt")

    if weights.is_all_zero():
        st.error("評価指標の重みを最低1つは0より大きくしてください。")
        return

    if mode == "Beam Search + CMA-ES":
        _render_beam_search(own_config, enemy_config, pool, weights)
    else:
        _render_cma_es(own_config, enemy_config, pool, weights)


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

    beam_widths = (beam_w, max(beam_w - 4, 4), max(beam_w - 6, 4), max(beam_w - 8, 4))
    cfg_preview = BeamSearchConfig(
        leader_top_k=leader_top_k, beam_widths=beam_widths,
        refine_top_k=refine_top_k, n_reeval_simulations=n_reeval,
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
        live_detail = st.empty()

        info = session.get_latest_info()
        if info is not None:
            _display_beam_progress(info, progress_bar, status_text, live_detail)

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

        _display_beam_results(result, n_reeval)

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
    live_detail.markdown(f"```\n{info.message}\n{info.best_label}\n```")


def _display_beam_results(result, n_reeval: int) -> None:
    """BeamSearchResultを表示。"""
    ci_lo, ci_hi = result.confidence_interval
    st.markdown("**最適化完了!**")

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
