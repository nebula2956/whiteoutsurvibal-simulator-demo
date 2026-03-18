"""Leader Enumeration + Member Beam Search + CMA-ES Ratio Refinement optimizer.

Phase 1: Enumerate all leader combinations, screen with expected-mode sims.
Phase 2: Beam search over member combinations for top-K leaders.
Phase 3: CMA-ES refinement on 2D ratio space for top candidates.
"""
from __future__ import annotations

import itertools
import math
import random
import numpy as np
import cma
from dataclasses import dataclass, field
from typing import Callable, Dict, FrozenSet, List, Optional, Tuple

from simulator.models import ArmyConfig, HeroConfig
from .genome import HeroPool, OwnedHero, build_army_config
from .fitness import FitnessEvaluator


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class BeamSearchConfig:
    # Phase 1: Leader screening
    leader_top_k: int = 20
    n_member_samples: int = 5
    ratio_grid_resolution: int = 3       # -> 10 simplex points
    ratio_dirichlet_extra: int = 15      # total ~25 ratio points
    # Phase 2: Beam search
    beam_widths: Tuple[int, ...] = (16, 12, 10, 8)
    beam_elite_ratio: float = 0.7       # stochastic beam: 70% elite + 30% random
    n_random_completions: int = 5
    coarse_grid_resolution: int = 2      # -> 6 simplex points (depth 1-2)
    # Phase 3: CMA-ES refinement
    refine_top_k: int = 5
    refine_cma_popsize: int = 8
    refine_cma_generations: int = 10
    refine_n_sim: int = 30
    # Final
    n_reeval_simulations: int = 200


# ---------------------------------------------------------------------------
# Callback info
# ---------------------------------------------------------------------------

@dataclass
class PhaseInfo:
    phase: str          # "leader_screen" | "beam_search" | "cma_refine" | "reeval"
    progress: float     # 0.0 - 1.0
    message: str
    best_label: str
    best_score: float
    detail: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class BeamSearchResult:
    best_config: ArmyConfig
    best_score: float
    best_score_reeval: float
    confidence_interval: Tuple[float, float]
    label: str
    top_candidates: List[Tuple[ArmyConfig, float, str]]
    cache_hits: int
    cache_misses: int


# ---------------------------------------------------------------------------
# Ratio grid helpers
# ---------------------------------------------------------------------------

def simplex_grid(resolution: int) -> List[Tuple[float, float, float]]:
    """Evenly-spaced points on the 2-simplex. resolution=3 -> 10 points."""
    points: List[Tuple[float, float, float]] = []
    for i in range(resolution + 1):
        for j in range(resolution + 1 - i):
            k = resolution - i - j
            points.append((i / resolution, j / resolution, k / resolution))
    return points


def _dirichlet_samples(n: int, alpha: float = 1.0) -> List[Tuple[float, float, float]]:
    samples = np.random.dirichlet([alpha, alpha, alpha], size=n)
    return [(float(r[0]), float(r[1]), float(r[2])) for r in samples]


def build_ratio_grid(cfg: BeamSearchConfig) -> List[Tuple[float, float, float]]:
    """Full ratio grid = simplex_grid + Dirichlet extras."""
    grid = simplex_grid(cfg.ratio_grid_resolution)
    grid += _dirichlet_samples(cfg.ratio_dirichlet_extra)
    return grid


def build_coarse_grid(cfg: BeamSearchConfig) -> List[Tuple[float, float, float]]:
    """Coarse grid for early beam depths."""
    return simplex_grid(cfg.coarse_grid_resolution)


def _local_refine_grid(
    center: Tuple[float, float, float],
    n_points: int = 9,
    spread: float = 0.08,
) -> List[Tuple[float, float, float]]:
    """Generate a local grid around a center ratio point on the simplex.

    Perturbs each component by [-spread, 0, +spread] combinations,
    then projects back onto the simplex (normalize to sum=1, clamp to >=0.01).
    """
    offsets = [-spread, 0.0, spread]
    points: List[Tuple[float, float, float]] = []
    seen: set = set()

    for di in offsets:
        for dj in offsets:
            raw = [center[0] + di, center[1] + dj, center[2] - di - dj]
            # Clamp to positive
            raw = [max(0.01, v) for v in raw]
            # Normalize to simplex
            total = sum(raw)
            pt = (round(raw[0] / total, 4), round(raw[1] / total, 4), round(raw[2] / total, 4))
            if pt not in seen:
                seen.add(pt)
                points.append(pt)
            if len(points) >= n_points:
                break
        if len(points) >= n_points:
            break

    return points


# ---------------------------------------------------------------------------
# Hero helpers
# ---------------------------------------------------------------------------

def _leaders_to_heroes(
    inf_leader: OwnedHero, lan_leader: OwnedHero, arc_leader: OwnedHero,
) -> List[HeroConfig]:
    return [
        HeroConfig(inf_leader.hero_id, inf_leader.gear_level, True, 0),
        HeroConfig(lan_leader.hero_id, lan_leader.gear_level, False, 1),
        HeroConfig(arc_leader.hero_id, arc_leader.gear_level, False, 2),
    ]


def _full_heroes(
    leader_heroes: List[HeroConfig], member_ids: List[str],
) -> List[HeroConfig]:
    heroes = list(leader_heroes)
    for i, mid in enumerate(member_ids):
        heroes.append(HeroConfig(mid, 0, False, 3 + i))
    return heroes


def candidate_label(
    leader_heroes: List[HeroConfig], member_ids: List[str],
    ratio: Tuple[float, float, float],
) -> str:
    leaders = "/".join(h.hero_id for h in leader_heroes[:3])
    members = "/".join(member_ids)
    return (f"盾{ratio[0]*100:.0f}% 槍{ratio[1]*100:.0f}% 弓{ratio[2]*100:.0f}% | "
            f"L: {leaders} | M: {members}")


# ---------------------------------------------------------------------------
# Diversity-aware beam selection
# ---------------------------------------------------------------------------

def _jaccard_distance(a: FrozenSet[str], b: FrozenSet[str]) -> float:
    """Jaccard distance between two member sets. 0=identical, 1=disjoint."""
    if not a and not b:
        return 0.0
    return 1.0 - len(a & b) / len(a | b)


def _diversity_select(
    candidates: List[Tuple[float, List[str], Tuple[float, float, float]]],
    already_selected: List[FrozenSet[str]],
    n_pick: int,
) -> List[List[str]]:
    """Select n_pick candidates that maximize diversity w.r.t. already_selected.

    Uses score-weighted sampling where weights are boosted by mean Jaccard
    distance to already-selected sets. This balances exploration (diverse
    member sets) with exploitation (high scores).
    """
    if n_pick <= 0 or not candidates:
        return []

    selected: List[List[str]] = []
    pool = list(candidates)

    for _ in range(n_pick):
        if not pool:
            break

        weights = []
        for score, members, ratio in pool:
            member_set = frozenset(members)
            # Mean Jaccard distance to all already-selected sets
            all_selected = already_selected + [frozenset(m) for m in selected]
            if all_selected:
                diversity = sum(_jaccard_distance(member_set, s) for s in all_selected) / len(all_selected)
            else:
                diversity = 1.0
            # Weight = score_component × diversity_component
            # Normalize score to [0.5, 1.5] range relative to pool
            # diversity in [0, 1], map to [0.5, 1.5]
            w = max(score, 0.001) * (0.5 + diversity)
            weights.append(w)

        total = sum(weights)
        if total <= 0:
            # Fallback to uniform random
            idx = random.randrange(len(pool))
        else:
            probs = [w / total for w in weights]
            idx = random.choices(range(len(pool)), weights=probs, k=1)[0]

        _, members, _ = pool.pop(idx)
        selected.append(members)

    return selected


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _evaluate_on_grid(
    heroes: List[HeroConfig],
    ratio_grid: List[Tuple[float, float, float]],
    evaluator: FitnessEvaluator,
    refine: bool = False,
) -> Tuple[float, Tuple[float, float, float]]:
    """Evaluate hero set across ratio grid (expected mode, batch parallel).

    If refine=True, performs a two-stage evaluation: first the given grid,
    then a local refinement grid around the top-2 ratio points.
    Returns (best_score, best_ratio).
    """
    template = evaluator.own_template
    configs = [build_army_config(heroes, ratio, template) for ratio in ratio_grid]
    scores = evaluator.evaluate_batch_expected(configs)

    if not refine:
        best_idx = int(np.argmax(scores))
        return scores[best_idx], ratio_grid[best_idx]

    # Two-stage: refine around top-2 ratio points
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    refine_grid: List[Tuple[float, float, float]] = []
    seen = set(ratio_grid)
    for idx, _ in ranked[:2]:
        for pt in _local_refine_grid(ratio_grid[idx]):
            if pt not in seen:
                refine_grid.append(pt)
                seen.add(pt)

    if refine_grid:
        refine_configs = [build_army_config(heroes, ratio, template) for ratio in refine_grid]
        refine_scores = evaluator.evaluate_batch_expected(refine_configs)
        # Merge with original scores
        all_ratios = list(ratio_grid) + refine_grid
        all_scores = list(scores) + list(refine_scores)
    else:
        all_ratios = list(ratio_grid)
        all_scores = list(scores)

    best_idx = int(np.argmax(all_scores))
    return all_scores[best_idx], all_ratios[best_idx]


def _evaluate_partial(
    leader_heroes: List[HeroConfig],
    partial_members: List[str],
    available_members: List[str],
    ratio_grid: List[Tuple[float, float, float]],
    evaluator: FitnessEvaluator,
    n_completions: int,
    refine: bool = False,
) -> Tuple[float, Tuple[float, float, float]]:
    """Evaluate partial member set by random completion. Returns (avg_best_score, best_ratio)."""
    needed = 4 - len(partial_members)
    if needed == 0:
        heroes = _full_heroes(leader_heroes, partial_members)
        return _evaluate_on_grid(heroes, ratio_grid, evaluator, refine=refine)

    # Candidates for completion: pool minus already chosen
    remaining = [m for m in available_members if m not in partial_members]
    if len(remaining) < needed:
        remaining = available_members  # fallback

    scores = []
    best_ratio_overall = ratio_grid[0]
    best_score_overall = float("-inf")

    for _ in range(n_completions):
        if len(remaining) >= needed:
            completion = random.sample(remaining, needed)
        else:
            completion = random.choices(remaining, k=needed)
        full_members = sorted(partial_members + completion)
        heroes = _full_heroes(leader_heroes, full_members)
        score, ratio = _evaluate_on_grid(heroes, ratio_grid, evaluator)
        scores.append(score)
        if score > best_score_overall:
            best_score_overall = score
            best_ratio_overall = ratio

    return float(np.mean(scores)), best_ratio_overall


# ---------------------------------------------------------------------------
# CMA-ES Ratio Refiner
# ---------------------------------------------------------------------------

class RatioRefiner:
    """CMA-ES on 2D ratio space for a fixed hero set."""

    def __init__(self, evaluator: FitnessEvaluator, heroes: List[HeroConfig],
                 cfg: BeamSearchConfig):
        self.evaluator = evaluator
        self.heroes = heroes
        self.cfg = cfg

    def run(self, initial_ratio: Tuple[float, float, float]) -> Tuple[Tuple[float, float, float], float]:
        eps = 1e-8
        x0 = np.log(np.array(initial_ratio) + eps)[:2]

        opts = {
            "popsize": self.cfg.refine_cma_popsize,
            "maxiter": self.cfg.refine_cma_generations,
            "verbose": -9,
            "bounds": [[-5.0, -5.0], [5.0, 5.0]],
            "tolfun": 0,
            "tolx": 0,
            "tolflatfitness": self.cfg.refine_cma_generations + 1,
            "tolfunhist": 0,
            "tolstagnation": self.cfg.refine_cma_generations + 1,
        }

        es = cma.CMAEvolutionStrategy(x0.tolist(), 0.3, opts)
        template = self.evaluator.own_template

        while not es.stop():
            solutions = es.ask()
            fitnesses = []
            for s in solutions:
                ratio = self._decode_ratio(np.array(s))
                config = build_army_config(self.heroes, ratio, template)
                score = self.evaluator.evaluate_config(config, n_sim=self.cfg.refine_n_sim)
                fitnesses.append(-score)
            es.tell(solutions, fitnesses)

        best_x = np.array(es.result.xbest)
        best_ratio = self._decode_ratio(best_x)
        best_score = -es.result.fbest
        return best_ratio, best_score

    @staticmethod
    def _decode_ratio(x2d: np.ndarray) -> Tuple[float, float, float]:
        log3 = np.array([x2d[0], x2d[1], 0.0])
        exp3 = np.exp(log3)
        r = exp3 / exp3.sum()
        return (float(r[0]), float(r[1]), float(r[2]))


# ---------------------------------------------------------------------------
# Main Optimizer
# ---------------------------------------------------------------------------

def estimate_cost(cfg: BeamSearchConfig, pool: HeroPool) -> dict:
    """推定シミュレーション回数を計算。UIの表示用。"""
    n_i, n_l, n_a = pool.leader_dims
    n_m = len(pool.members)
    total_leaders = n_i * n_l * n_a

    full_grid_size = len(simplex_grid(cfg.ratio_grid_resolution)) + cfg.ratio_dirichlet_extra
    coarse_grid_size = len(simplex_grid(cfg.coarse_grid_resolution))

    # Phase 1: SHA rounds (coarse→coarse→full)
    coarse_grid_size_p1 = len(simplex_grid(cfg.coarse_grid_resolution))
    sha_r1 = total_leaders * 1 * coarse_grid_size_p1
    sha_r2 = int(total_leaders * 0.5) * 2 * coarse_grid_size_p1
    sha_r3 = max(cfg.leader_top_k, int(total_leaders * 0.25)) * cfg.n_member_samples * full_grid_size
    phase1 = sha_r1 + sha_r2 + sha_r3

    # Phase 2: per leader_top_k, per depth, beam_w × n_m candidates × completions × grid
    phase2 = 0
    for d in range(4):
        bw = cfg.beam_widths[d] if d < len(cfg.beam_widths) else cfg.beam_widths[-1]
        grid_size = coarse_grid_size if d < 2 else full_grid_size
        completions = 3 if d < 2 else cfg.n_random_completions
        if d == 3:
            completions = 1  # depth 4 = complete set, no random completion
        phase2 += bw * n_m * completions * grid_size
    phase2 *= cfg.leader_top_k
    # Cache reduces this significantly (estimate ~60% cache hit)
    phase2_with_cache = int(phase2 * 0.4)

    # Phase 3: CMA-ES
    phase3 = cfg.refine_top_k * cfg.refine_cma_popsize * cfg.refine_cma_generations * cfg.refine_n_sim

    # Phase 4: re-evaluation
    phase4 = cfg.refine_top_k * cfg.n_reeval_simulations

    return {
        "phase1_expected": phase1,
        "phase2_expected_raw": phase2,
        "phase2_expected_cached": phase2_with_cache,
        "phase3_mc": phase3,
        "phase4_mc": phase4,
        "total_expected": phase1 + phase2_with_cache,
        "total_mc": phase3 + phase4,
    }


class BeamSearchOptimizer:
    """Leader Enumeration + Member Beam Search + CMA-ES Ratio Refinement."""

    def __init__(self, evaluator: FitnessEvaluator, pool: HeroPool,
                 config: BeamSearchConfig):
        self.evaluator = evaluator
        self.pool = pool
        self.config = config

        self.full_grid = build_ratio_grid(config)
        self.coarse_grid = build_coarse_grid(config)

    def run(self, callback: Optional[Callable[[PhaseInfo], None]] = None
            ) -> BeamSearchResult:
        cfg = self.config

        # --- Phase 1: Leader Screening ---
        leader_results = self._phase1_leader_screen(callback)

        # --- Phase 2: Beam Search ---
        beam_results = self._phase2_beam_search(leader_results, callback)

        # --- Phase 3: CMA-ES Refinement ---
        refined = self._phase3_cma_refine(beam_results, callback)

        # --- Final Re-evaluation ---
        best_config, best_score, best_label = refined[0]
        reeval_score = self.evaluator.evaluate_config(best_config, n_sim=cfg.n_reeval_simulations)

        # Confidence interval
        ci = self._confidence_interval(reeval_score, cfg.n_reeval_simulations)

        if callback:
            callback(PhaseInfo(
                phase="reeval", progress=1.0,
                message=f"再評価完了: {reeval_score:.4f} (95%CI: {ci[0]:.4f}-{ci[1]:.4f})",
                best_label=best_label, best_score=reeval_score,
            ))

        return BeamSearchResult(
            best_config=best_config,
            best_score=best_score,
            best_score_reeval=reeval_score,
            confidence_interval=ci,
            label=best_label,
            top_candidates=refined[:cfg.refine_top_k],
            cache_hits=self.evaluator.cache_hits,
            cache_misses=self.evaluator.cache_misses,
        )

    # -----------------------------------------------------------------------
    # Phase 1
    # -----------------------------------------------------------------------

    def _phase1_leader_screen(
        self, callback: Optional[Callable[[PhaseInfo], None]],
    ) -> List[Tuple[Tuple[int, int, int], float, Tuple[float, float, float]]]:
        """Successive Halving でリーダーをスクリーニング。
        Round 1: 全候補を少数sim(1 sample, coarse grid)で粗評価 → 上位50%に絞る
        Round 2: 残りを中程度sim(2 samples, coarse grid)で再評価 → 上位50%に絞る
        Round 3: 残りを完全評価(n_member_samples, full grid) → 上位 leader_top_k
        """
        cfg = self.config
        pool = self.pool

        leader_combos = list(itertools.product(
            range(len(pool.infantry_leaders)),
            range(len(pool.lancer_leaders)),
            range(len(pool.archer_leaders)),
        ))

        # SHA rounds: (n_samples, ratio_grid, keep_ratio)
        sha_rounds = [
            (1, self.coarse_grid, 0.5),
            (2, self.coarse_grid, 0.5),
            (cfg.n_member_samples, self.full_grid, None),  # final round keeps leader_top_k
        ]

        # Track cumulative scores per leader combo
        # key: leader_indices → (sum_of_scores, count, best_score, best_ratio)
        active: List[Tuple[int, int, int]] = [combo for combo in leader_combos]
        leader_data: Dict[Tuple[int, int, int], Tuple[float, int, float, Tuple[float, float, float]]] = {}

        total_work = sum(
            len(active) if r == 0 else int(len(leader_combos) * 0.5 ** r)
            for r in range(len(sha_rounds))
        )
        work_done = 0

        for round_i, (n_samples, grid, keep_ratio) in enumerate(sha_rounds):
            for combo in active:
                i_idx, l_idx, a_idx = combo
                leader_heroes = _leaders_to_heroes(
                    pool.infantry_leaders[i_idx],
                    pool.lancer_leaders[l_idx],
                    pool.archer_leaders[a_idx],
                )

                prev_sum, prev_count, prev_best_score, prev_best_ratio = leader_data.get(
                    combo, (0.0, 0, float("-inf"), grid[0])
                )

                round_scores = []
                best_score_this = prev_best_score
                best_ratio_this = prev_best_ratio

                for _ in range(n_samples):
                    if len(pool.members) >= 4:
                        members = sorted(random.sample(pool.members, 4))
                    else:
                        members = sorted(random.choices(pool.members, k=4))
                    heroes = _full_heroes(leader_heroes, members)
                    score, ratio = _evaluate_on_grid(heroes, grid, self.evaluator)
                    round_scores.append(score)
                    if score > best_score_this:
                        best_score_this = score
                        best_ratio_this = ratio

                new_sum = prev_sum + sum(round_scores)
                new_count = prev_count + len(round_scores)
                leader_data[combo] = (new_sum, new_count, best_score_this, best_ratio_this)

                work_done += 1
                if callback and work_done % max(1, total_work // 20) == 0:
                    # Find current best
                    best_combo = max(active, key=lambda c: leader_data.get(c, (0, 1, 0, grid[0]))[0] / max(leader_data.get(c, (0, 1, 0, grid[0]))[1], 1))
                    bd = leader_data[best_combo]
                    best_leaders = _leaders_to_heroes(
                        pool.infantry_leaders[best_combo[0]],
                        pool.lancer_leaders[best_combo[1]],
                        pool.archer_leaders[best_combo[2]],
                    )
                    label = candidate_label(best_leaders, ["?"] * 4, bd[3])
                    callback(PhaseInfo(
                        phase="leader_screen",
                        progress=work_done / total_work,
                        message=f"SHA Round {round_i+1}/{len(sha_rounds)}: "
                                f"{len(active)} 候補評価中",
                        best_label=label, best_score=bd[0] / max(bd[1], 1),
                        detail={"round": round_i + 1, "active": len(active)},
                    ))

            # Halve: keep top candidates
            if keep_ratio is not None:
                scored = [
                    (combo, leader_data[combo][0] / max(leader_data[combo][1], 1))
                    for combo in active
                ]
                scored.sort(key=lambda x: x[1], reverse=True)
                n_keep = max(cfg.leader_top_k, int(len(scored) * keep_ratio))
                active = [combo for combo, _ in scored[:n_keep]]
            else:
                # Final round: keep leader_top_k
                scored = [
                    (combo, leader_data[combo][0] / max(leader_data[combo][1], 1))
                    for combo in active
                ]
                scored.sort(key=lambda x: x[1], reverse=True)
                active = [combo for combo, _ in scored[:cfg.leader_top_k]]

        # Build final results
        results = []
        for combo in active:
            data = leader_data[combo]
            avg_score = data[0] / max(data[1], 1)
            best_ratio = data[3]
            results.append((combo, avg_score, best_ratio))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:cfg.leader_top_k]

    # -----------------------------------------------------------------------
    # Phase 2
    # -----------------------------------------------------------------------

    def _phase2_beam_search(
        self,
        leader_results: List[Tuple[Tuple[int, int, int], float, Tuple[float, float, float]]],
        callback: Optional[Callable[[PhaseInfo], None]],
    ) -> List[Tuple[List[HeroConfig], List[str], float, Tuple[float, float, float]]]:
        """Returns list of (leader_heroes, member_ids, score, best_ratio)."""
        cfg = self.config
        pool = self.pool
        all_candidates = []
        total_leaders = len(leader_results)

        for li, (leader_idx, _, _) in enumerate(leader_results):
            i_idx, l_idx, a_idx = leader_idx
            leader_heroes = _leaders_to_heroes(
                pool.infantry_leaders[i_idx],
                pool.lancer_leaders[l_idx],
                pool.archer_leaders[a_idx],
            )
            leader_key = (
                pool.infantry_leaders[i_idx].hero_id,
                pool.lancer_leaders[l_idx].hero_id,
                pool.archer_leaders[a_idx].hero_id,
            )

            # Partial evaluation cache for this leader set
            partial_cache: Dict[FrozenSet[str], Tuple[float, Tuple[float, float, float]]] = {}

            # Start beam search
            beam: List[List[str]] = [[]]

            for depth in range(4):
                beam_w = cfg.beam_widths[depth] if depth < len(cfg.beam_widths) else cfg.beam_widths[-1]
                # Adaptive: coarse grid for early depths, full grid for later
                grid = self.coarse_grid if depth < 2 else self.full_grid
                n_comp = 3 if depth < 2 else cfg.n_random_completions

                candidates: List[Tuple[float, List[str], Tuple[float, float, float]]] = []

                for partial in beam:
                    partial_set = set(partial)
                    for hero in pool.members:
                        if hero in partial_set:
                            continue
                        new_partial = sorted(partial + [hero])
                        cache_key = frozenset(new_partial)

                        if cache_key in partial_cache:
                            score, ratio = partial_cache[cache_key]
                        else:
                            # Adaptive: refine ratio grid at final depth (complete set)
                            use_refine = (depth == 3)
                            score, ratio = _evaluate_partial(
                                leader_heroes, new_partial, pool.members,
                                grid, self.evaluator, n_comp,
                                refine=use_refine,
                            )
                            partial_cache[cache_key] = (score, ratio)

                        candidates.append((score, new_partial, ratio))

                # Stochastic beam selection: elite + random
                candidates.sort(key=lambda x: x[0], reverse=True)
                # Deduplicate first
                seen: set = set()
                unique_candidates: List[Tuple[float, List[str], Tuple[float, float, float]]] = []
                for score, members, ratio in candidates:
                    key = frozenset(members)
                    if key not in seen:
                        seen.add(key)
                        unique_candidates.append((score, members, ratio))

                n_elite = max(1, int(beam_w * cfg.beam_elite_ratio))
                n_random = beam_w - n_elite

                beam = []
                # Elite selection
                for score, members, ratio in unique_candidates[:n_elite]:
                    beam.append(members)
                # Diversity-aware selection from remainder
                remainder = unique_candidates[n_elite:]
                if remainder and n_random > 0:
                    selected_sets = [frozenset(m) for m in beam]
                    n_pick = min(n_random, len(remainder))
                    diversity_selected = _diversity_select(
                        remainder, selected_sets, n_pick,
                    )
                    for members in diversity_selected:
                        beam.append(members)

            # Collect final beam results
            for members in beam:
                key = frozenset(members)
                if key in partial_cache:
                    score, ratio = partial_cache[key]
                else:
                    heroes = _full_heroes(leader_heroes, members)
                    score, ratio = _evaluate_on_grid(heroes, self.full_grid, self.evaluator, refine=True)
                all_candidates.append((leader_heroes, members, score, ratio))

            if callback:
                all_candidates_sorted = sorted(all_candidates, key=lambda x: x[2], reverse=True)
                best = all_candidates_sorted[0]
                label = candidate_label(best[0], best[1], best[3])
                callback(PhaseInfo(
                    phase="beam_search",
                    progress=(li + 1) / total_leaders,
                    message=f"ビームサーチ: {li+1}/{total_leaders} リーダーセット完了",
                    best_label=label, best_score=best[2],
                    detail={"leader_idx": li + 1, "total_leaders": total_leaders,
                            "cache_size": len(partial_cache)},
                ))

        # Sort all candidates and return top
        all_candidates.sort(key=lambda x: x[2], reverse=True)

        # Deduplicate by hero set
        seen_sets: set = set()
        unique: List[Tuple[List[HeroConfig], List[str], float, Tuple[float, float, float]]] = []
        for leader_h, members, score, ratio in all_candidates:
            hero_key = (tuple(h.hero_id for h in leader_h), tuple(members))
            if hero_key not in seen_sets:
                seen_sets.add(hero_key)
                unique.append((leader_h, members, score, ratio))

        return unique

    # -----------------------------------------------------------------------
    # Phase 3
    # -----------------------------------------------------------------------

    def _phase3_cma_refine(
        self,
        beam_results: List[Tuple[List[HeroConfig], List[str], float, Tuple[float, float, float]]],
        callback: Optional[Callable[[PhaseInfo], None]],
    ) -> List[Tuple[ArmyConfig, float, str]]:
        """CMA-ES ratio refinement for top-K candidates. Returns (config, score, label) list."""
        cfg = self.config
        top_k = beam_results[:cfg.refine_top_k]
        refined: List[Tuple[ArmyConfig, float, str]] = []

        for i, (leader_heroes, member_ids, beam_score, beam_ratio) in enumerate(top_k):
            heroes = _full_heroes(leader_heroes, member_ids)
            refiner = RatioRefiner(self.evaluator, heroes, cfg)
            best_ratio, best_score = refiner.run(beam_ratio)

            config = build_army_config(heroes, best_ratio, self.evaluator.own_template)
            label = candidate_label(leader_heroes, member_ids, best_ratio)
            refined.append((config, best_score, label))

            if callback:
                callback(PhaseInfo(
                    phase="cma_refine",
                    progress=(i + 1) / len(top_k),
                    message=f"CMA-ES比率最適化: {i+1}/{len(top_k)}",
                    best_label=label, best_score=best_score,
                    detail={"candidate": i + 1, "total": len(top_k),
                            "beam_score": beam_score, "refined_score": best_score},
                ))

        refined.sort(key=lambda x: x[1], reverse=True)
        return refined

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _confidence_interval(win_rate: float, n: int) -> Tuple[float, float]:
        """95% confidence interval for win rate (binomial proportion)."""
        se = math.sqrt(max(win_rate * (1 - win_rate), 0) / max(n, 1))
        margin = 1.96 * se
        return (max(0.0, win_rate - margin), min(1.0, win_rate + margin))
