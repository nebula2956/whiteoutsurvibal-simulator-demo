"""エピック vs レジェンド メンバー比較検証スクリプト"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from simulator.models import ArmyConfig, HeroConfig, StatBuff
from simulator import run_simulation

BASE_BUFF = StatBuff(atk=3.0, def_=3.0, hp=3.0, lethality=3.0)
LEADERS = [
    HeroConfig("Natalia", 5, True, 0),
    HeroConfig("Molly", 5, False, 1),
    HeroConfig("Zinman", 5, False, 2),
]

CONFIGS = {
    "Epic防御型": [
        HeroConfig("Sergey", 0, False, 3),
        HeroConfig("Bokan", 0, False, 4),
        HeroConfig("Patrick", 0, False, 5),
        HeroConfig("Ling Xue", 0, False, 6),
    ],
    "Epic攻撃型": [
        HeroConfig("Jessie", 0, False, 3),
        HeroConfig("Seo-yoon", 0, False, 4),
        HeroConfig("Bokan", 0, False, 5),
        HeroConfig("Sergey", 0, False, 6),
    ],
    "Legend防御型": [
        HeroConfig("Hector", 0, False, 3),
        HeroConfig("Norah", 0, False, 4),
        HeroConfig("Jeronimo", 0, False, 5),
        HeroConfig("Gwen", 0, False, 6),
    ],
    "Legend攻撃型": [
        HeroConfig("Jeronimo", 0, False, 3),
        HeroConfig("Gwen", 0, False, 4),
        HeroConfig("Flint", 0, False, 5),
        HeroConfig("Alonzo", 0, False, 6),
    ],
}

# 敵軍: Epic防御型で固定
ENEMY_MEMBERS = [
    HeroConfig("Sergey", 0, False, 3),
    HeroConfig("Bokan", 0, False, 4),
    HeroConfig("Patrick", 0, False, 5),
    HeroConfig("Ling Xue", 0, False, 6),
]

ENEMY = ArmyConfig(
    rally_capacity=300000,
    troop_ratio={"infantry": 0.34, "lancer": 0.33, "archer": 0.33},
    troop_tier="T11", fc_level=10,
    heroes=LEADERS + ENEMY_MEMBERS,
    base_buff=BASE_BUFF,
    role="defense",
)

N_RANDOM = 100


def run_comparison():
    print(f"{'編成':<16} {'mode':<10} {'勝率':>6} {'残兵率':>8} {'与キル率':>8} {'ターン':>6}")
    print("-" * 70)

    for name, members in CONFIGS.items():
        cfg = ArmyConfig(
            rally_capacity=300000,
            troop_ratio={"infantry": 0.34, "lancer": 0.33, "archer": 0.33},
            troop_tier="T11", fc_level=10,
            heroes=LEADERS + members,
            base_buff=BASE_BUFF,
            role="attack",
        )

        # Expected mode (1回)
        r = run_simulation(cfg, ENEMY, mode="expected", validate=False)
        surv = sum(r.turn_logs[-1].army_a_counts.values()) if r.turn_logs else 0
        kills = sum(r.total_kills_by_side.get("b", {}).values())
        print(f"{name:<16} {'expected':<10} {'win' if r.winner == 'a' else 'lose':>6} "
              f"{surv/300000*100:>7.1f}% {kills/300000*100:>7.1f}% {r.total_turns:>6}")

        # Random mode (N回)
        wins = 0
        total_surv = 0
        total_kills = 0
        total_turns = 0
        for _ in range(N_RANDOM):
            r = run_simulation(cfg, ENEMY, mode="random", validate=False)
            if r.winner == "a":
                wins += 1
            surv = sum(r.turn_logs[-1].army_a_counts.values()) if r.turn_logs else 0
            total_surv += surv
            total_kills += sum(r.total_kills_by_side.get("b", {}).values())
            total_turns += r.total_turns

        avg_surv = total_surv / N_RANDOM / 300000 * 100
        avg_kills = total_kills / N_RANDOM / 300000 * 100
        avg_turns = total_turns / N_RANDOM
        print(f"{'':<16} {'random':<10} {wins/N_RANDOM*100:>5.0f}% "
              f"{avg_surv:>7.1f}% {avg_kills:>7.1f}% {avg_turns:>6.1f}")
        print()


if __name__ == "__main__":
    run_comparison()
