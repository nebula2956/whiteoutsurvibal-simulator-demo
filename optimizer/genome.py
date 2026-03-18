from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

from simulator.models import ArmyConfig, HeroConfig


@dataclass
class OwnedHero:
    """所持英雄。gear_levelはリーダー時のみ使用。"""
    hero_id: str
    gear_level: int  # 0-10


@dataclass
class HeroPool:
    """
    所持英雄プール。
    - リーダー候補: 兵種別レジェンドのみ（各兵種から1体選出）
    - メンバー候補: 全英雄（重複選択可、スキル1のみ、装備なし）
    """
    infantry_leaders: List[OwnedHero]
    lancer_leaders: List[OwnedHero]
    archer_leaders: List[OwnedHero]
    members: List[str]  # hero_id のリスト

    @property
    def leader_dims(self) -> Tuple[int, int, int]:
        return (len(self.infantry_leaders),
                len(self.lancer_leaders),
                len(self.archer_leaders))

    @property
    def genome_dim(self) -> int:
        n_i, n_l, n_a = self.leader_dims
        n_m = len(self.members)
        return 3 + n_i + n_l + n_a + 4 * n_m


def _decode_ratios(x: np.ndarray) -> Tuple[float, float, float]:
    """x[0:3] → softmax → (infantry, lancer, archer) 比率。常に正・合計1。"""
    raw = np.clip(x[0:3], -20.0, 20.0)
    e = np.exp(raw)
    ratios = e / np.sum(e)
    return float(ratios[0]), float(ratios[1]), float(ratios[2])


def _decode_member(x_slice: np.ndarray, pool_members: List[str]) -> str:
    """メンバースロット用: n_m次元 → argmax でメンバー選出。"""
    return pool_members[int(np.argmax(x_slice))]


def decode_genome(x: np.ndarray, pool: HeroPool, template: ArmyConfig) -> ArmyConfig:
    """連続ベクトル → ArmyConfig にデコード。"""
    inf_r, lan_r, arc_r = _decode_ratios(x)

    # --- リーダー選出（argmax） ---
    idx = 3
    n_i, n_l, n_a = pool.leader_dims

    inf_leader = pool.infantry_leaders[int(np.argmax(x[idx:idx + n_i]))]
    idx += n_i
    lan_leader = pool.lancer_leaders[int(np.argmax(x[idx:idx + n_l]))]
    idx += n_l
    arc_leader = pool.archer_leaders[int(np.argmax(x[idx:idx + n_a]))]
    idx += n_a

    heroes: List[HeroConfig] = [
        HeroConfig(inf_leader.hero_id, inf_leader.gear_level, True, 0),
        HeroConfig(lan_leader.hero_id, lan_leader.gear_level, False, 1),
        HeroConfig(arc_leader.hero_id, arc_leader.gear_level, False, 2),
    ]

    # --- メンバー選出（各スロット n_m 次元 → argmax） ---
    n_m = len(pool.members)
    for i in range(4):
        member_id = _decode_member(x[idx:idx + n_m], pool.members)
        heroes.append(HeroConfig(member_id, 0, False, 3 + i))
        idx += n_m

    return ArmyConfig(
        rally_capacity=template.rally_capacity,
        troop_ratio={
            "infantry": inf_r,
            "lancer": lan_r,
            "archer": arc_r,
        },
        troop_tier=template.troop_tier,
        fc_level=template.fc_level,
        heroes=heroes,
        base_buff=template.base_buff,
        base_buff_per_type=template.base_buff_per_type,
        role=template.role,
    )


def genome_label(x: np.ndarray, pool: HeroPool) -> str:
    """ゲノムを可読ラベルに変換。"""
    inf_r, lan_r, arc_r = _decode_ratios(x)

    idx = 3
    n_i, n_l, n_a = pool.leader_dims
    inf_l = pool.infantry_leaders[int(np.argmax(x[idx:idx + n_i]))].hero_id
    idx += n_i
    lan_l = pool.lancer_leaders[int(np.argmax(x[idx:idx + n_l]))].hero_id
    idx += n_l
    arc_l = pool.archer_leaders[int(np.argmax(x[idx:idx + n_a]))].hero_id
    idx += n_a

    n_m = len(pool.members)
    members = []
    for i in range(4):
        members.append(_decode_member(x[idx:idx + n_m], pool.members))
        idx += n_m

    return (f"盾{inf_r*100:.0f}% 槍{lan_r*100:.0f}% 弓{arc_r*100:.0f}% | "
            f"L: {inf_l}/{lan_l}/{arc_l} | M: {'/'.join(members)}")


def build_army_config(
    heroes: List[HeroConfig],
    ratio: Tuple[float, float, float],
    template: ArmyConfig,
) -> ArmyConfig:
    """英雄リストと比率から直接ArmyConfigを構築する。Hybrid GA用。"""
    return ArmyConfig(
        rally_capacity=template.rally_capacity,
        troop_ratio={
            "infantry": ratio[0],
            "lancer": ratio[1],
            "archer": ratio[2],
        },
        troop_tier=template.troop_tier,
        fc_level=template.fc_level,
        heroes=heroes,
        base_buff=template.base_buff,
        base_buff_per_type=template.base_buff_per_type,
        role=template.role,
    )
