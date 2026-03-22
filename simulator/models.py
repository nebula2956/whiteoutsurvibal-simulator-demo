from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Dict

UNIT_TYPES: List[str] = ["infantry", "lancer", "archer"]

# =========================
# 兵士
# =========================
@dataclass
class TroopStats:
    atk: float
    def_: float
    hp: float
    lethality: float


@dataclass
class UnitGroup:
    unit_type: str          # "infantry" | "lancer" | "archer"
    tier: str               # "T10" | "T11"
    fc_level: int           # 0-10
    count: int
    base_stats: TroopStats
    current_hp: float
    attack_counter: int = 0  # every_n_attacks 用カウンター

    def effective_count(self) -> int:
        return max(0, int(self.current_hp / self.base_stats.hp))

    def is_alive(self) -> bool:
        return self.current_hp > 0 and self.count > 0


# =========================
# バフ（単一層・加算用）
# =========================
@dataclass
class StatBuff:
    """単一バフ層。層内は加算、層間は乗算（ArmyStatLayers参照）。"""
    atk: float = 0.0
    def_: float = 0.0
    hp: float = 0.0
    lethality: float = 0.0

    def merge(self, other: StatBuff) -> StatBuff:
        return StatBuff(
            atk=self.atk + other.atk,
            def_=self.def_ + other.def_,
            hp=self.hp + other.hp,
            lethality=self.lethality + other.lethality,
        )


# =========================
# ステータスバフ（3層・乗算）
# =========================
@dataclass
class ArmyStatLayers:
    """
    3層乗算ステータスバフ。
    eff_atk = base_atk
              × (1 + base[unit_type].atk)   ← 層1: 兵種別（研究・島・装備 + hero_base_stats + 兵種パッシブ）
              × (1 + hero.atk)               ← 層2: 全兵種共通（英雄スキルバフ ally_all）
              × (1 + gear.atk)               ← 層3: 全兵種共通（専用装備）
    """
    # 層1: 兵種別基礎バフ（研究/島/装備 + hero_base_stats(type-specific) + 兵種パッシブ）
    base: Dict[str, StatBuff] = field(
        default_factory=lambda: {
            "infantry": StatBuff(),
            "lancer":   StatBuff(),
            "archer":   StatBuff(),
        }
    )
    # 層2: 英雄スキルバフ（ally_all対象・全兵種共通）
    hero: StatBuff = field(default_factory=StatBuff)
    # 層3: 専用装備バフ（全兵種共通）
    gear: StatBuff = field(default_factory=StatBuff)


# =========================
# ダメージ倍率バフ
# =========================
@dataclass
class DamageMod:
    """
    ダメージ倍率の管理。
    - 上昇系: 加算してから × (1 + Σ)
    - 減少系: 各スキルを個別格納して Π(1 - each)
    """
    # 与ダメ上昇（additive）: Jeronimo, Zinman s3, Molly s2/s3 等
    dealt_up: float = 0.0
    # 与ダメ減少（multiplicative per skill）: Bokan 等
    dealt_down: List[float] = field(default_factory=list)
    # 被ダメ上昇デバフ（additive）: Gwen s1 等（攻撃側が対象に付与）
    taken_up: float = 0.0
    # 被ダメ減少（multiplicative per skill）: Molly s1, Natalia s1, Sergey 等
    taken_down: List[float] = field(default_factory=list)
    # 追加ダメージ枠（base_dmg × frac で加算後、倍率スキルが掛かる）
    # Crystal Gunpowder, Gwen s2, Flame Charge
    additional_fracs: List[float] = field(default_factory=list)
    # 兵種別与ダメ上昇（Flint S1: infantry +100% 等）
    dealt_up_per_type: Dict[str, float] = field(default_factory=dict)
    # ターン中の殺傷力バフ（Alonzo S1 等・on_turn_start duration管理）
    lethality_up: float = 0.0
    # 永続スタック攻撃力バフ（Rion S3: 攻撃毎に蓄積）
    extra_atk_up: float = 0.0


# =========================
# 英雄設定
# =========================
@dataclass
class HeroConfig:
    hero_id: str
    gear_level: int           # 0-10
    is_rally_leader: bool
    position: int             # 0=ラリーリーダー, 1-2=サブリーダー, 3-6=メンバー


# =========================
# 軍編成設定（入力）
# =========================
@dataclass
class ArmyConfig:
    rally_capacity: int
    troop_ratio: Dict[str, float]  # {"infantry": 0.5, "lancer": 0.3, "archer": 0.2}
    troop_tier: str                # "T10" | "T11"
    fc_level: int                  # 0-10
    heroes: List[HeroConfig]
    base_buff: StatBuff            # 全部隊共通バフ（研究・島・装備等）
    base_buff_per_type: Dict[str, StatBuff] = field(
        default_factory=lambda: {
            "infantry": StatBuff(),
            "lancer":   StatBuff(),
            "archer":   StatBuff(),
        }
    )                              # 兵種別バフ（盾/槍/弓それぞれ個別入力）
    role: str = "attack"           # "attack" | "defense"


# =========================
# 軍（実行時）
# =========================
@dataclass
class Army:
    units: Dict[str, UnitGroup]
    stat_layers: ArmyStatLayers      # 3層のステータスバフ（固定）
    damage_mod: DamageMod            # パッシブダメージ倍率（固定）
    active_damage_mod: DamageMod     # ターン毎の一時倍率（毎ターン更新）
    heroes: List[HeroConfig]
    role: str
    # 敵軍ステータスに影響するパッシブデバフ（Ling Xue: atk_down等）
    passive_stat_debuff_to_enemy: StatBuff = field(default_factory=StatBuff)
    # 敵軍の与ダメを削減するパッシブ（Bokan: Π(1-0.20)）
    passive_dealt_debuff_to_enemy: List[float] = field(default_factory=list)


# =========================
# スキル修飾子（1回の攻撃時に解決）
# =========================
@dataclass
class SkillModifiers:
    crystal_lance: bool = False          # True → base_ratio × 2
    additional_fracs: List[float] = field(default_factory=list)  # 追加ダメージ枠
    double_attack: bool = False          # Archer Volley: 2回攻撃
    bypass_target: Optional[str] = None  # Lancer Ambush: ターゲット上書き
    attack_skip: bool = False            # Akmos S1: 攻撃スキップ
    # 攻撃ロール時に発生した追加の与/被ダメ変化
    extra_dealt_up: float = 0.0
    extra_dealt_down: List[float] = field(default_factory=list)
    extra_taken_up: float = 0.0
    extra_taken_down: List[float] = field(default_factory=list)
    # スキル別ダメージ寄与率（base_dmg に対する倍率）→ キル按分に使用
    # Crystal Lance → {id: 1.0}（base_dmgを1倍分追加）
    # Gunpowder +50% → {id: 0.5}、Gwen s2 +100% → {id: 1.0}
    skill_damage_fracs: Dict[str, float] = field(default_factory=dict)


# =========================
# アクティブエフェクト（継続バフ/デバフ）
# =========================
@dataclass
class ActiveEffect:
    skill_id: str
    effect_type: str
    value: float
    remaining_turns: Optional[int]   # None=永続
    remaining_hits: Optional[int]    # None=ヒット制限なし
    target: str


# =========================
# 減衰バフ（Hector S2 等）
# =========================
@dataclass
class DecayingDamageBuff:
    """攻撃毎に decay_rate 倍で減衰する与ダメバフ。"""
    skill_id: str
    unit_type: str         # 対象兵種（"infantry" / "archer" 等）
    current_value: float   # 現在の damage_dealt_up 値
    decay_rate: float      # 毎攻撃後の乗率（例: 0.85）
    remaining_attacks: int # 残り有効攻撃回数


# =========================
# スキル状態（バトル全体で持続）
# =========================
@dataclass
class SkillState:
    unit_attack_counters: Dict[str, int] = field(
        default_factory=lambda: {"infantry": 0, "lancer": 0, "archer": 0}
    )
    turn_counters: Dict[str, int] = field(default_factory=dict)
    active_effects_a: List[ActiveEffect] = field(default_factory=list)
    active_effects_b: List[ActiveEffect] = field(default_factory=list)
    # Gwen S2 脆弱: 次の攻撃の追加ダメージ枠に加算（サイド別）
    # "a" = A軍が付与 → B軍ユニットへの次の攻撃に適用
    # "b" = B軍が付与 → A軍ユニットへの次の攻撃に適用
    vulnerability: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: {"a": {}, "b": {}}
    )
    # 減衰バフ（Hector S2 等）: A/B軍別
    decaying_buffs_a: List[DecayingDamageBuff] = field(default_factory=list)
    decaying_buffs_b: List[DecayingDamageBuff] = field(default_factory=list)


# =========================
# バトル状態
# =========================
@dataclass
class BattleState:
    turn: int
    army_a: Army
    army_b: Army
    skill_state: SkillState


# =========================
# ログ
# =========================
@dataclass
class SkillActivation:
    turn: int
    skill_id: str
    unit_type: str
    triggered_by: str
    effect_summary: str
    side: str = "a"   # "a" = 自軍, "b" = 敵軍


@dataclass
class DamageResult:
    raw_damage: float           # スキル修飾子なしのbase_dmg
    final_damage: float         # 全修飾子適用後の最終ダメージ
    kill_count: int             # 総キル数
    skill_kill_count: int       # スキルプロックによる追加キル数（合計）
    skill_procs: Dict[str, int] # {skill_id: kills} スキル別キル数


@dataclass
class TurnLog:
    turn: int
    # ターン開始時の兵数
    army_a_counts: Dict[str, int]
    army_b_counts: Dict[str, int]
    # キル数（ターゲット兵種別の損傷）
    a_kills_b: Dict[str, int]   # A軍 → B軍へのキル {target_type: count}
    b_kills_a: Dict[str, int]   # B軍 → A軍へのキル {target_type: count}
    # アタッカー別キル数 {attacker_type: {target_type: count}}
    a_kills_by_attacker: Dict[str, Dict[str, int]] = field(default_factory=dict)
    b_kills_by_attacker: Dict[str, Dict[str, int]] = field(default_factory=dict)
    # スキル発動ログ（このターン）
    skill_activations: List[SkillActivation] = field(default_factory=list)
    # スキル別キル寄与（このターン）A/B軍別
    a_skill_kills_turn: Dict[str, int] = field(default_factory=dict)
    b_skill_kills_turn: Dict[str, int] = field(default_factory=dict)

    @property
    def skill_kills_this_turn(self) -> Dict[str, int]:
        """後方互換: A+B合算"""
        merged = dict(self.a_skill_kills_turn)
        for k, v in self.b_skill_kills_turn.items():
            merged[k] = merged.get(k, 0) + v
        return merged


@dataclass
class BattleResult:
    turn_logs: List[TurnLog]
    total_kills_by_side: Dict[str, Dict[str, int]]
    skill_kill_contributions: Dict[str, Dict[str, int]]  # {"a": {skill_id: kills}, "b": {...}}
    skill_activation_log: List[SkillActivation]
    winner: str
    total_turns: int
    final_counts_a: Dict[str, int] = None  # ダメージ適用後の最終兵数
    final_counts_b: Dict[str, int] = None
