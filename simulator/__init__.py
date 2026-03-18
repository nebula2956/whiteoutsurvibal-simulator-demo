from .models import (
    TroopStats, StatBuff, UnitGroup, HeroConfig, ArmyConfig, Army,
    SkillModifiers, ActiveEffect, SkillState, BattleState,
    SkillActivation, DamageResult, TurnLog, BattleResult,
)
from .loaders import TroopLoader, HeroLoader, ArmyBuilder
from .buff_aggregator import BuffAggregator
from .damage_engine import DamageEngine
from .target_selector import TargetSelector
from .skill_engine import SkillEngine
from .battle_loop import BattleSimulator
from .logger import BattleLogger

def run_simulation(
    config_a: ArmyConfig,
    config_b: ArmyConfig,
    mode: str = "random",
    max_turns: int = 500,
    validate: bool = True,
) -> BattleResult:
    """共通エントリポイント。UIとGA最適化の両方から呼び出す。"""
    import os
    base = os.path.dirname(os.path.dirname(__file__))
    troop_defs = TroopLoader().load(os.path.join(base, "data", "troops.json"))
    hero_defs = HeroLoader().load_all(
        os.path.join(base, "data", "heroes.json"),
    )
    aggregator = BuffAggregator()
    builder = ArmyBuilder(troop_defs, hero_defs, aggregator)
    army_a = builder.build(config_a, validate=validate)
    army_b = builder.build(config_b, validate=validate)

    skill_engine = SkillEngine(mode=mode)
    damage_engine = DamageEngine()
    target_selector = TargetSelector(troop_defs)

    sim = BattleSimulator(
        army_a, army_b, skill_engine, damage_engine, target_selector,
        hero_defs=hero_defs, troop_data=troop_defs,
        max_turns=max_turns,
    )
    return sim.run()
