from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, List

from .models import ArmyConfig, HeroConfig

# position 0-2 に対応する兵種
LEADER_TROOP_TYPE = {0: "infantry", 1: "lancer", 2: "archer"}


@dataclass
class ValidationResult:
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def validate_army_config(
    config: ArmyConfig,
    hero_defs: Dict[str, Any],
) -> ValidationResult:
    """ArmyConfig の英雄編成を検証する。"""
    errors: List[str] = []
    warnings: List[str] = []
    seen_ids: List[str] = []

    for hc in config.heroes:
        # 1. 存在チェック
        if hc.hero_id not in hero_defs:
            errors.append(f"英雄 '{hc.hero_id}' はデータに存在しません")
            continue

        hdef = hero_defs[hc.hero_id]

        # 2. 重複チェック
        if hc.hero_id in seen_ids:
            errors.append(
                f"英雄 '{hc.hero_id}' が重複しています (position={hc.position})"
            )
        seen_ids.append(hc.hero_id)

        # 3. リーダー枠 (position 0-2) のバリデーション
        if hc.position in LEADER_TROOP_TYPE:
            expected_type = LEADER_TROOP_TYPE[hc.position]
            actual_type = hdef.get("troop_type")

            # Epic はリーダー不可
            if hdef.get("rarity") == "Epic":
                errors.append(
                    f"Epic英雄 '{hc.hero_id}' はリーダー枠 (position={hc.position}) に配置できません"
                )

            # 兵種一致チェック
            if actual_type != expected_type:
                errors.append(
                    f"英雄 '{hc.hero_id}' (兵種={actual_type}) は"
                    f" position={hc.position} ({expected_type}リーダー枠) に一致しません"
                )

        # 4. メンバー枠の装備レベル警告
        if hc.position > 2 and hc.gear_level > 0:
            warnings.append(
                f"メンバー '{hc.hero_id}' に gear_level={hc.gear_level} が設定されていますが無効です"
            )

    # 5. リーダー枠の重複 position チェック
    leader_positions = [hc.position for hc in config.heroes if hc.position <= 2]
    if len(leader_positions) != len(set(leader_positions)):
        errors.append("リーダー枠 (position 0-2) に同一 position が複数存在します")

    return ValidationResult(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
    )
