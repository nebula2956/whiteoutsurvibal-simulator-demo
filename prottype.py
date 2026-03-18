import random
from math import sqrt

# =========================
# ENUM
# =========================
INFANTRY = "infantry"
LANCER = "lancer"
MARKSMAN = "marksman"

TYPES = [INFANTRY, LANCER, MARKSMAN]

# =========================
# 仮ステータスデータ
# =========================
BASE_STATS = {
    INFANTRY: {"atk": 50, "def": 80, "hp": 120, "lethality": 40},
    LANCER: {"atk": 70, "def": 60, "hp": 90, "lethality": 60},
    MARKSMAN: {"atk": 90, "def": 40, "hp": 70, "lethality": 80},
}

# =========================
# 相性マトリクス
# =========================
COUNTER = {
    (INFANTRY, LANCER): 1.05,
    (INFANTRY, MARKSMAN): 0.95,
    (LANCER, MARKSMAN): 1.05,
    (LANCER, INFANTRY): 0.95,
    (MARKSMAN, INFANTRY): 1.05,
    (MARKSMAN, LANCER): 0.95,
}

def counter_multiplier(atk, df):
    return COUNTER.get((atk, df), 1.0)

# =========================
# Buffクラス
# =========================
class Buff:
    def __init__(self):
        self.attack = 0
        self.defense = 0
        self.hp = 0
        self.lethality = 0
        self.damage_up = 0
        self.damage_down = 0

# =========================
# UnitGroup
# =========================
class UnitGroup:
    def __init__(self, type_, count):
        self.type = type_
        self.count = count
        self.base = BASE_STATS[type_]
        self.current_hp = self.base["hp"] * count

    def is_alive(self):
        return self.count > 0

# =========================
# Army
# =========================
class Army:
    def __init__(self, infantry, lancer, marksman):
        self.units = {
            INFANTRY: infantry,
            LANCER: lancer,
            MARKSMAN: marksman
        }

# =========================
# DamageEngine
# =========================
class DamageEngine:

    def compute(self, atk_group, def_group):
        atk = atk_group.base["atk"]
        lethality = atk_group.base["lethality"]
        defense = def_group.base["def"]
        hp = def_group.base["hp"]

        ratio = atk * lethality / (defense * hp)

        # 人数 √スケーリング
        scale = sqrt(atk_group.count)

        # 相性
        counter = counter_multiplier(atk_group.type, def_group.type)

        dmg = scale * counter * ratio * atk_group.count

        return dmg

# =========================
# Battle Simulator
# =========================
class BattleSimulator:

    def __init__(self, armyA, armyB):
        self.A = armyA
        self.B = armyB
        self.engine = DamageEngine()

    def select_target(self, atk_type, enemy):
        order = [INFANTRY, LANCER, MARKSMAN]

        # 槍のすり抜け
        if atk_type == LANCER:
            if random.random() < 0.1 and enemy.units[MARKSMAN].is_alive():
                return MARKSMAN

        for t in order:
            if enemy.units[t].is_alive():
                return t

        return None

    def apply_damage(self, defender, dmg_map):
        for t, dmg in dmg_map.items():
            group = defender.units[t]

            if not group.is_alive():
                continue

            group.current_hp -= dmg

            # HP完全削りモデル
            if group.current_hp <= 0:
                group.count = 0
            else:
                group.count = int(group.current_hp / group.base["hp"])

    def compute_damage_map(self, atk, df):
        dmg_map = {INFANTRY:0, LANCER:0, MARKSMAN:0}

        for t, group in atk.units.items():
            if not group.is_alive():
                continue

            target_type = self.select_target(t, df)
            target = df.units[target_type]

            dmg = self.engine.compute(group, target)

            dmg_map[target_type] += dmg

        return dmg_map

    def is_end(self):
        a_alive = any(u.is_alive() for u in self.A.units.values())
        b_alive = any(u.is_alive() for u in self.B.units.values())
        return not (a_alive and b_alive)

    def run(self, turns=20):
        for turn in range(turns):

            dmgA = self.compute_damage_map(self.A, self.B)
            dmgB = self.compute_damage_map(self.B, self.A)

            self.apply_damage(self.B, dmgA)
            self.apply_damage(self.A, dmgB)

            print(f"Turn {turn+1}")
            print("A:", {t:u.count for t,u in self.A.units.items()})
            print("B:", {t:u.count for t,u in self.B.units.items()})
            print("----")

            if self.is_end():
                break

# =========================
# 実行サンプル
# =========================
if __name__ == "__main__":

    armyA = Army(
        UnitGroup(INFANTRY, 5000),
        UnitGroup(LANCER, 3000),
        UnitGroup(MARKSMAN, 2000)
    )

    armyB = Army(
        UnitGroup(INFANTRY, 4000),
        UnitGroup(LANCER, 4000),
        UnitGroup(MARKSMAN, 2000)
    )

    sim = BattleSimulator(armyA, armyB)
    sim.run()