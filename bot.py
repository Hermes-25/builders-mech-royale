"""Game policy for the mechfight/1 JSON observation stream."""
from collections import deque
import math
import sys
from pathlib import Path

# Also supports Python's isolated embedded distribution.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mechfight


def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def direction(a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    scale = max(1.0, math.hypot(dx, dy))
    return dx / scale, dy / scale


class Bot:
    name = "Afterglow"

    def __init__(self, hello):
        self.config = hello["config"]
        self.phase = (hello["slot"] * 2.39996) % math.tau
        self.positions = deque(maxlen=12)
        self.boosting = False
        self.detour_until = 0
        self.detour_sign = 1

    def destination(self, obs):
        me, zone = obs["self"], obs["zone"]
        p, center = me["pos"], zone["next_center"]
        speed = max(1, me["stats"]["walk_speed"])
        time_left = max(0, zone["shrink_end"] - obs["t"])
        # Reserve walking time for the closing circle before seeking rewards.
        def travel_slack(target):
            return time_left - (distance(p, target) +
                max(0, distance(target, center) - zone["next_radius"] + 8)) / speed

        crowd = [m["pos"] for m in obs["mechs"] if m["kind"] == "mech"]
        radius = min(0.62 * zone["next_radius"], 95)
        angle = self.phase + obs["t"] * 0.035
        goals = []
        for i in range(10):
            a = angle + i * math.tau / 10
            pt = (center[0] + radius * math.cos(a), center[1] + radius * math.sin(a))
            goals.append((pt, 0, 0))
        for obj in obs["cubes"]:
            if abs(obj["pos"][2] - p[2]) < 11:
                goals.append((obj["pos"], 52, 0))
        for obj in obs["drops"]:
            if obj["holder"] not in (None, me["id"]):
                continue
            arrival = distance(p, obj["pos"]) / speed
            wait = max(0, obj["lands_in"] - arrival)
            healing = 1 - me["hp"] / max(1, me["max_hp"])
            bonus = 84 + healing * 95 - min(100, wait * 4)
            goals.append((obj["pos"], bonus, obj["radius"] * 0.48))
        if me["weapons"][1] is None:
            for obj in obs["pickups"]:
                if obj["kind"] != "gun" and abs(obj["pos"][2] - p[2]) < 10:
                    goals.append((obj["pos"], 48, 0))
        for obj in obs["crates"]:
            if me["cubes"] < 6 and abs(obj["pos"][2] - p[2]) < 10:
                goals.append((obj["pos"], 25, 6))

        def score(goal):
            pt, reward, stop = goal
            slack = travel_slack(pt)
            boundary = distance(pt, zone["center"]) - zone["radius"] + 6
            exposure = sum(max(0, 72 - distance(pt, other)) for other in crowd)
            return (reward - 0.46 * max(0, distance(p, pt) - stop)
                    - 0.68 * exposure - 2 * max(0, boundary)
                    - 12 * max(0, 2 - slack))

        if distance(p, zone["center"]) >= zone["radius"] - 5 or travel_slack(p) <= 1:
            return center, 0
        best, _, stop = max(goals, key=score)
        return best, stop

    def act(self, obs):
        me = obs["self"]
        p, now = me["pos"], obs["t"]
        goal, stop = self.destination(obs)
        dx, dy = direction(p, goal)
        resting = distance(p, goal) <= max(0.6, stop)
        self.positions.append(tuple(p[:2]))
        blocked = (len(self.positions) == self.positions.maxlen and not resting
                   and distance(self.positions[0], p) < 1.25)
        if blocked and now >= self.detour_until:
            self.detour_sign *= -1
            self.detour_until = now + 1.3
        if resting:
            dx = dy = 0
        elif now < self.detour_until:
            dx, dy = 0.45 * dx - self.detour_sign * 0.88 * dy, 0.45 * dy + self.detour_sign * 0.88 * dx
        if me["energy"] < 0.24 * me["max_energy"]:
            self.boosting = False
        elif me["energy"] > 0.85 * me["max_energy"]:
            self.boosting = True
        action = {"tick": obs["tick"], "move": [dx, dy],
                  "boost": self.boosting and not resting and (me["grounded"] or blocked),
                  "jump": blocked and me["grounded"]}
        if obs["offer"]:
            preference = ("plating", "nanites", "armor", "gun_rate", "gun_damage",
                          "servos", "reactor", "jets", "capacitor")
            choices = obs["offer"]["choices"]
            action["pick"] = max(range(len(choices)), key=lambda i:
                len(preference) - preference.index(choices[i]["id"]) if choices[i]["id"] in preference else 0)
        if me["weapons"][1] is None:
            for item in obs["pickups"]:
                if (item["kind"] != "gun" and abs(item["pos"][2] - p[2]) < 4
                        and distance(p, item["pos"]) < self.config["loot"]["pickup_radius"]):
                    action["take"] = item["id"]
                    break

        # Use the game's basic point-and-click interaction on visible sprites.
        primary = me["weapons"][0]
        actors = [m for m in obs["mechs"] if distance(p, m["pos"]) < primary["stats"]["range"]]
        objects = actors or [c for c in obs["crates"] if distance(p, c["pos"]) < 80]
        if objects:
            target = min(objects, key=lambda obj: distance(p, obj["pos"]) + 0.023 * obj["hp"])
            q = target["pos"]
            height = {"mech": 4.5, "npc": 1.75}.get(target.get("kind"), 1.2)
            action["aim"] = [math.atan2(q[1] - p[1], q[0] - p[0]),
                             math.atan2(q[2] + height - p[2] - me["stats"]["muzzle_height"],
                                        distance(p, q))]
            item = me["weapons"][1]
            secondary = False
            if item:
                kind = item["kind"]
                if kind == "laser":
                    secondary = item.get("charge", 0) < item["stats"].get("charge_s", 0.8)
                elif kind == "lightning":
                    secondary = distance(p, q) <= item["stats"]["range"]
                else:
                    secondary = distance(p, q) > 30
            action["fire"] = [True, secondary]
        else:
            action["aim"] = [math.atan2(dy, dx) + 0.9 * math.sin(now * 0.7), 0]
            action["reload"] = primary.get("ammo", 0) < primary["stats"].get("mag", 0) * 0.7
        return action


if __name__ == "__main__":
    mechfight.run(Bot)
