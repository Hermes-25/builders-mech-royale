"""Game policy for the mechfight/1 JSON observation stream."""
from collections import deque
from array import array
import json
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


class RewardSurface:
    """Estimate elevation above the game's tile grid for pickup accessibility."""

    def __init__(self, folder):
        root = Path(folder) / "v2"
        grid = json.loads((root / "world.json").read_bytes())["terrain"]
        data = array("h")
        with (root / grid["file"]).open("rb") as stream:
            data.fromfile(stream, grid["nx"] * grid["ny"])
        if sys.byteorder != "little":
            data.byteswap()
        self.grid, self.data = grid, data

    def above_tile(self, point):
        g = self.grid
        gx = min(g["nx"] - 1.001, max(0, (point[0] - g["minX"]) / g["step"]))
        gy = min(g["ny"] - 1.001, max(0, (point[1] - g["minY"]) / g["step"]))
        x, y = int(gx), int(gy)
        i = y * g["nx"] + x
        fx, fy = gx - x, gy - y
        low = self.data[i] * (1 - fx) + self.data[i + 1] * fx
        j = i + g["nx"]
        high = self.data[j] * (1 - fx) + self.data[j + 1] * fx
        return point[2] - g["scale"] * (low * (1 - fy) + high * fy)


class Bot:
    name = "Afterglow"

    def __init__(self, hello):
        self.config = hello["config"]
        self.phase = (hello["slot"] * 2.39996) % math.tau
        self.positions = deque(maxlen=12)
        self.boosting = False
        self.contact = None
        self.preference = 1 if hello["slot"] % 2 else -1
        self.last_step = None
        self.obstruction = deque(maxlen=7)
        self.last_reverse = -10
        self.items = {"cubes": {}, "pickups": {}, "crates": {}}
        self.last_goal = None
        self.recover_until = -1
        self.last_health = None
        self.drop_progress = {}
        self.drop_cooldown = {}
        self.mobility_heading = (0, 0)
        self.headings = tuple((math.cos(a * math.tau / 16), math.sin(a * math.tau / 16)) for a in range(16))
        try:
            self.surface = RewardSurface(hello["map"]["dir"])
        except (OSError, KeyError, ValueError, EOFError):
            self.surface = None

    def reachable_reward(self, p, reward):
        if abs(reward[2] - p[2]) < 8:
            return True
        if self.surface is None:
            return abs(reward[2] - p[2]) < 11
        # A collectible on a downhill tile can be well below the avatar; only
        # the collectible's height *above its tile* signals a raised platform.
        return (self.surface.above_tile(reward) < 4
                or abs(self.surface.above_tile(reward) - self.surface.above_tile(p)) < 6)

    def recovery_route(self, obs, destination, stop):
        """Optimize a short walk against the future safe circle and crowding.

        A drop with almost-complete progress is worth finishing when damaged.
        Otherwise the regeneration timer gives the avatar an incentive to stay
        in quiet parts of the game circle continuously for several seconds.
        """
        me, zone, now = obs["self"], obs["zone"], obs["t"]
        p = me["pos"]
        health = me["hp"] / max(1, me["max_hp"])
        if self.last_health is not None and health + 0.014 < self.last_health:
            self.recover_until = now + me["stats"].get("regen_delay", 5) + 3
        self.last_health = health
        crowd = [m["pos"] for m in obs["mechs"] if m["kind"] == "mech" and distance(p, m["pos"]) < 160]
        close = min((distance(p, q) for q in crowd), default=200)
        if not crowd or (close > 110 and now >= self.recover_until):
            return destination, stop
        if any(d["holder"] == me["id"] and d["progress"] >= 1.4
               and self.drop_cooldown.get(d["id"], -1) < now for d in obs["drops"]):
            return destination, stop
        duration = max(0.1, zone["shrink_end"] - max(now, zone["shrink_start"]))
        step = min(39, max(5, zone["radius"] * 0.32))
        horizon = step / max(1, me["stats"]["walk_speed"])
        interest = direction(p, destination)
        recovering = now < self.recover_until or close < 62
        def circle(h):
            f = min(1, max(0, (now + h - max(now, zone["shrink_start"])) / duration))
            c = [a + (b - a) * f for a,b in zip(zone["center"], zone["next_center"])]
            radius = zone["radius"] + (zone["next_radius"] - zone["radius"]) * f
            return c, radius
        c1,r1 = circle(horizon / 2)
        c2,r2 = circle(horizon)
        directions = [interest, self.mobility_heading]
        directions.extend(self.headings)
        def utility(move):
            dx,dy=move
            mid = (p[0]+step*0.5*dx,p[1]+step*0.5*dy)
            end = (p[0]+step*dx,p[1]+step*dy)
            debt = max(0, distance(c1,mid)-max(0,r1-3))+max(0,distance(c2,end)-max(0,r2-4))
            distances = [distance(end,q) for q in crowd]
            clear = min(145,min(distances))
            clustered = sum(max(0,95-gap) for gap in distances)
            regain = (1.45 if recovering else 0.82) * clear - 0.23 * clustered
            toward = (interest[0]*dx + interest[1]*dy) * step
            smooth = (self.mobility_heading[0]*dx + self.mobility_heading[1]*dy)
            return regain + (0.34 if recovering else 0.94)*toward - 5*debt + 4*smooth
        move = max(directions,key=utility)
        self.mobility_heading = move
        if move == interest:
            return destination, stop
        return (p[0]+step*move[0],p[1]+step*move[1]),0

    def remember(self, obs):
        """Remember static collectibles across short gaps in visibility."""
        p, now = obs["self"]["pos"], obs["t"]
        for group, memory in self.items.items():
            visible = {obj["id"] for obj in obs[group]}
            for key, (obj, seen) in list(memory.items()):
                if key not in visible and (now - seen > 14 or distance(p, obj["pos"]) < 7):
                    del memory[key]
            for obj in obs[group]:
                memory[obj["id"]] = (obj, now)

    def contested_drops(self, obs):
        """Stop paying the time cost of a capture when its game clock freezes."""
        now, me = obs["t"], obs["self"]
        for obj in obs["drops"]:
            key = obj["id"]
            progress, since = self.drop_progress.get(key, (-1, now))
            clock = obj["progress"]
            if obj["holder"] != me["id"] or obj["lands_in"] > 0:
                self.drop_progress[key] = (clock, now)
            elif clock > progress + 0.015 or clock < progress - 0.015:
                self.drop_progress[key] = (clock, now)
                self.drop_cooldown.pop(key, None)
            elif now - since > 1.3 and any(
                    m["kind"] == "mech" and distance(m["pos"], obj["pos"]) <= obj["radius"] + 2
                    for m in obs["mechs"]):
                self.drop_cooldown[key] = now + 4.5

    def destination(self, obs):
        me, zone = obs["self"], obs["zone"]
        p, center = me["pos"], zone["next_center"]
        speed = max(1, me["stats"]["walk_speed"])
        self.remember(obs)
        self.contested_drops(obs)
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
        for obj, seen in self.items["cubes"].values():
            if self.reachable_reward(p, obj["pos"]):
                bonus = (75 if me["cubes"] < 3 else 54) - (obs["t"] - seen) * 0.9
                goals.append((obj["pos"], bonus, 0))
        for obj in obs["drops"]:
            if obj["holder"] not in (None, me["id"]) or self.drop_cooldown.get(obj["id"], -1) > obs["t"]:
                continue
            arrival = distance(p, obj["pos"]) / speed
            wait = max(0, obj["lands_in"] - arrival)
            healing = 1 - me["hp"] / max(1, me["max_hp"])
            bonus = 84 + healing * 95 - min(100, wait * 4)
            if obj["holder"] == me["id"]:
                bonus += 30 * obj["progress"]
            # Standing near the edge still captures. Pick a point reachable
            # from our side of the ring, preserving already banked time.
            rad = obj["radius"] * 0.70
            if distance(p, obj["pos"]) < rad:
                focus = p
            else:
                vx, vy = direction(obj["pos"], p)
                focus = (obj["pos"][0] + vx * rad, obj["pos"][1] + vy * rad)
            goals.append((focus, bonus, 0.7))
        held = me["weapons"][1]
        if not held or held["kind"] != "lightning":
            for obj, seen in self.items["pickups"].values():
                desired = obj["kind"] == "lightning" or (not held and obj["kind"] != "gun")
                if desired and self.reachable_reward(p, obj["pos"]):
                    value = 89 if obj["kind"] == "lightning" else 43
                    goals.append((obj["pos"], value - (obs["t"] - seen), 0))
        for obj, seen in self.items["crates"].values():
            if me["cubes"] < 6 and self.reachable_reward(p, obj["pos"]):
                goals.append((obj["pos"], 36 - (obs["t"] - seen), 6))

        # Pick local waypoints when the next circle is far away. They adapt to
        # the current avatar density while preserving time to reach that circle.
        if crowd:
            step = min(45, max(8, zone["radius"] * 0.20))
            for i in range(12):
                angle = i * math.tau / 12 + self.phase
                pt = (p[0] + step * math.cos(angle), p[1] + step * math.sin(angle))
                reserve = max(0, distance(pt, center) - 0.85 * zone["next_radius"])
                goals.append((pt, 14 - reserve * 0.36, 0))

        def score(goal):
            pt, reward, stop = goal
            slack = travel_slack(pt)
            boundary = distance(pt, zone["center"]) - zone["radius"] + 6
            exposure = sum(max(0, 72 - distance(pt, other)) for other in crowd)
            continuity = 8 if self.last_goal and distance(pt, self.last_goal) < 5 else 0
            return (reward + continuity - 0.46 * max(0, distance(p, pt) - stop)
                    - 0.68 * exposure - 2 * max(0, boundary)
                    - 12 * max(0, 2 - slack))

        # At the end of a phase the old travel-time estimate becomes zero
        # even when we are safely inside the next circle.
        if (distance(p, zone["center"]) >= zone["radius"] - 2
                or (travel_slack(p) <= 1 and distance(p, center) > max(2,zone["next_radius"] - 8))):
            return center, 0
        best, _, stop = max(goals, key=score)
        self.last_goal = tuple(best[:2])
        return best, stop

    def navigate(self, p, goal, resting, now, speed):
        """Trace a boundary from the avatar's observed displacement.

        Sliding along a tile's edge reveals which component of the previous
        command did not move the avatar. Preserve the side of the detour until
        it crosses the original contact plane.
        """
        dx, dy = direction(p, goal)
        if resting:
            self.contact = self.last_step = None
            self.obstruction.clear()
            self.positions.clear()
            return 0, 0, False
        self.positions.append(tuple(p[:2]))
        if self.last_step:
            old, cmd, previous_time = self.last_step
            dt = now - previous_time
            if 0.03 <= dt <= 0.25:
                expected = max(0.25, dt * speed)
                missing = [cmd[i] - (p[i] - old[i]) / expected for i in (0, 1)]
                self.obstruction.append(tuple(max(-1, min(1, v)) for v in missing))
            else:
                self.obstruction.clear()
        if self.contact:
            origin, normal, tangent, until, prior_goal = self.contact
            crossed = sum((p[i] - origin[i]) * normal[i] for i in (0, 1))
            if crossed > 3 or now >= until or distance(goal, prior_goal) > 35:
                self.contact = None
                self.obstruction.clear()
            elif (len(self.positions) == self.positions.maxlen and now - self.last_reverse > 2.3
                    and distance(p, self.positions[0]) < 1.6):
                tangent = (-tangent[0], -tangent[1])
                self.contact = (origin, normal, tangent, now + 14, prior_goal)
                self.last_reverse = now
        if not self.contact and len(self.obstruction) == self.obstruction.maxlen:
            loss = [sum(v[i] for v in self.obstruction) / len(self.obstruction) for i in (0, 1)]
            magnitude = math.hypot(*loss)
            if magnitude > 0.38 and loss[0] * dx + loss[1] * dy > 0.26:
                normal = (loss[0] / magnitude, loss[1] / magnitude)
                tangent = (-normal[1], normal[0])
                ahead = tangent[0] * dx + tangent[1] * dy
                side = self.preference if abs(ahead) < 0.18 else (1 if ahead > 0 else -1)
                tangent = (tangent[0] * side, tangent[1] * side)
                self.contact = (tuple(p[:2]), normal, tangent, now + 14, tuple(goal[:2]))
                self.last_reverse = now
        if self.contact:
            _, normal, tangent, _, _ = self.contact
            dx = 0.42 * normal[0] + 0.90 * tangent[0]
            dy = 0.42 * normal[1] + 0.90 * tangent[1]
        self.last_step = (tuple(p[:2]), (dx, dy), now)
        return dx, dy, self.contact is not None

    def act(self, obs):
        me = obs["self"]
        p, now = me["pos"], obs["t"]
        goal, stop = self.destination(obs)
        goal, stop = self.recovery_route(obs, goal, stop)
        resting = distance(p, goal) <= max(0.6, stop)
        dx, dy, blocked = self.navigate(p, goal, resting, now, me["stats"]["walk_speed"])
        if me["energy"] < 0.24 * me["max_energy"]:
            self.boosting = False
        elif me["energy"] > 0.85 * me["max_energy"]:
            self.boosting = True
        action = {"tick": obs["tick"], "move": [dx, dy],
                  "boost": self.boosting and not resting and (me["grounded"] or blocked),
                  "jump": blocked and me["grounded"]}
        held = me["weapons"][1]
        arc = held and held["kind"] == "lightning"
        active_arc = arc and any(distance(p, m["pos"]) < held["stats"]["range"] for m in obs["mechs"])
        if active_arc and me["energy"] < 0.65 * me["max_energy"]:
            action["boost"] = False
        if obs["offer"]:
            preference = ("plating", "nanites", "armor", "gun_rate", "gun_damage",
                          "servos", "reactor", "jets", "capacitor")
            if arc:
                preference = ("lightning_efficiency", "lightning_range", "plating",
                              "lightning_chain", "armor", "nanites", "reactor") + preference
            choices = obs["offer"]["choices"]
            action["pick"] = max(range(len(choices)), key=lambda i:
                len(preference) - preference.index(choices[i]["id"]) if choices[i]["id"] in preference else 0)
        if not arc:
            nearby = [item for item in obs["pickups"] if
                      (item["kind"] == "lightning" or (not held and item["kind"] != "gun"))
                      and abs(item["pos"][2] - p[2]) < 4
                      and distance(p, item["pos"]) < self.config["loot"]["pickup_radius"]]
            if nearby and not (held and held.get("charge", 0) > 0):
                action["take"] = max(nearby, key=lambda obj: obj["kind"] == "lightning")["id"]
                if held:
                    action["drop"] = 1

        # Use the game's basic point-and-click interaction on visible sprites.
        primary = me["weapons"][0]
        visible = [m for m in obs["mechs"] if distance(p, m["pos"]) < primary["stats"]["range"]]
        rivals = [m for m in visible if m["kind"] == "mech"]
        # Drones grant no placement point and become active when interacted
        # with. Harvest their cubes only in the opening when already nearby.
        useful_drones = [m for m in visible if m["kind"] == "npc" and
                         ((me["cubes"] < 3 and distance(p, m["pos"]) <= 44)
                          or distance(p, m["pos"]) <= 18)]
        actors = rivals or useful_drones
        if active_arc:
            arcade_range = [m for m in rivals if distance(p, m["pos"]) < held["stats"]["range"]]
            actors = arcade_range or actors
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
                    secondary = item.get("charge", 0) < item["stats"].get("charge", 0.75)
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
