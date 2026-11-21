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
        self.resource_targets = set()
        self.item_values = {}
        self.next_swap = 0
        self.sequence_at = -2
        self.sequence_key = None
        self.sequence_first = None
        self.discovered_tiles = set()
        self.scan_tick = -2
        self.explore_goal = None
        self.explore_at = -2
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

    def value_items(self, obs):
        """Value the two arcade abilities using their class-matching bonuses."""
        me=obs["self"]
        rivals=[m for m in obs["mechs"] if m["kind"]=="mech"]
        near=sum(distance(me["pos"],m["pos"]) <= self.config["weapons"]["lightning"]["range"] for m in rivals)
        bonus=any("mortar" in m.get("weapons",()) for m in rivals)
        self.item_values={"laser":88+12*bonus+5*(len(rivals)>near),
                          "lightning":72+15*min(near,3),
                          "mortar":34, "gun":-100}
        # Collecting both classes temporarily can cost an equip cooldown
        # each time. Give the all-distance item a stable inventory premium.
        self.item_values["laser"] += 23

    def destination(self, obs):
        me, zone = obs["self"], obs["zone"]
        p, center = me["pos"], zone["next_center"]
        speed = max(1, me["stats"]["walk_speed"])
        self.remember(obs)
        self.value_items(obs)
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
        resources = self.resource_targets
        resources.clear()
        cubes = list(self.items["cubes"].values())
        cube_groups = {}
        for c,_ in cubes:
            q=c["pos"];key=(math.floor(q[0]/13),math.floor(q[1]/13))
            cube_groups.setdefault(key,[]).append(q)
        milestones = self.config["cubes"]["upgrade_thresholds"]
        next_milestone = next((n for n in milestones if n>me["cubes"]),float("inf"))
        for obj, seen in cubes:
            if self.reachable_reward(p, obj["pos"]):
                bonus = (75 if me["cubes"] < 3 else 54) - (obs["t"] - seen) * 0.9
                q=obj["pos"];i,j=math.floor(q[0]/13),math.floor(q[1]/13)
                gain=0
                for di,dj in ((0,0),(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)):
                    for neighbor in cube_groups.get((i+di,j+dj),()):
                        if abs(neighbor[2]-q[2])<4 and distance(neighbor,q)<13:
                            gain+=1
                            if gain==4:break
                    if gain==4:break
                bonus += (gain-1)*18 + 20*(me["cubes"]+gain>=next_milestone)
                resources.add(tuple(obj["pos"][:2]))
                goals.append((obj["pos"], bonus, 0))
        for obj in obs["drops"]:
            if obj["holder"] not in (None, me["id"]):
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
            if obj["lands_in"] < arrival + 4:
                resources.add(tuple(focus[:2]))
                bonus += 18 * (me["cubes"]+3>=next_milestone)
            goals.append((focus, bonus, 0.7))
        held = me["weapons"][1]
        if not held or obs["t"] >= self.next_swap:
            for obj, seen in self.items["pickups"].values():
                value = self.item_values[obj["kind"]]
                desired = value > (self.item_values[held["kind"]]+11 if held else 0)
                if desired and self.reachable_reward(p, obj["pos"]):
                    if value >= 72:
                        resources.add(tuple(obj["pos"][:2]))
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
            # A brief visit that earns a cube/upgrade need not commit the
            # rest of the route to the crowded pickup zone.
            threat_weight=0.68
            if tuple(pt[:2]) in resources:
                our_time=distance(p,pt)/speed
                their_time=min((distance(q,pt)/speed for q in crowd),default=100)
                if our_time<2.8 and our_time <= their_time+0.4:
                    threat_weight=0.14 if me["hp"]>me["max_hp"]*.5 else 0.35
            return (reward + continuity - 0.46 * max(0, distance(p, pt) - stop)
                    - threat_weight * exposure - 2 * max(0, boundary)
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

    def resource_sequence(self, obs, original, stop):
        """Look three rewards ahead before changing the next pickup.

        Each route carries its own inventory, cube count and arrival time.
        Item eligibility, upgrade thresholds and public drop timers therefore
        depend on which reward is collected first.
        """
        me=obs["self"];p=me["pos"];now=obs["t"];zone=obs["zone"]
        if (not me["in_zone"] or zone["next_radius"]<35 or
                me["hp"]<0.42*me["max_hp"]):
            return original,stop
        speed=max(1,me["stats"]["walk_speed"])
        held=me["weapons"][1]
        equipped=held["kind"] if held else ""
        # Cluster only cubes already collectible from the representative
        # cube's actual location. Other cubes remain separate route stages.
        loose=sorted((c for c,seen in self.items["cubes"].values()
                      if now-seen<5 and self.reachable_reward(p,c["pos"])),
                     key=lambda c:distance(p,c["pos"]))[:16]
        nodes=[];used=set()
        for cube in loose:
            if cube["id"] in used:continue
            q=cube["pos"]
            group=[c for c in loose if c["id"] not in used and abs(q[2]-c["pos"][2])<3
                   and distance(q,c["pos"])<=self.config["cubes"]["pickup_radius"]-.5]
            used.update(c["id"] for c in group)
            nodes.append(("cube",q,len(group),0.,0.,cube["id"]))
        gear=sorted((item for item,seen in self.items["pickups"].values() if now-seen<5
                     and item["kind"] in ("laser","lightning") and self.reachable_reward(p,item["pos"])
                     and self.item_values[item["kind"]]>self.item_values.get(equipped,0)+11),
                    key=lambda item:distance(p,item["pos"]))[:3]
        for item in gear:
            nodes.append((item["kind"],item["pos"],0,0.,0.,item["id"]))
        for drop in obs["drops"]:
            if drop["holder"] in (None,me["id"]):
                nodes.append(("drop",drop["pos"],3,drop["lands_in"],drop["capture_s"]-drop["progress"],drop["id"]))
        nodes=sorted((n for n in nodes if distance(p,n[1])/speed<11.5),
                     key=lambda n:distance(p,n[1]))[:10]
        if len(nodes)<2:return original,stop
        nearest=min(range(len(nodes)),key=lambda i:distance(original,nodes[i][1]))
        if distance(original,nodes[nearest][1])>8:
            return original,stop
        if nodes[nearest][0]=="drop" and nodes[nearest][3]<4:
            # Honor the incumbent's choice to finish an imminent timed
            # capture. A detour can reset all banked capture time.
            return original,stop
        key=(tuple(sorted(n[-1] for n in nodes)),nodes[nearest][-1],me["cubes"],equipped)
        if key==self.sequence_key and now-self.sequence_at<.45:
            if self.sequence_first is None:return original,stop
            for n in nodes:
                if n[-1]==self.sequence_first:return n[1],2 if n[0]=="drop" else 0
        self.sequence_key=key;self.sequence_at=now;self.sequence_first=None
        rivals=[m["pos"] for m in obs["mechs"] if m["kind"]=="mech"]
        steps=self.config["cubes"]["upgrade_thresholds"]
        time_to=[distance(p,n[1])/speed for n in nodes]
        transit=[[distance(a[1],b[1])/speed for b in nodes] for a in nodes]
        contest=[min((distance(r,n[1])/speed for r in rivals),default=100) for n in nodes]
        best_by_first=[0.]*len(nodes)
        def extend(previous,elapsed,total,cubes,slot,first,visited,depth):
            for i,n in enumerate(nodes):
                if visited & (1<<i):continue
                kind,pt,gain,landing,dwell,_=n
                arrives=elapsed+(time_to[i] if previous<0 else transit[previous][i])
                end=max(arrives,landing)+dwell
                if end>13.5:continue
                remaining=zone["shrink_end"]-now-end
                if max(0,distance(pt,zone["next_center"])-zone["next_radius"]+5)/speed>remaining:continue
                ability=slot
                if kind in ("laser","lightning"):
                    upgrade=self.item_values[kind]-self.item_values.get(slot,0)
                    if upgrade<=11:continue
                    reward=0.69*upgrade;ability=kind
                else:
                    levels=sum(cubes<k<=cubes+gain for k in steps)
                    reward=13*(min(10,cubes+gain)-min(10,cubes))+levels*(26+11*(slot in ("laser","lightning")))
                    if kind=="drop":
                        reward+=65*(1-me["hp"]/max(1,me["max_hp"]))+12*(1-me["energy"]/max(1,me["max_energy"]))
                reliability=max(.42,min(1,.80+.13*(contest[i]-arrives)))
                nxt=total+reward*math.exp(-end/8.2)*reliability
                idx=i if first is None else first
                best_by_first[idx]=max(best_by_first[idx],nxt)
                if depth<3:
                    extend(i,end,nxt,cubes+gain,ability,idx,visited|(1<<i),depth+1)
        extend(-1,0.,0.,me["cubes"],equipped,None,0,1)
        chosen=max(range(len(nodes)),key=best_by_first.__getitem__)
        if chosen!=nearest and best_by_first[chosen]>best_by_first[nearest]*1.14+3:
            n=nodes[chosen];self.sequence_first=n[-1]
            return n[1],2 if n[0]=="drop" else 0
        return original,stop

    def act(self, obs):
        me = obs["self"]
        p, now = me["pos"], obs["t"]
        goal, stop = self.destination(obs)
        goal, stop = self.resource_sequence(obs,goal,stop)
        if now < 100 and (not me["weapons"][1] or me["weapons"][1]["kind"]=="mortar"):
            cell = 28
            if now-self.scan_tick >= .65:
                self.scan_tick = now
                x,y=math.floor(p[0]/cell),math.floor(p[1]/cell)
                yaw=me["aim"][0]
                ux,uy=math.cos(yaw),math.sin(yaw)
                for i in range(x-4,x+5):
                    for j in range(y-4,y+5):
                        dx,dy=(i+.5)*cell-p[0],(j+.5)*cell-p[1]
                        radius=math.hypot(dx,dy)
                        if radius<35 or (radius<116 and dx*ux+dy*uy>.6*radius):
                            self.discovered_tiles.add((i,j))
            waiting = any(distance(goal,point)<8 for point in self.resource_targets)
            known = any(it["kind"] in ("laser","lightning") and now-seen<9 and
                        distance(p,it["pos"])<125 and self.reachable_reward(p,it["pos"])
                        for it,seen in self.items["pickups"].values())
            near_box = any(distance(p,c["pos"])<44 for c in obs["crates"])
            z=obs["zone"]
            if (not (known or waiting or near_box) and me["in_zone"] and
                    me["hp"]>.52*me["max_hp"] and distance(p,z["center"])<z["radius"]-28):
                if self.explore_goal is None or now-self.explore_at>=2.5 or distance(p,self.explore_goal)<7:
                    options=[]
                    for i in range(16):
                        a=self.phase+i*math.tau/16
                        dx,dy=math.cos(a),math.sin(a)
                        pt=(p[0]+55*dx,p[1]+55*dy)
                        if distance(pt,z["center"])>z["radius"]-14:continue
                        gain=0
                        for depth in (45,75,105):
                            for side in (-35,0,35):
                                k=(math.floor((p[0]+depth*dx-side*dy)/cell),
                                   math.floor((p[1]+depth*dy+side*dx)/cell))
                                gain+=k not in self.discovered_tiles
                        continuity=0 if self.explore_goal is None else 1-.012*distance(pt,self.explore_goal)
                        cost=max(0,distance(pt,z["next_center"])-z["next_radius"])*.015
                        options.append((gain+continuity-cost-.004*distance(pt,goal),pt))
                    self.explore_goal=max(options)[1] if options else None
                    self.explore_at=now
                if self.explore_goal is not None:
                    goal,stop=self.explore_goal,0
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
            elif held and held["kind"] == "laser":
                preference = ("laser_damage", "plating", "laser_charge", "laser_pierce", "armor", "nanites") + preference
            choices = obs["offer"]["choices"]
            action["pick"] = max(range(len(choices)), key=lambda i:
                len(preference) - preference.index(choices[i]["id"]) if choices[i]["id"] in preference else 0)
        if not held or now >= self.next_swap:
            nearby = [item for item in obs["pickups"] if
                      self.item_values[item["kind"]] > (self.item_values[held["kind"]]+11 if held else 0)
                      and abs(item["pos"][2] - p[2]) < 4
                      and distance(p, item["pos"]) < self.config["loot"]["pickup_radius"]]
            if nearby and not (held and held.get("charge", 0) > 0):
                action["take"] = max(nearby, key=lambda obj: self.item_values[obj["kind"]])["id"]
                if held:
                    action["drop"] = 1
                    self.next_swap = now + 4

        # Use the game's basic point-and-click interaction on visible sprites.
        primary = me["weapons"][0]
        item_range = max(primary["stats"]["range"], held["stats"].get("range",0) if held else 0)
        actors = [m for m in obs["mechs"] if distance(p, m["pos"]) < item_range]
        rivals = [m for m in actors if m["kind"] == "mech"]
        if rivals:
            actors = rivals
        if active_arc:
            scored = [m for m in actors if m["kind"] == "mech" and distance(p, m["pos"]) < held["stats"]["range"]]
            actors = scored or actors
        objects = actors or [c for c in obs["crates"] if distance(p, c["pos"]) < 80]
        if objects:
            if held and held["kind"] == "laser" and rivals:
                # One HP counter reaching zero earns a placement. Prefer a
                # completed counter to spreading the same ability across many.
                def pulses(obj):
                    multiplier=1+held["stats"].get("cookoff",0)*("mortar" in obj.get("weapons",()))
                    power=max(1,held["stats"]["damage"]*multiplier)
                    return (math.ceil(obj["hp"]/power),obj["hp"]/power,distance(p,obj["pos"]))
                target=min(objects,key=pulses)
            else:
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
                    stats=item["stats"]
                    full=stats.get("charge",0.75)
                    frac=stats.get("min_frac",0.35)
                    power=stats["damage"]*(1+stats.get("cookoff",0)*("mortar" in target.get("weapons",())))
                    needed=full*(target["hp"]/max(1,power)-frac)/max(.01,1-frac)
                    threshold=max(stats.get("min_charge",0.15),min(full,needed+.025))
                    secondary = item.get("charge", 0) < threshold
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
