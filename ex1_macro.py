import ex1_check
import search
import utils
import math
from collections import deque
import search

# Student ID placeholder (set your id as requested by the assignment)
id = ["No numbers - I'm special!"]


class WateringProblem(search.Problem):
    """Plant watering problem.

    State representation (immutable): tuple(taps_tuple, plants_tuple, robots_tuple)
      - taps_tuple: tuple of ((r,c), amount) sorted
      - plants_tuple: tuple of ((r,c), demand) sorted
      - robots_tuple: tuple of (rid, r, c, load, cap) sorted by rid
    """

    def __init__(self, initial):
        # parse input dict
        size = tuple(initial['Size'])
        self.size = size
        walls = set(initial.get('Walls', set()))
        self.walls = set(walls)

        taps = tuple(sorted(initial.get('Taps', {}).items()))
        plants = tuple(sorted(initial.get('Plants', {}).items()))
        robots = tuple(sorted(((rid, r, c, load, cap)
                               for rid, (r, c, load, cap) in initial.get('Robots', {}).items()),
                              key=lambda x: x[0]))

        # immutable state
        state = (taps, plants, robots)

        # free cells
        max_r, max_c = size
        self.free_cells = {(r, c) for r in range(max_r) for c in range(max_c) if (r, c) not in self.walls}

        # neighbors map
        moves = [(-1, 0, 'UP'), (1, 0, 'DOWN'), (0, -1, 'LEFT'), (0, 1, 'RIGHT')]
        self.neighbors = {}
        for cell in self.free_cells:
            r, c = cell
            nb = []
            for dr, dc, _ in moves:
                nr, nc = r + dr, c + dc
                if (nr, nc) in self.free_cells:
                    nb.append((nr, nc))
            self.neighbors[cell] = nb

        # store static lists
        self.tap_positions = [pos for (pos, _) in taps]
        self.plant_positions = [pos for (pos, _) in plants]

        # Precompute APSP distances via BFS from every free cell (eager Seidel-like)
        # store as nested dict: self.dist_map[a][b] = distance
        self.dist_map = {}
        for src in self.free_cells:
            self.dist_map[src] = self._bfs_from(src)

        # precompute tap->plant and plant->tap distances (cached)
        self.tap_plant_dist = {}
        self.plant_tap_dist = {}
        for t in self.tap_positions:
            for p in self.plant_positions:
                d = self.dist(src=t, dst=p)
                self.tap_plant_dist[(t, p)] = d
                self.plant_tap_dist[(p, t)] = d

        # finalize
        search.Problem.__init__(self, state)

    def _bfs_from(self, source):
        dist = {cell: float('inf') for cell in self.free_cells}
        if source not in self.free_cells:
            return dist
        q = deque([source])
        dist[source] = 0
        while q:
            cur = q.popleft()
            for nb in self.neighbors[cur]:
                if dist[nb] == float('inf'):
                    dist[nb] = dist[cur] + 1
                    q.append(nb)
        return dist

    def dist(self, src, dst):
        """O(1) distance lookup between free cells; walls return inf."""
        if src not in self.free_cells or dst not in self.free_cells:
            return float('inf')
        return self.dist_map.get(src, {}).get(dst, float('inf'))

    def successor(self, state):
        """Generate successors: list of (action_str, next_state)
        Actions: UP{rid}, DOWN{rid}, LEFT{rid}, RIGHT{rid}, LOAD{rid}, POUR{rid}
        Additionally: MOVETO macro encoded as "MOVETO{rid}:<steps>:r,c"
        """
        taps_f, plants_f, robots_t = state
        taps = dict(taps_f)
        plants = dict(plants_f)
        robots = {rid: [r, c, load, cap] for (rid, r, c, load, cap) in robots_t}

        occupied = {(r, c) for (_, r, c, _, _) in robots_t}

        succs = []

        # primitive movement actions (single-step)
        for rid, (r, c, load, cap) in robots.items():
            for nb in self.neighbors.get((r, c), []):
                if nb in occupied and nb != (r, c):
                    continue
                nr, nc = nb
                # create new robots tuple with this robot moved one step
                new_robots = []
                for orid, orr, orc, ol, oc in robots_t:
                    if orid == rid:
                        new_robots.append((orid, nr, nc, ol, oc))
                    else:
                        new_robots.append((orid, orr, orc, ol, oc))
                new_robots = tuple(sorted(new_robots, key=lambda x: x[0]))
                new_state = (taps_f, plants_f, new_robots)
                # movement action string must match required format
                dr, dc = (nr - r, nc - c)
                if (dr, dc) == (-1, 0):
                    act = f"UP{{{rid}}}"
                elif (dr, dc) == (1, 0):
                    act = f"DOWN{{{rid}}}"
                elif (dr, dc) == (0, -1):
                    act = f"LEFT{{{rid}}}"
                else:
                    act = f"RIGHT{{{rid}}}"
                succs.append((act, new_state))

        # LOAD actions
        for rid, (r, c, load, cap) in robots.items():
            if (r, c) in taps and taps[(r, c)] > 0 and load < cap:
                new_taps = dict(taps)
                new_taps[(r, c)] -= 1
                new_taps_f = tuple(sorted(new_taps.items()))
                new_robots = []
                for orid, orr, orc, ol, oc in robots_t:
                    if orid == rid:
                        new_robots.append((orid, orr, orc, ol + 1, oc))
                    else:
                        new_robots.append((orid, orr, orc, ol, oc))
                new_robots = tuple(sorted(new_robots, key=lambda x: x[0]))
                succs.append((f"LOAD{{{rid}}}", (new_taps_f, plants_f, new_robots)))

        # POUR actions
        for rid, (r, c, load, cap) in robots.items():
            if (r, c) in plants and load > 0 and plants[(r, c)] > 0:
                new_plants = dict(plants)
                new_plants[(r, c)] -= 1
                new_plants_f = tuple(sorted(new_plants.items()))
                new_robots = []
                for orid, orr, orc, ol, oc in robots_t:
                    if orid == rid:
                        new_robots.append((orid, orr, orc, ol - 1, oc))
                    else:
                        new_robots.append((orid, orr, orc, ol, oc))
                new_robots = tuple(sorted(new_robots, key=lambda x: x[0]))
                succs.append((f"POUR{{{rid}}}", (taps_f, new_plants_f, new_robots)))

        # ---- MOVETO macros: one per (robot, target) where target in taps|plants ----
        # Conservative: only create macro when complete shortest path exists and all path cells (except src) are currently free.
        targets = list(self.tap_positions) + list(self.plant_positions)
        for rid, (r, c, load, cap) in robots.items():
            src = (r, c)
            for tgt in targets:
                if src == tgt:
                    continue
                # get precomputed distance from src to tgt
                dmap_src = self.dist_map.get(src)
                if not dmap_src:
                    continue
                dist_to_tgt = dmap_src.get(tgt)
                if dist_to_tgt is None:
                    continue  # unreachable

                # reconstruct shortest path greedily using dist_map: step to neighbor that reduces distance by 1
                path_cells = []
                cur = src
                steps = 0
                blocked = False
                while cur != tgt:
                    cur_dist_map = self.dist_map.get(cur, {})
                    cur_to_tgt = cur_dist_map.get(tgt)
                    if cur_to_tgt is None:
                        blocked = True
                        break
                    # find neighbor with dist = cur_to_tgt - 1
                    found_next = None
                    for nb in self.neighbors.get(cur, []):
                        nb_dist = self.dist_map.get(nb, {}).get(tgt)
                        if nb_dist is not None and nb_dist == cur_to_tgt - 1:
                            found_next = nb
                            break
                    if found_next is None:
                        blocked = True
                        break
                    path_cells.append(found_next)
                    cur = found_next
                    steps += 1
                    if steps > len(self.free_cells):
                        blocked = True
                        break
                if blocked or steps <= 1:
                    # skip if unreachable or single-step (we already add single-step primitives)
                    continue

                # check occupancy of entire path excluding source (conservative)
                occupied_block = False
                for cell in path_cells:
                    if cell in occupied and cell != src:
                        occupied_block = True
                        break
                if occupied_block:
                    continue

                # produce macro action string encoding steps so path_cost can charge it
                action = f"MOVETO{{{rid}}}:{steps}:{tgt[0]},{tgt[1]}"
                # create new state with robot at target (no change to taps/plants)
                new_robots = []
                for orid, orr, orc, ol, oc in robots_t:
                    if orid == rid:
                        new_robots.append((orid, tgt[0], tgt[1], ol, oc))
                    else:
                        new_robots.append((orid, orr, orc, ol, oc))
                new_robots = tuple(sorted(new_robots, key=lambda x: x[0]))
                new_state = (taps_f, plants_f, new_robots)
                succs.append((action, new_state))

        return succs

    def goal_test(self, state):
        _, plants_f, _ = state
        plants = dict(plants_f)
        return all(v == 0 for v in plants.values())

    def h_astar(self, node):
        """Admissible heuristic per provided "Optimistic Trips" logic.

        For each plant with demand>0 compute minimal trip cost across robots and sum.
        Trip cost per robot->plant via best tap:
          Effective_Capacity = min(robot_cap, total_water_remaining)
          Required_Trips = ceil(plant_demand / Effective_Capacity)
          Base_Cost = dist(robot, best_tap) + dist(best_tap, plant)
          Return_Trip_Cost = (Required_Trips - 1) * 2 * dist(best_tap, plant)
          Total = Base_Cost + Return_Trip_Cost
        Add small +1 penalty if any other robot currently lies on a shortest path used.
        """
        state = node.state
        taps_f, plants_f, robots_t = state
        taps = dict(taps_f)
        plants = dict(plants_f)
        robots = [(rid, r, c, load, cap) for (rid, r, c, load, cap) in robots_t]

        total_water_world = sum(taps.values())
        if sum(plants.values()) == 0:
            return 0

        # occupied positions
        occ = {(r, c) for (_, r, c, _, _) in robots_t}

        h_sum = 0
        for ppos, pdemand in plants.items():
            if pdemand <= 0:
                continue
            best_for_plant = float('inf')
            for (rid, rr, rc, rload, rcap) in robots:
                # effective capacity
                eff_cap = min(rcap, max(1, total_water_world))
                # required trips
                trips = int(math.ceil(pdemand / float(eff_cap)))

                # find best tap (closest tap that still may have water). We accept any tap position.
                # choose tap minimizing robot->tap + tap->plant (base trip cost)
                best_tap = None
                best_cost = float('inf')
                for tpos, tamt in taps.items():
                    if tamt <= 0:
                        continue
                    d_rt = self.dist((rr, rc), tpos)
                    d_tp = self.tap_plant_dist.get((tpos, ppos), self.dist(tpos, ppos))
                    total = d_rt + d_tp
                    if total < best_cost:
                        best_cost = total
                        best_tap = tpos

                # If no tap currently has water, check if this robot already carries
                # enough water to satisfy the plant; in that case it can go directly.
                if best_tap is None:
                    if any(rload >= pdemand for (_, _, _, rload, _) in robots):
                        # the robot with sufficient load will be handled below by using
                        # d_rt = dist(robot, plant) and return_trip_cost = 0
                        best_tap = None
                    else:
                        # no available tap and no robot currently carrying enough water
                        # -> this plant cannot be satisfied from this state (conservative)
                        return 10 ** 9

                d_rt = self.dist((rr, rc), best_tap)
                d_tp = self.dist(best_tap, ppos)
                if d_rt == float('inf') or d_tp == float('inf'):
                    continue

                base_cost = d_rt + d_tp
                return_trip_cost = (trips - 1) * 2 * d_tp if trips > 1 else 0
                total_cost = base_cost + return_trip_cost

                if total_cost < best_for_plant:
                    best_for_plant = total_cost

            if best_for_plant == float('inf'):
                # unreachable plant -> very large lower bound
                return 10 ** 9
            h_sum += best_for_plant

        # Finally, include minimal number of POUR actions (each unit needs 1 pour)
        pours_remaining = sum(plants.values())
        h_sum += pours_remaining

        return int(h_sum)

    def h_gbfs(self, node):
        """Greedy heuristic: sum distance from nearest robot to plants plus demand"""
        state = node.state
        taps_f, plants_f, robots_t = state
        plants = dict(plants_f)
        robots = [(r, c) for (_, r, c, _, _) in robots_t]
        h = 0
        for ppos, pd in plants.items():
            if pd <= 0:
                continue
            if not robots:
                dmin = 0
            else:
                dmin = min(self.dist(rpos, ppos) for rpos in robots)
                if dmin == float('inf'):
                    dmin = 0
            h += dmin + pd
        return int(h)

    def path_cost(self, c, state1, action, state2):
        """Charge MOVETO macros by their encoded step-count; primitives cost 1."""
        if isinstance(action, str) and action.startswith("MOVETO"):
            try:
                # format: MOVETO{rid}:<steps>:r,c
                parts = action.split(":", 2)
                steps = int(parts[1])
                return c + steps
            except Exception:
                return c + 1
        else:
            return c + 1


def create_watering_problem(game):
    print("<<create_watering_problem")
    return WateringProblem(game)


if __name__ == '__main__':
    import ex1_check
    ex1_check.main()
