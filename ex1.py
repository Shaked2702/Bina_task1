import ex1_check
import search
import utils
import math
import time

# Instrumentation: wrap search functions to record runtime and expanded nodes
try:
    _orig_astar = search.astar_search

    def _instrumented_astar(problem, h=None):
        start = time.time()
        if USE_MACRO:
            result = macro_astar(problem, h)
        else:
            result = _orig_astar(problem, h)
        end = time.time()
        # result may be (node, expanded) or node; graph_search returns (node, expanded)
        expanded = None
        node = None
        if isinstance(result, tuple) and len(result) >= 2:
            node, expanded = result[0], result[1]
        else:
            node = result
        # attach stats to problem for later inspection
        try:
            problem.last_search_time = end - start
            problem.last_search_expanded = expanded
        except Exception:
            pass
        # compute and attach extra info (solution length and cost) when available
        sol_len = None
        sol_cost = None
        try:
            if isinstance(node, search.Node):
                path = node.path()[::-1]
                actions = [pi.action for pi in path][1:]
                sol_len = len(actions)
                sol_cost = node.path_cost
                problem.last_search_solution_length = sol_len
                problem.last_search_solution_cost = sol_cost
        except Exception:
            pass

        # also print a short summary including cost and length
        mode = 'MACRO' if USE_MACRO else 'PRIMITIVE'
        print(f"[ASTAR:{mode}] time={end-start:.4f}s expanded={expanded} cost={sol_cost} len={sol_len}")
        return result

    search.astar_search = _instrumented_astar
except Exception:
    pass

try:
    _orig_gbfs = search.greedy_best_first_graph_search

    def _instrumented_gbfs(problem, h=None):
        start = time.time()
        result = _orig_gbfs(problem, h)
        end = time.time()
        expanded = None
        node = None
        if isinstance(result, tuple) and len(result) >= 2:
            node, expanded = result[0], result[1]
        else:
            node = result
        try:
            problem.last_search_time = end - start
            problem.last_search_expanded = expanded
        except Exception:
            pass
        # compute extra info for GBFS as well
        sol_len = None
        sol_cost = None
        try:
            if isinstance(node, search.Node):
                path = node.path()[::-1]
                actions = [pi.action for pi in path][1:]
                sol_len = len(actions)
                sol_cost = node.path_cost
                problem.last_search_solution_length = sol_len
                problem.last_search_solution_cost = sol_cost
        except Exception:
            pass
        print(f"[GBFS] time={end-start:.4f}s expanded={expanded} cost={sol_cost} len={sol_len}")
        return result

    search.greedy_best_first_graph_search = _instrumented_gbfs
except Exception:
    pass

# Toggle: enable macro planner when True (branch: macro-planner)
USE_MACRO = True


class MacroProblem(search.Problem):
    """Wrapper Problem that exposes macro actions (move directly to taps/plants,
    and bulk LOAD/POUR) while keeping the same underlying state representation.
    Actions are returned as tuples: (atype, cost, details) so `path_cost` can
    use the provided cost. The `macro_astar` will post-process the macro
    solution into primitive action strings before returning to callers.
    """

    def __init__(self, base_problem):
        super().__init__(base_problem.initial)
        self.base = base_problem

    def goal_test(self, state):
        return self.base.goal_test(state)

    def successor(self, state):
        # Generate macro successors from `state` using base problem data
        succs = []
        taps_fset, plants_fset, robots_tuple = state
        taps = dict(taps_fset)
        plants = dict(plants_fset)

        # interesting targets: taps and plants positions
        interesting = list(self.base.tap_positions) + list(self.base.plant_positions)

        for (rid, rr, rc, rload, rcap) in robots_tuple:
            robot_pos = (rr, rc)

            # Macro MOVE: to any interesting target
            for tgt in interesting:
                # compute distance using precomputed maps
                d = None
                if tgt in self.base.dist_from_plant:
                    d = self.base.dist_from_plant[tgt].get(robot_pos, float('inf'))
                if d is None or d == float('inf'):
                    if tgt in self.base.dist_from_tap:
                        d = self.base.dist_from_tap[tgt].get(robot_pos, float('inf'))
                if d is None or d == float('inf'):
                    continue
                if d == 0:
                    # already on target; we still add zero-cost MOVE so sequences are explicit
                    pass

                # create new robot tuple with position at tgt
                new_robots = []
                for (orid, or_, oc, oload, ocap) in robots_tuple:
                    if orid == rid:
                        new_robots.append((orid, tgt[0], tgt[1], oload, ocap))
                    else:
                        new_robots.append((orid, or_, oc, oload, ocap))
                new_robots = tuple(sorted(new_robots, key=lambda x: x[0]))

                new_state = (taps_fset, plants_fset, new_robots)
                action = ("MOVE", int(d), (rid, robot_pos, tgt))
                succs.append((action, new_state))

            # Macro LOAD: if on a tap, load as many units as possible in one macro action
            if robot_pos in taps and taps[robot_pos] > 0 and rload < rcap:
                can_load = min(rcap - rload, taps[robot_pos])
                new_taps = dict(taps)
                new_taps[robot_pos] = new_taps[robot_pos] - can_load
                new_taps_f = tuple(sorted(new_taps.items()))

                new_robots = []
                for (orid, or_, oc, oload, ocap) in robots_tuple:
                    if orid == rid:
                        new_robots.append((orid, or_, oc, oload + can_load, ocap))
                    else:
                        new_robots.append((orid, or_, oc, oload, ocap))
                new_robots = tuple(sorted(new_robots, key=lambda x: x[0]))

                new_state = (new_taps_f, plants_fset, new_robots)
                action = ("LOAD", int(can_load), (rid, robot_pos))
                succs.append((action, new_state))

            # Macro POUR: if on a plant, pour as many units as possible in one macro action
            if robot_pos in plants and plants[robot_pos] > 0 and rload > 0:
                can_pour = min(rload, plants[robot_pos])
                new_plants = dict(plants)
                new_plants[robot_pos] = new_plants[robot_pos] - can_pour
                new_plants_f = tuple(sorted(new_plants.items()))

                new_robots = []
                for (orid, or_, oc, oload, ocap) in robots_tuple:
                    if orid == rid:
                        new_robots.append((orid, or_, oc, oload - can_pour, ocap))
                    else:
                        new_robots.append((orid, or_, oc, oload, ocap))
                new_robots = tuple(sorted(new_robots, key=lambda x: x[0]))

                new_state = (taps_fset, new_plants_f, new_robots)
                action = ("POUR", int(can_pour), (rid, robot_pos))
                succs.append((action, new_state))

        return succs

    def path_cost(self, c, state1, action, state2):
        # action is a tuple (atype, cost, details)
        try:
            atype, acost, _ = action
            return c + int(acost)
        except Exception:
            return super().path_cost(c, state1, action, state2)


def reconstruct_primitive_moves(base_problem, rid, start, goal):
    """Reconstruct a shortest sequence of primitive move action strings
    from `start` to `goal` for robot `rid` using base_problem's distance maps.
    Returns list like ['UP{10}', 'RIGHT{10}', ...]."""
    if start == goal:
        return []
    # choose dist map of goal (plant or tap)
    if goal in base_problem.dist_from_plant:
        dmap = base_problem.dist_from_plant[goal]
    elif goal in base_problem.dist_from_tap:
        dmap = base_problem.dist_from_tap[goal]
    else:
        return []

    cur = start
    moves = []
    while cur != goal:
        curd = dmap.get(cur, float('inf'))
        # find neighbor with dist = curd - 1
        found = False
        for nb in base_problem.neighbors.get(cur, []):
            if dmap.get(nb, float('inf')) == curd - 1:
                dr, dc = nb[0] - cur[0], nb[1] - cur[1]
                if (dr, dc) == (-1, 0):
                    aname = 'UP'
                elif (dr, dc) == (1, 0):
                    aname = 'DOWN'
                elif (dr, dc) == (0, -1):
                    aname = 'LEFT'
                else:
                    aname = 'RIGHT'
                moves.append(f"{aname}{{{rid}}}")
                cur = nb
                found = True
                break
        if not found:
            # no path (shouldn't happen) — abort
            return []
    return moves


def macro_astar(problem, h=None):
    """Run A* on the macro action space and convert the macro solution to
    primitive actions (so callers receive the same primitive-action plan).
    Returns (node, expanded) matching the original signature.
    """
    # create MacroProblem wrapper that uses the same states
    mp = MacroProblem(problem)
    # use original astar implementation (saved as _orig_astar earlier)
    try:
        res = _orig_astar(mp, h or problem.h_astar)
    except Exception:
        # fallback to original if wrapper missing
        return _orig_astar(problem, h)

    # res may be (node, expanded) or node
    if isinstance(res, tuple) and len(res) >= 2:
        mnode, expanded = res[0], res[1]
    else:
        mnode, expanded = res, None

    if mnode is None:
        return res

    # Convert macro node path to a primitive action sequence and rebuild Nodes
    macro_path = mnode.path()[::-1]
    primitive_actions = []
    cur_state = problem.initial
    for mac in macro_path[1:]:
        action = mac.action
        atype, acost, details = action
        if atype == 'MOVE':
            rid, start, tgt = details[0], details[1], details[2]
            moves = reconstruct_primitive_moves(problem, rid, start, tgt)
            for ma in moves:
                primitive_actions.append(ma)
        elif atype == 'LOAD':
            rid = details[0]
            count = int(acost)
            for _ in range(count):
                primitive_actions.append(f"LOAD{{{rid}}}")
        elif atype == 'POUR':
            rid = details[0]
            count = int(acost)
            for _ in range(count):
                primitive_actions.append(f"POUR{{{rid}}}")

        # apply primitive actions to advance cur_state accordingly
        for pa in primitive_actions[:]:
            # find successor that matches pa
            found = False
            for (a, s2) in problem.successor(cur_state):
                if a == pa:
                    cur_state = s2
                    found = True
                    break
            if not found:
                # if we couldn't apply primitive action, stop conversion
                break
        # clear primitive_actions to avoid reapplying in next macro
        primitive_actions = []

    # Now rebuild a Node chain of primitive steps from initial to goal by re-simulating
    # using original successor function and creating Nodes with primitive actions.
    cur_state = problem.initial
    root = search.Node(cur_state, parent=None, action=None, path_cost=0)
    last = root
    full_actions = []

    # Reconstruct full primitive action sequence from the macro path again
    for mac in macro_path[1:]:
        action = mac.action
        atype, acost, details = action
        if atype == 'MOVE':
            rid, start, tgt = details
            moves = reconstruct_primitive_moves(problem, rid, start, tgt)
            for ma in moves:
                # apply ma
                for (a, s2) in problem.successor(cur_state):
                    if a == ma:
                        full_actions.append((ma, s2))
                        cur_state = s2
                        break
        elif atype == 'LOAD':
            rid = details[0]
            count = int(acost)
            for _ in range(count):
                act = f"LOAD{{{rid}}}"
                for (a, s2) in problem.successor(cur_state):
                    if a == act:
                        full_actions.append((act, s2))
                        cur_state = s2
                        break
        elif atype == 'POUR':
            rid = details[0]
            count = int(acost)
            for _ in range(count):
                act = f"POUR{{{rid}}}"
                for (a, s2) in problem.successor(cur_state):
                    if a == act:
                        full_actions.append((act, s2))
                        cur_state = s2
                        break

    # Build Node chain
    cur = root
    for (a, s2) in full_actions:
        pc = problem.path_cost(cur.path_cost, cur.state, a, s2)
        node = search.Node(s2, parent=cur, action=a, path_cost=pc)
        cur = node

    return (cur, expanded)


id = ["No numbers - I'm special!"]





class WateringProblem(search.Problem):
    """This class implements the Plant Watering problem."""

    def __init__(self, initial):
        """Convert the provided `initial` dict into an immutable, hashable
        internal state and store it as the problem initial state.

        State format (tuple):
          (size, walls_fset, taps_fset, plants_fset, robots_tuple)
        where taps_fset and plants_fset are frozensets of ((r,c), amt)
        and robots_tuple is a tuple of (rid, r, c, load, cap) sorted by rid.
        """
        size = tuple(initial['Size'])
        # Use sorted tuples instead of frozensets so the state is deterministic
        # while remaining immutable and hashable.
        walls = tuple(sorted(initial.get('Walls', set())))
        taps = tuple(sorted(initial.get('Taps', {}).items()))
        plants = tuple(sorted(initial.get('Plants', {}).items()))
        robots = tuple(sorted(((rid, r, c, load, cap)
                               for rid, (r, c, load, cap) in initial.get('Robots', {}).items()),
                              key=lambda x: x[0]))

        # Store immutable initial state (used by search). Keep static data
        # (size, walls, neighbors, distances) as instance attributes to keep
        # the per-state representation small.
        state = (taps, plants, robots)
        # Precompute useful static data to keep states small and speed successor/h.
        self.size = size
        self.walls = set(walls)
        # lists of positions
        self.tap_positions = [pos for (pos, _) in taps]
        self.plant_positions = [pos for (pos, _) in plants]

        # free cells (cells that are not walls)
        max_r, max_c = size
        self.free_cells = {(r, c) for r in range(max_r) for c in range(max_c) if (r, c) not in self.walls}

        # neighbors: for each free cell, allowed adjacent free cells (ignoring robots)
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

        # Precompute BFS shortest-path distances from each plant and each tap to all cells
        # (accounts for walls). Store per-target distance maps and also cell->min distance.
        from collections import deque

        def bfs_from(source):
            dist = {cell: float('inf') for cell in self.free_cells}
            q = deque()
            if source not in self.free_cells:
                return dist
            dist[source] = 0
            q.append(source)
            while q:
                cur = q.popleft()
                for nb in self.neighbors[cur]:
                    if dist[nb] == float('inf'):
                        dist[nb] = dist[cur] + 1
                        q.append(nb)
            return dist

        # distances from each plant/tap to all cells
        self.dist_from_plant = {p: bfs_from(p) for p in self.plant_positions}
        self.dist_from_tap = {t: bfs_from(t) for t in self.tap_positions}

        # assign stable integer IDs for plants and taps for convenient indexing
        self.plant_pos_by_id = list(self.plant_positions)
        self.plant_id_by_pos = {pos: i for i, pos in enumerate(self.plant_pos_by_id)}
        self.tap_pos_by_id = list(self.tap_positions)
        self.tap_id_by_pos = {pos: i for i, pos in enumerate(self.tap_pos_by_id)}

        # per-target maps keyed by id (same data, indexed by integer ids)
        self.dist_from_plant_id = {self.plant_id_by_pos[pos]: dmap
                                   for pos, dmap in self.dist_from_plant.items()}
        self.dist_from_tap_id = {self.tap_id_by_pos[pos]: dmap
                                 for pos, dmap in self.dist_from_tap.items()}

        # minimal distance from each plant to the nearest tap (used in heuristic)
        self.plant_min_tap_dist = {}
        for pid, pos in enumerate(self.plant_pos_by_id):
            # compute min distance from any tap position to this plant
            dmin = float('inf')
            for tpos in self.tap_positions:
                d = self.dist_from_plant[pos].get(tpos, float('inf'))
                if d < dmin:
                    dmin = d
            self.plant_min_tap_dist[pid] = dmin

        # record max robot capacity (used to compute optimistic #trips)
        # initial robots are in the `robots` tuple we stored in the starting state
        if robots:
            self.max_robot_cap = max(cap for (_, _, _, _, cap) in robots)
        else:
            self.max_robot_cap = 0

        # minimal distances per cell
        def min_dist_map(dist_map):
            if not dist_map:
                return {cell: float('inf') for cell in self.free_cells}
            result = {cell: float('inf') for cell in self.free_cells}
            for source, dmap in dist_map.items():
                for cell, d in dmap.items():
                    if d < result[cell]:
                        result[cell] = d
            return result

        self.min_dist_to_plant = min_dist_map(self.dist_from_plant)
        self.min_dist_to_tap = min_dist_map(self.dist_from_tap)

        # finalize parent initialization
        search.Problem.__init__(self, state)
        # Caches to avoid repeated work during search (keyed by immutable state)
        self._succ_cache = {}
        self._h_astar_cache = {}
        self._h_gbfs_cache = {}

    def successor(self, state):
        """Generate successor states: return list of (action_str, next_state).

        Actions: UP{rid}, DOWN{rid}, LEFT{rid}, RIGHT{rid}, LOAD{rid}, POUR{rid}.
        """
        # Use cache to avoid recomputing successors for the same state
        cached = self._succ_cache.get(state)
        if cached is not None:
            # return a fresh list so callers can mutate if they (incorrectly) want to
            return list(cached)

        successors = []

        taps_fset, plants_fset, robots_tuple = state
        max_r, max_c = self.size[0], self.size[1]

        walls = self.walls
        taps = dict(taps_fset)
        plants = dict(plants_fset)

        # robots mapping: rid -> [r, c, load, cap]
        robots = {rid: [r, c, load, cap] for (rid, r, c, load, cap) in robots_tuple}
        occupied = {(r2, c2) for (_, r2, c2, _, _) in robots_tuple}

        # movement via precomputed neighbors (ignores other robots; we still check occupancy)
        for rid, (r, c, load, cap) in robots.items():
            # Movement: iterate allowed neighbor cells (precomputed ignoring robots)
            for (nr, nc) in self.neighbors.get((r, c), []):
                # occupied by another robot?
                if (nr, nc) in occupied and not (nr == r and nc == c):
                    continue

                # determine action name from delta
                dr, dc = nr - r, nc - c
                if (dr, dc) == (-1, 0):
                    aname = 'UP'
                elif (dr, dc) == (1, 0):
                    aname = 'DOWN'
                elif (dr, dc) == (0, -1):
                    aname = 'LEFT'
                else:
                    aname = 'RIGHT'

                new_robots = []
                for (orid, or_, oc, oload, ocap) in robots_tuple:
                    if orid == rid:
                        new_robots.append((orid, nr, nc, oload, ocap))
                    else:
                        new_robots.append((orid, or_, oc, oload, ocap))
                new_robots = tuple(sorted(new_robots, key=lambda x: x[0]))

                new_state = (taps_fset, plants_fset, new_robots)
                action = f"{aname}{{{rid}}}"
                successors.append((action, new_state))

            # LOAD
            if (r, c) in taps and taps[(r, c)] > 0 and load < cap:
                new_taps = dict(taps)
                new_taps[(r, c)] = new_taps[(r, c)] - 1
                new_taps_f = tuple(sorted(new_taps.items()))

                new_robots = []
                for (orid, or_, oc, oload, ocap) in robots_tuple:
                    if orid == rid:
                        new_robots.append((orid, or_, oc, oload + 1, ocap))
                    else:
                        new_robots.append((orid, or_, oc, oload, ocap))
                new_robots = tuple(sorted(new_robots, key=lambda x: x[0]))

                new_state = (new_taps_f, plants_fset, new_robots)
                action = f"LOAD{{{rid}}}"
                successors.append((action, new_state))

            # POUR
            if (r, c) in plants and plants[(r, c)] > 0 and load > 0:
                new_plants = dict(plants)
                new_plants[(r, c)] = new_plants[(r, c)] - 1
                new_plants_f = tuple(sorted(new_plants.items()))

                new_robots = []
                for (orid, or_, oc, oload, ocap) in robots_tuple:
                    if orid == rid:
                        new_robots.append((orid, or_, oc, oload - 1, ocap))
                    else:
                        new_robots.append((orid, or_, oc, oload, ocap))
                new_robots = tuple(sorted(new_robots, key=lambda x: x[0]))

                new_state = (taps_fset, new_plants_f, new_robots)
                action = f"POUR{{{rid}}}"
                successors.append((action, new_state))

        # store as tuple (immutable) to safely cache
        self._succ_cache[state] = tuple(successors)
        return list(successors)

    def goal_test(self, state):
        """Return True iff all plants have received required water (remaining==0)."""
        taps_fset, plants_fset, robots_tuple = state
        plants = dict(plants_fset)
        return all(amt == 0 for amt in plants.values())

    def h_astar(self, node):
        """Admissible heuristic: lower bound = pours needed + loads needed."""
        state = node.state
        cached = self._h_astar_cache.get(state)
        if cached is not None:
            return cached
        taps_fset, plants_fset, robots_tuple = state
        plants = dict(plants_fset)

        total_needed = sum(v for v in plants.values())
        total_loaded = sum(r[3] for r in robots_tuple)
        loads_needed = max(0, total_needed - total_loaded)

        if total_needed == 0:
            return 0

        # If no robots or no capacity, return large heuristic (unreachable)
        if not robots_tuple or self.max_robot_cap <= 0:
            return 10 ** 9

        # Build per-unit plant minimal costs (tap->plant) repeated per required unit
        plant_unit_costs = []
        for (pos, amt) in plants.items():
            if amt <= 0:
                continue
            pid = self.plant_id_by_pos.get(pos, None)
            if pid is None:
                return 10 ** 9
            d_plant = self.plant_min_tap_dist.get(pid, float('inf'))
            if d_plant == float('inf'):
                return 10 ** 9
            plant_unit_costs.extend([d_plant] * int(amt))

        plant_unit_costs.sort()  # ascending

        # Loaded units: compute optimistic robot->plant distances for each loaded unit
        loaded_dists = []
        for (_, rr, rc, rload, _) in robots_tuple:
            robot_pos = (rr, rc)
            if rload <= 0:
                continue
            # distance to closest plant (any plant needing water)
            pdmap_min = float('inf')
            for ppos in plants.keys():
                pdmap = self.dist_from_plant.get(ppos)
                if pdmap is None:
                    continue
                d = pdmap.get(robot_pos, float('inf'))
                if d < pdmap_min:
                    pdmap_min = d
            if pdmap_min == float('inf'):
                continue
            loaded_dists.extend([pdmap_min] * int(rload))

        loaded_dists.sort()

        loaded_used = min(len(loaded_dists), total_needed)
        # Assign loaded units optimistically to the most expensive plant-unit deliveries
        # Remove largest plant_unit_costs entries as they can be satisfied by loaded units
        remaining_units = total_needed
        if loaded_used > 0:
            # remove largest loaded_used elements from plant_unit_costs
            plant_unit_costs = plant_unit_costs[:max(0, len(plant_unit_costs) - loaded_used)]
            remaining_units = total_needed - loaded_used

        movement_loaded = sum(loaded_dists[:loaded_used]) if loaded_used > 0 else 0

        # For remaining units, each requires robot->tap + tap->plant
        # robot->tap distance: optimistic min over robots for their distance to nearest tap
        robot_to_tap_min = float('inf')
        for (_, rr, rc, _, _) in robots_tuple:
            robot_pos = (rr, rc)
            d = self.min_dist_to_tap.get(robot_pos, float('inf'))
            if d < robot_to_tap_min:
                robot_to_tap_min = d

        if robot_to_tap_min == float('inf'):
            return 10 ** 9

        # Sum the smallest remaining_units plant unit costs (optimistic assignment)
        plant_cost_for_unloaded = sum(plant_unit_costs[:remaining_units]) if remaining_units > 0 else 0
        movement_unloaded = remaining_units * robot_to_tap_min + plant_cost_for_unloaded

        movement_lb = movement_loaded + movement_unloaded

        h = total_needed + loads_needed + movement_lb
        val = int(h)
        self._h_astar_cache[state] = val
        return val

    def h_gbfs(self, node):
        """Greedy heuristic: sum of distances from robots to plants plus pours."""
        state = node.state
        cached = self._h_gbfs_cache.get(state)
        if cached is not None:
            return cached
        taps_fset, plants_fset, robots_tuple = state
        plants = {pos: amt for (pos, amt) in plants_fset}
        robots = [(r, c) for (_, r, c, _, _) in robots_tuple]

        h = 0
        for (pos, amt) in plants.items():
            if amt <= 0:
                continue
            # distance from nearest robot to this plant (using precomputed BFS distances)
            if robots and pos in self.dist_from_plant:
                d = min(self.dist_from_plant[pos].get(robot_pos, float('inf')) for robot_pos in robots)
                if d == float('inf'):
                    d = 0
            else:
                # no robots or no precomputed map -> fallback to 0
                d = 0
            h += d + amt

        loaded = sum(r[3] for r in robots_tuple)
        h = max(0, h - loaded)
        val = int(h)
        self._h_gbfs_cache[state] = val
        return val


def create_watering_problem(game):
    print("<<create_watering_problem")
    """ Create a pressure plate problem, based on the description.
    game - tuple of tuples as described in pdf file"""
    return WateringProblem(game)


if __name__ == '__main__':
    ex1_check.main()
