import ex1_check
import search
import utils
import math

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

    def successor(self, state):
        """Generate successor states: return list of (action_str, next_state).

        Actions: UP{rid}, DOWN{rid}, LEFT{rid}, RIGHT{rid}, LOAD{rid}, POUR{rid}.
        """
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

        return successors

    def goal_test(self, state):
        """Return True iff all plants have received required water (remaining==0)."""
        taps_fset, plants_fset, robots_tuple = state
        plants = dict(plants_fset)
        return all(amt == 0 for amt in plants.values())

    def h_astar(self, node):
        """Admissible heuristic: lower bound = pours needed + loads needed."""
        state = node.state
        taps_fset, plants_fset, robots_tuple = state
        plants = dict(plants_fset)

        total_needed = sum(v for v in plants.values())
        total_loaded = sum(r[3] for r in robots_tuple)
        loads_needed = max(0, total_needed - total_loaded)

        # If no robots or no capacity, return large heuristic (unreachable)
        if not robots_tuple or self.max_robot_cap <= 0:
            return 10 ** 9

        # Movement lower bound improved by accounting for loaded units and unloaded units separately.
        # 1) Loaded units: optimistic minimal movement is distance from loaded robot to nearest plant.
        loaded_dists = []
        for (_, rr, rc, rload, rcap) in robots_tuple:
            if rload <= 0:
                continue
            robot_pos = (rr, rc)
            # distance from this robot to nearest plant needing water
            dmin = float('inf')
            for ppos, need in plants.items():
                if need <= 0:
                    continue
                # dist from plant to robot_pos (symmetric in grid)
                pdmap = self.dist_from_plant.get(ppos)
                if pdmap is None:
                    continue
                d = pdmap.get(robot_pos, float('inf'))
                if d < dmin:
                    dmin = d
            if dmin == float('inf'):
                # this robot cannot reach any plant (ignore its loaded units)
                continue
            # add rload copies (optimistic per-unit cost)
            for _ in range(rload):
                loaded_dists.append(dmin)

        loaded_dists.sort()
        # consider at most total_needed loaded units
        loaded_units_considered = min(sum(r[3] for r in robots_tuple), total_needed)
        movement_loaded = sum(loaded_dists[:loaded_units_considered]) if loaded_dists else 0

        # 2) Unloaded units: each requires at least robot->tap + tap->plant movement.
        # optimistic estimate: use minimal robot->nearest_tap and minimal plant_min_tap
        robot_to_tap_min = float('inf')
        for (_, rr, rc, _, _) in robots_tuple:
            robot_pos = (rr, rc)
            d = self.min_dist_to_tap.get(robot_pos, float('inf'))
            if d < robot_to_tap_min:
                robot_to_tap_min = d

        plant_min_tap_min = min(self.plant_min_tap_dist.values()) if self.plant_min_tap_dist else float('inf')

        if robot_to_tap_min == float('inf') or plant_min_tap_min == float('inf'):
            # unreachable parts
            return 10 ** 9

        movement_unloaded = loads_needed * (robot_to_tap_min + plant_min_tap_min)

        movement_lb = movement_loaded + movement_unloaded

        h = total_needed + loads_needed + movement_lb
        return int(h)

    def h_gbfs(self, node):
        """Greedy heuristic: sum of distances from robots to plants plus pours."""
        state = node.state
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
        return int(h)


def create_watering_problem(game):
    print("<<create_watering_problem")
    """ Create a pressure plate problem, based on the description.
    game - tuple of tuples as described in pdf file"""
    return WateringProblem(game)


if __name__ == '__main__':
    ex1_check.main()
