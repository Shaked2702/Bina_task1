import ex1_check
import search
import utils

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
        walls = frozenset(initial.get('Walls', set()))
        taps = frozenset(((pos, amt) for pos, amt in initial.get('Taps', {}).items()))
        plants = frozenset(((pos, amt) for pos, amt in initial.get('Plants', {}).items()))
        robots = tuple(sorted(((rid, r, c, load, cap)
                               for rid, (r, c, load, cap) in initial.get('Robots', {}).items()),
                              key=lambda x: x[0]))

        # Store immutable initial state (used by search). All heavy precomputations
        # are kept as attributes on the problem instance (not part of the state).
        state = (size, walls, taps, plants, robots)
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

        size, walls_fset, taps_fset, plants_fset, robots_tuple = state
        max_r, max_c = size[0], size[1]

        walls = set(walls_fset)
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

                new_state = (size, walls_fset, taps_fset, plants_fset, new_robots)
                action = f"{aname}{{{rid}}}"
                successors.append((action, new_state))

            # LOAD
            if (r, c) in taps and taps[(r, c)] > 0 and load < cap:
                new_taps = dict(taps)
                new_taps[(r, c)] = new_taps[(r, c)] - 1
                new_taps_f = frozenset(new_taps.items())

                new_robots = []
                for (orid, or_, oc, oload, ocap) in robots_tuple:
                    if orid == rid:
                        new_robots.append((orid, or_, oc, oload + 1, ocap))
                    else:
                        new_robots.append((orid, or_, oc, oload, ocap))
                new_robots = tuple(sorted(new_robots, key=lambda x: x[0]))

                new_state = (size, walls_fset, new_taps_f, plants_fset, new_robots)
                action = f"LOAD{{{rid}}}"
                successors.append((action, new_state))

            # POUR
            if (r, c) in plants and plants[(r, c)] > 0 and load > 0:
                new_plants = dict(plants)
                new_plants[(r, c)] = new_plants[(r, c)] - 1
                new_plants_f = frozenset(new_plants.items())

                new_robots = []
                for (orid, or_, oc, oload, ocap) in robots_tuple:
                    if orid == rid:
                        new_robots.append((orid, or_, oc, oload - 1, ocap))
                    else:
                        new_robots.append((orid, or_, oc, oload, ocap))
                new_robots = tuple(sorted(new_robots, key=lambda x: x[0]))

                new_state = (size, walls_fset, taps_fset, new_plants_f, new_robots)
                action = f"POUR{{{rid}}}"
                successors.append((action, new_state))

        return successors

    def goal_test(self, state):
        """Return True iff all plants have received required water (remaining==0)."""
        _, _, _, plants_fset, _ = state
        plants = dict(plants_fset)
        return all(amt == 0 for amt in plants.values())

    def h_astar(self, node):
        """Admissible heuristic: lower bound = pours needed + loads needed."""
        state = node.state
        _, _, _, plants_fset, robots_tuple = state
        plants = dict(plants_fset)
        total_needed = sum(v for v in plants.values())
        total_loaded = sum(r[3] for r in robots_tuple)
        loads_needed = max(0, total_needed - total_loaded)
        return int(total_needed + loads_needed)

    def h_gbfs(self, node):
        """Greedy heuristic: sum of distances from robots to plants plus pours."""
        state = node.state
        _, _, _, plants_fset, robots_tuple = state
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
