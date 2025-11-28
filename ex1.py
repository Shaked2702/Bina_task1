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

        state = (size, walls, taps, plants, robots)
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
        occupied = {(r, c) for (r, c, _, _) in ((r, c, l, cap) for (_, r, c, l, cap) in robots_tuple)}

        moves = [(-1, 0, 'UP'), (1, 0, 'DOWN'), (0, -1, 'LEFT'), (0, 1, 'RIGHT')]

        for rid, (r, c, load, cap) in robots.items():
            # Movement
            for dr, dc, aname in moves:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < max_r and 0 <= nc < max_c):
                    continue
                if (nr, nc) in walls:
                    continue
                # occupied by other robot?
                if any((nr == r2 and nc == c2) for (r2, c2) in occupied if not (r2 == r and c2 == c)):
                    continue

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

        def manhattan(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        h = 0
        for (pos, amt) in plants.items():
            if amt <= 0:
                continue
            if robots:
                d = min(manhattan(pos, rob) for rob in robots)
            else:
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
