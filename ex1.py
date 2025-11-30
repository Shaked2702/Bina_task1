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
        Exactly one robot moves per step.
        """
        taps_f, plants_f, robots_t = state
        taps = dict(taps_f)
        plants = dict(plants_f)
        robots = {rid: [r, c, load, cap] for (rid, r, c, load, cap) in robots_t}

        occupied = {(r, c) for (_, r, c, _, _) in robots_t}

        succs = []

        # movement actions
        for rid, (r, c, load, cap) in robots.items():
            for nb in self.neighbors.get((r, c), []):
                if nb in occupied and nb != (r, c):
                    continue
                nr, nc = nb
                # create new robots tuple
                new_robots = []
                for orid, orr, orc, ol, oc in robots_t:
                    if orid == rid:
                        new_robots.append((orid, nr, nc, ol, oc))
                    else:
                        new_robots.append((orid, orr, orc, ol, oc))
                new_robots = tuple(sorted(new_robots, key=lambda x: x[0]))
                new_state = (taps_f, plants_f, new_robots)
                dr, dc = nr - r, nc - c
                if (dr, dc) == (-1, 0):
                    an = f"UP{{{rid}}}"
                elif (dr, dc) == (1, 0):
                    an = f"DOWN{{{rid}}}"
                elif (dr, dc) == (0, -1):
                    an = f"LEFT{{{rid}}}"
                else:
                    an = f"RIGHT{{{rid}}}"
                succs.append((an, new_state))

            # LOAD
            if (r, c) in taps and taps[(r, c)] > 0 and load < cap:
                new_taps = dict(taps)
                new_taps[(r, c)] = new_taps[(r, c)] - 1
                if new_taps[(r, c)] == 0:
                    del new_taps[(r, c)]
                new_taps_f = tuple(sorted(new_taps.items()))
                new_robots = []
                for orid, orr, orc, ol, oc in robots_t:
                    if orid == rid:
                        new_robots.append((orid, orr, orc, ol + 1, oc))
                    else:
                        new_robots.append((orid, orr, orc, ol, oc))
                new_robots = tuple(sorted(new_robots, key=lambda x: x[0]))
                new_state = (new_taps_f, plants_f, new_robots)
                an = f"LOAD{{{rid}}}"
                succs.append((an, new_state))

            # POUR
            if (r, c) in plants and plants[(r, c)] > 0 and load > 0:
                new_plants = dict(plants)
                new_plants[(r, c)] = new_plants[(r, c)] - 1
                if new_plants[(r, c)] == 0:
                    del new_plants[(r, c)]
                new_plants_f = tuple(sorted(new_plants.items()))
                new_robots = []
                for orid, orr, orc, ol, oc in robots_t:
                    if orid == rid:
                        new_robots.append((orid, orr, orc, ol - 1, oc))
                    else:
                        new_robots.append((orid, orr, orc, ol, oc))
                new_robots = tuple(sorted(new_robots, key=lambda x: x[0]))
                new_state = (taps_f, new_plants_f, new_robots)
                an = f"POUR{{{rid}}}"
                succs.append((an, new_state))

        return succs

    def goal_test(self, state):
        _, plants_f, _ = state
        return len(plants_f) == 0

    def h_astar(self, node):
        """Admissible heuristic: Work / Capacity relaxation.
        
        1. Action costs: We need 1 POUR per unit of demand.
           We need 1 LOAD per unit of demand that isn't currently loaded.
        2. Movement costs:
           Calculate 'transport work' = sum(dist(Source, Plant) * Demand).
           Source is the closest loaded robot OR closest tap.
           Divide total work by Max_Capacity to account for batching.
           If we need to fetch water (Load < Demand), add min dist(Robot, Tap).
        """
        state = node.state
        taps_f, plants_f, robots_t = state
        taps = dict(taps_f)
        plants = dict(plants_f)
        
        # Basic stats
        total_demand = sum(plants.values())
        if total_demand == 0:
            return 0
            
        total_load = sum(load for (_, _, _, load, _) in robots_t)
        max_cap = max((cap for (_, _, _, _, cap) in robots_t), default=1)
        
        # 1. Action Costs
        # Pours needed
        h_actions = total_demand
        # Loads needed (if we don't have enough water)
        needed_load = max(0, total_demand - total_load)
        h_actions += needed_load
        
        # 2. Movement Costs (Relaxation)
        # We calculate the minimal "distance units" the water must travel.
        # For each unit of demand at a plant, it must come from somewhere.
        # If we have loaded robots, they are sources. Taps are also sources.
        # We take the optimistic view that any source can supply any plant.
        
        loaded_robot_positions = [(r, c) for (_, r, c, load, _) in robots_t if load > 0]
        tap_positions = [pos for pos, amt in taps.items() if amt > 0]
        
        # If no water sources available at all (and we need water), return infinity
        if not tap_positions and not loaded_robot_positions and total_demand > 0:
             return 10**9

        transport_work = 0
        for ppos, pdemand in plants.items():
            if pdemand <= 0: continue
            
            # Distance from nearest tap
            dist_tap = float('inf')
            if tap_positions:
                # Use cached tap_plant_dist if available, else direct lookup
                dist_tap = min(self.tap_plant_dist.get((t, ppos), self.dist(t, ppos)) 
                             for t in tap_positions)
            
            # Distance from nearest loaded robot
            dist_robot = float('inf')
            if loaded_robot_positions:
                dist_robot = min(self.dist(r, ppos) for r in loaded_robot_positions)
                
            # Optimistic: take the better of the two sources
            # (If we have load, we can use it. If not, we must use tap)
            # But we can't use robot source if we don't have load...
            # However, we are aggregating total work.
            # If we have L units of load, we can save L * (dist_tap - dist_robot) work?
            # Simpler admissible bound:
            # Assume all current load is magically at the BEST position for the demands.
            # Actually, just use dist_tap for ALL demand, because eventually water comes from taps.
            # Unless it's already in a robot closer than the tap.
            
            cost_per_unit = dist_tap
            if loaded_robot_positions:
                cost_per_unit = min(cost_per_unit, dist_robot)
                
            if cost_per_unit == float('inf'):
                return 10**9
                
            transport_work += cost_per_unit * pdemand

        # Divide by max capacity because one robot can carry multiple units
        h_move = math.ceil(transport_work / max_cap)
        
        # 3. Fetch Cost
        # If we don't have enough load, at least one robot must go to a tap.
        if total_load < total_demand:
            # Min dist from any robot to any tap
            min_fetch = float('inf')
            robot_positions = [(r, c) for (_, r, c, _, _) in robots_t]
            if robot_positions and tap_positions:
                # This can be optimized, but N is small
                for rpos in robot_positions:
                    for tpos in tap_positions:
                        d = self.dist(rpos, tpos)
                        if d < min_fetch:
                            min_fetch = d
            
            if min_fetch != float('inf'):
                h_move += min_fetch
                
        return int(h_actions + h_move)

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
        """Support multi-step MOVETO actions by using their step count as cost."""
        return c + 1


def create_watering_problem(game):
    print("<<create_watering_problem")
    return WateringProblem(game)


if __name__ == '__main__':
    import ex1_check
    ex1_check.main()
