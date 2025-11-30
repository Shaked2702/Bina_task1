import ex1_check
import search
import utils
import math
from collections import deque
import time

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
        # Performance stats
        self.succ_calls = 0
        self.succ_time = 0
        self.h_calls = 0
        self.h_time = 0
        
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

    def path_cost(self, c, state1, action, state2):
        return c + 1

    def successor(self, state):
        self.succ_calls += 1
        start_time = time.time()
        
        taps_f, plants_f, robots_t = state
        taps = dict(taps_f)
        plants = dict(plants_f)
        
        succs = []
        
        # 1. MOVE
        # Try all 4 directions for each robot
        for rid, r, c, load, cap in robots_t:
            # Check if other robots are blocking
            # We need to know positions of all other robots
            other_robots_pos = { (or_r, or_c) for (orid, or_r, or_c, _, _) in robots_t if orid != rid }
            
            for dr, dc, name in [(-1, 0, 'UP'), (1, 0, 'DOWN'), (0, -1, 'LEFT'), (0, 1, 'RIGHT')]:
                nr, nc = r + dr, c + dc
                if (nr, nc) in self.free_cells and (nr, nc) not in other_robots_pos:
                    # Valid move
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

        end_time = time.time()
        self.succ_time += (end_time - start_time)
        
        return succs

    def goal_test(self, state):
        _, plants_f, _ = state
        return len(plants_f) == 0

    def _h_astar_impl(self, node):
        """Admissible heuristic: Work / Capacity relaxation.
        
        1. Action costs: We need 1 POUR per unit of demand.
           We need 1 LOAD per unit of demand that isn't currently loaded.
        2. Movement costs:
           Calculate 'transport work' = sum(dist(Source, Plant) * Amount).
           We greedily assign the cheapest available water units (from taps or loaded robots)
           to each plant's demand, respecting the limited amount of water in each source.
           This provides a tighter lower bound than assuming infinite water at the nearest tap.
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
        
        # Single Robot Optimization
        if len(robots_t) == 1:
            robot = robots_t[0]
            rid, r_r, r_c, r_load, r_cap = robot
            r_pos = (r_r, r_c)
            
            # Identify tap positions
            tap_positions = [pos for pos, amt in taps.items() if amt > 0]
            if not tap_positions and needed_load > 0:
                return 10**9 # Unsolvable
            
            # Expand demands into individual units with their locations
            # List of (dist_to_tap, dist_to_robot)
            unit_demands = []
            
            for ppos, pdemand in plants.items():
                if pdemand <= 0: continue
                
                # Dist to nearest tap
                d_tap = float('inf')
                for tpos in tap_positions:
                    d = self.tap_plant_dist.get((tpos, ppos))
                    if d is None: d = self.dist(tpos, ppos)
                    if d < d_tap:
                        d_tap = d
                
                if d_tap == float('inf'):
                    return 10**9
                
                d_robot = self.dist(r_pos, ppos)
                
                # Add for each unit of demand
                unit_demands.extend([(d_tap, d_robot)] * pdemand)
            
            if not unit_demands:
                return 0

            # Helper to calculate Batch Cost for a set of demands (given by their dist_to_tap)
            def calculate_batch_cost(dists_tap):
                if not dists_tap: return 0
                
                # Sort once
                d_asc = sorted(dists_tap)
                
                # Option 1: Ascending
                sum_asc = 0
                for i in range(0, len(d_asc), r_cap):
                    sum_asc += d_asc[i:i+r_cap][-1]
                
                # Option 2: Descending
                # Equivalent to taking chunks from the end of the sorted array
                sum_desc = 0
                n = len(d_asc)
                for i in range(n, 0, -r_cap):
                    sum_desc += d_asc[i-1]
                
                best_sum = min(sum_asc, sum_desc)
                
                # Cost is 2 * sum(maxes) - max(all_dists)
                return 2 * best_sum - d_asc[-1]

            # Strategy 1: Go to Tap first (Fill up / Dump)
            # Cost = Dist(Robot, Tap) + BatchCost(All Demands)
            # We need dist to nearest tap
            d_robot_tap = float('inf')
            for tpos in tap_positions:
                d = self.dist(r_pos, tpos)
                if d < d_robot_tap:
                    d_robot_tap = d
            
            all_dists_tap = [u[0] for u in unit_demands]
            cost_go_tap = d_robot_tap + calculate_batch_cost(all_dists_tap)
            
            # Strategy 2: Deliver current load first
            # We can only do this if we have load
            cost_deliver = float('inf')
            
            if r_load > 0:
                # We need to choose WHICH r_load units to deliver.
                # We try two heuristics:
                # A. Deliver the ones furthest from Tap (Maximize batch savings)
                # B. Deliver the ones closest to Robot (Minimize detour)
                
                # Sort by dist_tap descending
                unit_demands.sort(key=lambda x: x[0], reverse=True)
                s_furthest = unit_demands[:r_load]
                r_furthest = [u[0] for u in unit_demands[r_load:]]
                
                # Sort by dist_robot ascending
                unit_demands.sort(key=lambda x: x[1])
                s_closest = unit_demands[:r_load]
                r_closest = [u[0] for u in unit_demands[r_load:]]
                
                for s_set, r_dists_tap in [(s_furthest, r_furthest), (s_closest, r_closest)]:
                    # Cost = Visit S + (Go Tap if R not empty) + BatchCost(R)
                    
                    # Lower bound for visiting S: Max(dist(Robot, p))
                    # If R is not empty, we must also go to Tap: Max(dist(Robot, p) + dist(p, Tap))
                    
                    if not r_dists_tap:
                        # No remaining demands. Just deliver S.
                        # Cost is max dist from robot to any p in S
                        move_cost = 0
                        for _, d_r in s_set:
                            if d_r > move_cost: move_cost = d_r
                    else:
                        # Must go to tap after.
                        # Lower bound: max(dist(Robot, p) + dist(p, Tap)) for p in S
                        move_cost = 0
                        for d_t, d_r in s_set:
                            trip = d_r + d_t
                            if trip > move_cost: move_cost = trip
                        
                        move_cost += calculate_batch_cost(r_dists_tap)
                    
                    if move_cost < cost_deliver:
                        cost_deliver = move_cost

            h_move = min(cost_go_tap, cost_deliver)
                
            return int(h_actions + h_move)

        # 2. Movement Costs (Relaxation) - Multi Robot
        transport_work = 0
        
        # Identify all water sources
        # Robots with load
        robot_sources = []
        for (_, r, c, load, _) in robots_t:
            if load > 0:
                robot_sources.append(((r, c), load))
                
        # Taps with water
        tap_sources = []
        for tpos, amt in taps.items():
            if amt > 0:
                tap_sources.append((tpos, amt))
        
        # If no water available and we have demand, it's unsolvable
        if not tap_sources and not robot_sources and total_demand > 0:
             return 10**9

        for ppos, pdemand in plants.items():
            if pdemand <= 0: continue
            
            # Collect (distance, amount) for all sources relative to this plant
            sources = []
            
            # Taps
            for tpos, amt in tap_sources:
                # Use cached distance if available
                dist = self.tap_plant_dist.get((tpos, ppos))
                if dist is None:
                    dist = self.dist(tpos, ppos)
                sources.append((dist, amt))
                
            # Robots
            for rpos, load in robot_sources:
                dist = self.dist(rpos, ppos)
                sources.append((dist, load))
            
            # Sort by distance (cheapest water first)
            sources.sort(key=lambda x: x[0])
            
            remaining_demand = pdemand
            plant_transport_cost = 0
            
            for dist, amt in sources:
                if dist == float('inf'): continue
                
                take = min(remaining_demand, amt)
                plant_transport_cost += take * dist
                remaining_demand -= take
                
                if remaining_demand == 0:
                    break
            
            # If we exhausted all sources and still have demand, this plant cannot be fully watered
            # (in this relaxed view where we don't compete with other plants).
            # However, since we check this per plant, and we know total water > total demand is not guaranteed globally here,
            # we should be careful. But if a single plant can't be satisfied by *all* world water, it's definitely unsolvable.
            if remaining_demand > 0:
                return 10**9
                
            transport_work += plant_transport_cost

        # Divide by max capacity because one robot can carry multiple units
        h_move = math.ceil(transport_work / max_cap)
        
        # 3. Fetch Cost
        # If we don't have enough load, at least one robot must go to a tap.
        if total_load < total_demand:
            # Min dist from any robot to any tap
            min_fetch = float('inf')
            robot_positions = [(r, c) for (_, r, c, _, _) in robots_t]
            tap_positions = [pos for pos, amt in taps.items() if amt > 0]
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

    def h_astar(self, node):
        return self._h_astar_impl(node)


def create_watering_problem(game):
    print("<<create_watering_problem")
    return WateringProblem(game)


if __name__ == '__main__':
    import ex1_check
    ex1_check.main()
