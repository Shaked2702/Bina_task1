import search
import utils
import math
from collections import deque

id = "208018853"

class WateringProblem(search.Problem):
    """Plant watering problem.

    State representation (immutable): tuple(taps_tuple, plants_tuple, robots_tuple)
      - taps_tuple: tuple of ((r,c), amount) sorted
      - plants_tuple: tuple of ((r,c), demand) sorted
      - robots_tuple: tuple of (rid, r, c, load, cap) sorted by rid
    """

    # Initialize the problem, parsing the input dictionary and precomputing distances.
    def __init__(self, initial):
        # Performance stats
        self.succ_calls = 0
        self.h_calls = 0
        
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

    # Perform BFS from a source cell to all reachable cells to compute distances.
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

    # Return the precomputed distance between two cells (O(1) lookup).
    def dist(self, src, dst):
        """O(1) distance lookup between free cells; walls return inf."""
        if src not in self.free_cells or dst not in self.free_cells:
            return float('inf')
        return self.dist_map.get(src, {}).get(dst, float('inf'))

    # Return the cost of an action (always 1 in this problem).
    def path_cost(self, c, state1, action, state2):
        return c + 1

    # Generate all valid successor states from the current state.
    def successor(self, state):
        self.succ_calls += 1
        
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

        return succs

    # Check if the goal is reached (all plants have 0 demand).
    def goal_test(self, state):
        _, plants_f, _ = state
        return len(plants_f) == 0

    # Implementation of the admissible heuristic using relaxation and single-robot optimization.
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
            
            # Calculate distance from each demand unit to the nearest tap
            # We expand the demand into individual units
            demand_dists = []
            active_plants = []
            min_trip = float('inf')
            max_dist_R_P = 0

            for ppos, pdemand in plants.items():
                if pdemand <= 0: continue
                active_plants.append(ppos)
                
                # Dist to nearest tap
                d_tap = float('inf')
                for tpos in tap_positions:
                    d = self.tap_plant_dist.get((tpos, ppos))
                    if d is None: d = self.dist(tpos, ppos)
                    if d < d_tap:
                        d_tap = d
                
                if d_tap == float('inf'):
                    return 10**9
                
                # Add d_tap for each unit of demand
                demand_dists.extend([d_tap] * pdemand)

                # Track min trip (R -> P -> Tap) and max dist (R -> P)
                d_R_P = self.dist(r_pos, ppos)
                if d_R_P > max_dist_R_P:
                    max_dist_R_P = d_R_P
                
                trip = d_R_P + d_tap
                if trip < min_trip:
                    min_trip = trip
            
            # Sort demand distances descending (furthest first)
            demand_dists.sort(reverse=True)
            
            # Helper for batch cost
            def get_batch_cost(demands):
                if not demands: return 0
                d_desc = sorted(demands, reverse=True)
                cost = d_desc[0]
                remaining = d_desc[r_cap:]
                if remaining:
                    rem_asc = sorted(remaining)
                    for i in range(0, len(rem_asc), r_cap):
                        cost += 2 * rem_asc[i:i+r_cap][-1]
                return cost

            # Option 1: Dump load at Tap (or just go to Tap)
            d_robot_tap = float('inf')
            for tpos in tap_positions:
                d = self.dist(r_pos, tpos)
                if d < d_robot_tap:
                    d_robot_tap = d
            
            h_dump = d_robot_tap + get_batch_cost(demand_dists)
            
            # Option 2: Deliver to k plants (1 <= k <= r_load)
            h_del = float('inf')
            
            if len(demand_dists) <= r_load:
                h_del = max_dist_R_P
            else:
                # Try delivering k items first
                for k in range(1, r_load + 1):
                    rem = demand_dists[k:]
                    if not rem:
                        cost = max_dist_R_P
                    else:
                        cost = min_trip + get_batch_cost(rem)
                    
                    if cost < h_del:
                        h_del = cost
            
            h_move = min(h_dump, h_del)

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

        if h_actions == float('inf') or h_move == float('inf'):
            return float('inf')
        return int(h_actions + h_move)

    # Greedy heuristic: prioritizes visiting taps if empty, then delivering to plants.
    def h_gbfs(self, node):
        """Greedy heuristic: sum distance from nearest robot to plants plus demand.
        Improved to account for empty robots needing to visit a tap first.
        For single robot, delegates to the more accurate A* heuristic.
        """
        state = node.state
        _, _, robots_t = state
        
        # Use the advanced heuristic for single robot (Problem 17 optimization)
        if len(robots_t) == 1:
            return self._h_astar_impl(node)

        taps_f, plants_f, _ = state
        plants = dict(plants_f)
        taps = dict(taps_f)
        
        # Precompute active taps
        active_taps = [pos for pos, amt in taps.items() if amt > 0]
        
        h = 0
        for ppos, pd in plants.items():
            if pd <= 0:
                continue
            
            min_dist = float('inf')
            
            if not robots_t:
                min_dist = 0
            else:
                for _, r_r, r_c, r_load, _ in robots_t:
                    rpos = (r_r, r_c)
                    if r_load > 0:
                        # Robot has water, can go directly
                        d = self.dist(rpos, ppos)
                        if d < min_dist:
                            min_dist = d
                    else:
                        # Robot empty, must go R -> Tap -> Plant
                        # Estimate as: dist(R, nearest_Tap) + dist(nearest_Tap, P)
                        
                        # 1. Dist to nearest tap
                        d_r_t = float('inf')
                        for tpos in active_taps:
                            d = self.dist(rpos, tpos)
                            if d < d_r_t:
                                d_r_t = d
                        
                        # 2. Dist from nearest tap to plant
                        d_t_p = float('inf')
                        for tpos in active_taps:
                            d = self.tap_plant_dist.get((tpos, ppos))
                            if d is None: d = self.dist(tpos, ppos)
                            if d < d_t_p:
                                d_t_p = d
                                
                        if d_r_t != float('inf') and d_t_p != float('inf'):
                            trip = d_r_t + d_t_p
                            if trip < min_dist:
                                min_dist = trip
            
            if min_dist == float('inf'):
                min_dist = 0
                
            h += min_dist + pd
            
        return int(h)

    # Wrapper for the A* heuristic function.
    def h_astar(self, node):
        self.h_calls += 1
        return self._h_astar_impl(node)


# Factory function to create a WateringProblem instance from the input dictionary.
def create_watering_problem(game):
    print("<<create_watering_problem")
    return WateringProblem(game)


if __name__ == '__main__':
    import ex1_check
    ex1_check.main()