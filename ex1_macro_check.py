import time

import ex1_macro as ex1
import search
import problems


def run_problem(func, targs=(), kwargs=None):
    if kwargs is None:
        kwargs = {}
    result = (-3, "default")
    try:
        result = func(*targs, **kwargs)

    except Exception as e:
        result = (-3, e)
    return result


# check_problem: problem, search_method, timeout
# timeout_exec: search_method, targs=[problem], timeout_duration=timeout
def solve_problems(problem, algorithm):
    

    try:
        p = ex1.create_watering_problem(problem)
    except Exception as e:
        print("Error creating problem: ", e)
        return None

    if algorithm == "gbfs":
        result = run_problem((lambda p: search.greedy_best_first_graph_search(p, p.h_gbfs)),targs=[p])
    else:
        result = run_problem((lambda p: search.astar_search(p, p.h_astar)), targs=[p])

    if result and isinstance(result[0], search.Node):
        solve = result[0].path()[::-1]
        raw_solution = [pi.action for pi in solve][1:]
        solution = []
        for act in raw_solution:
            if act.startswith("MOVETO"):
                # Parse "MOVETO{rid}:steps:r,c;ACT1,ACT2..."
                parts = act.split(";")
                if len(parts) > 1:
                    sub_actions = parts[1].split(",")
                    solution.extend(sub_actions)
                else:
                    # Fallback if no path found (should not happen with updated macro)
                    solution.append(parts[0])
            else:
                solution.append(act)
        print(len(solution), solution)
    else:
        print("no solution")



def main():
    start = time.time()
    problem = [
        problems.problem1, 
        problems.problem2, 
        problems.problem3, 
        problems.problem4, 
        problems.problem6, 
        problems.problem7
    ]
    for i, p in enumerate(problem):
        print(f"--- Problem {i+1} ---")
        for a in ['astar']:
            print(f"Algorithm: {a}")
            solve_problems(p, a)
    end = time.time()
    print('Submission took:', end-start, 'seconds.')


if __name__ == '__main__':
    main()
