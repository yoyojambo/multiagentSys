# https://en.wikipedia.org/wiki/A*_search_algorithm#Pseudocode

import agentpy as ap

# Heuristic for estimating the distance between a point and the
# goal. For A*, a heuristic is valid if it won't OVERestimate the
# distance, so this is it, this is the best case for 4-way moves.
def h_func(a, b):
    g = a.model.tracks
    
    pos = g.positions[a]
    goal = g.positions[b]
    
    return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])


def reconstruct_path(cameFrom, cur, grid):
    total_path = [grid.positions[cur]]
    while cur in cameFrom:
        cur = cameFrom[cur]
        total_path.append(grid.positions[cur])
    return total_path[::-1]


def four_way_filter(cur_pos):
    # Returns the function to be passed to `filter`.  Only keeps the
    # positions that are (exclusive) either to the 'north', 'south',
    # 'east' or 'west' of the current position.
    def f(track_pos):
        x_diff = track_pos[0] - cur_pos[0]
        y_diff = track_pos[1] - cur_pos[1]

        return (abs(x_diff) + abs(y_diff)) == 1

    return f

def FWF_neighbors(cur):
    grid = cur.model.tracks
    f = four_way_filter(grid.positions[cur])
    return [n for n in grid.neighbors(cur) if f(grid.positions[n])] # All 8 potential neighbors included
    

def A_Star(start, goal):
    ## `start_obj` and `goal_obj` are `agentpy.Agent` objects returns
    ## the sequence of positions that the `start` agent will take to
    ## get to the position of the `goal` agent.

    # h function for this specific goal
    def h(pos):
        return h_func(pos, goal)
    
    grid = start.model.tracks
    openSet = {start}
    cameFrom = dict()

    # gScore[n] = Cheapest known cost to get to `n` position.
    gScore = {start: 0}
    goal_pos = grid.positions[goal]

    # fScore[n] = Tentative distance to goal from `n` position.
    # fScore = {start: h_func(start, goal)}

    while openSet:
        # This agent is the object in openSet that seems closest to the goal
        cur = min(openSet, key=lambda p: gScore[p] + h(p))
        if grid.positions[cur] == goal_pos:
            return reconstruct_path(cameFrom, cur, grid)

        openSet.remove(cur)
        
        for neighbor in FWF_neighbors(cur):
            tentative_gScore = gScore[cur] + 1
            if tentative_gScore < gScore.get(neighbor, float('inf')):
                cameFrom[neighbor] = cur
                gScore[neighbor] = tentative_gScore

                # If it wasn't in line to be tested (or be tested
                # again), it is now that it may be the fastest route.
                openSet.add(neighbor)

    raise Exception("Could not find a valid route to 'goal'")
        
