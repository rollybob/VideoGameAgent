import collections

def shortest_path_len(grid, start, goal):
    rows = len(grid)
    cols = len(grid[0])
    start_r, start_c = start
    goal_r, goal_c = goal

    # Check if start is out of bounds or on a wall
    if start_r < 0 or start_r >= rows or start_c < 0 or start_c >= cols or grid[start_r][start_c] != 0:
        return -1
    
    # Check if goal is out of bounds or on a wall
    if goal_r < 0 or goal_r >= rows or goal_c < 0 or goal_c >= cols or grid[goal_r][goal_c] != 0:
        return -1
    
    # If start and goal are the same free cell
    if start == goal:
        return 0

    # BFS initialization
    queue = collections.deque()
    visited = set()
    queue.append((start_r, start_c, 0))
    visited.add((start_r, start_c))
    
    # Directions: up, down, left, right
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    
    while queue:
        r, c, steps = queue.popleft()
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            # If the neighbor is the goal, return steps + 1
            if nr == goal_r and nc == goal_c:
                return steps + 1
            # Check if the new cell is within bounds, free, and not visited
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 0 and (nr, nc) not in visited:
                visited.add((nr, nc))
                queue.append((nr, nc, steps + 1))
    
    # Goal not reachable
    return -1
