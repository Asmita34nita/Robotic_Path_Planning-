import numpy as np
import matplotlib.pyplot as plt
import heapq

def create_environment(width, height, obstacles):
    env = np.zeros((height, width))
    for (x, y, w, h) in obstacles:
        env[y:y+h, x:x+w] = 1  # Mark the obstacle cells as occupied
    return env

def adaptive_decomposition(env, min_cell_size):
    def decompose_cell(x, y, w, h):
        if w <= min_cell_size or h <= min_cell_size:
            status = 'occupied' if np.any(env[y:y+h, x:x+w] == 1) else 'free'
            return [(x, y, w, h, status)]
        if np.all(env[y:y+h, x:x+w] == 1):
            return [(x, y, w, h, 'occupied')]
        if np.all(env[y:y+h, x:x+w] == 0):
            return [(x, y, w, h, 'free')]
        
        cells = []
        if w >= h:
            mid = w // 2
            cells += decompose_cell(x, y, mid, h)
            cells += decompose_cell(x + mid, y, w - mid, h)
        else:
            mid = h // 2
            cells += decompose_cell(x, y, w, mid)
            cells += decompose_cell(x, y + mid, w, h - mid)
        
        return cells
    
    return decompose_cell(0, 0, env.shape[1], env.shape[0])

def check_cells(cells):
    cell_status = []
    for (x, y, w, h, status) in cells:
        cell_status.append((x, y, w, h, status))
    return cell_status

def find_neighbors(cells, idx):
    x, y, w, h, status = cells[idx]
    neighbors = []
    for nidx, (nx, ny, nw, nh, nstatus) in enumerate(cells):
        if nstatus == 'free':
            if (x < nx < x + w or x < nx + nw < x + w or nx <= x < nx + nw) and (y < ny < y + h or y < ny + nh < y + h or ny <= y < ny + nh):
                neighbors.append(nidx)
    return neighbors

def dijkstra_cells(env, cells, start, goal):
    start_cell = None
    goal_cell = None
    for idx, (x, y, w, h, status) in enumerate(cells):
        if status == 'free':
            if x <= start[0] < x + w and y <= start[1] < y + h:
                start_cell = idx
            if x <= goal[0] < x + w and y <= goal[1] < y + h:
                goal_cell = idx
    
    if start_cell is None or goal_cell is None:
        return []

    graph = {}
    for idx, (x, y, w, h, status) in enumerate(cells):
        if status == 'free':
            graph[idx] = find_neighbors(cells, idx)

    dist = {start_cell: 0}
    prev = {start_cell: None}
    pq = [(0, start_cell)]

    while pq:
        current_dist, current = heapq.heappop(pq)
        if current == goal_cell:
            break

        for neighbor in graph[current]:
            distance = current_dist + 1
            if neighbor not in dist or distance < dist[neighbor]:
                dist[neighbor] = distance
                prev[neighbor] = current
                heapq.heappush(pq, (distance, neighbor))

    path = []
    current = goal_cell
    while current is not None:
        x, y, w, h, _ = cells[current]
        path.append((x + w // 2, y + h // 2))
        current = prev.get(current)
    path.reverse()
    return path

def visualize_decomposition_and_path(env, cell_status, path, start, goal):
    plt.imshow(env, cmap='gray')

    for (x, y, w, h, status) in cell_status:
        color = 'red' if status == 'occupied' else 'green'
        rect = plt.Rectangle((x, y), w, h, linewidth=1, edgecolor=color, facecolor='none')
        plt.gca().add_patch(rect)

    if path:
        path_x, path_y = zip(*path)
        plt.plot(path_x, path_y, color='blue', linewidth=2, marker='o', markersize=3)
        
        for (px, py) in path:
            plt.plot(px, py, 'bo')

    plt.plot(start[0], start[1], 'go', markersize=10, label='Start')  # Mark start
    plt.plot(goal[0], goal[1], 'ro', markersize=10, label='Goal')     # Mark goal

    plt.gca().invert_yaxis()  # Invert y-axis
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    plt.title('Adaptive Cell Decomposition with Path')
    plt.savefig('DesktopAdap')
    plt.show()

# Define environment dimensions
width, height = 100, 100

# Define obstacles as (x, y, width, height)
obstacles = [
    (0, 0, 100, 1),
    (99, 0, 1, 20),
    (50, 20, 50, 1),
    (50, 15, 1, 20),
    (50, 60, 1, 20),
    (35, 80, 45, 1),
    (79, 80, 1, 20),
    (0, 99, 80, 1),
    (10, 40, 30, 1),
    (0, 60, 1, 40),
    (0, 80, 5, 1),
    (40, 40, 1, 20),
    (0, 59, 40, 1),
    (50, 0, 1, 5),
    (0, 0, 1, 100),
    (20, 10, 10, 20),
    (20, 90, 20, 10),
]

# Create environment
env = create_environment(width, height, obstacles)

# Adaptive decomposition with minimum cell size 5x5
min_cell_size = 5
cells = adaptive_decomposition(env, min_cell_size)

# Check cells
cell_status = check_cells(cells)

# Define start and goal points
start = (10, 10)
goal = (10, 20)

# Find the shortest path using Dijkstra's algorithm for cells
path = dijkstra_cells(env, cells, start, goal)
for i in path:
   print (i)

# Visualize decomposition and path
visualize_decomposition_and_path(env, cell_status, path, start, goal)

