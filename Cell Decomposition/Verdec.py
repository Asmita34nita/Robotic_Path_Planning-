import numpy as np
import matplotlib.pyplot as plt
import heapq

def create_environment(width, height, obstacles):
    env = np.zeros((height, width))
    for (x, y, w, h) in obstacles:
        env[y:y+h, x:x+w] = 1  # Mark the obstacle cells as occupied
    return env

def vertical_cell_decomposition(env):
    height, width = env.shape
    cells = []

    def get_vertical_cells(x_start, x_end):
        y_start = 0
        while y_start < height:
            if np.all(env[y_start:y_start+1, x_start:x_end] == 0):
                y_end = y_start
                while y_end < height and np.all(env[y_end:y_end+1, x_start:x_end] == 0):
                    y_end += 1
                cells.append((x_start, y_start, x_end - x_start, y_end - y_start))
                y_start = y_end
            else:
                y_start += 1

    x_boundaries = sorted(set([0] + [x for (x, y, w, h) in obstacles] + [x+w for (x, y, w, h) in obstacles] + [width]))

    for i in range(len(x_boundaries) - 1):
        get_vertical_cells(x_boundaries[i], x_boundaries[i+1])

    return cells

def heuristic(a, b):
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def a_star_cells(env, cells, start, goal):
    cell_map = {idx: (x, y, w, h) for idx, (x, y, w, h) in enumerate(cells)}
    start_cell = None
    goal_cell = None

    for idx, (x, y, w, h) in enumerate(cells):
        if x <= start[0] < x + w and y <= start[1] < y + h:
            start_cell = idx
        if x <= goal[0] < x + w and y <= goal[1] < y + h:
            goal_cell = idx

    if start_cell is None or goal_cell is None:
        return []

    graph = {i: [] for i in cell_map.keys()}

    for i, (x1, y1, w1, h1) in cell_map.items():
        for j, (x2, y2, w2, h2) in cell_map.items():
            if i != j:
                if x1 <= x2 < x1 + w1 or x2 <= x1 < x2 + w2:
                    if y1 <= y2 < y1 + h1 or y2 <= y1 < y2 + h2:
                        graph[i].append(j)

    open_set = []
    heapq.heappush(open_set, (0, start_cell))
    g_score = {cell: float('inf') for cell in cell_map.keys()}
    g_score[start_cell] = 0
    f_score = {cell: float('inf') for cell in cell_map.keys()}
    f_score[start_cell] = heuristic(start, goal)
    came_from = {}

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal_cell:
            path = []
            while current in came_from:
                x, y, w, h = cell_map[current]
                path.append((x + w // 2, y + h // 2))
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path

        for neighbor in graph[current]:
            tentative_g_score = g_score[current] + 1

            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(cell_map[neighbor][:2], goal)
                if neighbor not in [i[1] for i in open_set]:
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return []

def visualize_decomposition_and_path(env, cells, path, start, goal):
    plt.imshow(env, cmap='gray')

    for (x, y, w, h) in cells:
        color = 'green'
        rect = plt.Rectangle((x, y), w, h, linewidth=1, edgecolor=color, facecolor='none')
        plt.gca().add_patch(rect)

    if path:
        path_x, path_y = zip(*path)
        plt.plot(path_x, path_y, color='blue', linewidth=2, marker='o', markersize=3)

    plt.plot(start[0], start[1], 'go', markersize=10, label='Start')  # Green for start
    plt.plot(goal[0], goal[1], 'ro', markersize=10, label='Goal')    # Red for goal
    plt.gca().invert_yaxis()
    plt.title('Vertical Cell Decomposition with Path')
    plt.legend()
    plt.show()

# Define environment dimensions and obstacles
width, height = 100, 100
obstacles = [
    (0, 0, 100, 1), (99, 0, 1, 20), (50, 20, 50, 1), (50, 15, 1, 20), (50, 60, 1, 20),
    (35, 80, 45, 1), (79, 80, 1, 20), (0, 99, 80, 1), (10, 40, 30, 1), (0, 60, 1, 40),
    (0, 80, 5, 1), (40 ,40, 1, 20), (0, 59, 40, 1), (50, 0, 1, 5), (0, 0, 1, 100),
    (20, 10, 10, 20), (20, 90, 20, 10),
]

# Create environment and perform vertical cell decomposition
env = create_environment(width, height, obstacles)
cells = vertical_cell_decomposition(env)

# Define start and goal points
start = (10, 10)
goal = (10, 90)

# Ensure start and goal points are not inside obstacles
if env[start[1], start[0]] == 1 or env[goal[1], goal[0]] == 1:
    raise ValueError("Start or goal is inside an obstacle.")

# Find the shortest path using A* algorithm for cells
path = a_star_cells(env, cells, start, goal)
print (len(path))
# Visualize decomposition and path
visualize_decomposition_and_path(env, cells, path, start, goal)

