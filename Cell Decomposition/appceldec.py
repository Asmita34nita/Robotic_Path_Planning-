import numpy as np
import matplotlib.pyplot as plt
import heapq

def create_environment(width, height, obstacles):
    env = np.zeros((height, width))
    for (x, y, w, h) in obstacles:
        env[y:y+h, x:x+w] = 1  # Mark the obstacle cells as occupied
    return env

def decompose_environment(env, cell_size):
    height, width = env.shape
    cells = []
    for i in range(0, height, cell_size):
        for j in range(0, width, cell_size):
            cell = env[i:i+cell_size, j:j+cell_size]
            cells.append((i, j, cell))
    return cells

def check_cells(cells):
    cell_status = []
    for (i, j, cell) in cells:
        if np.any(cell == 1):
            cell_status.append((i, j, 'occupied'))
        else:
            cell_status.append((i, j, 'free'))
    return cell_status

def dijkstra_cells(env, cell_size, start, goal):
    height, width = env.shape
    cell_height, cell_width = height // cell_size, width // cell_size
    start_cell = (start[0] // cell_size, start[1] // cell_size)
    goal_cell = (goal[0] // cell_size, goal[1] // cell_size)

    dist = {start_cell: 0}
    prev = {start_cell: None}
    pq = [(0, start_cell)]  # Priority queue of (distance, node)

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (1, 1), (-1, -1)]

    while pq:
        current_dist, current = heapq.heappop(pq)
        if current == goal_cell:
            break

        for dx, dy in directions:
            neighbor = (current[0] + dx, current[1] + dy)
            if 0 <= neighbor[0] < cell_width and 0 <= neighbor[1] < cell_height:
                if np.all(env[neighbor[1]*cell_size:(neighbor[1]+1)*cell_size, neighbor[0]*cell_size:(neighbor[0]+1)*cell_size] == 0):
                    distance = current_dist + 1
                    if neighbor not in dist or distance < dist[neighbor]:
                        dist[neighbor] = distance
                        prev[neighbor] = current
                        heapq.heappush(pq, (distance, neighbor))

    # Reconstruct the path
    path = []
    current = goal_cell
    while current is not None:
        path.append((current[0] * cell_size + cell_size // 2, current[1] * cell_size + cell_size // 2))
        current = prev.get(current)
    path.reverse()
    return path

def visualize_decomposition_and_path(env, cell_status, cell_size, path):
    plt.imshow(env, cmap='gray')

    for (i, j, status) in cell_status:
        color = 'red' if status == 'occupied' else 'green'
        rect = plt.Rectangle((j, i), cell_size, cell_size, linewidth=1, edgecolor=color, facecolor='none')
        plt.gca().add_patch(rect)

    # Draw the path
    if path:
        path_x, path_y = zip(*path)
        plt.plot(path_x, path_y, color='blue', linewidth=2, marker='o', markersize=3)
        path_length = sum(np.linalg.norm(np.array(path[i+1]) - np.array(point)) for i, point in enumerate(path[:-1]))
        print(f"Length of the path: {path_length:.2f}")

    plt.gca().invert_yaxis()  # Invert y-axis
    plt.gca().set_aspect('equal', adjustable='box')  # Maintain equal aspect ratio
    plt.title('Approximate Cell Decomposition with Path')
    plt.savefig('DesktopApp')
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
    (40 ,40, 1, 20),
    (0, 59, 40, 1),
    (50, 0, 1, 5),
    (0, 0, 1, 100),
    (20, 10, 10, 20),
    (20, 90, 20, 10),
]

# Create environment
env = create_environment(width, height, obstacles)

# Decompose environment into cells of size 5x5
cell_size = 5
cells = decompose_environment(env, cell_size)

# Check cells
cell_status = check_cells(cells)

# Define start and goal points
start = (10, 10)
goal = (60, 10)

# Find the shortest path using Dijkstra's algorithm for cells
path = dijkstra_cells(env, cell_size, start, goal)


# Visualize decomposition and path with inverted y-axis
visualize_decomposition_and_path(env, cell_status, cell_size, path)

