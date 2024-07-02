import numpy as np
import matplotlib.pyplot as plt
import heapq
import itertools

def create_environment(width, height, obstacles):
    env = np.zeros((height, width))
    for (x, y, w, h) in obstacles:
        env[y:y+h, x:x+w] = 1  # Mark the obstacle cells as occupied
    return env

def get_obstacle_corners(obstacles):
    corners = set()
    for (x, y, w, h) in obstacles:
        corners.add((x, y))
        corners.add((x + w, y))
        corners.add((x, y + h))
        corners.add((x + w, y + h))
    return list(corners)

def bresenham_line(x1, y1, x2, y2):
    """Bresenham's Line Algorithm to get the points on a line."""
    points = []
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy

    while True:
        points.append((x1, y1))
        if x1 == x2 and y1 == y2:
            break
        e2 = err * 2
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy

    return points

def is_visible(env, point1, point2):
    line = bresenham_line(point1[0], point1[1], point2[0], point2[1])
    for (x, y) in line:
        if not (0 <= x < env.shape[1] and 0 <= y < env.shape[0]) or env[y, x] == 1:
            return False
    return True

def build_visibility_graph(env, corners):
    graph = {corner: [] for corner in corners}
    for point1, point2 in itertools.combinations(corners, 2):
        if is_visible(env, point1, point2):
            distance = np.linalg.norm(np.array(point1) - np.array(point2))
            graph[point1].append((distance, point2))
            graph[point2].append((distance, point1))
    return graph

def dijkstra_visibility_graph(graph, start, goal):
    pq = [(0, start)]
    dist = {start: 0}
    prev = {start: None}

    while pq:
        current_dist, current = heapq.heappop(pq)
        if current == goal:
            break
        for neighbor_dist, neighbor in graph[current]:
            distance = current_dist + neighbor_dist
            if neighbor not in dist or distance < dist[neighbor]:
                dist[neighbor] = distance
                prev[neighbor] = current
                heapq.heappush(pq, (distance, neighbor))

    path = []
    current = goal
    while current is not None:
        path.append(current)
        current = prev[current]
    path.reverse()
    return path

def visualize_visibility_graph(env, corners, graph, path):
    plt.imshow(env, cmap='gray')

    for point1 in graph:
        for _, point2 in graph[point1]:
            plt.plot([point1[0], point2[0]], [point1[1], point2[1]], color='gray', linestyle='--', linewidth=1)

    if path:
        path_x, path_y = zip(*path)
        plt.plot(path_x, path_y, color='blue', linewidth=3, marker='o', markersize=5)
        path_length = sum(np.linalg.norm(np.array(path[i+1]) - np.array(point)) for i, point in enumerate(path[:-1]))
        print(f"Length of the path: {path_length:.2f}")

    plt.gca().invert_yaxis()  # Invert y-axis
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title('Visibility Map Decomposition with Path')
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

# Get corners of obstacles
corners = get_obstacle_corners(obstacles)

# Add start and goal to corners
start = (10, 10)
goal = (10, 90)
corners.append(start)
corners.append(goal)

# Build visibility graph
graph = build_visibility_graph(env, corners)

# Find the shortest path using Dijkstra's algorithm on the visibility graph
path = dijkstra_visibility_graph(graph, start, goal)

# Visualize visibility graph and path
visualize_visibility_graph(env, corners, graph, path)

