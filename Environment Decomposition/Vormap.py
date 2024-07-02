import numpy as np
import matplotlib.pyplot as plt
import heapq
from scipy.spatial import Voronoi, voronoi_plot_2d

def create_environment(width, height, obstacles):
    env = np.zeros((height, width))
    for (x, y, w, h) in obstacles:
        env[y:y+h, x:x+w] = 1  # Mark obstacle cells as occupied
    return env

def generate_random_points(num_points, width, height, fixed_points=[]):
    points = np.random.rand(num_points - len(fixed_points), 2) * np.array([width, height])
    fixed_points.extend([(10, 10), (10, 90)])  # Ensure fixed points are included
    points = np.vstack([points, fixed_points])
    return points

def add_boundary_points(width, height):
    boundary_points = [
        (0, 0), (width - 1, 0), (width - 1, height - 1), (0, height - 1),
        (width // 2, 0), (width // 2, height - 1), (0, height // 2), (width - 1, height // 2)
    ]
    return np.array(boundary_points)

def decompose_environment_with_voronoi(env, num_points):
    height, width = env.shape
    points = generate_random_points(num_points, width, height)
    boundary_points = add_boundary_points(width, height)
    all_points = np.vstack([points, boundary_points])
    vor = Voronoi(all_points)
    return vor

def build_graph_from_voronoi(vor):
    graph = {}
    for point_idx, point in enumerate(vor.points):
        graph[tuple(np.round(point, decimals=6))] = []
    for simplex in vor.ridge_vertices:
        simplex = np.array(simplex)
        if np.all(simplex >= 0):
            p1 = tuple(np.round(vor.vertices[simplex[0]], decimals=6))
            p2 = tuple(np.round(vor.vertices[simplex[1]], decimals=6))
            if p1 not in graph:
                graph[p1] = []
            if p2 not in graph:
                graph[p2] = []
            distance = np.linalg.norm(np.array(p1) - np.array(p2))
            graph[p1].append((distance, p2))
            graph[p2].append((distance, p1))
    return graph

def find_closest_point(graph, point):
    closest_point = min(graph.keys(), key=lambda p: np.linalg.norm(np.array(p) - np.array(point)))
    return closest_point

def dijkstra_shortest_path(graph, start, goal):
    start = find_closest_point(graph, start)
    goal = find_closest_point(graph, goal)

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
        current = prev.get(current)
    path.reverse()
    return path

def visualize_voronoi_diagram(env, vor, start, goal, path):
    fig, ax = plt.subplots()
    plt.imshow(env, cmap='gray', extent=(0, env.shape[1], 0, env.shape[0]))
    voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='blue', line_width=2)

    # Mark start and goal points
    plt.plot(start[0], start[1], 'go', markersize=10, label='Start')  # Green for start
    plt.plot(goal[0], goal[1], 'ro', markersize=10, label='Goal')    # Red for goal

    # Draw the path
    if path:
        path_x, path_y = zip(*path)
        plt.plot(path_x, path_y, color='green', linewidth=3, marker='o', markersize=5, label='Path')  # Green for path
    plt.gca().invert_yaxis()  # Invert y-axis

    plt.gca().set_aspect('equal', adjustable='box')
    plt.title('Voronoi Decomposition of Environment with Path')
    plt.legend()
    plt.savefig('DesktopVor')
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

# Number of random points to generate for Voronoi decomposition
num_points = 100

# Decompose environment using Voronoi diagram
vor = decompose_environment_with_voronoi(env, num_points)

# Build graph from Voronoi diagram
graph = build_graph_from_voronoi(vor)

# Define start and goal points
start = (10, 10)
goal = (10, 90)

# Find the shortest path using Dijkstra's algorithm on the graph
path = dijkstra_shortest_path(graph, start, goal)

# Visualize Voronoi diagram with start, goal, and path
visualize_voronoi_diagram(env, vor, start, goal, path)

