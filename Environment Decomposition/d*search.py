import matplotlib.pyplot as plt
import numpy as np
from heapq import heappop, heappush

# Define the grid size
grid_size = (100, 100)

# Define the start and goal positions
start = (10, 10)
goal = (10, 90)

# Define obstacles
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

# Create the grid
grid = np.zeros(grid_size)
for obstacle in obstacles:
    x, y, width, height = obstacle
    grid[x:x+width, y:y+height] = 1

# Define heuristic function
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# D* Lite algorithm
class DStarLite:
    def __init__(self, grid, start, goal):
        self.grid = grid
        self.start = start
        self.goal = goal
        self.rhs = {}
        self.g = {}
        self.u = []
        self.km = 0
        self.init()
    
    def init(self):
        self.rhs[self.goal] = 0
        self.g[self.goal] = float('inf')
        heappush(self.u, (self.calculate_key(self.goal), self.goal))
    
    def calculate_key(self, s):
        return (min(self.g.get(s, float('inf')), self.rhs.get(s, float('inf'))) + heuristic(self.start, s) + self.km,
                min(self.g.get(s, float('inf')), self.rhs.get(s, float('inf'))))

    def update_vertex(self, u):
        if u != self.goal:
            self.rhs[u] = min([self.g.get(s, float('inf')) + 1 for s in self.get_neighbors(u)])
        if u in [i[1] for i in self.u]:
            self.u = [i for i in self.u if i[1] != u]
        if self.g.get(u, float('inf')) != self.rhs.get(u, float('inf')):
            heappush(self.u, (self.calculate_key(u), u))
    
    def get_neighbors(self, u):
        neighbors = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (1, 1), (-1, -1)]:
            x, y = u[0] + dx, u[1] + dy
            if 0 <= x < grid_size[0] and 0 <= y < grid_size[1] and self.grid[x, y] == 0:
                neighbors.append((x, y))
        return neighbors
    
    def compute_shortest_path(self):
        while self.u and (self.u[0][0] < self.calculate_key(self.start) or self.rhs[self.start] != self.g.get(self.start, float('inf'))):
            k_old, u = heappop(self.u)
            if k_old < self.calculate_key(u):
                heappush(self.u, (self.calculate_key(u), u))
            elif self.g.get(u, float('inf')) > self.rhs.get(u, float('inf')):
                self.g[u] = self.rhs[u]
                for s in self.get_neighbors(u):
                    self.update_vertex(s)
            else:
                self.g[u] = float('inf')
                for s in self.get_neighbors(u) + [u]:
                    self.update_vertex(s)

    def find_path(self):
        self.compute_shortest_path()
        path = [self.start]
        current = self.start
        while current != self.goal:
            neighbors = self.get_neighbors(current)
            current = min(neighbors, key=lambda s: self.g.get(s, float('inf')))
            path.append(current)
        return path

# Instantiate the D* Lite algorithm
dstar = DStarLite(grid, start, goal)

# Find the path using D* Lite
path = dstar.find_path()

# Plotting the grid and the path
plt.figure(figsize=(8, 8))
plt.imshow(grid, cmap='gray')
ax = plt.gca()

# Mark the start and goal points
ax.plot(start[1], start[0], "ro")  # Start is marked with a red circle
ax.plot(goal[1], goal[0], "go")    # Goal is marked with a green circle

# Plot the path
for i in range(len(path) - 1):
    p1 = path[i]
    p2 = path[i + 1]
    plt.plot([p1[1], p2[1]], [p1[0], p2[0]], "b-")

plt.title("D* Lite Algorithm Path Finding with Obstacles")
plt.savefig('dstarlite_with_obstacles.png')
plt.show()

