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

# Define heuristic function (Manhattan distance)
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# A* algorithm
class AStar:
    def __init__(self, grid, start, goal):
        self.grid = grid
        self.start = start
        self.goal = goal
        self.frontier = []
        self.came_from = {}
        self.cost_so_far = {}
        self.run_search()

    def run_search(self):
        heappush(self.frontier, (0, self.start))
        self.came_from[self.start] = None
        self.cost_so_far[self.start] = 0

        while self.frontier:
            current_cost, current_node = heappop(self.frontier)

            if current_node == self.goal:
                break

            for next_x, next_y in [(current_node[0] + 1, current_node[1]),
                                   (current_node[0] - 1, current_node[1]),
                                   (current_node[0], current_node[1] + 1),
                                   (current_node[0], current_node[1] - 1)]:
                if 0 <= next_x < grid_size[0] and 0 <= next_y < grid_size[1] and self.grid[next_x, next_y] == 0:
                    new_cost = self.cost_so_far[current_node] + 1
                    if (next_x, next_y) not in self.cost_so_far or new_cost < self.cost_so_far[(next_x, next_y)]:
                        self.cost_so_far[(next_x, next_y)] = new_cost
                        priority = new_cost + heuristic(self.goal, (next_x, next_y))
                        heappush(self.frontier, (priority, (next_x, next_y)))
                        self.came_from[(next_x, next_y)] = current_node

    def reconstruct_path(self):
        if self.goal not in self.came_from:
            return None
        path = []
        current = self.goal
        while current != self.start:
            path.append(current)
            current = self.came_from[current]
        path.append(self.start)
        path.reverse()
        return path

# Instantiate the A* algorithm
astar = AStar(grid, start, goal)

# Find the path using A*
path = astar.reconstruct_path()

# Plotting the grid and the path
plt.figure(figsize=(8, 8))
plt.imshow(grid, cmap='gray')
ax = plt.gca()

# Mark the start and goal points
ax.plot(start[1], start[0], "ro")  # Start is marked with a red circle
ax.plot(goal[1], goal[0], "go")    # Goal is marked with a green circle

# Plot the path
if path is not None:
    for i in range(len(path) - 1):
        p1 = path[i]
        p2 = path[i + 1]
        plt.plot([p1[1], p2[1]], [p1[0], p2[0]], "b-")
else:
    print("No path found.")

plt.title("A* Algorithm Path Finding with Obstacles")
plt.savefig('astar_with_obstacles.png')
plt.show()

