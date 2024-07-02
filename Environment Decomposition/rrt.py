import math
import random
import numpy as np
import matplotlib.pyplot as plt

show_animation = True

class Node:
    """
    RRT Node
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None

class RRT:
    """
    Class for RRT Planning
    """

    def __init__(self, start, goal, obstacle_list, rand_area, expand_dis=5.0, goal_sample_rate=5, max_iter=800):
        self.start = Node(start[0], start[1])
        self.end = Node(goal[0], goal[1])
        self.min_rand = rand_area[0]
        self.max_rand = rand_area[1]
        self.expand_dis = expand_dis
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.obstacle_list = obstacle_list

    def planning(self, animation=True):
        self.node_list = [self.start]
        for i in range(self.max_iter):
            rnd_node = self.get_random_node()
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd_node)
            nearest_node = self.node_list[nearest_ind]

            new_node = self.steer(nearest_node, rnd_node, self.expand_dis)

            if not self.check_collision(new_node):
                self.node_list.append(new_node)

            if animation:
                self.draw_graph(rnd_node)

            if self.calc_dist_to_goal(self.node_list[-1].x, self.node_list[-1].y) <= self.expand_dis:
                final_node = self.steer(self.node_list[-1], self.end, self.expand_dis)
                if not self.check_collision(final_node):
                    return self.generate_final_course(len(self.node_list) - 1)

            if animation:
                plt.pause(0.01)

        return None

    def steer(self, from_node, to_node, extend_length=float("inf")):
        new_node = Node(from_node.x, from_node.y)
        d, theta = self.calc_distance_and_angle(new_node, to_node)

        new_node.x += extend_length * math.cos(theta)
        new_node.y += extend_length * math.sin(theta)

        new_node.parent = from_node

        return new_node

    def get_random_node(self):
        if random.randint(0, 100) > self.goal_sample_rate:
            rnd = [random.uniform(self.min_rand, self.max_rand),
                   random.uniform(self.min_rand, self.max_rand)]
        else:
            rnd = [self.end.x, self.end.y]
        return Node(rnd[0], rnd[1])

    def get_nearest_node_index(self, node_list, rnd_node):
        dlist = [(node.x - rnd_node.x) ** 2 + (node.y - rnd_node.y) ** 2 for node in node_list]
        minind = dlist.index(min(dlist))
        return minind

    def check_collision(self, node):
        if node is None or node.parent is None:
            return False

        x1, y1 = int(node.x), int(node.y)
        x2, y2 = int(node.parent.x), int(node.parent.y)
        
        # Use Bresenham's line algorithm to check for collision
        for x, y in self.bresenham(x1, y1, x2, y2):
            for (ox, oy, w, h) in self.obstacle_list:
                if ox <= x <= ox + w and oy <= y <= oy + h:
                    return True  # collision
        return False  # safe

    def bresenham(self, x1, y1, x2, y2):
        """
        Bresenham's Line Algorithm to generate points on a line.
        """
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

    def generate_final_course(self, goal_ind):
        path = [[self.end.x, self.end.y]]
        node = self.node_list[goal_ind]
        while node.parent is not None:
            path.append([node.x, node.y])
            node = node.parent
        path.append([node.x, node.y])
        return path

    def calc_dist_to_goal(self, x, y):
        dx = x - self.end.x
        dy = y - self.end.y
        return math.hypot(dx, dy)

    def calc_distance_and_angle(self, from_node, to_node):
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        d = math.hypot(dx, dy)
        theta = math.atan2(dy, dx)
        return d, theta

    def draw_graph(self, rnd=None):
        plt.clf()
        if rnd is not None:
            plt.plot(rnd.x, rnd.y, "^k")
        for node in self.node_list:
            if node.parent:
                plt.plot([node.x, node.parent.x], [node.y, node.parent.y], "-g")
        for (ox, oy, w, h) in self.obstacle_list:
            rect = plt.Rectangle((ox, oy), w, h, linewidth=1, edgecolor='black', facecolor='gray')
            plt.gca().add_patch(rect)
        plt.plot(self.start.x, self.start.y, "^r")
        plt.plot(self.end.x, self.end.y, "^c")
        plt.grid(True)
        plt.axis("equal")

def main():
    print("Start RRT planning")

    obstacle_list = [
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

    # Set Initial parameters
    rrt = RRT(start=[10, 10], goal=[60, 10],
              rand_area=[-2, 100], obstacle_list=obstacle_list)
    path = rrt.planning(animation=show_animation)

    if path is None:
        print("Cannot find path")
    else:
        print("Found path!!")
        if show_animation:
            rrt.draw_graph()
            plt.plot([x for (x, y) in path], [y for (x, y) in path], '-r')
            plt.grid(True)
            plt.show()

        # Calculate the Euclidean length of the path
        path_length = calculate_path_length(path)
        print(f"Path length: {path_length:.2f} units")

def calculate_path_length(path):
    length = 0.0
    for i in range(1, len(path)):
        length += math.hypot(path[i][0] - path[i-1][0], path[i][1] - path[i-1][1])
    return length

if __name__ == '__main__':
    main()

