import random
import sys
import json
import time
import math


def read_instance_json(file_path):
    with open(file_path) as f:
        return json.load(f)


def write_instance_json(solution, file_path):
    with open(file_path, 'w') as f:
        json.dump(solution, f)

class LNSTSPSolver:
    def __init__(self, distance_matrix, time_out, initial_temp=100, end_temp=1, alpha=0.995, destroy_ratio=0.2, temp):
        self.distance_matrix = distance_matrix
        self.time_out = time_out
        self.alpha = alpha
        self.num_nodes = len(self.distance_matrix)
        self.current_solution = self.get_initial_solution()
        self.best_solution = self.current_solution[:]
        self.current_cost = self.calculate_cost(self.current_solution)
        self.best_solution_cost = self.current_cost
        self.stage_time = self.time_out // 3
        self.number_to_remove = int(destroy_ratio * self.num_nodes)
        self.initial_temp = initial_temp
        self.end_temp = end_temp

    def get_initial_solution(self):
        start_node = random.randint(0, self.num_nodes - 1)
        solution = [start_node]
        unvisited = set(range(self.num_nodes))
        unvisited.remove(start_node)

        current_node = start_node
        while unvisited:
            nearest_node = min(unvisited, key=lambda node: self.distance_matrix[current_node][node])
            solution.append(nearest_node)
            unvisited.remove(nearest_node)
            current_node = nearest_node
        return solution

    def calculate_cost(self, solution):
        return sum(self.distance_matrix[solution[i], solution[i + 1]] for i in range(len(solution) - 1)) + self.distance_matrix[solution[-1], solution[0]]

    def accept(self):
        return self.current_cost < self.best_solution_cost

    def random_destroy(self):
        removed_indices = random.sample(range(self.num_nodes), self.number_to_remove)
        removed = set(self.current_solution[i] for i in removed_indices)
        solution = [node for i, node in enumerate(self.current_solution) if i not in removed_indices]
        return removed, solution

    def clustered_destroy(self):
        starting_city = random.choice(self.current_solution)
        distances = self.distance_matrix[starting_city]
        nearest_indices = sorted(distances)[:self.number_to_remove]
        removed = set(self.current_solution[i] for i in nearest_indices)
        solution = [city for i, city in enumerate(self.current_solution) if i not in nearest_indices]
        return removed, solution

    def worst_destroy(self):
        costs = sorted(self.distance_matrix[self.current_solution[i], self.current_solution[(i + 1) % self.num_nodes]] for i in range(len(self.current_solution)))
        worst_indices = costs[-self.number_to_remove:]
        removed = set(self.current_solution[i] for i in worst_indices)
        solution = [city for i, city in enumerate(self.current_solution) if i not in worst_indices]
        return removed, solution

    def greedy_repair(self, destroyed_solution, removed):
        for node in removed:
            best_pos = 0
            best_increase = float('inf')
            for i in range(len(destroyed_solution) + 1):
                solution = destroyed_solution[:i] + [node] + destroyed_solution[i:]
                increase = self.calculate_cost(solution) - self.calculate_cost(destroyed_solution)
                if increase < best_increase:
                    best_increase = increase
                    best_pos = i
            destroyed_solution.insert(best_pos, node)
        return destroyed_solution

    def nearest_insertion_repair(self, destroyed_solution, removed):
        for city in removed:
            best_pos = 0
            best_increase = float('inf')
            for i, current_city in enumerate(destroyed_solution):
                distance_to_current = self.distance_matrix[city, current_city]
                distance_to_next = self.distance_matrix[city, destroyed_solution[(i + 1) % len(destroyed_solution)]]
                increase = distance_to_current + distance_to_next
                if increase < best_increase:
                    best_increase = increase
                    best_pos = i + 1
            destroyed_solution.insert(best_pos, city)
        return destroyed_solution


    def two_opt(self, destroyed_solution):
        improvement = True
        while improvement:
            improvement = False
            for i in range(1, len(destroyed_solution) - 2):
                for j in range(i + 1, len(destroyed_solution)):
                    if j - i == 1:
                        continue
                    new_solution = destroyed_solution[:]
                    new_solution[i:j] = destroyed_solution[j - 1:i - 1:-1]
                    if self.calculate_cost(new_solution) < self.calculate_cost(destroyed_solution):
                        destroyed_solution = new_solution[:]
                        improvement = True
        return destroyed_solution

    def accept(self, new_cost, current_cost, temp):
        if new_cost < current_cost:
            return True
        else:
            return random.random() < math.exp((current_cost - new_cost) / temp)



instance_path = sys.argv[1]
output_path = sys.argv[2]

instance = read_instance_json(instance_path)
naive_solution = [i for i in range(len(instance['Matrix']))] # TODO - implement something better
write_instance_json(naive_solution, output_path)


#######################################################################
# Example of the required timeout mechanism within the LNS structure: #
#######################################################################
# ...
# time_limit = instance['Timeout']
# start_time = time.time()
# for iteration in range(9999999999):
#     ...logic of one search iteration...
#     if time.time() - start_time >= time_limit:
#         break
# ...
#######################################################################
#######################################################################

