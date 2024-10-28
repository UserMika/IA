import random
import sys
import json
import time
import math


def read_instance_json(file_path):
    with open(file_path) as f:
        return json.load(f)


def write_instance_json(solver, instance, solution, file_path):
    with open(file_path, 'w') as f:
        json.dump(solution, f)
        # f.write('\n')
        # json.dump(instance['GlobalBest'], f)
        # f.write('\n')
        # f.write('Value of the best solution found by the solver: ')
        # json.dump(solver.best_solution_cost, f)
        # f.write('\n')
        # f.write('Value of the best solution found by the instance: ')
        # json.dump(instance['GlobalBestVal'], f)

class LNSTSPSolver:
    def __init__(self, distance_matrix, time_out, initial_temp=100, end_temp=1, alpha=0.995, destroy_ratio=0.2):
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
        self.destroy_functions = [self.random_destroy, self.clustered_destroy, self.worst_destroy]
        self.repair_functions = [self.greedy_repair, self.nearest_insertion_repair]
        self.destroy_weights = [1, 1, 1]
        self.repair_weights = [1, 1]

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
        total_cost = 0
        for i in range(len(solution) - 1):
            total_cost += self.distance_matrix[solution[i]][solution[i + 1]]
        total_cost += self.distance_matrix[solution[-1]][solution[0]]
        return total_cost

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
        nearest_indices = sorted(range(len(distances)), key=lambda i: distances[i])[:self.number_to_remove]
        removed = set(self.current_solution[i] for i in nearest_indices)
        solution = [city for i, city in enumerate(self.current_solution) if i not in nearest_indices]
        return removed, solution

    def worst_destroy(self):
        costs = sorted([(self.distance_matrix[self.current_solution[i]][self.current_solution[(i + 1) % self.num_nodes]], i) for i in range(len(self.current_solution))])
        worst_indices = [index for cost, index in costs[-self.number_to_remove:]]
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
                distance_to_current = self.distance_matrix[city][current_city]
                distance_to_next = self.distance_matrix[city][destroyed_solution[(i + 1) % len(destroyed_solution)]]
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




    def k_opt(self, destroyed_solution, k):
        def swap_k_segments(solution, indices):
            segments = [solution[indices[i]:indices[i + 1]] for i in range(len(indices) - 1)]
            new_solution = []
            for segment in segments:
                new_solution.extend(segment)
            return new_solution

        improvement = True
        while improvement:
            improvement = False
            for i in range(len(destroyed_solution) - k):
                for j in range(i + 1, len(destroyed_solution) - k + 1):
                    indices = sorted(random.sample(range(len(destroyed_solution)), k + 1))
                    new_solution = swap_k_segments(destroyed_solution, indices)
                    if self.calculate_cost(new_solution) < self.calculate_cost(destroyed_solution):
                        destroyed_solution = new_solution[:]
                        improvement = True
        return destroyed_solution

    def accept(self, new_cost, current_cost, temp):
        if new_cost < current_cost:
            return True
        else:
            return random.random() < math.exp((current_cost - new_cost) / temp)

    def ALNS(self, instance, output_path, w1, w2, w3, w4, decay):
        start_time = time.time()
        while time.time() - start_time < self.time_out:
            x1,x2,x3,x4 = False, False, False, False
            best_probability = 0
            for i in range(len(self.destroy_functions)):
                destroy_probability = self.destroy_weights[i] / sum(self.destroy_weights)
                if destroy_probability > best_probability:
                    best_probability = destroy_probability
                    best_destroy_index = i
            destroy_selected = self.destroy_functions[best_destroy_index]
            best_probability = 0
            for i in range(len(self.repair_functions)):
                repair_probability = self.repair_weights[i] / sum(self.repair_weights)
                if repair_probability > best_probability:
                    best_probability = repair_probability
                    best_repair_index = i
            repair_selected = self.repair_functions[best_repair_index]
            destroyed, destroyed_solution = destroy_selected()
            repaired_solution = repair_selected(destroyed_solution, destroyed)
            new_cost = self.calculate_cost(repaired_solution)

            if self.accept(new_cost, self.current_cost, self.initial_temp):
                self.current_solution = repaired_solution[:]
                self.current_cost = new_cost
                print("the length of the current solution is", len(self.current_solution))
                print("and the length of the best solution is", len(self.best_solution))
                print(self.current_cost)
                x2 = True
                x3 = True
            else:
                x4 = True
            if self.best_solution_cost > new_cost:
                x1 = True
                self.best_solution = repaired_solution[:]
                self.best_solution_cost = new_cost
            adjustment = max(x1*w1, x2*w2, x3*w3, x4*w4)
            self.destroy_weights[best_destroy_index] = decay*self.destroy_weights[best_destroy_index] + (1-decay)*adjustment
            self.repair_weights[best_repair_index] = decay*self.repair_weights[best_repair_index] + (1-decay)*adjustment
        write_instance_json(self, instance, self.best_solution, output_path)

            
                




def main():
    instance_path = sys.argv[1]
    output_path = sys.argv[2]
    instance = read_instance_json(instance_path)
    distance_matrix = instance['Matrix']
    time_out = instance['Timeout']
    solver = LNSTSPSolver(distance_matrix, time_out)
    solver.ALNS(instance, output_path, 0.5, 0.4, 0.3, 0.1, 1)


# def main():
#     instance_path = sys.argv[1]
#     output_path = sys.argv[2]
#     instance = read_instance_json(instance_path)
#     distance_matrix = instance['Matrix']
#     time_out = instance['Timeout']
#     solver = LNSTSPSolver(distance_matrix, time_out)
#     start_time = time.time()
#     while time.time() - start_time < time_out:
#         destroyed, destroyed_solution = solver.random_destroy()
#         repaired_solution = solver.nearest_insertion_repair(destroyed_solution, destroyed)
#         # repaired_solution = solver.two_opt(repaired_solution)
#         repaired_cost = solver.calculate_cost(repaired_solution)
#         if repaired_cost < solver.current_cost:
#             print(repaired_cost)
#             solver.current_solution = repaired_solution[:]
#             solver.current_cost = repaired_cost
#         if repaired_cost < solver.best_solution_cost:
#             solver.best_solution = repaired_solution[:]
#             solver.best_solution_cost = repaired_cost
#     write_instance_json(solver, instance, solver.best_solution, output_path)





                
            


        
    

    





if __name__ == '__main__':
    main()

# instance_path = sys.argv[1]
# output_path = sys.argv[2]

# instance = read_instance_json(instance_path)
# naive_solution = [i for i in range(len(instance['Matrix']))] # TODO - implement something better
# write_instance_json(instance, naive_solution, output_path)


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



