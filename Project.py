import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import random



number_of_cities = int(input("How many cities do you want to test? 10, 30, 50: "))

# Allow the user to choose how many transport modes to use
num_transports = int(input("How many transport modes do you want to use? (1, 2, or 3): "))

# Verify which transport modes the user wants to use
if num_transports == 1:
    transports = [input("Which transport mode do you want to use? Plane, Train, Bus: ").capitalize()]
elif num_transports == 2:
    transports = input("Which transport modes do you want to use? (separated by commas): ").split(", ")
    transports = [transport.capitalize() for transport in transports]
else:
    transports = ['Plane', 'Train', 'Bus']

# Prompt the user to select time, cost, or both
time_cost = str(input("Do you want time, cost, or both? "))

# Initialize dictionaries to store the time and cost matrices
matrices_time = {}
matrices_cost = {}
matrices = {}

# Load the matrices for each chosen transport mode separately
for transport in transports:
    if number_of_cities != 50:
        time_matrix_file = f'{number_of_cities} Cities/{transport}/time{transport.lower()}{number_of_cities}.csv'
        cost_matrix_file = f'{number_of_cities} Cities/{transport}/cost{transport.lower()}{number_of_cities}.csv'
    else:
        time_matrix_file = f'{number_of_cities} Cities/{transport}/time{transport.lower()}.csv'
        cost_matrix_file = f'{number_of_cities} Cities/{transport}/cost{transport.lower()}.csv'
    
    # Load time matrix if 'time' or 'both' is chosen
    if time_cost in ['time', 'both']:
        try:
            matrices_time[transport] = pd.read_csv(time_matrix_file, index_col=0)
            matrices[transport] = pd.read_csv(time_matrix_file, index_col=0)
        except FileNotFoundError:
            print(f"File not found for {transport} (time): {time_matrix_file}. Please check if the file exists.")
    
    # Load cost matrix if 'cost' or 'both' is chosen
    if time_cost in ['cost', 'both']:
        try:
            matrices_cost[transport] = pd.read_csv(cost_matrix_file, index_col=0)
            matrices[transport] = pd.read_csv(cost_matrix_file, index_col=0)
        except FileNotFoundError:
            print(f"File not found for {transport} (cost): {cost_matrix_file}. Please check if the file exists.")


# Load the city coordinates file
if number_of_cities != 50:
    cities_file = f'{number_of_cities} Cities/xy{number_of_cities}.csv'
else:
    cities_file = f'{number_of_cities} Cities/xy.csv'

cities = pd.read_csv(cities_file)

#--------------------------------------------------------------//-------------------------------------------------------------------------------------------
#SINGLE OBJECTIVE
# Function to calculate total distance/cost of a solution
def calculate_distance(individual, matrices):
    total_distance = 0
    for i in range(len(individual) - 1):
        city_a = individual[i]
        city_b = individual[i + 1]

        shortest_distance = float('inf')
        for transport in matrices:
            distance = matrices[transport].iloc[city_a, city_b]
            if distance < shortest_distance:
                shortest_distance = distance
        total_distance += shortest_distance

    city_a = individual[-1]
    city_b = individual[0]
    shortest_distance = float('inf')
    
    for transport in matrices:
        distance = matrices[transport].iloc[city_a, city_b]
        if distance < shortest_distance:
            shortest_distance = distance
    
    total_distance += shortest_distance
    return total_distance

# Function to draw the map of the best route in Europe
def plot_route(ax, individual, cities, m):
    ax.clear()
    m.drawcoastlines(ax=ax)
    m.drawcountries(ax=ax)
    m.fillcontinents(color='lightgray', lake_color='aqua', ax=ax)
    m.drawparallels(range(30, 75, 5), labels=[1,0,0,0], linewidth=0.2, ax=ax)
    m.drawmeridians(range(-30, 50, 5), labels=[0,0,0,1], linewidth=0.2, ax=ax)
    
    latitudes = cities['Latitude'].values
    longitudes = cities['Longitude'].values

    for i in range(len(individual) - 1):
        city_a = individual[i]
        city_b = individual[i + 1]
        x1, y1 = m(longitudes[city_a], latitudes[city_a])
        x2, y2 = m(longitudes[city_b], latitudes[city_b])
        ax.plot([x1, x2], [y1, y2], 'b-', markersize=5)

    city_a = individual[-1]
    city_b = individual[0]
    x1, y1 = m(longitudes[city_a], latitudes[city_a])
    x2, y2 = m(longitudes[city_b], latitudes[city_b])
    ax.plot([x1, x2], [y1, y2], 'b-', markersize=5)

    for i, city in enumerate(individual):
        x, y = m(longitudes[city], latitudes[city])
        ax.plot(x, y, 'ro', markersize=5)
        ax.text(x, y, cities['City'].iloc[city], fontsize=8, ha='right')

    ax.set_title(f"Best Route - Current Generation")

# Order crossover function (OX)
def order_crossover(parent1, parent2):
    size = len(parent1)
    start, end = sorted([random.randint(0, size), random.randint(0, size)])
    
    child = [None] * size
    child[start:end] = parent1[start:end]

    pointer = end
    for city in parent2:
        if city not in child:
            if pointer >= size:
                pointer = 0
            child[pointer] = city
            pointer += 1
    return child

# Mutation function: swap two cities in the route
def swap_mutation(individual):
    a, b = random.sample(range(len(individual)), 2)
    individual[a], individual[b] = individual[b], individual[a]
    return individual

# Evaluation function: calculates total distance/cost for the population
def evaluate_population(population, matrices):
    fitness_scores = [calculate_distance(individual, matrices) for individual in population]
    return fitness_scores

# Tournament selection
def tournament_selection(population, fitness_scores, k=3):
    selected = []
    for _ in range(len(population)):
        tournament = random.sample(range(len(population)), k)
        selected.append(population[min(tournament, key=lambda i: fitness_scores[i])])
    return selected

def heuristic_solution(cities):
    # Divide the space based on longitude (right/left of Europe)
    middle_longitude = cities['Longitude'].median()
    left_side = cities[cities['Longitude'] <= middle_longitude].sort_values(by=['Latitude', 'Longitude'], ascending=[False, False])
    right_side = cities[cities['Longitude'] > middle_longitude].sort_values(by=['Latitude', 'Longitude'], ascending=[True, True])

    # Create solution with cities on the left side first, then right side
    solution = list(left_side.index) + list(right_side.index)
    return solution

# Function for the Genetic Algorithm with the variable show_plot
def genetic_algorithm(matrices, cities, time_cost="time", num_generations=101, pop_size=100, cx_prob=0.7, mut_prob=0.2, plot_interval_generation=2, plot_interval_route=10, show_plot=1, heuristic=0):
    # Initialize plotting if show_plot is 1
    if show_plot == 1:
        plt.ion()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        m = Basemap(projection='merc', llcrnrlat=35, urcrnrlat=70, llcrnrlon=-25, urcrnrlon=45, resolution='i')
    
    # Create initial solution using the heuristic (if heuristic is activated)
    if heuristic == 1:
        initial_solution = heuristic_solution(cities)
        population = [initial_solution] + [random.sample(range(len(cities)), len(cities)) for _ in range(pop_size - 1)]
        
        if show_plot == 1:
            # Plot the initial solution (route generated by the heuristic)
            print("Plotting initial solution based on the heuristic...")
            plot_route(ax2, initial_solution, cities, m)
            plt.show()
            plt.pause(3)
    else:
        population = [random.sample(range(len(cities)), len(cities)) for _ in range(pop_size)]

    best_solution = None
    best_fitness = float('inf')
    avg_fitness_history = []
    best_fitness_history = []

    for generation in range(num_generations):
        fitness_scores = evaluate_population(population, matrices)
        avg_fitness = np.mean(fitness_scores)
        best_idx = np.argmin(fitness_scores)
        best_individual = population[best_idx]
        best_individual_fitness = fitness_scores[best_idx]

        if best_individual_fitness < best_fitness:
            best_fitness = best_individual_fitness
            best_solution = best_individual

        avg_fitness_history.append(avg_fitness)
        best_fitness_history.append(best_fitness)

        selected_population = tournament_selection(population, fitness_scores)

        # Crossover
        offspring = []
        for i in range(0, pop_size, 2):
            parent1 = selected_population[i]
            parent2 = selected_population[i + 1]
            if random.random() < cx_prob:
                child1 = order_crossover(parent1, parent2)
                child2 = order_crossover(parent2, parent1)
            else:
                child1, child2 = parent1[:], parent2[:]
            offspring.extend([child1, child2])

        # Mutation
        population = [swap_mutation(child) if random.random() < mut_prob else child for child in offspring]

        # Plot the progress of the algorithm if show_plot is 1
        if show_plot == 1 and generation % plot_interval_generation == 0:
            ax1.clear()
            ax1.plot(avg_fitness_history, label="Population Average")
            ax1.plot(best_fitness_history, label="Best Solution")
            ax1.set_xlabel("Generation")
            ax1.set_ylabel("Distance")
            ax1.legend()

            best_text = f"Best Time: {best_fitness:.2f} h" if time_cost == 'time' else f"Best Cost: {best_fitness:.2f} €"
            ax1.text(0.5, 1.1, f"Generation: {generation}, {best_text}\n"
                                f"{len(cities)} Cities, Transports: {', '.join(transports)}",
                                horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes, fontsize=12)

            if generation % plot_interval_route == 0:
                plot_route(ax2, best_solution, cities, m)

            plt.pause(0.01)

        if generation % 10 == 0:
            if time_cost == 'time':
                print(f"Generation {generation}: Best Time: {best_fitness:.2f} h")
            elif time_cost == 'cost':
                print(f"Generation {generation}: Best Cost: {best_fitness:.2f} €")

    # Disable plotting
    if show_plot == 1:
        plt.ioff()
        plt.show()

    return best_solution, best_fitness


#--------------------------------------------------------------//-------------------------------------------------------------------------------------------
#MULTI-OBJECTIVE
# Calculate both time and cost for a solution (route)
def calculate_time_and_cost(individual, matrices_time, matrices_cost):
    """
    Calculate the minimum total time and cost for a given solution (route) by selecting
    the shortest option across all transport modes for each leg of the journey.

    :param individual: A candidate solution (sequence of city indices)
    :param matrices_time: Dictionary of time matrices for each transport mode
    :param matrices_cost: Dictionary of cost matrices for each transport mode
    :return: A tuple (total_time, total_cost)
    """
    total_time = 0
    total_cost = 0
    
    for i in range(len(individual) - 1):
        city_from = individual[i]
        city_to = individual[i + 1]

        # Initialize minimum time and cost for this segment as infinity
        min_time = float('inf')
        min_cost = float('inf')

        # Evaluate all transport modes to find the minimum time and cost
        for transport in matrices_time:
            time = matrices_time[transport].iloc[city_from, city_to]
            cost = matrices_cost[transport].iloc[city_from, city_to]
            
            if time < min_time:
                min_time = time
            if cost < min_cost:
                min_cost = cost

        # Add the minimum time and cost for this segment to the totals
        total_time += min_time
        total_cost += min_cost

    # Also connect the last city back to the start to complete the loop
    city_from = individual[-1]
    city_to = individual[0]
    min_time = float('inf')
    min_cost = float('inf')

    # Again, find the minimum time and cost for the loop closure
    for transport in matrices_time:
        time = matrices_time[transport].iloc[city_from, city_to]
        cost = matrices_cost[transport].iloc[city_from, city_to]

        if time < min_time:
            min_time = time
        if cost < min_cost:
            min_cost = cost

    # Add the loop closure's minimum time and cost
    total_time += min_time
    total_cost += min_cost
    
    return total_time, total_cost


# Evaluate population based on both objectives (time and cost)
def evaluate_population_time_cost(population, matrices_time, matrices_cost):
    fitness_values = []
    for individual in population:
        total_time, total_cost = calculate_time_and_cost(individual, matrices_time, matrices_cost)
        fitness_values.append((total_time, total_cost))
    return fitness_values

# Implement Pareto dominance to find non-dominated solutions
def is_dominated(individual, population):
    """Return True if individual is dominated by any other individual in the population."""
    for other_individual in population:
        # Check if another individual dominates this one (better in both objectives)
        if (other_individual[0] <= individual[0] and other_individual[1] < individual[1]) or \
           (other_individual[0] < individual[0] and other_individual[1] <= individual[1]):
            return True
    return False

def pareto_selection(population, fitness_scores):
    """Select the Pareto front from the population based on the fitness scores."""
    pareto_front = []
    for i, individual_fitness in enumerate(fitness_scores):
        if not is_dominated(individual_fitness, fitness_scores):
            pareto_front.append(population[i])
    return pareto_front

def genetic_algorithm_multi(matrices_time, matrices_cost, cities, num_generations=100, pop_size=100, cx_prob=0.7, mut_prob=0.2, plot_interval_generation=2, plot_interval_route=10, show_plot=1, heuristic=0):
    """
    Genetic Algorithm for Multi-Objective Optimization (Time and Cost).
    
    :param matrices_time: Time matrix
    :param matrices_cost: Cost matrix
    :param cities: List of city coordinates
    :param num_generations: Number of generations for the algorithm to run
    :param pop_size: Population size
    :param cx_prob: Crossover probability
    :param mut_prob: Mutation probability
    :param plot_interval_generation: Interval for updating fitness plot
    :param plot_interval_route: Interval for updating route plot
    :param show_plot: Whether to show plot (1 = show, 0 = no)
    :param heuristic: Whether to initialize with a heuristic-based solution
    """
    # Initialize plot if show_plot is enabled
    if show_plot == 1:
        plt.ion()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        m = Basemap(projection='merc', llcrnrlat=35, urcrnrlat=70, llcrnrlon=-25, urcrnrlon=45, resolution='i')

    # Generate initial solution (with or without heuristic)
    if heuristic == 1:
        initial_solution = heuristic_solution(cities)
        population = [initial_solution] + [random.sample(range(len(cities)), len(cities)) for _ in range(pop_size - 1)]
        print("Plotting initial solution based on heuristic...")
        plot_route(ax2, initial_solution, cities, m)
        plt.show()
        plt.pause(3)
    else:
        population = [random.sample(range(len(cities)), len(cities)) for _ in range(pop_size)]

    best_solution = None
    best_fitness = (float('inf'), float('inf'))
    avg_fitness_history = []
    best_fitness_history = []
    pareto_history = []
    
    for generation in range(num_generations):
        # Evaluate population (both time and cost)
        fitness_scores = evaluate_population_time_cost(population, matrices_time, matrices_cost)
        pareto_front = pareto_selection(population, fitness_scores)

        # Log Pareto front points for history tracking
        pareto_history.append([(fitness_scores[i][0], fitness_scores[i][1]) for i in range(len(population)) if population[i] in pareto_front])

        # **Select the "Best" Individual from Pareto Front**
        # Example approach: Minimizing the sum of time and cost as a representative solution from Pareto front
        best_individual = min(pareto_front, key=lambda ind: fitness_scores[population.index(ind)][0] + fitness_scores[population.index(ind)][1])
        best_individual_fitness = fitness_scores[population.index(best_individual)]

        # Update best fitness and best solution history
        if best_individual_fitness[0] + best_individual_fitness[1] < sum(best_fitness):
            best_fitness = best_individual_fitness
            best_solution = best_individual

        # Track average fitness
        avg_fitness = np.mean([sum(f) for f in fitness_scores]) 
        avg_fitness_history.append(avg_fitness)
        best_fitness_history.append(sum(best_fitness))

        # Selection using tournament or another method
        selected_population = tournament_selection(population, fitness_scores)
        population = pareto_front + selected_population[:pop_size - len(pareto_front)]

        # Crossover and mutation
        offspring = []
        for i in range(0, pop_size, 2):
            parent1 = selected_population[i]
            parent2 = selected_population[i + 1]
            if random.random() < cx_prob:
                child1 = order_crossover(parent1, parent2)
                child2 = order_crossover(parent2, parent1)
            else:
                child1, child2 = parent1[:], parent2[:]
            offspring.extend([child1, child2])

        population = [swap_mutation(child) if random.random() < mut_prob else child for child in offspring]

        # Plot progress if show_plot is enabled
        if show_plot == 1 and generation % plot_interval_generation == 0:
            ax1.clear()
            ax1.plot(avg_fitness_history, label="Population Average")
            ax1.plot(best_fitness_history, label="Best Solution")
            ax1.set_xlabel("Generation")
            ax1.set_ylabel("Fitness")
            ax1.legend()

            ax1.text(0.5, 1.1, f"Generation: {generation}, Best Time: {best_fitness[0]:.2f} h, Best Cost: {best_fitness[1]:.2f} €\n"
                                f"{len(cities)} Cities, Transports: {', '.join(transports)}",
                                horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes, fontsize=12)

            if generation % plot_interval_route == 0:
                plot_route(ax2, best_solution, cities, m)

            plt.pause(0.01)

        if generation % 10 == 0:
            print(f"Generation {generation}: Best Time: {best_fitness[0]:.2f} h, Best Cost: {best_fitness[1]:.2f} €")

    # Finalize plot if show_plot is enabled
    if show_plot == 1:
        plt.ioff()
        plt.show()

    return best_solution, best_fitness


#--------------------------------------------------------------//-------------------------------------------------------------------------------------------
# Run the Genetic Algorithm with the option to use the heuristic
results_GA = []
number_of_runs = 0
use_heuristic = int(input("Do you want to use the heuristic? 1 for yes, 0 for no: "))
runs = 1


if time_cost == "both":
    while number_of_runs < runs:
        print("\n\nRun: ", number_of_runs + 1)
        _, best_fitness = genetic_algorithm_multi(matrices_time, matrices_cost, cities, num_generations=251, pop_size=40, plot_interval_generation=1, plot_interval_route=20, show_plot=1, heuristic=use_heuristic)
        results_GA.append(best_fitness)  # Store only (time, cost)
        number_of_runs += 1

    # Calculating the mean and standard deviation for time and cost
    mean_time = np.mean([result[0] for result in results_GA])  
    std_time = np.std([result[0] for result in results_GA])
    mean_cost = np.mean([result[1] for result in results_GA])  
    std_cost = np.std([result[1] for result in results_GA])

    # Printing the results
    print(f'Mean Time: {mean_time} h')
    print(f'Std Time: {std_time} h')
    print(f'Mean Cost: {mean_cost} €')
    print(f'Std Cost: {std_cost} €')

else:
    while number_of_runs < runs:
        print("\n\nRun: ", number_of_runs + 1)
        _, best_fitness = genetic_algorithm(matrices, cities, time_cost=time_cost, num_generations=251, pop_size=40, plot_interval_generation=1, plot_interval_route=20, show_plot=1, heuristic=use_heuristic)
        results_GA.append(best_fitness) 
        number_of_runs += 1

    # Calculating the mean and standard deviation based on time_cost
    if time_cost == "time":
        mean = np.mean(results_GA)
        std = np.std(results_GA)
        print(f'Mean Time: {mean} h')
        print(f'Std Time: {std} h')
    elif time_cost == "cost":
        mean = np.mean(results_GA)
        std = np.std(results_GA)
        print(f'Mean Cost: {mean} €')
        print(f'Std Cost: {std} €')
