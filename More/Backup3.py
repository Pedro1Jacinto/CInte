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

# Choose the metric (time or cost)
time_cost = str(input("Do you want time or cost? time, cost: "))

# Initialize a dictionary to store the time/cost matrices
matrices = {}

# Load the matrices for each chosen transport mode separately
for transport in transports:
    if number_of_cities != 50:
        time_matrix_file = f'{number_of_cities} Cities/{transport}/{time_cost}{transport.lower()}{number_of_cities}.csv'
    else:
        time_matrix_file = f'{number_of_cities} Cities/{transport}/{time_cost}{transport.lower()}.csv'
    
    try:
        # Store the time/cost matrix in the dictionary
        matrices[transport] = pd.read_csv(time_matrix_file, index_col=0)
    except FileNotFoundError:
        print(f"File not found for {transport}: {time_matrix_file}. Please check if the file exists.")

# Load the city coordinates file
if number_of_cities != 50:
    cities_file = f'{number_of_cities} Cities/xy{number_of_cities}.csv'
else:
    cities_file = f'{number_of_cities} Cities/xy.csv'

cities = pd.read_csv(cities_file)


# Função para calcular distância total/custo de uma solução
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

# Função para desenhar o mapa da melhor rota na Europa
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


# Função de cruzamento por ordem (OX)
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

# Função de mutação: troca duas cidades na rota
def swap_mutation(individual):
    a, b = random.sample(range(len(individual)), 2)
    individual[a], individual[b] = individual[b], individual[a]
    return individual

# Função de avaliação: calcula a distância/custo total para a população
def evaluate_population(population, matrices):
    fitness_scores = [calculate_distance(individual, matrices) for individual in population]
    return fitness_scores

# Seleção por torneio
def tournament_selection(population, fitness_scores, k=3):
    selected = []
    for _ in range(len(population)):
        tournament = random.sample(range(len(population)), k)
        selected.append(population[min(tournament, key=lambda i: fitness_scores[i])])
    return selected


def heuristic_solution(cities):
    # Dividir o espaço com base na longitude (direita/esquerda da Europa)
    middle_longitude = cities['Longitude'].median()
    left_side = cities[cities['Longitude'] <= middle_longitude].sort_values(by=['Latitude', 'Longitude'], ascending=[False, False])
    right_side = cities[cities['Longitude'] > middle_longitude].sort_values(by=['Latitude', 'Longitude'], ascending=[True, True])

    # Criar solução com cidades do lado esquerdo primeiro, depois lado direito
    solution = list(left_side.index) + list(right_side.index)
    return solution


# Função para o Algoritmo Genético com a variável show_plot
def genetic_algorithm(matrices, cities, num_generations=100, pop_size=100, cx_prob=0.7, mut_prob=0.2, plot_interval_generation=2, plot_interval_route=10, show_plot=1, heuristic=0):
    # Inicializar a plotagem se show_plot for 1
    if show_plot == 1:
        plt.ion()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        m = Basemap(projection='merc', llcrnrlat=35, urcrnrlat=70, llcrnrlon=-25, urcrnrlon=45, resolution='i')
    
    # Criar solução inicial usando a heurística (se heurística for ativada)
    if heuristic == 1:
        initial_solution = heuristic_solution(cities)
        population = [initial_solution] + [random.sample(range(len(cities)), len(cities)) for _ in range(pop_size - 1)]
        
        # Plotar a solução inicial (rota gerada pela heurística)
        print("Plotando solução inicial baseada na heurística...")
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

        # Cruzamento
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

        # Mutação
        population = [swap_mutation(child) if random.random() < mut_prob else child for child in offspring]

        # Plotar o progresso do algoritmo se show_plot for 1
        if show_plot == 1 and generation % plot_interval_generation == 0:
            ax1.clear()
            ax1.plot(avg_fitness_history, label="Population Average")
            ax1.plot(best_fitness_history, label="Best Solution")
            ax1.set_xlabel("Generation")
            ax1.set_ylabel("Distance")
            ax1.legend()

            ax1.text(0.5, 1.1, f"Generation: {generation}, Best: {best_fitness:.2f}\n"
                                f"{number_of_cities} Cities, Transports: {', '.join(transports)}",
                                horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes, fontsize=12)

            if generation % plot_interval_route == 0:
                plot_route(ax2, best_solution, cities, m)

            plt.pause(0.01)

        if generation % 10 == 0:
            print("Generation:", generation)

    # Desativar a plotagem
    if show_plot == 1:
        plt.ioff()
        plt.show()

    return best_fitness


# Executar o Algoritmo Genético com a opção de usar a heurística
results_GA = []
number_of_runs = 0
use_heuristic = int(input("Do you want to use the heuristic? 1 for yes, 0 for no: "))
runs = 3

while number_of_runs < runs:
    print("\n\nRun: ", number_of_runs + 1)
    result_GA = genetic_algorithm(matrices, cities, num_generations=101, pop_size=100, plot_interval_generation=1, plot_interval_route=20, show_plot=1, heuristic=use_heuristic)
    results_GA.append(result_GA)
    number_of_runs += 1

# Calculando a média e o desvio padrão
mean = np.mean(results_GA)
std = np.std(results_GA)

# Imprimindo os resultados
print(f'Mean: {mean}')
print(f'Std: {std}')



