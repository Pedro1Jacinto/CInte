import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import random

number_of_cities = int(input("Quantas cidades deseja testar? 10, 30, 50: "))
transport = str(input("Qual meio de locomoção? Plane, Train, Bus: "))
time_cost = str(input("Quer time ou cost? time, cost: "))
if number_of_cities !=50:
    # Gerar os nomes dos arquivos com base nos inputs
    time_matrix_file = f'{number_of_cities} Cities/{transport}/{time_cost}{transport.lower()}{number_of_cities}.csv'
    cities_file = f'{number_of_cities} Cities/xy{number_of_cities}.csv'
else:
    # Gerar os nomes dos arquivos com base nos inputs
    time_matrix_file = f'{number_of_cities} Cities/{transport}/{time_cost}{transport.lower()}.csv'
    cities_file = f'{number_of_cities} Cities/xy.csv'

# Carregar os arquivos
time_matrix = pd.read_csv(time_matrix_file, index_col=0)
cities = pd.read_csv(cities_file)

# Função para calcular a distância total de uma solução
def calculate_distance(individual, time_matrix):
    distance = 0
    for i in range(len(individual) - 1):
        city_a = individual[i]
        city_b = individual[i + 1]
        distance += time_matrix.iloc[city_a, city_b]
    distance += time_matrix.iloc[individual[-1], individual[0]]  # Volta para a cidade inicial
    return distance

# Função para desenhar o gráfico do melhor caminho no mapa da Europa
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

    ax.set_title(f"Melhor Caminho - Geração Atual")


# Crossover de Ordem (Order Crossover)
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

# Função de mutação: troca duas cidades de posição
def swap_mutation(individual):
    a, b = random.sample(range(len(individual)), 2)
    individual[a], individual[b] = individual[b], individual[a]
    return individual

# Função de avaliação: calcula a distância total
def evaluate_population(population, time_matrix):
    fitness_scores = [calculate_distance(individual, time_matrix) for individual in population]
    return fitness_scores

# Seleção por torneio
def tournament_selection(population, fitness_scores, k=3):
    selected = []
    for _ in range(len(population)):
        tournament = random.sample(range(len(population)), k)
        selected.append(population[min(tournament, key=lambda i: fitness_scores[i])])
    return selected

# Variáveis usadas para exibir informações na anotação
unit = "horas" if time_cost == "time" else "€"  # Define unidade de medida com base no time_cost
transport_mode = transport.capitalize()  # Nome do transporte escolhido com capitalização correta

# Função para o Algoritmo Genético
def genetic_algorithm(time_matrix, cities, num_generations=100, pop_size=100, cx_prob=0.7, mut_prob=0.2, plot_interval_generation=2, plot_interval_route=10):
    plt.ion()  # Ativar modo interativo do matplotlib
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    m = Basemap(projection='merc', llcrnrlat=35, urcrnrlat=70, llcrnrlon=-25, urcrnrlon=45, resolution='i')

    # Inicializar população
    population = [random.sample(range(len(cities)), len(cities)) for _ in range(pop_size)]
    best_solution = None
    best_fitness = float('inf')

    # Dados para gráfico
    avg_fitness_history = []
    best_fitness_history = []

    for generation in range(num_generations):
        # Avaliar a população
        fitness_scores = evaluate_population(population, time_matrix)
        avg_fitness = np.mean(fitness_scores)
        best_idx = np.argmin(fitness_scores)
        best_individual = population[best_idx]
        best_individual_fitness = fitness_scores[best_idx]

        # Atualizar a melhor solução encontrada
        if best_individual_fitness < best_fitness:
            best_fitness = best_individual_fitness
            best_solution = best_individual

        # Armazenar para gráficos
        avg_fitness_history.append(avg_fitness)
        best_fitness_history.append(best_fitness)

        # Seleção
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

        if generation % plot_interval_generation == 0:
            # Limpar e redesenhar os gráficos em tempo real
            ax1.clear()
            ax1.plot(avg_fitness_history, label="Média da população")
            ax1.plot(best_fitness_history, label="Melhor solução")
            ax1.set_xlabel("Geração")
            ax1.set_ylabel("Distância")
            ax1.legend()
            
            # Adicionar texto com número de cidades, transporte e unidade correta
            ax1.text(0.5, 1.1, 
                     f"Geração: {generation}, Melhor: {best_fitness:.2f} {unit}\n"
                     f"{number_of_cities} cidades, Transporte: {transport_mode}", 
                     horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes, fontsize=12)
        
            # Atualizar o gráfico de caminho a cada 'plot_interval' gerações
            if generation % plot_interval_route == 0:
                plot_route(ax2, best_solution, cities, m)

            plt.pause(0.01)  # Pequena pausa para permitir atualização gráfica

    plt.ioff()  # Desativar modo interativo do matplotlib
    plt.show()

# Executar o Algoritmo Genético
genetic_algorithm(time_matrix, cities, num_generations=101, pop_size=100, plot_interval_generation=1, plot_interval_route=20)
