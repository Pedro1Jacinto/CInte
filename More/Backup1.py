import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import random

# Carregar o arquivo timeplane.csv
time_matrix = pd.read_csv('timeplane.csv', index_col=0)

# Carregar as coordenadas das cidades
cities = pd.read_csv('xy.csv')

# Função para calcular a distância total de uma solução
def calculate_distance(individual, time_matrix):
    distance = 0
    for i in range(len(individual) - 1):
        city_a = individual[i]
        city_b = individual[i + 1]
        distance += time_matrix.iloc[city_a, city_b]
    # Adiciona a volta para a cidade inicial
    distance += time_matrix.iloc[individual[-1], individual[0]]
    return distance

# Função para desenhar o gráfico do melhor caminho no mapa da Europa
def plot_route(individual, cities, title="Melhor Caminho"):
    plt.figure(figsize=(10, 8))
    m = Basemap(projection='merc', llcrnrlat=35, urcrnrlat=70, llcrnrlon=-25, urcrnrlon=45, resolution='i')
    m.drawcoastlines()
    m.drawcountries()
    m.fillcontinents(color='lightgray', lake_color='aqua')
    m.drawparallels(range(30, 75, 5), labels=[1,0,0,0], linewidth=0.2)
    m.drawmeridians(range(-30, 50, 5), labels=[0,0,0,1], linewidth=0.2)
    
    # Pegar coordenadas das cidades
    latitudes = cities['Latitude'].values
    longitudes = cities['Longitude'].values

    # Desenhar o caminho no mapa
    for i in range(len(individual) - 1):
        city_a = individual[i]
        city_b = individual[i + 1]
        x1, y1 = m(longitudes[city_a], latitudes[city_a])
        x2, y2 = m(longitudes[city_b], latitudes[city_b])
        plt.plot([x1, x2], [y1, y2], 'b-', markersize=5)
    
    # Desenhar último segmento para voltar à cidade inicial
    city_a = individual[-1]
    city_b = individual[0]
    x1, y1 = m(longitudes[city_a], latitudes[city_a])
    x2, y2 = m(longitudes[city_b], latitudes[city_b])
    plt.plot([x1, x2], [y1, y2], 'b-', markersize=5)

    for i, city in enumerate(individual):
        x, y = m(longitudes[city], latitudes[city])
        plt.plot(x, y, 'ro', markersize=5)
        plt.text(x, y, cities['City'].iloc[city], fontsize=8, ha='right')

    plt.title(title)
    plt.show()








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

# Função para o Algoritmo Genético
def genetic_algorithm(time_matrix, cities, num_generations=100, pop_size=100, cx_prob=0.7, mut_prob=0.2):
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

        # Mostrar progresso
        print(f"Geração {generation}: Melhor distância = {best_fitness}, Média da população = {avg_fitness}")

    # Plotar evolução
    plt.figure(figsize=(10, 6))
    plt.plot(avg_fitness_history, label="Média da população")
    plt.plot(best_fitness_history, label="Melhor solução")
    plt.xlabel("Geração")
    plt.ylabel("Distância")
    plt.legend()
    plt.show()

    # Plotar o melhor caminho
    plot_route(best_solution, cities, title="Melhor Caminho Encontrado")

# Executar o Algoritmo Genético
genetic_algorithm(time_matrix, cities, num_generations=100, pop_size=100)





