import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from math import radians, sin, cos, sqrt, atan2


# Lê o arquivo xy.csv
data = pd.read_csv('xy.csv')






def rodar(o_que_fazer):
    if o_que_fazer == 1:
        

        # Configura o gráfico com o mapa da Europa
        plt.figure(figsize=(10, 8))

        # Cria o mapa da Europa com projeção 'mercator'
        m = Basemap(projection='merc', llcrnrlat=35, urcrnrlat=70, llcrnrlon=-25, urcrnrlon=45, resolution='i')

        # Desenha os contornos do mapa
        m.drawcoastlines()
        m.drawcountries()
        m.fillcontinents(color='lightgray', lake_color='aqua')

        # Configura a grade e os rótulos nos eixos
        m.drawparallels(range(30, 75, 5), labels=[1,0,0,0], linewidth=0.2)  # Linhas de latitude
        m.drawmeridians(range(-30, 50, 5), labels=[0,0,0,1], linewidth=0.2)  # Linhas de longitude

        # Pega as coordenadas e plota no mapa
        for index, row in data.iterrows():
            lat = row['Latitude']
            lon = row['Longitude']
            x, y = m(lon, lat)
            plt.plot(x, y, 'ro', markersize=5)  # Plota os pontos das cidades como círculos vermelhos

        # Título e exibição do gráfico
        plt.title('Cidades Selecionadas na Europa com Latitude e Longitude')
        plt.show()


    if o_que_fazer == 2: 
        # Função para calcular a distância usando a fórmula de Haversine
        def haversine(lat1, lon1, lat2, lon2):
            R = 6371.0  # Raio da Terra em km
            # Converte latitude e longitude de graus para radianos
            lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
            
            # Diferença entre as coordenadas
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            
            # Fórmula de Haversine
            a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
            c = 2 * atan2(sqrt(a), sqrt(1 - a))
            
            distance = R * c
            return distance



        # Inicializa uma tabela de distâncias vazia
        n_cities = len(data)
        distance_matrix = np.zeros((n_cities, n_cities))

        # Calcula a distância entre todas as cidades
        for i in range(n_cities):
            for j in range(n_cities):
                if i != j:
                    lat1, lon1 = data.iloc[i]['Latitude'], data.iloc[i]['Longitude']
                    lat2, lon2 = data.iloc[j]['Latitude'], data.iloc[j]['Longitude']
                    distance_matrix[i, j] = round(haversine(lat1, lon1, lat2, lon2),2)
                else:
                    distance_matrix[i, j] = 0  # Distância de uma cidade para si mesma é 0

        # Cria um DataFrame com a matriz de distâncias
        distance_df = pd.DataFrame(distance_matrix, index=data['City'], columns=data['City'])

        # Salva o arquivo distance.csv
        distance_df.to_csv('distance.csv')

        print("Arquivo distance.csv gerado com sucesso!")



rodar(int(input("O que fazer? 1-Mapa Europa, 2-Planilha Distância: ")))