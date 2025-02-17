import csv
import pandas as pd
rodar = 3
#rodar =  int(input("Digite 1 para converter , em . na planilha. 2 para recortar a planilha: "))
if rodar == 1:
    arquivo='costplane.csv'
    # Função para converter string com vírgula para float com ponto
    def convert_to_float(value):
        return float(value.replace(',', '.'))

    # Abrir o arquivo CSV para leitura
    with open(arquivo, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        data = list(reader)

    # Converter os valores de string para float
    for i in range(1, len(data)):
        for j in range(1, len(data[i])):
            data[i][j] = convert_to_float(data[i][j])

    # Abrir o arquivo CSV para escrita e salvar as alterações
    with open(arquivo, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)

    print("Arquivo atualizado com sucesso!")

if rodar == 2:

    # Nome dos arquivos
    arquivo_costplane = "50 Cities/Bus/timebus.csv"
    arquivo_xy = "10 Cities/xy10.csv"
    arquivo_modificado = "10 Cities/Bus/timebus10.csv"

    # Ler as cidades do arquivo xy.csv
    with open(arquivo_xy, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        cidades_xy = {row['City'] for row in reader}

    # Ler o arquivo costplane.csv
    with open(arquivo_costplane, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        data = list(reader)

    # Filtrar as linhas e colunas que não são cidades em xy.csv
    cidades_indices = [i for i, cidade in enumerate(data[0]) if cidade in cidades_xy]
    filtered_data = [[row[0]] + [row[i] for i in cidades_indices] for row in data if row[0] in cidades_xy or row == data[0]]

    # Salvar o arquivo filtrado com um nome diferente
    with open(arquivo_modificado, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(filtered_data)


    # Nome dos arquivos
    arquivo_costplane = "50 Cities/Bus/costbus.csv"
    arquivo_xy = "10 Cities/xy10.csv"
    arquivo_modificado = "10 Cities/Bus/costbus10.csv"

    # Ler as cidades do arquivo xy.csv
    with open(arquivo_xy, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        cidades_xy = {row['City'] for row in reader}

    # Ler o arquivo costplane.csv
    with open(arquivo_costplane, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        data = list(reader)

    # Filtrar as linhas e colunas que não são cidades em xy.csv
    cidades_indices = [i for i, cidade in enumerate(data[0]) if cidade in cidades_xy]
    filtered_data = [[row[0]] + [row[i] for i in cidades_indices] for row in data if row[0] in cidades_xy or row == data[0]]

    # Salvar o arquivo filtrado com um nome diferente
    with open(arquivo_modificado, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(filtered_data)


















    # Nome dos arquivos
    arquivo_costplane = "50 Cities/Plane/timeplane.csv"
    arquivo_xy = "10 Cities/xy10.csv"
    arquivo_modificado = "10 Cities/Plane/timeplane10.csv"

    # Ler as cidades do arquivo xy.csv
    with open(arquivo_xy, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        cidades_xy = {row['City'] for row in reader}

    # Ler o arquivo costplane.csv
    with open(arquivo_costplane, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        data = list(reader)

    # Filtrar as linhas e colunas que não são cidades em xy.csv
    cidades_indices = [i for i, cidade in enumerate(data[0]) if cidade in cidades_xy]
    filtered_data = [[row[0]] + [row[i] for i in cidades_indices] for row in data if row[0] in cidades_xy or row == data[0]]

    # Salvar o arquivo filtrado com um nome diferente
    with open(arquivo_modificado, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(filtered_data)


    # Nome dos arquivos
    arquivo_costplane = "50 Cities/Plane/costplane.csv"
    arquivo_xy = "10 Cities/xy10.csv"
    arquivo_modificado = "10 Cities/Plane/costplane10.csv"

    # Ler as cidades do arquivo xy.csv
    with open(arquivo_xy, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        cidades_xy = {row['City'] for row in reader}

    # Ler o arquivo costplane.csv
    with open(arquivo_costplane, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        data = list(reader)

    # Filtrar as linhas e colunas que não são cidades em xy.csv
    cidades_indices = [i for i, cidade in enumerate(data[0]) if cidade in cidades_xy]
    filtered_data = [[row[0]] + [row[i] for i in cidades_indices] for row in data if row[0] in cidades_xy or row == data[0]]

    # Salvar o arquivo filtrado com um nome diferente
    with open(arquivo_modificado, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(filtered_data)
























    # Nome dos arquivos
    arquivo_costplane = "50 Cities/Train/timetrain.csv"
    arquivo_xy = "10 Cities/xy10.csv"
    arquivo_modificado = "10 Cities/Train/timetrain10.csv"

    # Ler as cidades do arquivo xy.csv
    with open(arquivo_xy, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        cidades_xy = {row['City'] for row in reader}

    # Ler o arquivo costplane.csv
    with open(arquivo_costplane, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        data = list(reader)

    # Filtrar as linhas e colunas que não são cidades em xy.csv
    cidades_indices = [i for i, cidade in enumerate(data[0]) if cidade in cidades_xy]
    filtered_data = [[row[0]] + [row[i] for i in cidades_indices] for row in data if row[0] in cidades_xy or row == data[0]]

    # Salvar o arquivo filtrado com um nome diferente
    with open(arquivo_modificado, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(filtered_data)


    # Nome dos arquivos
    arquivo_costplane = "50 Cities/Train/costtrain.csv"
    arquivo_xy = "10 Cities/xy10.csv"
    arquivo_modificado = "10 Cities/Train/costtrain10.csv"

    # Ler as cidades do arquivo xy.csv
    with open(arquivo_xy, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        cidades_xy = {row['City'] for row in reader}

    # Ler o arquivo costplane.csv
    with open(arquivo_costplane, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        data = list(reader)

    # Filtrar as linhas e colunas que não são cidades em xy.csv
    cidades_indices = [i for i, cidade in enumerate(data[0]) if cidade in cidades_xy]
    filtered_data = [[row[0]] + [row[i] for i in cidades_indices] for row in data if row[0] in cidades_xy or row == data[0]]

    # Salvar o arquivo filtrado com um nome diferente
    with open(arquivo_modificado, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(filtered_data)

    
    print("Arquivo atualizado e salvo como", arquivo_modificado)

if rodar == 3:
    def reordenar_csv(nome_arquivo):
        # Ler o arquivo CSV
        df = pd.read_csv(nome_arquivo, index_col=0)

        # Pegar a primeira cidade em ordem alfabética
        cidades = list(df.columns)
        primeira_cidade = sorted(cidades)[0]

        # Reordenar a tabela
        df = df.reindex([primeira_cidade] + [cidade for cidade in cidades if cidade != primeira_cidade])
        df = df[[primeira_cidade] + [cidade for cidade in cidades if cidade != primeira_cidade]]

        # Salvar o CSV reordenado
        df.to_csv(nome_arquivo)
        print(nome_arquivo,"Feito")

    # Exemplo de uso
    reordenar_csv('10 Cities/Bus/costbus10.csv')
    reordenar_csv('10 Cities/Bus/timebus10.csv')

    reordenar_csv('10 Cities/Train/costtrain10.csv')
    reordenar_csv('10 Cities/Train/timetrain10.csv')

    reordenar_csv('10 Cities/Plane/costplane10.csv')
    reordenar_csv('10 Cities/Plane/timeplane10.csv')

    


    reordenar_csv('30 Cities/Bus/costbus30.csv')
    reordenar_csv('30 Cities/Bus/timebus30.csv')

    reordenar_csv('30 Cities/Train/costtrain30.csv')
    reordenar_csv('30 Cities/Train/timetrain30.csv')

    reordenar_csv('30 Cities/Plane/costplane30.csv')
    reordenar_csv('30 Cities/Plane/timeplane30.csv')

    


    reordenar_csv('50 Cities/Bus/costbus.csv')
    reordenar_csv('50 Cities/Bus/timebus.csv')

    reordenar_csv('50 Cities/Train/costtrain.csv')
    reordenar_csv('50 Cities/Train/timetrain.csv')

    reordenar_csv('50 Cities/Plane/costplane.csv')
    reordenar_csv('50 Cities/Plane/timeplane.csv')

    