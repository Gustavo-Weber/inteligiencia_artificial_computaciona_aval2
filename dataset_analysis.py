import pandas as pd
import numpy as np

dataset = pd.read_csv("movies.csv")

#Adicionar a colunea score_class para armazenar a classificação do score (Baixo, Médio ou Alto)
bins = [0, 5.0, 7.5, np.inf]
labels = ['Baixo', 'Médio', 'Alto']
#Adiciona coluna Classe de score
dataset['score_class'] = pd.cut(dataset['score'], bins=bins, labels=labels, right=False)

print("Quantidade de entradas: " + str(len(dataset)))

print("\n-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-\n")

print("Tipos de dados: \n" + str(dataset.dtypes))

print("\n-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-\n")

print("Descrição: \n" + str(dataset.describe()))

print("\n-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-\n")

# Verificar qual atributo é mais desbalanceado 

desbalanceamento = {}

for i in dataset.columns:
    counts = dataset[i].value_counts()
    if len(counts) > 1:          
        proportion = counts.max() / counts.min()
        desbalanceamento[i] = proportion

atributo_mais_desbalanceado = max(desbalanceamento, key=desbalanceamento.get)

print(f"Atributo mais desbalanceado: {atributo_mais_desbalanceado}")
print(f"Razão de desbalanceamento: {desbalanceamento[atributo_mais_desbalanceado]}")
print(f"Valor mais frequente: {dataset[atributo_mais_desbalanceado].mode()[0]}")

print("\nProporção de valores:")
print(dataset[atributo_mais_desbalanceado].value_counts())

print("\n-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-\n")

print("Atributos com valores faltando: \n" + str(dataset.isna().sum()))