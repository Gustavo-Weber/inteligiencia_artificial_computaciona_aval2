import pandas as pd

dataset = pd.read_csv("movies.csv")

print("Quantidade de entradas: " + str(len(dataset)))

print("\n-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-\n")

print("Tipos de dados: \n" + str(dataset.dtypes))

print("\n-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-\n")

print("Descrição: \n" + str(dataset.describe()))

print("\n-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-\n")

# Verificar qual atributo é mais desbalanceado 

desbalanceamento = {}

for i in dataset.columns:
    if dataset[i].dtype == 'object':
        counts = dataset[i].value_counts()
        if len(counts) > 1:          
            proportion = counts.max() / counts.min()
            desbalanceamento[i] = proportion

atributo_mais_desbalanceado = max(desbalanceamento, key=desbalanceamento.get)

print("Atributo mais desbalanceado:", atributo_mais_desbalanceado)
print("Razão de desbalanceamento:", desbalanceamento[atributo_mais_desbalanceado])
print("Valor mais frequente:", dataset[atributo_mais_desbalanceado].mode()[0])

print("\n-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-\n")

print("Atributos com valores faltando: \n" + str(dataset.isna().sum()))