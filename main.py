import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from time import time
import numpy as np

# Carregar o dataset
dataset = pd.read_csv('movies.csv')

# Pré-processamento

# Remover coluna 'name', pois não é necessária
dataset.drop(columns=['name'], inplace=True)

# Remover linhas sem um Score
dataset.dropna(subset=['score'], inplace=True)

bins = [0, 5.0, 7.5, np.inf]
labels = ['Baixo', 'Médio', 'Alto']
#Adiciona coluna Classe de score
dataset['score_class'] = pd.cut(dataset['score'], bins=bins, labels=labels, right=False)

#Remoção da coluna score
dataset.drop(columns=['score'], inplace=True)

label_encoder = LabelEncoder()

dataset['rating'] = label_encoder.fit_transform(dataset['rating'])
dataset['released'] = label_encoder.fit_transform(dataset['released'])
dataset['genre'] = label_encoder.fit_transform(dataset['genre'])
dataset['director'] = label_encoder.fit_transform(dataset['director'])
dataset['writer'] = label_encoder.fit_transform(dataset['writer'])
dataset['star'] = label_encoder.fit_transform(dataset['star'])
dataset['country'] = label_encoder.fit_transform(dataset['country'])
dataset['company'] = label_encoder.fit_transform(dataset['company'])
dataset['score_class'] = label_encoder.fit_transform(dataset['score_class'])


#Remover linhas em que alguma cluna esteja vazia
dataset.dropna(inplace=True)

X = dataset.drop(columns=['score_class'])
y = dataset['score_class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True,  random_state=42)

def train_and_evaluate_model(model, X_train, X_test, y_train, y_test):
    start_time = time()
    
    model = model.fit(X_train, y_train)  
    
    end_time = time()
    training_time = end_time - start_time
    
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    print(f"Modelo: {model.__class__.__name__}")
    print(f"Tempo de Treinamento: {training_time:.4f} segundos")
    print(f"Acurácia: {accuracy:.4f}")
    print(f"Precisão: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F-Measure (F1-Score): {f1:.4f}")
    print("\nReporte de classificação:")
    print(classification_report(y_test, y_pred, target_names=labels))
    print("-" * 50)

   
# 1. Árvore de Decisão
dt_model = DecisionTreeClassifier(random_state=42)
dt_results = train_and_evaluate_model(dt_model, X_train, X_test, y_train, y_test)

# 2. Random Forest
rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
rf_results = train_and_evaluate_model(rf_model, X_train, X_test, y_train, y_test)

# 3. Support Vector (SVC)
svm_model = SVC(kernel='rbf', random_state=42)
svm_results = train_and_evaluate_model(svm_model, X_train, X_test, y_train, y_test)