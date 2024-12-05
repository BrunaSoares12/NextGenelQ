from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from scripts.processamento import ProcessamentoDNA
from scripts.modelo import ModeloML
from scripts.metricas import Metricas
from scripts.download_dados import Downloader
import pandas as pd
import numpy as np

if __name__ == "__main__":
    # 1. Baixar os dados
    downloader = Downloader("mukund23/predict-the-genetic-disorder", "data")
    downloader.autenticar()
    downloader.baixar_dados()

    # 2. Carregar e processar os dados
    processamento = ProcessamentoDNA("data/dataset.csv")
    dados = processamento.carregar_dados()
    dados = processamento.preprocessar(dados)

    # 3. Separar variáveis de entrada (X) e saída (y)
    X = dados.iloc[:, :-1]  # Todas as colunas menos a última (entradas)
    y = dados.iloc[:, -1]  # Última coluna (rótulo de saída)

    # 4. Dividir os dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    print("Forma de X_train:", X_train.shape)
    print("Forma de y_train:", y_train.shape)

    # 5. Reconfigurar os índices dos dados de treino
    X_train.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)

    # 6. Garantir que y_train seja uma série, se for um DataFrame de uma coluna
    y_train = y_train.squeeze()

    # RANDOM FOREST
    # 7. Treinar o modelo Random Forest
    modelo_rf = ModeloML(algoritmo="random_forest")
    modelo_rf.treinar(X_train, y_train)

    # 8. Testar o modelo Random Forest
    y_pred_rf = modelo_rf.testar(X_test)

    # 9. Calcular e exibir as métricas para Random Forest
    metricas = Metricas()
    sensibilidade_rf = metricas.calcular_sensibilidade(y_test, y_pred_rf)
    especificidade_rf = metricas.calcular_especificidade(y_test, y_pred_rf)
    precisao_rf = metricas.calcular_precisao(y_test, y_pred_rf)

    print("\nMétricas para Random Forest:")
    print(f"Sensibilidade: {sensibilidade_rf:.2f}")
    print(f"Especificidade: {especificidade_rf:.2f}")
    print(f"Precisão: {precisao_rf:.2f}")

    # 10. Fazer uma nova previsão com Random Forest
    X_novo = pd.DataFrame([[5.1, 3.5, 1.4]], columns=X_train.columns)
    previsao_rf = modelo_rf.modelo.predict(X_novo)
    probabilidades_rf = modelo_rf.modelo.predict_proba(X_novo)

    print("\nPrevisão Random Forest:")
    if previsao_rf[0] == 1:
        print("A amostra tem chance de ter a doença (Random Forest).")
    else:
        print("A amostra não tem chance de ter a doença (Random Forest).")
    print(f"Probabilidades (Random Forest): {probabilidades_rf}")

    # K-MEANS
    # 11. Treinar o modelo K-Means
    n_clusters = len(y_train.unique())  # Número de clusters com base nos rótulos
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X_train)

    # 12. Mapear os clusters para os rótulos reais
    clusters_train = kmeans.predict(X_train)
    cluster_to_label = {}
    for cluster in range(n_clusters):
        mask = clusters_train == cluster
        cluster_to_label[cluster] = y_train[mask].mode()[0]

    # 13. Prever no conjunto de teste usando K-Means
    clusters_test = kmeans.predict(X_test)
    y_pred_kmeans = np.array([cluster_to_label[cluster] for cluster in clusters_test])

    # 14. Calcular e exibir as métricas para K-Means
    sensibilidade_kmeans = metricas.calcular_sensibilidade(y_test, y_pred_kmeans)
    especificidade_kmeans = metricas.calcular_especificidade(y_test, y_pred_kmeans)
    precisao_kmeans = metricas.calcular_precisao(y_test, y_pred_kmeans)

    print("\nMétricas para K-Means:")
    print(f"Sensibilidade: {sensibilidade_kmeans:.2f}")
    print(f"Especificidade: {especificidade_kmeans:.2f}")
    print(f"Precisão: {precisao_kmeans:.2f}")

    # 15. Fazer uma nova previsão com K-Means incluindo pseudo-probabilidades
    distancias_novo = kmeans.transform(X_novo)[0]  # Distâncias aos centros
    total_distancias = sum(distancias_novo)
    pseudo_probabilidades = 1 - (distancias_novo / total_distancias)  # Inverter para aproximar probabilidade

    pseudo_probabilidades /= pseudo_probabilidades.sum()

    # Determinar a previsão com base no cluster mais próximo
    cluster_novo = np.argmin(distancias_novo)
    previsao_kmeans = cluster_to_label[cluster_novo]

    print("\nPrevisão K-Means:")
    if previsao_kmeans == 1:
        print("A amostra tem chance de ter a doença (K-Means).")
    else:
        print("A amostra não tem chance de ter a doença (K-Means).")

    print(f"Pseudo-Probabilidades (K-Means): {pseudo_probabilidades}")
