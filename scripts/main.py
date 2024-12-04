# Classe main ajustada
from sklearn.model_selection import train_test_split
from scripts.processamento import ProcessamentoDNA
from scripts.modelo import ModeloML
from scripts.metricas import Metricas
from scripts.download_dados import Downloader
import pandas as pd

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

    # 7. Treinar o modelo
    modelo = ModeloML(algoritmo="random_forest")
    modelo.treinar(X_train, y_train)

    # 8. Testar o modelo
    y_pred = modelo.testar(X_test)

    # 9. Calcular e exibir as métricas
    metricas = Metricas()
    sensibilidade = metricas.calcular_sensibilidade(y_test, y_pred)
    especificidade = metricas.calcular_especificidade(y_test, y_pred)
    precisao = metricas.calcular_precisao(y_test, y_pred)

    print(f"Sensibilidade: {sensibilidade:.2f}")
    print(f"Especificidade: {especificidade:.2f}")
    print(f"Precisão: {precisao:.2f}")

    # 10. Fazer uma nova previsão
    # Suponha que você tenha uma amostra de entrada (X_novo) com as mesmas características de entrada usadas no treinamento
    X_novo = pd.DataFrame([[5.1, 3.5, 1.4]], columns=X_train.columns)

    # Fazer a previsão
    previsao = modelo.modelo.predict(X_novo)
    probabilidades = modelo.modelo.predict_proba(X_novo)

    # Exibir o resultado da previsão
    if previsao[0] == 1:
        print("A amostra tem chance de ter a doença.")
    else:
        print("A amostra não tem chance de ter a doença.")

    # Exibir a probabilidade de cada classe
    print(f"Probabilidades: {probabilidades}")
