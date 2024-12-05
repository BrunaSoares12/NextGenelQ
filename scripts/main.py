from sklearn.model_selection import train_test_split
from scripts.processamento import ProcessamentoDNA
from scripts.modelo import ModeloML
from scripts.metricas import Metricas
from scripts.download_dados import Downloader
from sklearn.metrics import confusion_matrix
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

    # 7. Treinar o modelo Random Forest
    modelo_rf = ModeloML(algoritmo="random_forest")
    modelo_rf.treinar(X_train, y_train)
    y_pred_rf = modelo_rf.testar(X_test)

    # 8. Treinar o modelo K-Means
    modelo_kmeans = ModeloML(algoritmo="k_means")
    modelo_kmeans.treinar(X_train, y_train)
    y_pred_kmeans = modelo_kmeans.testar(X_test)

    # 9. Calcular as métricas para Random Forest
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    tp_rf, fn_rf, fp_rf, tn_rf = cm_rf.ravel()

    sensibilidade_rf = tp_rf / (tp_rf + fn_rf)
    especificidade_rf = tn_rf / (tn_rf + fp_rf)
    precisao_rf = tp_rf / (tp_rf + fp_rf)
    f1_rf = 2 * (precisao_rf * sensibilidade_rf) / (precisao_rf + sensibilidade_rf)

    print("\nMétricas para Random Forest:")
    print(f"Sensibilidade: {sensibilidade_rf:.2f}")
    print(f"Especificidade: {especificidade_rf:.2f}")
    print(f"Precisão: {precisao_rf:.2f}")
    print(f"F1-Score: {f1_rf:.2f}")

    # 10. Calcular as métricas para K-Means
    cm_kmeans = confusion_matrix(y_test, y_pred_kmeans)
    tp_km, fn_km, fp_km, tn_km = cm_kmeans.ravel()

    sensibilidade_km = tp_km / (tp_km + fn_km)
    especificidade_km = tn_km / (tn_km + fp_km)
    precisao_km = tp_km / (tp_km + fp_km)
    f1_km = 2 * (precisao_km * sensibilidade_km) / (precisao_km + sensibilidade_km)

    print("\nMétricas para K-Means:")
    print(f"Sensibilidade: {sensibilidade_km:.2f}")
    print(f"Especificidade: {especificidade_km:.2f}")
    print(f"Precisão: {precisao_km:.2f}")
    print(f"F1-Score: {f1_km:.2f}")

    # 11. Fazer uma nova previsão para ambos os modelos
    X_novo = pd.DataFrame([[5.1, 3.5, 1.4]], columns=X_train.columns)

    # Previsão com Random Forest
    previsao_rf = modelo_rf.modelo.predict(X_novo)
    probabilidades_rf = modelo_rf.modelo.predict_proba(X_novo)
    print("\nPrevisão Random Forest:")
    if previsao_rf[0] == 1:
        print("A amostra tem chance de ter a doença (Random Forest).")
    else:
        print("A amostra não tem chance de ter a doença (Random Forest).")
    print(f"Probabilidades (Random Forest): {probabilidades_rf}")

    # Previsão com K-Means
    previsao_kmeans = modelo_kmeans.modelo.predict(X_novo)
    print("\nPrevisão K-Means:")
    if previsao_kmeans[0] == 1:
        print("A amostra tem chance de ter a doença (K-Means).")
    else:
        print("A amostra não tem chance de ter a doença (K-Means).")