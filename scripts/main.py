from sklearn.model_selection import train_test_split

from scripts.processamento import ProcessamentoDNA
from scripts.modelo import ModeloML
from scripts.metricas import Metricas
from scripts.download_dados import Downloader

if __name__ == "__main__":
    # Baixar os dados
    downloader = Downloader("singhakash/dna-sequencing-with-machine-learning", "data")
    downloader.autenticar()
    downloader.baixar_dados()


    # Processar os dados
    processamento = ProcessamentoDNA("data/dataset.csv")
    dados = processamento.carregar_dados()
    dados = processamento.preprocessar(dados)

    # Separar variáveis de entrada e saída
    X = dados.iloc[:, :-1]  # Todas as colunas menos a última
    y = dados.iloc[:, -1]   # Última coluna

    # Dividir os dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Treinar o modelo
    modelo = ModeloML()
    modelo.treinar(X_train, y_train)

    # Testar o modelo
    y_pred = modelo.testar(X_test)

    # Calcular as métricas
    metricas = Metricas()
    sensibilidade = metricas.calcular_sensibilidade(y_test, y_pred)
    especificidade = metricas.calcular_especificidade(y_test, y_pred)
    precisao = metricas.calcular_precisao(y_test, y_pred)

    print(f"Sensibilidade: {sensibilidade}")
    print(f"Especificidade: {especificidade}")
    print(f"Precisão: {precisao}")