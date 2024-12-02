from scripts.modelo import ModeloML
from scripts.processamento import ProcessamentoDNA
from scripts.metricas import Metricas
from utils.download_dados import download_dados


def main():
    # Baixar os dados
    download_dados()

    # Processar os dados
    processador = ProcessamentoDNA('data/dataset.csv')
    dados_treinamento, dados_teste = processador.carregar_dados()

    # Treinamento do modelo
    modelo = ModeloML()
    modelo.treinar(dados_treinamento)

    # Avaliar o modelo
    predições = modelo.prever(dados_teste)
    metricas = Metricas(dados_teste, predições)
    resultado_metricas = metricas.calcular_metricas()

    print("Resultados das Métricas:", resultado_metricas)


if __name__ == "__main__":
    main()

    # Explicação para bruno e bibi
    # dataset.csv: É o arquivo de dados.Para um teste inicial, criei um arquivo CSV com colunas e dados fictícios.
    # main.py: O script principal que orquestra o processo de download, processamento, treinamento e avaliação.
    # metricas.py: Tem a classe Metricas para calcular precisão, sensibilidade e acurácia.
    # modelo.py: Define a classe ModeloML com funções para treinar e prever.
    # processamento.py: Carrega e divide o dataset.
    # download_dados.py: Baixa o arquivo CSV do Kaggle para a pasta data/.
    # Para baixar as dependencias, é só clicar em cima de cada import em uma lupa vermelha e instalar o que pedir (fazer isso para todos os arquivos menos o CSV)