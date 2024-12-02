import pandas as pd
from sklearn.model_selection import train_test_split

class ProcessamentoDNA:
    def __init__(self, caminho_arquivo):
        self.caminho_arquivo = caminho_arquivo

    def carregar_dados(self):
        dados = pd.read_csv(self.caminho_arquivo) # precisamos do caminho do arquivo
        dados_treinamento, dados_teste = train_test_split(dados, test_size=0.2, random_state=42)
        return dados_treinamento, dados_teste