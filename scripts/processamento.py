import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

class ProcessamentoDNA:
    def __init__(self, caminho_arquivo):
        self.caminho_arquivo = caminho_arquivo

    def carregar_dados(self):
        # Carrega o dataset com informações de DNA e doenças
        dados = pd.read_csv(self.caminho_arquivo)
        return dados

    def preprocessar(self, dados):
        # 1. Remover valores ausentes
        dados = dados.dropna()

        # 2. Codificar sequências de DNA (exemplo: one-hot encoding para 'A, T, G, C')
        colunas_entrada = [col for col in dados.columns if 'DNA' in col]
        coluna_saida = 'Doenca'

        # Codificação para variáveis categóricas
        if dados[coluna_saida].dtype == 'object':
            le = LabelEncoder()
            dados[coluna_saida] = le.fit_transform(dados[coluna_saida])

        ohe = OneHotEncoder()
        for col in colunas_entrada:
            one_hot = ohe.fit_transform(dados[[col]]).toarray()
            one_hot_cols = [f"{col}_{i}" for i in range(one_hot.shape[1])]
            dados = pd.concat([dados, pd.DataFrame(one_hot, columns=one_hot_cols)], axis=1)
            dados = dados.drop(columns=[col])

        # Retorna apenas o DataFrame processado
        return dados