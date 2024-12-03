class ProcessamentoDNA:
    def __init__(self, caminho_arquivo):
        self.caminho_arquivo = caminho_arquivo

    def carregar_dados(self):
        import pandas as pd
        dados = pd.read_csv(self.caminho_arquivo)
        return dados

    def preprocessar(self, dados):
        # 1. Remover valores ausentes
        dados = dados.dropna()

        # 2. Selecionar apenas colunas relevantes (exemplo fictício)
        colunas_entrada = [col for col in dados.columns if
                           'SNP' in col]  # Supondo que as colunas de SNPs contêm 'SNP' no nome
        coluna_saida = 'Doenca'  # Nome fictício da variável de saída
        dados = dados[colunas_entrada + [coluna_saida]]

        # 3. Codificar variáveis categóricas (se houver)
        if dados[coluna_saida].dtype == 'object':
            dados[coluna_saida] = dados[coluna_saida].astype('category').cat.codes

        # 4. Normalizar os dados de entrada (opcional)
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        dados[colunas_entrada] = scaler.fit_transform(dados[colunas_entrada])

        return dados
