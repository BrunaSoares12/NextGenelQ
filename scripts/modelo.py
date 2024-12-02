from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd

class ModeloML:
    def __init__(self):
        self.modelo = LogisticRegression()

    def treinar(self, dados_treinamento):
        X = dados_treinamento.iloc[:, :-1]  # Features
        y = dados_treinamento.iloc[:, -1]   # Target
        self.modelo.fit(X, y)

    def prever(self, dados_teste):
        X_teste = dados_teste.iloc[:, :-1]  # Features
        return self.modelo.predict(X_teste)