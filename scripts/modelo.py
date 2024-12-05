from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans

class ModeloML:
    def __init__(self, algoritmo="random_forest"):
        if algoritmo == "random_forest":
            self.modelo = RandomForestClassifier()
        elif algoritmo == "rede_neural":
            self.modelo = MLPClassifier(max_iter=1000)
        elif algoritmo == "k_means":
            self.modelo = KMeans(n_clusters=2, random_state=42)
        else:
            raise ValueError("Algoritmo inválido. Escolha 'random_forest', 'rede_neural' ou 'k_means'.")

    def treinar(self, X, y=None):
        """
        Treina o modelo. Para K-Means, 'y' não é necessário.
        """
        if hasattr(self.modelo, 'fit_predict'):
            # Para K-Means
            self.modelo.fit(X)
        else:
            self.modelo.fit(X, y)

    def testar(self, X):
        """
        Realiza previsões. Para K-Means, retorna os clusters.
        """
        if hasattr(self.modelo, 'predict'):
            return self.modelo.predict(X)
        else:
            return self.modelo.labels_
