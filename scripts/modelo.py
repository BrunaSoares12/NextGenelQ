from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

class ModeloML:
    def __init__(self, algoritmo="random_forest"):
        """
        Inicializa o modelo com base no algoritmo escolhido.
        :param algoritmo: 'random_forest' ou 'rede_neural'
        """
        if algoritmo == "random_forest":
            self.modelo = RandomForestClassifier(n_estimators=100, random_state=42)
        elif algoritmo == "rede_neural":
            self.modelo = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
        else:
            raise ValueError("Algoritmo inválido. Escolha 'random_forest' ou 'rede_neural'.")

    def treinar(self, X_train, y_train):
        """
        Treina o modelo nos dados de treinamento.
        :param X_train: Dados de entrada para treino
        :param y_train: Labels correspondentes ao treino
        """
        self.modelo.fit(X_train, y_train)

    def testar(self, X_test):
        """
        Realiza previsões nos dados de teste.
        :param X_test: Dados de entrada para teste
        :return: Previsões do modelo
        """
        return self.modelo.predict(X_test)
