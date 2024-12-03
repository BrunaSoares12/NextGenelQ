from sklearn.ensemble import RandomForestClassifier

class ModeloML:
    def __init__(self):
        self.modelo = RandomForestClassifier()

    def treinar(self, X_train, y_train):
        self.modelo.fit(X_train, y_train)

    def testar(self, X_test):
        return self.modelo.predict(X_test)
