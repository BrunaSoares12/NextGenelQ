from sklearn.metrics import accuracy_score, precision_score, recall_score

class Metricas:
    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred

    def calcular_metricas(self):
        return {
            'Precisao': precision_score(self.y_true, self.y_pred, average='binary'),
            'Sensibilidade': recall_score(self.y_true, self.y_pred, average='binary'),
            'Acuracia': accuracy_score(self.y_true, self.y_pred)
        }