from sklearn.metrics import recall_score, precision_score, accuracy_score

class Metricas:
    def calcular_sensibilidade(self, y_true, y_pred):
        return recall_score(y_true, y_pred)

    def calcular_especificidade(self, y_true, y_pred):
        tn, fp, fn, tp = self._confusion_matrix(y_true, y_pred)
        return tn / (tn + fp)

    def calcular_precisao(self, y_true, y_pred):
        return precision_score(y_true, y_pred)

    def _confusion_matrix(self, y_true, y_pred):
        from sklearn.metrics import confusion_matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return tn, fp, fn, tp