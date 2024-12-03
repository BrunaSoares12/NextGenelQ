import os
import kaggle

class Downloader:
    def __init__(self, dataset, pasta_destino):
        self.dataset = dataset
        self.pasta_destino = pasta_destino

    def autenticar(self):
        kaggle.api.authenticate()

    def baixar_dados(self):
        if not os.path.exists(self.pasta_destino):
            os.makedirs(self.pasta_destino)
        kaggle.api.dataset_download_files(self.dataset, path=self.pasta_destino, unzip=True)