import os
from kaggle.api.kaggle_api_extended import KaggleApi


def download_dados():
    api = KaggleApi()
    api.authenticate()
    dataset_path = 'data/dataset.csv'

    if not os.path.exists(dataset_path):
        print("Baixando dataset...")
        api.dataset_download_file('nome-do-usuario/nome-do-dataset', file_name='dataset.csv', path='data/')
        print("Download completo.")
    else:
        print("O arquivo já existe.")