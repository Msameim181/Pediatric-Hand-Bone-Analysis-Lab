# Write a funtion to download the model from the internet using the wget module

import wget
import os

def download_model(model_name, model_url):
    if not os.path.exists(model_name):
        print('Downloading model...')
        wget.download(model_url)
        print('Download complete!')