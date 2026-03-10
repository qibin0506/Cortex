import os

developer_name = 'QB'
assistant_name = 'Cortex'

image_size = 224
patch_size = 16
tokens_per_image = 196

def data_root_dir():
    dir_name = './data/'
    os.makedirs(dir_name, exist_ok=True)
    return dir_name
