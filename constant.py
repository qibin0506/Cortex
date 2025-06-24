developer_name = 'QB'
assistant_name = 'Cortex'

system_prompt_content = f"""你是{assistant_name}，由{developer_name}训练和开发。你的目标是理解用户的需求，并提供准确、相关、清晰且有价值的信息和帮助"""

image_size = 224
patch_size = 16
tokens_per_image = 196

data_root_dir = './data/'
max_seq_len = 1024