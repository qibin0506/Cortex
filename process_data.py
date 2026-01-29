import subprocess
import orjson

from tqdm import tqdm
from utils import init_env
import glob
import re
from sklearn.utils import shuffle
from llm_trainer.dataset import *
from constant import *


def _extra_think_and_answer(text: str):
    match = re.search(r"<think>(.*?)</think>(.*)", text, re.DOTALL)
    # 提取 <think> 和 </think> 中间的内容 (第一个捕获组)
    think_data = match.group(1)
    # 提取 </think> 后面的内容 (第二个捕获组)
    content = match.group(2)
    if '<answer>' in content and '</answer>' in content:
        match = re.search(r"<answer>(.*?)</answer>(.*)", content, re.DOTALL)
        # 提取 <think> 和 </think> 中间的内容 (第一个捕获组)
        content = match.group(1)

    return think_data, content


def _sft_to_text(line):
    text = ''
    conversations = json.loads(line)['conversations']
    for conversation in conversations:
        content = conversation['content']
        if '<think>' in content:
            think_data, content = _extra_think_and_answer(content)
            text = f'{text}\n\n{think_data}\n\n{content}'
        else:
            text = f'{text}\n\n{content}'

    return text.strip()


def _get_file_line_count(file_path):
    with open(file_path, 'r') as f:
        return sum(1 for _ in f)


def _get_self_cognition(dtype):
    tokens = []
    with open('./data/raw/self_cognition.jsonl', 'r') as f:
        for line in f:
            json_ = json.loads(line)
            user = f"{json_['query']}"
            content = json_['response'].replace('{{AUTHOR}}', developer_name).replace('{{NAME}}', assistant_name)

            chat_template = [
                {'role': 'system', 'content': ' '},
                {'role': 'user', 'content': user},
                {'role': 'assistant', 'content': content.strip()}
            ]

            token = TrainerTools().tokenizer.apply_chat_template(chat_template, add_answer_tag_for_assistant=False)
            tokens.append(np.array(token, dtype=dtype))

    return tokens


def download_raw_dataset():
    from modelscope import dataset_snapshot_download
    dataset_snapshot_download(
        'gongjy/minimind_dataset',
        local_dir='./data/raw',
        allow_file_pattern='*.jsonl'
    )

    dataset_snapshot_download(
        'swift/self-cognition',
        local_dir='./data/raw',
        allow_file_pattern='*.jsonl'
    )


def shuffle_all_data():
    all_files = glob.glob("./data/raw/*.jsonl")
    for file in all_files:
        file_name = file.split('/')[-1]
        subprocess.run(f'terashuf < {file} > ./data/tmp/shuffle_{file_name}', shell=True, check=True)


def split_sft_2048():
    sft_2048_line_count = _get_file_line_count('./data/tmp/shuffle_sft_mini_512.jsonl')
    sft = []
    pretrain = []

    with open('./data/tmp/shuffle_sft_2048.jsonl', 'r') as f:
        for idx, line in enumerate(f):
            if idx <= sft_2048_line_count:
                sft.append(line)
            else:
                pretrain.append(line)

    with open('./data/tmp/shuffle_sft_mini_2048.jsonl', 'w') as f:
        f.writelines(sft)

    with open('./data/tmp/shuffle_pretrain_2048.jsonl', 'w') as f:
        f.writelines(pretrain)

    os.remove('./data/tmp/shuffle_sft_2048.jsonl')


def preprocess_pretrain_data():
    print('preprocess_pretrain_data')
    short_data = []
    long_data = []
    short_threshold = 768

    with open('./data/tmp/shuffle_pretrain_hq.jsonl', 'r') as f:
        for line in f:
            text = json.loads(line)['text'].replace('<|im_end|>', '')
            if len(text) <= short_threshold:
                short_data.append(f"{json.dumps({'text': text}, ensure_ascii=False)}\n")
            else:
                long_data.append(f"{json.dumps({'text': text}, ensure_ascii=False)}\n")

    os.remove('./data/tmp/shuffle_pretrain_hq.jsonl')

    sft_files = ['shuffle_pretrain_2048.jsonl', 'shuffle_sft_512.jsonl', 'shuffle_sft_1024.jsonl', 'shuffle_r1_mix_1024.jsonl']

    for file_idx, file in enumerate(sft_files):
        with open(f'./data/tmp/{file}', 'r') as f:
            for line in f:
                text = _sft_to_text(line)
                text = text.replace('MiniMind-R1', assistant_name).replace('MiniMind', assistant_name)
                if len(text) <= short_threshold:
                    short_data.append(f"{json.dumps({'text': text}, ensure_ascii=False)}\n")
                else:
                    long_data.append(f"{json.dumps({'text': text}, ensure_ascii=False)}\n")

        os.remove(f'./data/tmp/{file}')

    with open('./data/tmp/pretrain_data_short.jsonl', 'w') as f:
        f.writelines(short_data)
    del short_data

    with open('./data/tmp/pretrain_data_long.jsonl', 'w') as f:
        f.writelines(long_data)
    del long_data


def shuffle_pretrain_data():
    print('shuffle_data')
    pretrain_files = ['pretrain_data_short.jsonl', 'pretrain_data_long.jsonl']
    for file in pretrain_files:
        subprocess.run(f'terashuf < ./data/tmp/{file} > ./data/tmp/shuffle_{file}', shell=True, check=True)

    for file in pretrain_files:
        os.remove(f'./data/tmp/{file}')


def encode_pretrain_data():
    print('encode_pretrain_data')
    input_file = './data/tmp/shuffle_pretrain_data_short.jsonl'

    total_lines = _get_file_line_count(input_file)
    print(f"Total lines: {total_lines}")

    vocab_size = TrainerTools().tokenizer.vocab_size
    dtype = np.uint16 if vocab_size < 65535 else np.uint32

    batch_size = 50000
    text_buffer = []

    def process_and_save_stream(input_path, output_prefix, split_count=1):
        file_lines = _get_file_line_count(input_path)
        lines_per_chunk = file_lines // split_count if split_count > 0 else file_lines

        current_chunk_idx = 0
        current_token_count = 0

        temp_bin_path = f"{output_prefix}_{current_chunk_idx}.bin"
        bin_file = open(temp_bin_path, "wb")

        print(f"Processing {input_path}...")

        with open(input_path, 'r') as f:
            for idx, line in tqdm(enumerate(f), total=file_lines):
                text = f"{orjson.loads(line)['text'].strip()}</s>"
                text_buffer.append(text)

                if len(text_buffer) >= batch_size:
                    batch_encodings = TrainerTools().tokenizer.batch_encode(text_buffer)

                    for input_ids in batch_encodings:
                        arr = np.array(input_ids, dtype=dtype)
                        bin_file.write(arr.tobytes())
                        current_token_count += len(arr)

                    text_buffer.clear()

                is_chunk_boundary = (split_count > 1 and
                                     current_chunk_idx < split_count - 1 and
                                     (idx + 1) % lines_per_chunk == 0)

                if is_chunk_boundary:
                    if text_buffer:
                        batch_encodings = TrainerTools().tokenizer.batch_encode(text_buffer)
                        for input_ids in batch_encodings:
                            arr = np.array(input_ids, dtype=dtype)
                            bin_file.write(arr.tobytes())
                            current_token_count += len(arr)
                        text_buffer.clear()

                    bin_file.close()
                    print(f"Finished raw chunk {current_chunk_idx}, tokens: {current_token_count}")

                    convert_bin_to_npy(temp_bin_path, f"{output_prefix}_{current_chunk_idx}.npy", dtype,
                                       current_token_count)
                    os.remove(temp_bin_path)

                    current_chunk_idx += 1
                    current_token_count = 0
                    temp_bin_path = f"{output_prefix}_{current_chunk_idx}.bin"
                    bin_file = open(temp_bin_path, "wb")

        if text_buffer:
            batch_encodings = TrainerTools().tokenizer.batch_encode(text_buffer)
            for input_ids in batch_encodings:
                arr = np.array(input_ids, dtype=dtype)
                bin_file.write(arr.tobytes())
                current_token_count += len(arr)
            text_buffer.clear()

        bin_file.close()

        if current_token_count > 0:
            print(f"Finished final chunk {current_chunk_idx}, tokens: {current_token_count}")
            convert_bin_to_npy(temp_bin_path, f"{output_prefix}_{current_chunk_idx}.npy", dtype, current_token_count)
            os.remove(temp_bin_path)

    def convert_bin_to_npy(bin_path, npy_path, dtype, shape):
        import shutil

        with open(bin_path, 'rb') as f_bin:
            with open(npy_path, 'wb') as f_npy:
                header_data = {
                    'descr': np.dtype(dtype).str,
                    'fortran_order': False,
                    'shape': (shape,),
                }

                np.lib.format.write_array_header_1_0(f_npy, header_data)
                shutil.copyfileobj(f_bin, f_npy)

    process_and_save_stream(
        './data/tmp/shuffle_pretrain_data_short.jsonl',
        './data/pretrain_data',
        split_count=2
    )

    process_and_save_stream(
        './data/tmp/shuffle_pretrain_data_long.jsonl',
        './data/midtrain_data',
        split_count=1
    )


def merge_sft_data():
    print('merge_sft_data')
    input_files = ['shuffle_sft_mini_2048.jsonl', 'shuffle_sft_mini_512.jsonl']
    lines = []

    for input_file in input_files:
        with open(f'./data/tmp/{input_file}', 'r') as f:
            for line in f:
                lines.append(line)

    with open('./data/tmp/sft_data.jsonl', 'w') as f:
        f.writelines(lines)


def encode_sft_data():
    print('encode_sft_data')

    tokenizer = TrainerTools().tokenizer
    vocab_size = tokenizer.vocab_size
    dtype = np.uint16 if vocab_size < 65535 else np.uint32
    batch_size = 10000
    tokens = []
    text_buffer = []

    self_cognition = _get_self_cognition(dtype)

    input_path = './data/tmp/sft_data.jsonl'
    total_lines = _get_file_line_count(input_path)

    with open(input_path, 'r') as f:
        for line in tqdm(f, total=total_lines):
            conversations = orjson.loads(line)['conversations']

            chat_template = [{'role': 'system', 'content': ' '}]
            for item in conversations:
                chat_template.append({
                    'role': item['role'],
                    'content': item['content'].strip()
                })

            formatted_text = tokenizer.apply_chat_template(
                chat_template,
                tokenizer=False,
                add_answer_tag_for_assistant=False
            )
            text_buffer.append(formatted_text)

            if len(text_buffer) >= batch_size:
                batch_encodings = tokenizer.batch_encode(text_buffer)

                for token_ids in batch_encodings:
                    if len(token_ids) <= 2048:
                        tokens.append(np.array(token_ids, dtype=dtype))

                text_buffer.clear()

    if text_buffer:
        batch_encodings = tokenizer.batch_encode(text_buffer)
        for token_ids in batch_encodings:
            if len(token_ids) <= 2048:
                tokens.append(np.array(token_ids, dtype=dtype))
        text_buffer.clear()

    tokens.extend(self_cognition * 20)

    print(f"Shuffling {len(tokens)} samples...")
    tokens = shuffle(tokens)

    token_array = np.array(tokens, dtype=object)
    output_path = f'./data/sft_data.npy'
    print(f"Saving to {output_path}...")
    np.save(output_path, token_array)


def encode_ppo_data():
    print('encode_ppo_data')

    tokens = []
    with open('./data/raw/rlaif-mini.jsonl', 'r') as f:
        for line in f:
            user_content = json.loads(line)['conversations'][0]['content']
            chat_template = [
                {'role': 'system', 'content': ' '},
                {'role': 'user', 'content': user_content.strip()}
            ]

            item = TrainerTools().tokenizer.apply_chat_template(chat_template, tokenizer=False)
            tokens.append({'prompt': TrainerTools().tokenizer.encode(f'{item}<assistant>')})

    tokens = shuffle(tokens)

    token_array = np.array(tokens, dtype=object)
    np.save(f'./data/ppo_data.npy', token_array)

if __name__ == '__main__':
    init_env()
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    os.makedirs('./data/raw/', exist_ok=True)
    os.makedirs('./data/tmp/', exist_ok=True)

    download_raw_dataset()
    shuffle_all_data()
    split_sft_2048()
    preprocess_pretrain_data()
    shuffle_pretrain_data()
    encode_pretrain_data()
    merge_sft_data()
    encode_sft_data()
    encode_ppo_data()

