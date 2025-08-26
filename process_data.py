import os.path
import re
import json

from sklearn.utils import shuffle
from llm_trainer import TrainerTools
from constant import *
import pickle
import random
import pandas as pd

def _init():
    from utils import init_env
    init_env()
    os.environ["TOKENIZERS_PARALLELISM"] = "true"


def _remove_urls(text: str):
    url_pattern = re.compile(r'(?:(?:https?|ftp):\/\/)?[\w\/\-?=%.]+\.[\w\/\-&?=%.]+')
    return url_pattern.sub('', text)


def _remove_brackets(text: str):
    return (text.replace('[]', '')
     .replace('{}', '')
     .replace('()', '')
     .replace('<>', '')
     .replace('【】', '')
     .replace('《》', '')
     .replace('（）', '')
     .replace('（，）', '')
     # .replace('\"\"', '')
     # .replace("\'\'", '')
     )


def _filter_content(content: str) -> str:
    content = _remove_brackets(_remove_urls(content))
    content = (content.replace("{{assistant_name}}", assistant_name)
               .replace('Qwen', assistant_name)
               .replace('qwen', assistant_name)
               .replace('DeepSeek', assistant_name)
               .replace('deepseek', assistant_name)
               .replace('ChatGPT', assistant_name)
               .replace('chatgpt', assistant_name)
               .replace('阿里巴巴', developer_name)
               .replace('OpenAI', developer_name)
               .replace('openai', developer_name))
    return content


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


# download from http://share.mobvoi.com:5000/sharing/O91blwPkY
def split_mobvoi():
    short_threshold = 3072

    with open('./data/raw/mobvoi_seq_monkey_general_open_corpus.jsonl', 'r') as f:
        for line in f:
            text = json.loads(line)['text']
            # item = TrainerTools().tokenizer.encode(text.strip())

            if len(text) <= short_threshold:
                with open('./data/raw/mobvoi_seq_monkey_short.jsonl', 'a') as f:
                    f.write(line)
            else:
                with open('./data/raw/mobvoi_seq_monkey_long.jsonl', 'a') as f:
                    f.write(line)


def encode_mobvoi_short():
    encoded = []
    cur_len = 0
    suffix = 0

    with open('./data/raw/mobvoi_seq_monkey_short.jsonl', 'r') as f:
        for line in f:
            item = TrainerTools().tokenizer.encode(f"{json.loads(line)['text']}</s>")
            item_len = len(item)

            encoded.extend(item)
            cur_len += item_len

            if cur_len >= 8e8:
                with open(f'./data/mobvoi_seq_monkey_short_{suffix}.pkl', 'wb') as short:
                    pickle.dump(encoded, short)

                encoded.clear()
                cur_len = 0
                suffix += 1

    with open(f'./data/mobvoi_seq_monkey_short_{suffix}.pkl', 'wb') as short:
        pickle.dump(encoded, short)


def encode_mobvoi_long():
    encoded = []
    cur_len = 0
    suffix = 0

    with open('./data/raw/mobvoi_seq_monkey_long.jsonl', 'r') as f:
        for line in f:
            item = TrainerTools().tokenizer.encode(f"{json.loads(line)['text']}</s>")
            item_len = len(item)

            encoded.extend(item)
            cur_len += item_len

            if cur_len >= 8e8:
                with open(f'./data/mobvoi_seq_monkey_long_{suffix}.pkl', 'wb') as l:
                    pickle.dump(encoded, l)

                encoded.clear()
                cur_len = 0
                suffix += 1

    with open(f'./data/mobvoi_seq_monkey_long_{suffix}.pkl', 'wb') as l:
        pickle.dump(encoded, l)


def encode_wikipedia():
    encoded = []

    with open('./data/raw/wikipedia-cn-20230720-filtered.json', 'r') as f:
        json_ = json.loads(f.read())
        for item in json_:
            item = TrainerTools().tokenizer.encode(f"{item['completion']}</s>")
            encoded.extend(item)

    with open(f'./data/wikipedia.pkl', 'wb') as f:
        pickle.dump(encoded, f)


# download from https://modelscope.cn/datasets/liucong/Chinese-DeepSeek-R1-Distill-data-110k-SFT
# and https://huggingface.co/datasets/shareAI/Alpaca-Distill-R1-ZH/
def preprocess_cot_data():
    # system_prompt = random.choice(GENERAL_SYSTEM_PROMPTS)
    result = []

    print('encode distill_r1_110k_sft')
    with open('./data/raw/distill_r1_110k_sft.jsonl', 'r') as f:
        for line in f:
            json_ = json.loads(line)
            user = json_['instruction']
            output = json_['output']

            think, content = _extra_think_and_answer(output)
            think = _filter_content(think)
            content = _filter_content(content)

            chat_template = [
                {'role': 'system', 'content': random.choice(GENERAL_SYSTEM_PROMPTS)},
                {'role': 'user', 'content': user.strip()},
                {'role': 'assistant', 'think': think.strip(), 'content': content.strip()}
            ]

            encoded = TrainerTools().tokenizer.apply_chat_template(chat_template)
            if len(encoded) > 2048:
                continue

            result.append(encoded)

    print('encode alpaca_r1_data_zh-localpost')
    with open('./data/raw/alpaca_r1_data_zh-localpost.json', 'r') as f:
        json_ = json.loads(f.read())
        for line in json_:
            user = line['instruction']
            output = line['output']

            think, content = _extra_think_and_answer(output)
            think = _filter_content(think)
            content = _filter_content(content)

            chat_template = [
                {'role': 'system', 'content': random.choice(GENERAL_SYSTEM_PROMPTS)},
                {'role': 'user', 'content': user},
                {'role': 'assistant', 'think': think.strip(), 'content': content.strip()}
            ]

            encoded = TrainerTools().tokenizer.apply_chat_template(chat_template)
            if len(encoded) > 2048:
                continue

            result.append(encoded)

    result = shuffle(result)

    print('dump')
    with open('./data/cot_sft.pkl', 'wb') as f:
        pickle.dump(result, f)


# download from https://huggingface.co/datasets/swulling/gsm8k_chinese
def preprocess_grpo_data():
    qas = []
    for file_name in ['train-00000-of-00001.parquet', 'test-00000-of-00001.parquet']:
        df = pd.read_parquet(f"./data/raw/gsm8k_chinese/{file_name}", engine="pyarrow")
        for q, a in zip(df['question_zh-cn'].values, df['answer_only'].values):
            q_template = [
                {'role': 'system', 'content': random.choice(GENERAL_SYSTEM_PROMPTS)},
                {'role': 'user', 'content': f'{str(q)}'}
            ]

            prompt = TrainerTools().tokenizer.apply_chat_template(q_template)
            if len(prompt) > 2048:
                continue

            qas.append({
                'prompt': prompt,
                'answer': TrainerTools().tokenizer.encode(str(a))
            })

        qas = shuffle(qas)
        with open(f'./data/grpo.pkl', 'wb') as f:
            pickle.dump(qas, f)


# download from https://www.modelscope.cn/datasets/swift/self-cognition
# and https://www.modelscope.cn/datasets/gongjy/minimind_dataset/file/view/master/r1_mix_1024.jsonl?id=68909&status=2
def preprocess_mix_data():
    # 添加自我认知数据集
    # 加入/think 和 /no think
    result = []

    with open('./data/raw/self_cognition.jsonl', 'r') as f:
        for line in f:
            json_ = json.loads(line)
            user = f"{json_['query']} /no think"
            content = json_['response'].replace('{{AUTHOR}}', developer_name).replace('{{NAME}}', assistant_name)

            chat_template = [
                {'role': 'system', 'content': random.choice(GENERAL_SYSTEM_PROMPTS)},
                {'role': 'user', 'content': user},
                {'role': 'assistant', 'think': ' ', 'content': f"{content.strip()}"}
            ]

            encoded = TrainerTools().tokenizer.apply_chat_template(chat_template)
            result.append(encoded)

    with open('./data/raw/r1_mix_1024.jsonl', 'r') as f:
        for line in f:
            json_ = json.loads(line)
            conversations = json_['conversations']

            chat_template = [{'role': 'system', 'content': random.choice(GENERAL_SYSTEM_PROMPTS)}]
            for conversation in conversations:
                if conversation['role'] == 'user':
                    chat_template.append({'role': 'user', 'content': conversation['content'].strip()})
                elif conversation['role'] == 'assistant':
                    if 'think' in conversation['content']:
                        chat_template[-1]['content'] = f"{chat_template[-1]['content']} /think"
                        chat_template.append({'role': 'assistant', 'content': _filter_content(conversation['content'].strip())})
                    else:
                        chat_template[-1]['content'] = f"{chat_template[-1]['content']} /no think"
                        chat_template.append({'role': 'assistant', 'think': ' ', 'content': f"<answer>{_filter_content(conversation['content'].strip())}</answer>"})

            encoded = TrainerTools().tokenizer.apply_chat_template(chat_template, add_answer_tag_for_assistant=False)
            if len(encoded) > 2048:
                continue

            result.append(encoded)

        result = shuffle(result)
        print('dump')
        with open('./data/mix_sft.pkl', 'wb') as f:
            pickle.dump(result, f)


# download from https://huggingface.co/datasets/shibing624/DPO-En-Zh-20k-Preference
def preprocess_dpo_data():
    dpo_list = []

    for file_item in ['dpo_zh.json', 'dpo_en.json']:
        with open(f'./data/raw/dpo/{file_item}', 'r') as f:
            json_ = json.loads(f.read())

            for item in json_:
                system = random.choice(GENERAL_SYSTEM_PROMPTS)

                conversations = item['conversations']

                chosen = item['chosen']
                rejected = item['rejected']

                chat_template = [{'role': 'system', 'content': system}]
                for conversation in conversations:
                    if conversation['from'] == 'system':
                        continue

                    if conversation['from'] == 'human':
                        chat_template.append({'role': 'user', 'content': f"{conversation['value']} /no think"})
                    else:
                        chat_template.append({'role': 'assistant', 'think': ' ', 'content': _filter_content(conversation['value'])})

                chosen_template = []
                chosen_template.extend(chat_template)
                chosen_template.append({'role': 'assistant', 'think':' ', 'content': _filter_content(chosen['value'])})

                rejected_template = []
                rejected_template.extend(chat_template)
                rejected_template.append({'role': 'assistant', 'think':' ', 'content': _filter_content(rejected['value'])})

                chosen = TrainerTools().tokenizer.apply_chat_template(chosen_template)
                rejected = TrainerTools().tokenizer.apply_chat_template(rejected_template)
                if len(chosen) > 2048 or len(rejected) > 2048:
                    continue

                encode_item = {
                    'chosen': chosen,
                    'rejected': rejected,
                }

                dpo_list.append(encode_item)

    dpo_list = shuffle(dpo_list)
    with open(f'./data/dpo.pkl', 'wb') as f:
        pickle.dump(dpo_list, f)


if __name__ == '__main__':
    _init()

    split_mobvoi()
    encode_mobvoi_short()
    encode_mobvoi_long()
    encode_wikipedia()

    preprocess_cot_data()
    preprocess_grpo_data()
    preprocess_mix_data()
    preprocess_dpo_data()

    # upload_to_ms()


