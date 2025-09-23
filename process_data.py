import os.path
import re
import json

from sklearn.utils import shuffle
from llm_trainer import TrainerTools
from constant import *
import pickle
import itertools
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
    content = content.replace("{{assistant_name}}", assistant_name)
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


def preprocess_wikipedia():
    print('preprocess_wikipedia')
    encoded = []

    with open('./data/raw/wikipedia-cn-20230720-filtered.json', 'r') as f:
        json_ = json.loads(f.read())
        for item in json_:
            item = TrainerTools().tokenizer.encode(f"{item['completion']}{TrainerTools().tokenizer.text_end}")
            encoded.append(item)

    with open(f'./data/tmp/wikipedia.pkl', 'wb') as f:
        pickle.dump(encoded, f)


def preprocess_cmm_math():
    print('preprocess_cmm_math')
    def is_empty(text):
        return len(text) == 0 or text == 'null'

    result = []
    with open('./data/raw/CMM-Math.jsonl', 'r') as f:
        for line in f:
            json_ = json.loads(line)
            if len(json_['image']) == 0:
                question = json_['question']
                options = json_['options']
                analysis = json_['analysis']
                answer = json_['answer']

                content = f'{question}\n'
                if not is_empty(options):
                    content += f'{options}\n'

                if not is_empty(analysis):
                    content += f'{analysis}\n'

                if not is_empty(answer):
                    content += f'答案：{answer}'

                content = f'{content}{TrainerTools().tokenizer.text_end}'
            result.append(TrainerTools().tokenizer.encode(content))

    with open('./data/tmp/cmm_math.pkl', 'wb') as f:
        pickle.dump(result, f)


def sample_github_code():
    print('sample_github_code')
    from modelscope import dataset_snapshot_download
    encoded = []

    # 只是有一个文件中的1/4
    include_files = ['train-00019-of-01126.parquet']

    for include_file in include_files:
        dataset_snapshot_download(
            'swift/github-code',
            allow_file_pattern=[f'data/{include_file}'],
            local_dir=f'./data/tmp/'
        )

        local_file_name = f'./data/tmp/data/{include_file}'
        df = pd.read_parquet(local_file_name, engine="pyarrow")
        values = df['content'].values[:len(df['content'].values)//4]

        for v in values:
            v = f'{v}{TrainerTools().tokenizer.text_end}'
            encoded.append(TrainerTools().tokenizer.encode(v.strip()))

    with open(f'./data/tmp/github_code.pkl', 'wb') as f:
        pickle.dump(encoded, f)


def preprocess_pretrain_data():
    tag_list = ['zh', 'en']
    short_thresholds = [1536, 3072]

    for file_idx in range(len(tag_list)):
        result_short = []
        result_long = []
        tokens_count_short = 0
        tokens_count_long = 0
        suffix_short = 0
        suffix_long = 0

        file = f'./data/raw/sft_data_{tag_list[file_idx]}.jsonl'
        print(f'encode file {file}')

        with open(file, 'r') as f:
            for idx, line in enumerate(f):
                json_ = json.loads(line)
                history = ''
                for his in json_['history']:
                    if len(his) != 0:
                        history = f'{history}{"\n".join(his)}'

                if len(history) == 0:
                    item = _filter_content(
                        f"{json_['input'].strip()}\n{json_['output'].strip()}{TrainerTools().tokenizer.text_end}")
                else:
                    item = _filter_content(
                        f"{history}{json_['input'].strip()}\n{json_['output'].strip()}{TrainerTools().tokenizer.text_end}")

                item = TrainerTools().tokenizer.encode(item.strip())
                item_count = len(item)

                if item_count > short_thresholds[file_idx]:
                    result_long.append(item)
                    tokens_count_long += item_count
                else:
                    result_short.append(item)
                    tokens_count_short += item_count

                if tokens_count_long >= 4e8:
                    with open(f'./data/tmp/pretrain_long_{tag_list[file_idx]}_{suffix_long}.pkl', 'wb') as f:
                        pickle.dump(result_long, f)
                        result_long.clear()
                        tokens_count_long = 0
                        suffix_long += 1

                if tokens_count_short >= 4e8:
                    with open(f'./data/tmp/pretrain_short_{tag_list[file_idx]}_{suffix_short}.pkl', 'wb') as f:
                        pickle.dump(result_short, f)
                        result_short.clear()
                        tokens_count_short = 0
                        suffix_short += 1

        with open(f'./data/tmp/pretrain_short_{tag_list[file_idx]}.pkl', 'wb') as f:
            pickle.dump(result_short, f)

        with open(f'./data/tmp/pretrain_long_{tag_list[file_idx]}.pkl', 'wb') as f:
            pickle.dump(result_long, f)


def get_self_cognition(add_think_tag=False):
    result = []

    with open('./data/raw/self_cognition.jsonl', 'r') as f:
        for line in f:
            json_ = json.loads(line)
            user = f"{json_['query']}"

            if add_think_tag:
                user = f"{user} /no think"

            content = json_['response'].replace('{{AUTHOR}}', developer_name).replace('{{NAME}}', assistant_name)

            chat_template = [
                {'role': 'system', 'content': " "},
                {'role': 'user', 'content': user},
                {'role': 'assistant', 'think': ' ', 'content': f"{content.strip()}"}
            ]

            encoded = TrainerTools().tokenizer.apply_chat_template(chat_template)
            result.append(encoded)

    return result


def merge_pretrain_data():
    print('start merge short data')
    # 将en_0 merge到zh_0和zh_1中
    with open('./data/tmp/pretrain_short_en_0.pkl', 'rb') as f:
        en = pickle.load(f)
        en_0_mid = len(en) // 2
        en_0 = en[:en_0_mid]
        en_1 = en[en_0_mid:]
        del en

    merge_froms = [en_0, en_1]
    merge_tos = [0, 1]

    for merge_from, merge_to in zip(merge_froms, merge_tos):
        result = merge_from
        with open(f'./data/tmp/pretrain_short_zh_{merge_to}.pkl', 'rb') as f:
            to_content = pickle.load(f)
            result.extend(to_content)

        flat_result = list(itertools.chain.from_iterable(shuffle(result)))
        with open(f'./data/pretrain_short_{merge_to}.pkl', 'wb') as f:
            pickle.dump(flat_result, f)

    short_zh_list = [
        'pretrain_short_zh_2.pkl',
        'pretrain_short_zh_3.pkl',
        'pretrain_short_zh_4.pkl',
        'pretrain_short_zh_5.pkl',
        'pretrain_short_zh_6.pkl',
        'pretrain_short_zh.pkl',
    ]

    short_en_list = [
        'pretrain_short_en_1.pkl',
        'pretrain_short_en_2.pkl',
        'pretrain_short_en_3.pkl',
        'pretrain_short_en_4.pkl',
        'pretrain_short_en_5.pkl',
        'pretrain_short_en.pkl',
    ]

    for idx in range(len(short_zh_list)):
        result = []

        with open(f'./data/tmp/{short_zh_list[idx]}', 'rb') as f:
            zh = pickle.load(f)
            result.extend(zh)
            del zh

        with open(f'./data/tmp/{short_en_list[idx]}', 'rb') as f:
            en = pickle.load(f)
            result.extend(en)
            del en

        flat_result = list(itertools.chain.from_iterable(shuffle(result)))
        with open(f'./data/pretrain_short_{idx + 2}.pkl', 'wb') as f:
            pickle.dump(flat_result, f)

        del flat_result

    print('start merge long data')
    long_list = [
        'pretrain_long_en_0.pkl',
        'pretrain_long_en.pkl',
        'pretrain_long_zh_0.pkl',
        'pretrain_long_zh.pkl',
        'cmm_math.pkl',
        'wikipedia.pkl',
        'github_code.pkl'
    ]

    result = []
    for idx in range(len(long_list)):
        with open(f'./data/tmp/{long_list[idx]}', 'rb') as f:
            temp = pickle.load(f)
            result.extend(temp)

    result = shuffle(result)
    results = [result[:len(result)//2], result[len(result)//2:]]

    for idx, result in enumerate(results):
        print(f'start dump long {idx}')

        flat_result = list(itertools.chain.from_iterable(result))
        with open(f'./data/pretrain_long_{idx}.pkl', 'wb') as f:
            pickle.dump(flat_result, f)

        print(f'end dump long {idx}')

    print('finish...')


def preprocess_cot_data():
    result = get_self_cognition()

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
                {'role': 'system', 'content': " "},
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
                {'role': 'system', 'content': " "},
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


def preprocess_grpo_data():
    qas = []
    for file_name in ['train-00000-of-00001.parquet', 'test-00000-of-00001.parquet']:
        df = pd.read_parquet(f"./data/raw/gsm8k_chinese/{file_name}", engine="pyarrow")
        for q, a in zip(df['question_zh-cn'].values, df['answer_only'].values):
            q_template = [
                {'role': 'system', 'content': " "},
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


def preprocess_mix_data():
    # 添加自我认知数据集
    # 加入/think 和 /no think
    result = get_self_cognition(True)

    with open('./data/raw/r1_mix_1024.jsonl', 'r') as f:
        for line in f:
            json_ = json.loads(line)
            conversations = json_['conversations']

            chat_template = [{'role': 'system', 'content': " "}]
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


def preprocess_dpo_data():
    dpo_list = []

    for file_item in ['dpo_zh.json', 'dpo_en.json']:
        with open(f'./data/raw/dpo/{file_item}', 'r') as f:
            json_ = json.loads(f.read())

            for item in json_:
                system = " "

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
  
    sample_github_code()
    preprocess_wikipedia()
    preprocess_cmm_math()
    preprocess_pretrain_data()
    merge_pretrain_data()
    preprocess_cot_data()
    preprocess_grpo_data()
    preprocess_mix_data()
    preprocess_dpo_data()
   