import torch
from utils import init_env, get_model_config, get_eval_prompt
from llm_model import LlmModel
from llm_trainer import TrainerTools, streaming_generate
from llm_trainer.utils import set_seed
from modelscope import snapshot_download
from transformers import AutoModel, AutoTokenizer

set_seed()

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"

init_env()

# 模型配置
MODEL_ID = "Shanghai_AI_Laboratory/internlm2-1_8b-reward"
LOCAL_CACHE_DIR = "./rm_models"

model_dir = snapshot_download(
    MODEL_ID,
    cache_dir=LOCAL_CACHE_DIR,
    revision='master'
)

rm = AutoModel.from_pretrained(
    model_dir,
    torch_dtype=torch.float16,
    device_map='cpu',
    trust_remote_code=True
).eval()

tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

prompts = [
    # --- 第一组：创意写作与生成 ---
    "请写一个关于一只在大城市里迷路的流浪猫的短篇故事，结局要温馨。",
    "帮我构思一个悬疑故事的开头，背景设定在一家深夜的便利店。",
    # --- 第二组：生活建议与实用助手 ---
    "我最近总是失眠，有什么非药物的助眠小技巧吗？",
    "给一个刚毕业的大学生三条关于职场沟通的建议。",
    # --- 第三组：知识解释与常识 ---
    "请用通俗易懂的语言解释一下为什么天空是蓝色的。",
    "什么是“蝴蝶效应”？请举个例子说明。",
    # --- 第四组：逻辑推理与常识判断 ---
    "树上骑个猴，地上一个猴，一共几个猴？",
    "为什么在高速公路上不能突然停车？",
    # --- 第五组：角色扮演与特定语境 ---
    "扮演一个外星人，第一次吃到冰淇淋时的反应。",
    "你是一个耐心的心理咨询师，安慰一个因为考试失利而沮丧的学生。",
    # --- 第六组：文本处理与摘要 ---
    "将“快乐、公园、跑步、遇见、老朋友”这几个词串成一个通顺的句子。",
    "给“人工智能的发展”这个主题拟定三个不同的文章标题。",
]

def eval_model(model_type):
    print(f'Start eval {model_type}')
    ckpt = 'sft.bin' if model_type == 'sft' else 'ppo_policy.bin'
    model = LlmModel(get_model_config(long_context=True)).to(device)
    model.load_state_dict(torch.load(f'./bin/{ckpt}', weights_only=True))
    model.eval()

    batch_eval_template = []
    for prompt in prompts:
        chat_template = get_eval_prompt(prompt)
        chat_template = TrainerTools().tokenizer.encode(chat_template, covert_tensor=True)

        generator = streaming_generate(
            model=model,
            prompt=chat_template,
            max_new_tokens=2048 - chat_template.shape[0],
            temperature=1.0,
            p=0.95,
            suppress_tokens=None,
            device=device,
            return_token=True
        )

        response_tokens = []
        for item in generator:
            response_tokens.append(item)

        response = TrainerTools().tokenizer.decode(torch.tensor(response_tokens))
        batch_eval_template.append(
            tokenizer.apply_chat_template([
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response.replace('</s>', '')}
            ], tokenize=False)
        )

    inputs = tokenizer(
        batch_eval_template,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048
    ).to(device)

    with torch.no_grad():
        output = rm(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask
        )

        scores_tensor = output.logits
        print(f'{model_type} avg score = {torch.mean(scores_tensor).item()}')

# sft avg score = -0.73046875
# ppo avg score = 0.8203125
eval_model(model_type='sft')
eval_model(model_type='ppo')
