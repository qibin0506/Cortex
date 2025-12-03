import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from typing import List, Optional
import torch
from llm_trainer import PPOTrainer, TrainerTools
from utils import init_env, get_ppo_config, get_eval_prompt
from modelscope import AutoModelForSequenceClassification, AutoTokenizer

init_env()


# Load model and tokenizer
model_name = "Skywork/Skywork-Reward-V2-Qwen3-0.6B"
rm = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    dtype=torch.bfloat16,
    device_map=TrainerTools().parallel.device,
    # attn_implementation="flash_attention_2",
    num_labels=1,
)

tokenizer = AutoTokenizer.from_pretrained(model_name)


def replace_spec_tokens(text: str) -> str:
    text = text.replace('<system> </s>', '')
    spec_tokens = TrainerTools().tokenizer.get_special_tokens_dict().keys()
    for spec_token in spec_tokens:
        text = text.replace(spec_token, '')

    return text

def reward_func(
        prompt_ids: List[torch.Tensor],
        completion_ids: torch.Tensor,
        answers: List[Optional[torch.Tensor]]) -> List[float]:
    prompts_text = TrainerTools().tokenizer.batch_decode(prompt_ids)
    completions_text = TrainerTools().tokenizer.batch_decode(completion_ids)

    scores = []
    need_log = True
    for prompt, completion in zip(prompts_text, completions_text):
        prompt = replace_spec_tokens(prompt)
        completion = replace_spec_tokens(completion)

        template = [
            {"role": "user", "content": prompt},
            {"role": "assistant", 'content': completion}
        ]

        chat_template = tokenizer.apply_chat_template(template, tokenize=False)
        tokenizer_chat_template = tokenizer(chat_template, return_tensors="pt").to(TrainerTools().parallel.device)

        with torch.no_grad():
            score = rm(**tokenizer_chat_template).logits[0][0].item()

        scores.append(score)

        if TrainerTools().parallel.is_main_process and need_log:
            with open('./log/reward.txt', 'a') as f:
                f.write(f'{prompt} -> {completion} -> {score}\n')
        need_log = False

    return scores


if __name__ == '__main__':
    eval_prompts = [
        get_eval_prompt('写一篇介绍太阳系行星的科普文章'),
        get_eval_prompt('请问今天北京天气如何？？'),
        get_eval_prompt('哪吒和孙悟空谁更厉害？'),
        get_eval_prompt('保持健康的三个提示是什么？'),
        get_eval_prompt('你是谁？'),
        get_eval_prompt('你叫什么？')
    ]

    trainer = PPOTrainer(
        train_config=get_ppo_config(),
        reward_func=reward_func,
        eval_prompts=eval_prompts
    )

    trainer.train()