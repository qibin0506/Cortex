import torch
from typing import List, Optional, Tuple
from llm_trainer import PPOTrainer, TrainerTools
from utils import init_env, get_ppo_config, get_eval_prompt

from modelscope import snapshot_download
from transformers import AutoModel, AutoTokenizer

init_env()

rm_device = TrainerTools().parallel.device

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


def replace_spec_tokens(text: str) -> str:
    text = text.replace('<system> </s>', '')
    spec_tokens = TrainerTools().tokenizer.get_special_tokens_dict().keys()
    for spec_token in spec_tokens:
        text = text.replace(spec_token, '')
    return text.strip()


def reward_func(
        prompt_ids: List[torch.Tensor],
        completion_ids: torch.Tensor,
        answers: List[Optional[torch.Tensor]]) -> List[float]:
    prompts_text = TrainerTools().tokenizer.batch_decode(prompt_ids, skip_special_tokens=True)
    completions_text = TrainerTools().tokenizer.batch_decode(completion_ids, skip_special_tokens=False)

    batch_size = len(prompts_text)
    total_scores = [0.0] * batch_size

    rm_inputs_text = []
    rm_indices = []
    log_details = {}

    # 参数设置
    SCORE_EOS_PENALTY = -5.0  # 没有结束符的惩罚
    RM_WEIGHT = 0.5  # RM 分数权重

    debug_scores = {
        "eos_score": 0.0,
        "rm_raw": 0.0,
        "rm_weighted": 0.0
    }

    for idx, (prompt, completion) in enumerate(zip(prompts_text, completions_text)):
        # 1. 检查是否以 </s> 结尾
        completion = completion.replace("<pad>", '')
        has_eos = completion.endswith('</s>')
        current_score = 0.0

        if not has_eos:
            current_score += SCORE_EOS_PENALTY

        # 2. 准备 RM 的输入
        # 清理 prompt 和 completion
        clean_prompt = replace_spec_tokens(prompt)
        clean_completion = replace_spec_tokens(completion)

        # 构建对话格式
        chat = [
            {"role": "user", "content": clean_prompt},
            {"role": "assistant", "content": clean_completion}
        ]
        formatted_input = tokenizer.apply_chat_template(chat, tokenize=False)
        rm_inputs_text.append(formatted_input)
        rm_indices.append(idx)

        total_scores[idx] = current_score

        # 记录第一条数据的调试信息
        if idx == 0:
            debug_scores["eos_score"] = 0.0 if has_eos else SCORE_EOS_PENALTY
            log_details = {
                "prompt_preview": clean_prompt[:100].replace('\n', ' '),
                "answer_preview": clean_completion[:100].replace('\n', ' '),
                "has_eos": has_eos,
                "pre_rm_score": current_score
            }

    # 3. 计算 RM 分数
    if len(rm_inputs_text) > 0:
        rm.to(rm_device)
        try:
            inputs = tokenizer(
                rm_inputs_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            ).to(rm_device)

            with torch.no_grad():
                output = rm(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask
                )

                scores_tensor = output.logits
                batch_rm_scores = scores_tensor.float().cpu().numpy().flatten()

            for i, original_idx in enumerate(rm_indices):
                raw_rm_val = float(batch_rm_scores[i])

                # 截断与加权
                clipped_rm_val = max(min(raw_rm_val, 5.0), -5.0)
                weighted_rm_val = clipped_rm_val * RM_WEIGHT

                total_scores[original_idx] += weighted_rm_val

                if original_idx == 0:
                    debug_scores["rm_raw"] = raw_rm_val
                    debug_scores["rm_weighted"] = weighted_rm_val

        except Exception as e:
            print(f"RM Error: {e}")
            for original_idx in rm_indices:
                total_scores[original_idx] -= 2.0
        finally:
            rm.to('cpu')
            torch.cuda.empty_cache()

    if log_details:
        log_details["final_total"] = total_scores[0]

    # 4. 写日志
    if TrainerTools().parallel.is_main_process and log_details:
        with open('./log/reward.txt', 'a', encoding='utf-8') as f:
            f.write("-" * 65 + "\n")
            f.write(f"Prompt: {log_details['prompt_preview']}...\n")
            f.write(f"Answer: {log_details['answer_preview']}...\n")

            eos_status = "✅" if log_details['has_eos'] else "❌"
            f.write(
                f"Reward: {log_details['final_total']:.4f} | "
                f"Breakdown: [EOS Check({eos_status}): {debug_scores['eos_score']}] + "
                f"[RM Raw: {debug_scores['rm_raw']:.2f} * {RM_WEIGHT} -> {debug_scores['rm_weighted']:.2f}]\n"
            )

    return total_scores


if __name__ == '__main__':
    eval_prompts = [
        get_eval_prompt('写一篇介绍太阳系行星的科普文章'),
        get_eval_prompt('生态环境是人类的生存和发展的空间，所以人类是不是应当尽可能地去改变生态环境？'),
        get_eval_prompt('水资源主要是被工业用水消耗，我在生活中节约用水有意义吗？'),
        get_eval_prompt('作为历史初学者，我该如何开始我的历史学习之旅？'),
        get_eval_prompt('如果Python中的父类和子类分别定义在不同的文件里，怎样导入才能避免出现循环导入的问题呢？'),
        get_eval_prompt('你叫什么？'),
        get_eval_prompt('你是谁？')
    ]

    trainer = PPOTrainer(
        train_config=get_ppo_config(),
        reward_func=reward_func,
        eval_prompts=eval_prompts
    )

    trainer.train()