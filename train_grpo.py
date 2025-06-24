from typing import List, Optional
import re
import torch
from llm_trainer import GRPOTrainer, TrainerTools
from utils import init_env, get_grpo_config, get_eval_prompt


def extract_answer_from_completion(completion_text: str)-> str:
    # <reasoning>思考</reasoning><answer>回答</answer></s>
    parts = completion_text.split("<answer>")
    if len(parts) < 2:
        return ''

    # 回答</answer></s>
    last_part = parts[-1]

    # Extract content up to </answer>
    if "</answer>" not in last_part:
        return ''

    # 回答
    answer = last_part.split("</answer>")[0].strip()
    return '' if answer == "..." else answer


def get_last_number(response_answer: str)-> Optional[str]:
    numbers = re.findall(r'-?\d+\.?\d*', response_answer)
    if numbers:
        last_num = numbers[-1]
        return last_num

    return None


def get_reward(completion_text: str, correct_answer: str)-> float:
    """
        为一个给定的模型输出文本计算奖励分数，旨在引导模型进行更好的推理并提高准确性。

        奖励逻辑如下:
        1.  **格式遵循**: 模型输出必须同时包含 <reasoning>...</reasoning> 和 <answer>...</answer> 标签。
            否则，奖励为 0。这是为了确保模型遵循我们期望的思考-回答格式。

        2.  **推理质量 (代理指标)**: 根据推理过程的文本长度计算“推理分数”。
            这会激励模型产出更详细、更丰富的思考过程，而不仅仅是空标签或一句话。

        3.  **答案准确性**: 为正确的最终答案提供一个较高的基础分数。这是模型的核心任务。

        4.  **协同奖励**: 当一个正确的答案由详细的推理过程支撑时，给予最高奖励。
            对于导致错误答案的推理过程，只给予非常小的奖励，以继续鼓励模型进行思考尝试。

        Args:
            completion_text: 模型生成的完整文本。
            correct_answer: 标准的正确答案字符串。

        Returns:
            一个浮点数奖励分数，通常在 0.0 到 10.0 之间。
    """

    reasoning_match = re.search(r'<reasoning>(.*?)</reasoning>', completion_text, re.DOTALL)
    if reasoning_match:
        reasoning_text = reasoning_match.group(1).strip()
    else:
        reasoning_text = ''

    answer_match = re.search(r'<answer>(.*?)</answer>', completion_text, re.DOTALL)
    if answer_match:
        answer_text = answer_match.group(1).strip()
    else:
        answer_text = ''

    # 如果必要的标签缺失，说明输出格式不正确，直接返回0分。
    if not reasoning_match or not answer_match:
        return 0.0

    # --- 步骤 2: 计算“推理分数”，作为“思考”的代理指标 ---
    # 基于推理文本的长度给予奖励，上限为 2.0 分。
    # 这会激励模型生成至少150个字符的思考过程。
    # len(reasoning_text) / 75.0 是一个平滑的奖励函数，长度越长奖励越高，直到达到上限。
    reasoning_score = min(2.0, len(reasoning_text) / 75.0)

    # --- 步骤 3: 评估答案的准确性 ---
    answer_score = 0.0
    response_answer = extract_answer_from_completion(completion_text)
    response_last_number = get_last_number(response_answer)

    if response_last_number is not None:
        # 注意：这里我们假设 correct_answer 是一个字符串形式的数字，以便直接比较。
        if response_last_number == correct_answer:
            # 答案完全正确，给予8分的基础分。
            answer_score = 8.0
        elif correct_answer in answer_text:
            # 如果最终答案不对，但正确答案出现在回答文本中，给予4分的部分分。
            answer_score = 4.0

    # --- 步骤 4: 组合分数，得出最终奖励 ---
    # 最终的奖励是推理过程和答案正确性的协同结果。
    if answer_score > 0:
        # 如果答案是正确或部分正确的，则将完整的“推理分数”加到“答案分数”上。
        # 这为“在得出正确结论时展现思考过程”的行为提供了强大的激励。
        # 理想情况 (正确答案 + 充分推理) = 8.0 + 2.0 = 10.0
        reward = answer_score + reasoning_score
    else:
        # 如果答案是错误的，说明推理过程存在缺陷。
        # 即便如此，我们仍然为“尝试推理”这一行为提供少量奖励。
        # 这可以鼓励模型在面对难题时不要放弃思考，直接输出一个猜测的答案。
        # 错误答案下的最高奖励为 2.0 * 0.5 = 1.0
        reward = reasoning_score * 0.5

    return reward


def reward_func(prompt_ids: torch.Tensor, completion_ids: torch.Tensor, answers: torch.Tensor) -> List[float]:
    # 1. 如果回答包含思考部分，则奖励1.25分
    # 2. 如果正确答案相同，则奖励1分
    # 3. 如果正确答案在回答中，则奖励0.5分

    rewards = []
    for completion_id, answer in zip(completion_ids, answers):
        completion_text = TrainerTools().tokenizer.decode(completion_id)
        completion_text = completion_text.replace('<pad>', '').strip()
        correct_answer = TrainerTools().tokenizer.decode(answer)

        rewards.append(get_reward(completion_text, correct_answer))

    return rewards


if __name__ == '__main__':
    init_env()

    eval_prompts = [
        get_eval_prompt('朱莉正在读一本 120 页的书。昨天，她能读12页，今天，她读的页数是昨天的两倍。如果她明天想读剩下的一半页，她应该读多少页？'),
        get_eval_prompt('詹姆斯从事教学工作 40 年。他的搭档教书的时间比他少了10年。他们的综合经验有多长？'),
        get_eval_prompt('赫克托买了一盒口香糖。他给了托德 4 个，然后他给了艾丽莎的是托德的两倍，然后他给了鲍比 5 个，比他给艾丽莎的四倍还少。如果赫克托还剩下 6 个口香糖，那么赫克托总共购买了多少个口香糖？'),
        get_eval_prompt('如果艾琳每周工作 40 小时，她将赚取 500 美元，并且每加班一小时即可额外获得 20 美元。如果她上周工作了 50 小时，请计算她的总收入。'),
    ]

    trainer = GRPOTrainer(
        train_config=get_grpo_config(),
        reward_func=reward_func,
        eval_prompts=eval_prompts
    )

    trainer.train()