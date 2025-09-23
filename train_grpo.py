from typing import List, Optional
import re
import torch
from llm_trainer import GRPOTrainer, TrainerTools
from utils import init_env, get_grpo_config, get_eval_prompt
import math


def extract_answer_from_completion(completion_text: str)-> str:
    # <think>思考</think><answer>回答</answer></s>
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


def get_answer_closeness_reward(response_text: str, correct_answer_str: str) -> float:
    """计算答案的接近度奖励，返回一个0到1之间的连续值。"""
    last_num_str = get_last_number(response_text)
    if last_num_str is None:
        return 0.0

    try:
        response_num = float(last_num_str)
        correct_num = float(correct_answer_str)
        error = abs(response_num - correct_num)
        # 使用一个更平滑的函数，例如指数衰减
        # a=0.1意味着误差为10时，奖励大约衰减到0.36
        reward = math.exp(-0.1 * error)
        return reward
    except (ValueError, TypeError):
        return 0.0


def get_think_complexity_reward(think_text: str) -> float:
    """
    根据思考文本的长度计算复杂度奖励，使用对数函数使其平滑。
    """
    if not think_text:
        return 0.0

    # log(1+x) 保证了当长度为0时，奖励也为0
    # 除以 log(500) 进行归一化，假设500个字符是一个比较理想的思考长度
    # 这意味着长度为499时，分数约等于1。你可以调整这个基准值。
    normalized_reward = math.log1p(len(think_text)) / math.log1p(500)

    # 将奖励限制在最大值1，防止过长的无意义文本获得过高奖励
    return min(1.0, normalized_reward)


def get_reward(completion_text: str, correct_answer: str)-> float:
    """
        为一个给定的模型输出文本计算奖励分数，旨在引导模型进行更好的推理并提高准确性。

        奖励逻辑如下:
        1.  **格式遵循**: 模型输出必须同时包含 <think>...</think> 和 <answer>...</answer> 标签。
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

    think_match = re.search(r'<think>(.*?)</think>', completion_text, re.DOTALL)
    if think_match:
        think_text = think_match.group(1).strip()
    else:
        think_text = ''

    answer_match = re.search(r'<answer>(.*?)</answer>', completion_text, re.DOTALL)
    if answer_match:
        answer_text = answer_match.group(1).strip()
    else:
        answer_text = ''

    # 如果必要的标签缺失，说明输出格式不正确，直接返回0分。
    if not think_match or not answer_match:
        return 0.0

    # --- 步骤 2: 计算“推理分数”，作为“思考”的代理指标 ---
    # 基于推理文本的长度给予奖励，上限为 2.0 分。
    # 这会激励模型生成至少150个字符的思考过程。
    # len(think_text) / 75.0 是一个平滑的奖励函数，长度越长奖励越高，直到达到上限。
    think_score = min(2.0, len(think_text) / 75.0)

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
        reward = answer_score + think_score
    else:
        # 如果答案是错误的，说明推理过程存在缺陷。
        # 即便如此，我们仍然为“尝试推理”这一行为提供少量奖励。
        # 这可以鼓励模型在面对难题时不要放弃思考，直接输出一个猜测的答案。
        # 错误答案下的最高奖励为 2.0 * 0.5 = 1.0
        reward = think_score * 0.5

    return reward


def get_reward_v2(completion_text: str, correct_answer: str) -> float:
    """
    修改版奖励函数，旨在提供更密集、更平滑的奖励信号。
    """
    # --- 1. 格式分数 (Format Score) ---
    # 为每个正确出现的标签都给予奖励，不再一票否决
    format_score = 0.0
    think_match = re.search(r'<think>(.*?)</think>', completion_text, re.DOTALL)
    answer_match = re.search(r'<answer>(.*?)</answer>', completion_text, re.DOTALL)

    if think_match:
        format_score += 1.0  # 出现<think>标签，奖励1分
    if answer_match:
        format_score += 1.0  # 出现<answer>标签，奖励1分

    # --- 2. 思考分数 (Think Score) ---
    # 只有当思考标签存在时，才计算思考内容的长度奖励
    think_score = 0.0
    if think_match:
        think_text = think_match.group(1).strip()
        # 鼓励有内容的思考，最高2分
        think_score = min(2.0, len(think_text) / 75.0)

    # --- 3. 答案分数 (Answer Score) ---
    # 这是核心任务，分值最高，且梯度更平滑
    answer_score = 0.0
    if answer_match:
        answer_text = answer_match.group(1).strip()
        response_last_number = get_last_number(answer_text)

        if response_last_number is not None:
            if response_last_number == correct_answer:
                # 最终答案正确，给予最高分
                answer_score = 5.0
            elif correct_answer in answer_text:
                # 最终答案错误，但正确答案在文本中，给予部分分
                answer_score = 2.5
        # 如果模型没有提取出数字，或者数字错误，但文本中有正确答案，也给予少量分数
        elif correct_answer in completion_text:
            answer_score = 1.0

    # --- 4. 组合最终奖励 ---
    # 将各部分分数加权求和
    # 权重可以调整，这里我们让答案准确性占最大比重
    # 总分范围大约在 0 到 10
    # format_score (max 2) + think_score (max 2) + answer_score (max 5)
    # 稍微调整权重，让总分在10左右
    final_reward = (format_score * 0.5) + (think_score * 1.0) + (answer_score * 2.0)

    return float(final_reward)


def get_reward_v3(completion_text: str, correct_answer_str: str) -> float:
    """
    计算最终的、带权重的奖励分数，无需黄金思考文本。
    """
    # --- 权重配置 ---
    WEIGHT_ANSWER_CLOSENESS = 7.0  # 答案准确度绝对核心
    WEIGHT_THINK_COMPLEXITY = 2.0  # 鼓励模型展现思考过程
    WEIGHT_FORMAT_AND_CONTENT = 1.0  # 对基本格式和非空内容给予奖励

    # --- 数据提取 ---
    think_match = re.search(r'<think>(.*?)</think>', completion_text, re.DOTALL)
    answer_match = re.search(r'<answer>(.*?)</answer>', completion_text, re.DOTALL)

    think_text = think_match.group(1).strip() if think_match else ""
    answer_text = answer_match.group(1).strip() if answer_match else ""

    # --- 分项计分 ---
    # 1. 格式与内容基础分
    # 只有当两个标签都存在，并且里面的内容不为空时，才得1分
    format_and_content_score = 1.0 if think_text and answer_text else 0.0

    # 2. 思考过程复杂度分 (0-1)
    think_complexity_score = get_think_complexity_reward(think_text)

    # 3. 答案接近度分 (0-1)
    answer_closeness_score = get_answer_closeness_reward(completion_text, correct_answer_str)

    # --- 加权求和 ---
    final_reward = (
            answer_closeness_score * WEIGHT_ANSWER_CLOSENESS +
            think_complexity_score * WEIGHT_THINK_COMPLEXITY +
            format_and_content_score * WEIGHT_FORMAT_AND_CONTENT
    )

    # 当答案完全正确时，给予一个额外的“大奖”，以强化正确行为
    if answer_closeness_score > 0.999:
        final_reward += 1.0  # Bonus point

    return float(final_reward)

def reward_func(prompt_ids: torch.Tensor, completion_ids: torch.Tensor, answers: torch.Tensor) -> List[float]:
    # 1. 如果回答包含思考部分，则奖励1.25分
    # 2. 如果正确答案相同，则奖励1分
    # 3. 如果正确答案在回答中，则奖励0.5分

    rewards = []
    for i, (prompt_id, completion_id, answer) in enumerate(zip(prompt_ids, completion_ids, answers)):
        # prompt = TrainerTools().tokenizer.decode(prompt_id)
        completion_text = TrainerTools().tokenizer.decode(completion_id)
        completion_text = completion_text.replace('<pad>', '').strip()
        correct_answer = TrainerTools().tokenizer.decode(answer)

        reward = get_reward_v3(completion_text, correct_answer)
        rewards.append(reward)

        if TrainerTools().parallel.is_main_process:
            with open("./reward.txt", 'a') as f:
                f.write(f"Calculated Reward: {reward}\n")
    
    if TrainerTools().parallel.is_main_process:
        with open("./reward.txt", 'a') as f:
            f.write(f"--------\n")

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

    # origin_compute_group_relative_advantages = trainer._compute_group_relative_advantages
    #
    # def replace(rewards):
    #     advantages = origin_compute_group_relative_advantages(rewards)
    #     print(f'advantages = {advantages}')
    #     return advantages
    #
    # trainer._compute_group_relative_advantages = replace

    trainer.train()