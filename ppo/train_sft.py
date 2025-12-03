from llm_trainer import SFTTrainer, TrainerTools
from utils import init_env, get_sft_config, get_eval_prompt


if __name__ == '__main__':
    init_env()

    eval_prompts = [
        get_eval_prompt('介绍一道好吃的家常菜的做法'),
        get_eval_prompt('请描述一下如何正确规划个人理财。'),
        get_eval_prompt('鉴别两种不同类型的葡萄酒。'),
        get_eval_prompt('保持健康的三个提示是什么？'),
        get_eval_prompt('回答问题并给出详细的推理过程：为什么在地球上的物体会受到重力而不会飘起来？'),
        get_eval_prompt('你叫什么？')
    ]

    trainer = SFTTrainer(
        train_config=get_sft_config(),
        eval_prompts=eval_prompts
    )

    trainer.train()