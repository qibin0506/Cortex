from llm_trainer import SFTTrainer
from utils import init_env, get_mix_config, get_eval_prompt


if __name__ == '__main__':
    init_env()

    eval_prompts = [
        get_eval_prompt('写一篇介绍太阳系行星的科普文章', add_think_tag=True),
        get_eval_prompt('请问今天北京天气如何？？', add_think_tag=True, no_think=True),
        get_eval_prompt('哪吒和孙悟空谁更厉害？', add_think_tag=True),
        get_eval_prompt('保持健康的三个提示是什么？', add_think_tag=True, no_think=True),
        get_eval_prompt('你是谁？', add_think_tag=True),
        get_eval_prompt('你叫什么？', add_think_tag=True, no_think=True)
    ]

    trainer = SFTTrainer(
        train_config=get_mix_config(),
        eval_prompts=eval_prompts
    )

    trainer.train()