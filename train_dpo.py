from llm_trainer import DPOTrainer
from utils import init_env, get_dpo_config, get_eval_prompt

if __name__ == '__main__':
    init_env()

    eval_prompts = [
        get_eval_prompt('告诉我世界上最大的湖是哪个？', add_think_tag=True, no_think=True),
        get_eval_prompt('请问今天北京天气如何？', add_think_tag=True),
        get_eval_prompt('哪吒和孙悟空谁更厉害？', add_think_tag=True, no_think=True),
        get_eval_prompt('保持健康的三个提示是什么？', add_think_tag=True),
        get_eval_prompt('你是谁？', add_think_tag=True, no_think=True)
    ]

    trainer = DPOTrainer(
        train_config=get_dpo_config(),
        eval_prompts=eval_prompts
    )

    trainer.train()