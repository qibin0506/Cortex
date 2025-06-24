from llm_trainer import SFTTrainer
from utils import init_env, get_reasoning_config, get_eval_prompt

if __name__ == '__main__':
    init_env()

    eval_prompts = [
        get_eval_prompt('告诉我世界上最大的湖是哪个？', no_reasoning=False),
        get_eval_prompt('请问今天北京天气如何？', no_reasoning=True),
        get_eval_prompt('哪吒和孙悟空谁更厉害？', no_reasoning=False),
        get_eval_prompt('保持健康的三个提示是什么？', no_reasoning=True),
        get_eval_prompt('你是谁？', no_reasoning=False),
        get_eval_prompt('你叫什么', no_reasoning=True)
    ]

    trainer = SFTTrainer(
        train_config=get_reasoning_config(),
        eval_prompts=eval_prompts
    )

    trainer.train()