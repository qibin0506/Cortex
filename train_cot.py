from llm_trainer import SFTTrainer, TrainerTools
from utils import init_env, get_cot_config, get_eval_prompt


if __name__ == '__main__':
    init_env()

    eval_prompts = [
        get_eval_prompt('写一篇介绍太阳系行星的科普文章'),
        get_eval_prompt('请问今天北京天气如何？？'),
        get_eval_prompt('哪吒和孙悟空谁更厉害？'),
        get_eval_prompt('保持健康的三个提示是什么？'),
        get_eval_prompt('你是谁？'),
        get_eval_prompt('你叫什么？')
    ]

    trainer = SFTTrainer(
        train_config=get_cot_config(),
        eval_prompts=eval_prompts
    )

    trainer.train()