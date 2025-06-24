from llm_trainer import Trainer
from utils import init_env, get_pretrain_config

if __name__ == '__main__':
    init_env()
    eval_prompts = [
        '请描述一下如何正确规划个人理财。',
        'A公司去年亏损了500万美元，今年净利润增长了50%，今年的净利润是多少？',
        '列举出五种古代建筑的设计特点',
        '你是谁？'
    ]

    trainer = Trainer(
        train_config=get_pretrain_config(),
        eval_prompts=eval_prompts
    )

    trainer.train()