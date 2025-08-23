from llm_trainer import Trainer
from utils import init_env, get_pretrain_stage0_config

if __name__ == '__main__':
    init_env()
    eval_prompts = [
        '初中阶段是学生身心发育的一个突变期',
        '癜风病人调节心理要偶尔也要屈服',
        '列举出五种古代建筑的设计特点',
        '吕宽，西汉末年平帝时期人'
    ]

    trainer = Trainer(
        train_config=get_pretrain_stage0_config(),
        eval_prompts=eval_prompts
    )

    trainer.train()