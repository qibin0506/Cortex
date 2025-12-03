from llm_trainer import Trainer
from utils import init_env, get_pretrain_config

if __name__ == '__main__':
    init_env()
    eval_prompts = [
        '基于以下角色信息完成一段对话餐厅老板：一个友好开朗的中年男子，身着统一的工作服，面容和善，喜欢与顾客交流。',
        '给你两个角色信息如下：Mike：一位天真活泼的大学生John：一位严谨认真的教授生成他们之间的一段对话，要求对话内容详细丰富。',
        'Lisa：一位职业顾问，专业帮助人们找到自己的事业方向。Peter：一名大学生，正在考虑自己的职业发展方向。生成一段他们的对话内容。',
        '小明：一名初中生，成绩优秀，但在社交方面不太擅长。王老师：小明所在学校的语文老师，经验丰富。生成一段他们的对话内容。'
    ]

    trainer = Trainer(
        train_config=get_pretrain_config(),
        eval_prompts=eval_prompts
    )

    trainer.train()