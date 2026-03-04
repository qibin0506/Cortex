import torch
import math
from llm_trainer import train_configs
from llm_model import ModelConfig, RoPEConfig
from file_dataset import *
import os


def init_env():
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    os.environ['TOKEN_DIR'] = './tokens'
    os.environ['LOG_DIR'] = './log/'

    os.environ['DIST_CHECKPOINT_DIR'] = 'ckpt_dir'
    os.environ['CHECKPOINT_NAME'] = 'ckpt.pth'

    os.environ['CKPT_MAX_TO_KEEP'] = '2'
    os.environ['SAVE_BEST_CHECKPOINT'] = '0'  # or '1'


def get_eval_prompt(content: str) -> str:
    chat_template = [
        {'role': 'system', 'content': ' '},
        {'role': 'user', 'content': content}
    ]

    chat_template = TrainerTools().tokenizer.apply_chat_template(chat_template, tokenizer=False)
    return f'{chat_template}<assistant>'


def get_model_config(long_context=False):
    # max_position_embeddings: 512 -> 2048
    max_position_embeddings = 2048 if long_context else 512
    original_max_position_embeddings = 512 if long_context else None
    rope_type = 'yarn' if long_context else 'default'

    return ModelConfig(
        vocab_size=TrainerTools().tokenizer.vocab_size,
        hidden_size=768,
        intermediate_size=2048,

        num_hidden_layers=12,
        num_attention_heads=12,
        num_key_value_heads=4,

        max_position_embeddings=max_position_embeddings,
        original_max_position_embeddings=original_max_position_embeddings,
        attention_dropout=0.0,
        tie_word_embeddings=True,

        rope_config=RoPEConfig(
            rope_type=rope_type,
            rope_theta=10000.0,
        )
    )


def calc_lr_schedular_args(
        train_stage: str,
        epochs: int,
        all_data_size: int,
        batch_size: int,
        gradient_accumulation_steps: int,
        **kwargs
):
    world_size = TrainerTools().parallel.world_size

    # 基础 dataloader 的总 batch 数量（每个 GPU 上的 batch 数）
    dataloader_batches_per_gpu = epochs * (all_data_size // (batch_size * world_size))

    if train_stage in ['pretrain', 'midtrain', 'sft', 'dpo']:
        # DPO 和常规的 SFT/Pretrain 更新逻辑一致：直接在 dataloader batch 级别上做梯度累积
        train_batch_per_world = dataloader_batches_per_gpu / gradient_accumulation_steps
    elif train_stage == 'ppo':
        # PPO 算法特性：
        # - 数据加载：每次 dataloader 给出 batch_size 条数据进行 1 次 Rollout。
        # - 训练拆分：对 Rollout 数据训练 ppo_epochs 次，每次按 ppo_batch_size 拆分成 micro_batch 进行 forward+backward。
        # - 梯度累积：每 gradient_accumulation_steps 个 micro_batch 执行一次 step()。
        ppo_epochs = kwargs.get('ppo_epochs', 1)
        ppo_batch_size = kwargs.get('ppo_batch_size', 1)

        updates_per_dataloader_batch = (ppo_epochs * batch_size / ppo_batch_size) / gradient_accumulation_steps
        train_batch_per_world = dataloader_batches_per_gpu * updates_per_dataloader_batch
    elif train_stage == 'grpo':
        # GRPO 算法特性：
        # - 数据加载：每次 dataloader 给出 batch_size 个 prompt，内部生成 batch_size * group_size 条数据。
        # - 训练拆分：对这批扩增后的数据训练 grpo_epochs 次，按 grpo_batch_size 拆分为 micro_batch。
        # - 梯度累积：每 gradient_accumulation_steps 个 micro_batch 执行一次 step()。
        grpo_epochs = kwargs.get('grpo_epochs', 1)
        group_size = kwargs.get('group_size', 1)
        grpo_batch_size = kwargs.get('grpo_batch_size', 1)

        updates_per_dataloader_batch = (grpo_epochs * batch_size * group_size / grpo_batch_size) / gradient_accumulation_steps
        train_batch_per_world = dataloader_batches_per_gpu * updates_per_dataloader_batch
    else:
        train_batch_per_world = dataloader_batches_per_gpu / gradient_accumulation_steps

    train_batch_per_world = math.floor(train_batch_per_world)
    warmup_iters = int(0.1 * train_batch_per_world)
    cosine_annealing_batches = math.ceil(train_batch_per_world - warmup_iters)

    if TrainerTools().parallel.is_main_process:
        print(f'stage={train_stage}, total_updates={train_batch_per_world}, warmup_iters={warmup_iters}, cosine_annealing_batches={cosine_annealing_batches}')

    return warmup_iters, cosine_annealing_batches


def _get_train_config(
        n_epochs: int,
        real_batch_size: int,
        file_dataset: FileDataset,
        model_config: ModelConfig,
        train_stage: str
):
    init_state_dict = torch.load('./last_checkpoint.bin', weights_only=True) if os.path.exists('./last_checkpoint.bin') else None
    ref_checkpoint = torch.load('./sft.bin', weights_only=True) if os.path.exists('./sft.bin') else None
    if train_stage != 'pretrain':
        assert init_state_dict is not None

    if train_stage == 'ppo':
        assert ref_checkpoint is not None

    gradient_accumulation_steps = 3
    eval_batch_interval = 10 if train_stage == 'grpo' or train_stage == 'ppo' else 100

    min_lr_ratio = 0.1
    max_lr = -1
    warmup_iters = -1
    period = -1
    enable_lr_scheduler = False

    if train_stage == 'ppo':
        enable_lr_scheduler = True

        ppo_epochs = 2
        ppo_batch_size = 5
        gradient_accumulation_steps = 10

        max_lr = 5e-6
        initial_lr = 5e-7
        min_lr_ratio = 1.0

        warmup_iters, period = calc_lr_schedular_args(
            train_stage=train_stage,
            epochs=n_epochs,
            all_data_size=10000,
            batch_size=real_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            ppo_epochs=ppo_epochs,
            ppo_batch_size=ppo_batch_size
        )
    elif train_stage == 'dpo':
        enable_lr_scheduler = True

        max_lr = 1e-5
        initial_lr = 1e-6
        warmup_iters, period = calc_lr_schedular_args(
            train_stage=train_stage,
            epochs=n_epochs,
            all_data_size=100000,
            batch_size=real_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps
        )
    elif train_stage == 'grpo':
        enable_lr_scheduler = True

        grpo_epochs = 2
        grpo_batch_size = 4
        grpo_group_size = 12

        max_lr = 1e-5
        initial_lr = 1e-6
        warmup_iters, period = calc_lr_schedular_args(
            train_stage=train_stage,
            epochs=n_epochs,
            all_data_size=100000,
            batch_size=real_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            grpo_epochs=grpo_epochs,
            grpo_batch_size=grpo_batch_size,
            group_size=grpo_group_size
        )
    elif train_stage == 'sft':
        enable_lr_scheduler = True
        max_lr = 2e-5
        initial_lr = 1e-7

        warmup_iters, period = calc_lr_schedular_args(
            train_stage=train_stage,
            epochs=n_epochs,
            all_data_size=2430000,  # 2430781
            batch_size=real_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )
    elif train_stage == 'midtrain':
        enable_lr_scheduler = True
        max_lr = 8e-5
        initial_lr = 1e-7

        warmup_iters, period = calc_lr_schedular_args(
            train_stage=train_stage,
            epochs=n_epochs,
            all_data_size=1147000,  # 1147192
            batch_size=real_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )
    else:
        enable_lr_scheduler = True
        max_lr = 6e-4
        initial_lr = 1e-7

        warmup_iters, period = calc_lr_schedular_args(
            train_stage=train_stage,
            epochs=n_epochs,
            all_data_size=6532000,  # 6532762
            batch_size=real_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )

    ds_config = train_configs.DsConfig(
        zero_config=train_configs.DsZero1Config()
    )

    pretrain_config = train_configs.PretrainConfig(
        gradient_accumulation_steps=gradient_accumulation_steps,
        kd_config=None
    ) if train_stage == 'pretrain' or train_stage == 'midtrain' else None

    sft_config = train_configs.SFTConfig(
        mask_prompt=True,
        gradient_accumulation_steps=gradient_accumulation_steps,
        kd_config=None
    ) if train_stage == 'sft' else None

    ppo_config = train_configs.PPOConfig(
        ppo_epochs=2,
        ppo_batch_size=5,
        gradient_accumulation_steps=gradient_accumulation_steps,
        value_optim_config=train_configs.OptimConfig(
            enable_lr_scheduler=enable_lr_scheduler,
            initial_lr=1e-6,
            warmup_iters=warmup_iters,
            max_lr=2e-5,
            min_lr=2e-5,
            cosine_annealing_period=period
        ),
        vf_coef=0.5,
        kl_beta=0.02,
        kl_estimator='k3',
        normalize_rewards=True,
        normalize_method='RunningMeanStd',
        ref_model_checkpoint=ref_checkpoint,
        gen_max_seq_len=2048,
        gen_temperature=0.7,
        gen_p=0.9,
    ) if train_stage == 'ppo' else None

    dpo_config = train_configs.DPOConfig(
        ref_model_checkpoint=ref_checkpoint,
        mask_prompt=True,
        gradient_accumulation_steps=gradient_accumulation_steps,
        loss_beta=0.1,
        loss_label_smoothing=0.0,
        nll_loss_coef=0.2
    ) if train_stage == 'dpo' else None

    grpo_config = train_configs.GRPOConfig(
        grpo_epochs=2,
        grpo_batch_size=4,
        group_size=12,
        gradient_accumulation_steps=3,
        loss_beta=0.0,
        loss_clip_eps=3e-4,
        loss_clip_eps_high=4e-4,
        gen_max_seq_len=1024,
        loss_importance_sampling_level='seq',
        gen_temperature=1.0,
        gen_k=None,
        gen_p=0.95,
        gen_suppress_tokens=None,
    ) if train_stage == 'grpo' else None

    optim_config = train_configs.OptimConfig(
        enable_lr_scheduler=enable_lr_scheduler,
        initial_lr=initial_lr,
        warmup_iters=warmup_iters,
        max_lr=max_lr,
        min_lr=max_lr * min_lr_ratio,
        cosine_annealing_period=period
    )

    data_loader_config = train_configs.DataLoaderConfig(
        pin_memory=True,
        num_workers=0,
        shuffle=False,
    )

    train_config = train_configs.TrainConfig(
        n_epochs=n_epochs,
        batch_size=real_batch_size,
        model_config=model_config,
        file_dataset=file_dataset,
        dataset_block_size=model_config.max_position_embeddings,
        loss_config=train_configs.LossConfig(),
        optim_config=optim_config,
        ds_config=ds_config,
        data_loader_config=data_loader_config,
        init_state_dict=init_state_dict,
        eval_config=train_configs.EvalConfig(
            max_seq_len=model_config.max_position_embeddings,
            eval_batch_interval=eval_batch_interval,
        ),
        pretrain_config=pretrain_config,
        sft_config=sft_config,
        ppo_config=ppo_config,
        dpo_config=dpo_config,
        grpo_config=grpo_config
    )

    return train_config


def get_pretrain_config():
    return _get_train_config(
        n_epochs=1,
        real_batch_size=76,
        file_dataset=PretrainFileDataset(),
        model_config=get_model_config(long_context=False),
        train_stage='pretrain'
    )


def get_midtrain_config():
    return _get_train_config(
        n_epochs=1,
        real_batch_size=18,
        file_dataset=MidtrainFileDataset(),
        model_config=get_model_config(long_context=True),
        train_stage='midtrain'
    )


def get_sft_config():
    return _get_train_config(
        n_epochs=1,
        real_batch_size=15,
        file_dataset=SFTFileDataset(),
        model_config=get_model_config(long_context=True),
        train_stage='sft'
    )


def get_ppo_config():
    return _get_train_config(
        n_epochs=2,
        real_batch_size=50,
        file_dataset=PPOFileDataset(),
        model_config=get_model_config(long_context=True),
        train_stage='ppo'
    )
