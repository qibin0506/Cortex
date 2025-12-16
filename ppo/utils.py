import torch
import math
from llm_trainer import train_configs, TrainerTools
from llm_model import ModelConfig, RoPEConfig, MoEConfig
from file_dataset import *
import os


def init_env():
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    os.environ['TOKEN_DIR'] = './tiny_tokenizer'

    os.environ['LOG_DIR'] = './log/'

    os.environ['DIST_CHECKPOINT_DIR'] = 'ckpt_dir'
    os.environ['CHECKPOINT_NAME'] = 'ckpt.pth'

    os.environ['CKPT_MAX_TO_KEEP'] = '2'
    os.environ['SAVE_BEST_CHECKPOINT'] = '0' # or '0'


def get_eval_prompt(content: str) -> str:
    chat_template = [
        {'role': 'system', 'content': ' '},
        {'role': 'user', 'content': content}
    ]

    chat_template = TrainerTools().tokenizer.apply_chat_template(chat_template, tokenizer=False, add_answer_tag_for_assistant=False)
    return chat_template


def get_model_config():
    return ModelConfig(
        vocab_size=TrainerTools().tokenizer.vocab_size,
        hidden_size=768,
        intermediate_size=2048,
        num_hidden_layers=16,
        num_attention_heads=8,
        num_key_value_heads=2,
        max_position_embeddings=512,
        attention_implementation='auto',
        tie_word_embeddings=True,
        rope_config=RoPEConfig(
            rope_type='default',
            rope_theta=1e6
        )
    )


def calc_lr_schedular_args(
        epochs,
        all_data_size,
        batch_size,
        gradient_accumulation_steps
):
    world_size = TrainerTools().parallel.world_size

    train_batch_per_world = epochs * (all_data_size / batch_size) / world_size / gradient_accumulation_steps
    warmup_iters = int(0.1 * train_batch_per_world)
    cosine_annealing_batches = math.ceil(train_batch_per_world - warmup_iters)

    if TrainerTools().parallel.is_main_process:
        print(f'warmup_iters={warmup_iters}, cosine_annealing_batches={cosine_annealing_batches}')

    return warmup_iters, cosine_annealing_batches


def _get_train_config(
        n_epochs: int,
        real_batch_size: int,
        file_dataset: FileDataset,
        model_config: ModelConfig,
        train_stage: str
):
    last_checkpoint = './last_checkpoint.bin'
    if train_stage == 'ppo':
        assert os.path.exists(last_checkpoint)

    init_state_dict = torch.load(last_checkpoint, weights_only=True) if os.path.exists(last_checkpoint) else None

    gradient_accumulation_steps = 3
    eval_batch_interval = 100 if train_stage != 'ppo' else 10

    min_lr_ratio = 0.1
    if train_stage == 'ppo' or train_stage == 'dpo' or train_stage == 'grpo':
        initial_lr = 5e-6
        max_lr = -1
        warmup_iters = -1
        period = -1
    elif train_stage == 'sft':
        initial_lr = 1e-5
        max_lr = 5e-5
        warmup_iters, period = calc_lr_schedular_args(
            epochs=n_epochs,
            all_data_size=2260000, # 2266904
            batch_size=real_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )
    else:
        initial_lr = 1e-4
        max_lr = 5e-4
        warmup_iters, period = calc_lr_schedular_args(
            epochs=n_epochs,
            all_data_size=7960000, # 7962535
            batch_size=real_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )

    optim_config = train_configs.OptimConfig(
        enable_lr_scheduler=train_stage != 'ppo',
        initial_lr=initial_lr,
        warmup_iters=warmup_iters,
        max_lr=max_lr,
        min_lr=initial_lr * min_lr_ratio,
        cosine_annealing_period=period
    )

    data_loader_config = train_configs.DataLoaderConfig(
        data_loader_pin_memory=True,
        data_loader_num_workers=0,
        data_loader_shuffle=False,
        data_loader_drop_last=True
    )

    ds_config = train_configs.DsConfig(
        zero_config=train_configs.DsZero3Config(
            offload_param=train_configs.DsOffloadConfig() if train_stage == 'ppo' else None,
            offload_optimizer=train_configs.DsOffloadConfig() if train_stage == 'ppo' else None
        )
    )

    ppo_config = train_configs.PPOConfig(
        ppo_epochs=2,
        ppo_batch_size=8,
        vf_coef=0.5,
        kl_beta=0.02,
        kl_estimator='k3',
        normalize_rewards=True,
        ref_model_checkpoint=init_state_dict,
        gen_max_new_tokens=512,
        gen_temperature=1.0,
        gen_p=0.95,
    ) if train_stage == 'ppo' else None

    dpo_config = train_configs.DPOConfig(
        ref_model_checkpoint=init_state_dict,
        loss_beta=0.1,
        loss_label_smoothing=0.0,
        nll_loss_coef=0.2
    ) if train_stage == 'dpo' else None

    grpo_config = train_configs.GRPOConfig(
        grpo_steps=2,
        group_size=4,
        loss_beta=0.0,
        loss_clip_eps=3e-4,
        loss_clip_eps_high=4e-4,
        loss_importance_sampling_level='seq',
        gen_max_new_tokens=512,
        gen_temperature=1.0,
        gen_k=None,
        gen_p=0.95,
        gen_suppress_tokens=None,
    ) if train_stage == 'grpo' else None

    train_config = train_configs.TrainConfig(
        n_epochs=n_epochs,
        batch_size=real_batch_size,
        model_config=model_config,
        file_dataset=file_dataset,
        max_seq_len=512,
        gradient_accumulation_steps=gradient_accumulation_steps,
        eval_batch_interval=eval_batch_interval,
        loss_config=train_configs.LossConfig(),
        dpo_config=dpo_config,
        ppo_config=ppo_config,
        grpo_config=grpo_config,
        optim_config=optim_config,
        ds_config=ds_config,
        data_loader_config=data_loader_config,
        init_state_dict=init_state_dict,
        eval_config=train_configs.EvalConfig(
            max_new_tokens=512
        ),
    )

    return train_config


def get_pretrain_config():
    return _get_train_config(
        n_epochs=1,
        real_batch_size=48,
        file_dataset=PretrainFileDataset(),
        model_config=get_model_config(),
        train_stage='pretrain'
    )


def get_sft_config():
    return _get_train_config(
        n_epochs=2,
        real_batch_size=32,
        file_dataset=SFTFileDataset(),
        model_config=get_model_config(),
        train_stage='sft'
    )


def get_ppo_config():
    return _get_train_config(
        n_epochs=2,
        real_batch_size=32,
        file_dataset=PPOFileDataset(),
        model_config=get_model_config(),
        train_stage='ppo'
    )


def get_dpo_config():
    return _get_train_config(
        n_epochs=2,
        real_batch_size=24,
        file_dataset=DPOFileDataset(),
        model_config=get_model_config(),
        train_stage='dpo'
    )


def get_grpo_config():
    return _get_train_config(
        n_epochs=1,
        real_batch_size=8,
        file_dataset=GRPOFileDataset(),
        model_config=get_model_config(),
        train_stage='grpo'
    )
