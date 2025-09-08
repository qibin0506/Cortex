import torch
import math
from llm_trainer import train_configs
from llm_model import ModelConfig, RoPEConfig, MoEConfig
from constant import *
from file_dataset import *
import os
import random


def init_env():
    #  Of the allocated memory 33.98 GiB is allocated by PyTorch,
    #  and 8.89 GiB is reserved by PyTorch but unallocated.
    #  If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.
    #  See documentation for Memory Management
    #  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    os.environ['TOKENIZERS_TYPE'] = 'zh_llama'  # or qwen
    os.environ['TOKEN_DIR'] = './tokens/'

    os.environ['LOG_DIR'] = './log/'

    os.environ['DIST_CHECKPOINT_DIR'] = 'ckpt_dir'
    os.environ['CHECKPOINT_NAME'] = 'ckpt.pth'

    os.environ['CKPT_MAX_TO_KEEP'] = '2'
    os.environ['SAVE_BEST_CHECKPOINT'] = '0' # or '0'


def get_eval_prompt(content: str, add_think_tag = False, no_think = False) -> str:
    if add_think_tag:
        content = f'{content} /no think' if no_think else f'{content} /think'

    chat_template = [
        {'role': 'system', 'content': random.choice(GENERAL_SYSTEM_PROMPTS)},
        {'role': 'user', 'content': content}
    ]

    chat_template = TrainerTools().tokenizer.apply_chat_template(chat_template, tokenizer=False)
    return f'{chat_template}<assistant>'


def get_model_config(long_context = False):
    # max_position_embeddings: 512 -> 2048
    max_position_embeddings = 2048 if long_context else 512
    original_max_position_embeddings = 512 if long_context else None
    rope_type = 'yarn' if long_context else 'default'

    return ModelConfig(
        vocab_size=TrainerTools().tokenizer.vocab_size,
        hidden_size=768,
        intermediate_size=2048,
        moe_intermediate_size=1024,
        moe_n_dense_layer=1,
        num_hidden_layers=24,
        num_attention_heads=12,
        num_key_value_heads=4,
        max_position_embeddings=max_position_embeddings,
        original_max_position_embeddings=original_max_position_embeddings,
        attention_implementation='auto',
        rope_config=RoPEConfig(
            rope_type=rope_type,
            rope_theta=1e6
        ),
        moe_config=MoEConfig(
            num_experts_per_tok=2,
            n_routed_experts=8,
            n_shared_experts=1,
            aux_loss_alpha=0.1,
            seq_aux=True,
            norm_topk_prob=True
        )
    )


def get_small_model_config():
    max_position_embeddings = 2048

    return ModelConfig(
        vocab_size=TrainerTools().tokenizer.vocab_size,
        hidden_size=512,
        intermediate_size=1024,
        moe_intermediate_size=-1,
        moe_n_dense_layer=-1,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=2,
        max_position_embeddings=max_position_embeddings,
        attention_implementation='auto',
        rope_config=RoPEConfig(
            rope_type='default',
            rope_theta=1e6
        ),
    )


def calc_lr_schedular_args(
        epochs,
        all_data_size,
        batch_size,
        gradient_accumulation_steps,
        grpo_steps
):
    world_size = TrainerTools().parallel.world_size
    # epochs * all_data_size / batch_size / world_size / gradient_accumulation_steps
    if grpo_steps == -1:
        train_batch_per_world = epochs * all_data_size / batch_size / world_size / gradient_accumulation_steps
    else:
        train_batch_per_world = epochs * (all_data_size / batch_size / world_size) * grpo_steps

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
    init_state_dict = torch.load('./last_checkpoint.bin', weights_only=True)\
        if os.path.exists('./last_checkpoint.bin') and train_stage != 'distill' else None

    gradient_accumulation_steps = 3
    eval_batch_interval = 10 if train_stage == 'grpo' else 100

    ds_config = train_configs.DsConfig(
        zero_config=train_configs.DsZero3Config(
            offload_param=train_configs.DsOffloadConfig() if train_stage == 'grpo' else None,
            offload_optimizer=train_configs.DsOffloadConfig() if train_stage == 'grpo' else None
        )
    )

    dpo_config = train_configs.DPOConfig(
        loss_beta=0.1,
        loss_label_smoothing=0.0,
        nll_loss_coef=0.2
    ) if train_stage == 'dpo' else None

    grpo_config = train_configs.GRPOConfig(
        grpo_steps=4,
        group_size=16,
        loss_beta=0.0,
        loss_clip_eps=3e-4,
        loss_clip_eps_high=4e-4,
        loss_importance_sampling_level='seq',
        gen_max_new_tokens=1024,
        gen_temperature=1.0,
        gen_k=None,
        gen_p=0.85,
        gen_suppress_tokens=None,
    ) if train_stage == 'grpo' else None

    lr_mul = TrainerTools().parallel.world_size
    min_lr_ratio = 0.1

    if train_stage == 'grpo':
        initial_lr = 1e-6
        max_lr = 5e-6
        warmup_iters, period = calc_lr_schedular_args(
            epochs=n_epochs,
            all_data_size=8792,
            batch_size=real_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            grpo_steps=1
        )
    elif train_stage == 'dpo':
        initial_lr = 1e-6
        max_lr = 5e-6
        warmup_iters, period = calc_lr_schedular_args(
            epochs=n_epochs,
            all_data_size=19942,
            batch_size=real_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            grpo_steps=-1
        )
    elif train_stage == 'cot':
        initial_lr = 1e-5 * lr_mul
        max_lr = 5e-5 * lr_mul
        warmup_iters, period = calc_lr_schedular_args(
            epochs=n_epochs,
            all_data_size=107041,
            batch_size=real_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            grpo_steps=-1
        )
    elif train_stage == 'mix':
        initial_lr = 1e-5 * lr_mul
        max_lr = 5e-5 * lr_mul
        warmup_iters, period = calc_lr_schedular_args(
            epochs=n_epochs,
            all_data_size=190247,
            batch_size=real_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            grpo_steps=-1
        )
    elif train_stage == 'pretrain_stage0':
        initial_lr = 1e-4 * lr_mul
        max_lr = 5e-4 * lr_mul
        warmup_iters, period = calc_lr_schedular_args(
            epochs=n_epochs,
            all_data_size=10_000_000, # 14,062,509
            batch_size=real_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            grpo_steps=-1
        )
    elif train_stage == 'distill':
        initial_lr = 1e-5 * lr_mul
        max_lr = 5e-5 * lr_mul
        warmup_iters, period = calc_lr_schedular_args(
            epochs=n_epochs,
            all_data_size=297288,
            batch_size=real_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            grpo_steps=-1
        )
    else: # pretrain_stage1 230087
        initial_lr = 1e-4 * lr_mul
        max_lr = 5e-4 * lr_mul
        warmup_iters, period = calc_lr_schedular_args(
            epochs=n_epochs,
            all_data_size=700000, # 714311
            batch_size=real_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            grpo_steps=-1
        )

    optim_config = train_configs.OptimConfig(
        optim_type='lion',
        enable_lr_scheduler=True,
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

    if train_stage == 'distill':
        from llm_model import LlmModel
        teacher_model = LlmModel(get_model_config(long_context=True))
        teacher_model.to(device=TrainerTools().parallel.device, dtype=torch.float16)
        teacher_model.load_state_dict(torch.load('./last_checkpoint.bin', weights_only=True), strict=False)
        teacher_model.eval()
        teacher_model.requires_grad_(False)

        def kd_teacher_logits_provider(inputs: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
            return teacher_model(inputs, attention_mask=attention_mask)['logits']

        kd_config = train_configs.KDConfig(
            teacher_logits_provider=kd_teacher_logits_provider
        )
    else:
        kd_config = None

    train_config = train_configs.TrainConfig(
        n_epochs=n_epochs,
        batch_size=real_batch_size,
        model_config=model_config,
        file_dataset=file_dataset,
        gradient_accumulation_steps=gradient_accumulation_steps,
        eval_batch_interval=eval_batch_interval,
        loss_config=train_configs.LossConfig(),
        dpo_config=dpo_config,
        grpo_config=grpo_config,
        optim_config=optim_config,
        ds_config=ds_config,
        data_loader_config=data_loader_config,
        kd_config=kd_config,
        init_state_dict=init_state_dict,
        eval_config=train_configs.EvalConfig()
    )

    return train_config


def get_pretrain_stage0_config():
    return _get_train_config(
        n_epochs=1,
        real_batch_size=20,
        file_dataset=PretrainStage0FileDataset(),
        model_config=get_model_config(long_context=False),
        train_stage='pretrain_stage0'
    )


def get_pretrain_stage1_config():
    return _get_train_config(
        n_epochs=1,
        real_batch_size=4,
        file_dataset=PretrainStage1FileDataset(),
        model_config=get_model_config(long_context=True),
        train_stage='pretrain_stage1'
    )


def get_cot_config():
    return _get_train_config(
        n_epochs=2,
        real_batch_size=4,
        file_dataset=COTFileDataset(),
        model_config=get_model_config(long_context=True),
        train_stage='cot'
    )


def get_grpo_config():
    return _get_train_config(
        n_epochs=1,
        real_batch_size=1,
        file_dataset=GRPOFileDataset(),
        model_config=get_model_config(long_context=True),
        train_stage='grpo'
    )


def get_mix_config():
    return _get_train_config(
        n_epochs=2,
        real_batch_size=6,
        file_dataset=MixFileDataset(),
        model_config=get_model_config(long_context=True),
        train_stage='mix'
    )


def get_dpo_config():
    return _get_train_config(
        n_epochs=2,
        real_batch_size=2,
        file_dataset=DPOFileDataset(),
        model_config=get_model_config(long_context=True),
        train_stage='dpo'
    )


def get_distill_config():
    return _get_train_config(
        n_epochs=2,
        real_batch_size=6,
        file_dataset=DistillDataset(),
        model_config=get_small_model_config(),
        train_stage="distill"
    )