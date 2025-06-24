import torch
import math
from llm_trainer import train_configs
from llm_model import ModelConfig, VLMConfig, RoPEConfig, MoEConfig
from constant import *
from file_dataset import *
import os


def init_env():
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    os.environ['TOKENIZERS_TYPE'] = 'zh_llama'  # or qwen
    os.environ['TOKEN_DIR'] = './tokens/'

    os.environ['LOG_DIR'] = './log/'

    os.environ['ENABLE_DCP'] = '1'
    os.environ['DIST_CHECKPOINT_DIR'] = 'ckpt_dir'
    os.environ['CHECKPOINT_NAME'] = 'ckpt.pth'
    os.environ['EVAL_CHECKPOINT_NAME'] = 'eval_ckpt.pth'

    # os.environ['DTYPE'] = 'float32'


def get_eval_prompt(content: str, no_reasoning=False) -> str:
    system_prompt = {'role': 'system', 'content': system_prompt_content}
    user_prompt = {'role': 'user', 'content': content}

    template = TrainerTools().tokenizer.apply_chat_template([system_prompt, user_prompt], tokenizer=False)
    if no_reasoning:
        template = f'{template}<assistant><reasoning> </reasoning>'

    return template


def get_model_config():
    return ModelConfig(
        vocab_size=TrainerTools().tokenizer.vocab_size,
        hidden_size=768,
        intermediate_size=2048,
        moe_intermediate_size=1024,
        moe_n_dense_layer=1,
        num_hidden_layers=24,
        num_attention_heads=12,
        num_key_value_heads=4,
        max_position_embeddings=max_seq_len,
        attention_implementation='auto',
        rope_config=RoPEConfig(
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

def calc_lr_schedular_args(
        epochs,
        all_data_size,
        batch_size,
        gradient_accumulation_steps,
        grpo_steps
):
    world_size = TrainerTools().parallel.world_size
    if grpo_steps == -1:
        train_batch_per_world = epochs * all_data_size / batch_size / world_size / gradient_accumulation_steps
    else:
        train_batch_per_world = epochs * (all_data_size / batch_size / world_size) * grpo_steps

    warmup_iters = int(0.2 * train_batch_per_world)
    cosine_annealing_batches = math.ceil(train_batch_per_world - warmup_iters)

    if TrainerTools().parallel.is_main_process:
        print(f'warmup_iters={warmup_iters}, cosine_annealing_batches={cosine_annealing_batches}')

    return warmup_iters, cosine_annealing_batches


def _get_train_config(
        n_epochs: int,
        train_reasoning_model: bool,
        is_sft: bool,
        is_dpo: bool,
        is_grpo: bool,
        real_batch_size: int,
        file_dataset: FileDataset,
        model_config: ModelConfig
):
    gradient_accumulation_steps = 3
    eval_batch_interval = 10 if is_grpo else 100

    ds_config = train_configs.DsConfig(zero_config=train_configs.DsZero3Config())

    loss_config = train_configs.LossConfig(
        critical_tokens=[
            TrainerTools().tokenizer.reasoning_start,
            TrainerTools().tokenizer.reasoning_end,
            TrainerTools().tokenizer.answer_start,
            TrainerTools().tokenizer.answer_end
        ],
        critical_alpha=10.0
    ) if train_reasoning_model else train_configs.LossConfig()

    dpo_config = train_configs.DPOConfig(
        loss_beta=0.1,
        loss_label_smoothing=0.0,
        nll_loss_coef=0.2
    ) if is_dpo else None

    grpo_config = train_configs.GRPOConfig(
        grpo_steps=1,
        clip_eps=0.1,
        kl_weight=0.04,
        group_size=8,
        gen_max_new_tokens=max_seq_len,
        gen_temperature=0.7,
        gen_k=None,
        gen_p=0.5,
        gen_suppress_tokens=None,
    ) if is_grpo else None

    lr_mul = TrainerTools().parallel.world_size
    min_lr_ratio = 0.1

    if is_grpo:
        # 8792
        initial_lr = 5e-6 * lr_mul
        max_lr = 1e-5 * lr_mul
        min_lr = initial_lr * min_lr_ratio
        warmup_iters, period = calc_lr_schedular_args(
            epochs=n_epochs,
            all_data_size=8792,
            batch_size=real_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            grpo_steps=1
        )
        period_mul = 1
    elif is_dpo:
        initial_lr = 1e-8 * lr_mul
        max_lr = 5e-8 * lr_mul
        min_lr = initial_lr * min_lr_ratio
        warmup_iters, period = calc_lr_schedular_args(
            epochs=n_epochs,
            all_data_size=227336,
            batch_size=real_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            grpo_steps=-1
        )
        period_mul = 1
    elif train_reasoning_model:
        initial_lr = 1e-5 * lr_mul
        max_lr = 5e-5 * lr_mul
        min_lr = initial_lr * min_lr_ratio
        warmup_iters, period = calc_lr_schedular_args(
            epochs=n_epochs,
            all_data_size=214457,
            batch_size=real_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            grpo_steps=-1
        )
        period_mul = 1
    elif is_sft:
        initial_lr = 1e-5 * lr_mul
        max_lr = 5e-5 * lr_mul
        min_lr = initial_lr * min_lr_ratio
        warmup_iters, period = calc_lr_schedular_args(
            epochs=n_epochs,
            all_data_size=2253111,
            batch_size=real_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            grpo_steps=-1
        )
        period_mul = 1
    else:
        initial_lr = 1e-4 * lr_mul
        max_lr = 5e-4 * lr_mul
        min_lr = initial_lr * min_lr_ratio
        warmup_iters, period = calc_lr_schedular_args(
            epochs=n_epochs,
            all_data_size=5906957,
            batch_size=real_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            grpo_steps=-1
        )
        period_mul = 1

    lr_config = train_configs.LrConfig(
        enable_lr_scheduler=True,
        initial_lr=initial_lr,
        max_lr=max_lr,
        min_lr=min_lr,
        period=period,
        period_mul=period_mul,
        warmup_iters=warmup_iters
    )

    data_loader_config = train_configs.DataLoaderConfig(
        data_loader_pin_memory=True,
        data_loader_num_workers=0,
        data_loader_shuffle=False,
        data_loader_drop_last=True
    )

    init_state_dict = torch.load('./last_checkpoint.bin', weights_only=True) if os.path.exists('./last_checkpoint.bin') else None

    train_config = train_configs.TrainConfig(
        n_epochs=n_epochs,
        batch_size=real_batch_size,
        model_config=model_config,
        file_dataset=file_dataset,
        gradient_accumulation_steps=gradient_accumulation_steps,
        eval_batch_interval=eval_batch_interval,
        loss_config=loss_config,
        dpo_config=dpo_config,
        grpo_config=grpo_config,
        lr_config=lr_config,
        ds_config=ds_config,
        data_loader_config=data_loader_config,
        kd_config=None,
        init_state_dict=init_state_dict
    )

    return train_config


def get_pretrain_config():
    # warmup_iters=19689, cosine_annealing_batches=78761
    return _get_train_config(
        n_epochs=2,
        train_reasoning_model=False,
        is_sft=False,
        is_dpo=False,
        is_grpo=False,
        real_batch_size=10,
        file_dataset=PretrainFileDataset(),
        model_config=get_model_config()
    )


def get_sft_config():
    return _get_train_config(
        n_epochs=2,
        train_reasoning_model=False,
        is_sft=True,
        is_dpo=False,
        is_grpo=False,
        real_batch_size=10,
        file_dataset=SFTFileDataset(),
        model_config=get_model_config()
    )


def get_reasoning_config():
    return _get_train_config(
        n_epochs=3,
        train_reasoning_model=True,
        is_dpo=False,
        is_sft=True,
        is_grpo=False,
        real_batch_size=10,
        file_dataset=ReasoningFileDataset(),
        model_config=get_model_config()
    )


def get_dpo_config():
    return _get_train_config(
        n_epochs=2,
        train_reasoning_model=False,
        is_sft=False,
        is_dpo=True,
        is_grpo=False,
        real_batch_size=4,
        file_dataset=DPOFileDataset(),
        model_config=get_model_config()
    )


def get_grpo_config():
    return _get_train_config(
        n_epochs=2,
        train_reasoning_model=False,
        is_dpo=False,
        is_sft=False,
        is_grpo=True,
        real_batch_size=3,
        file_dataset=GRPOFileDataset(),
        model_config=get_model_config()
    )