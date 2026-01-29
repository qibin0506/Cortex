import torch

from utils import init_env, get_model_config
from llm_model import LlmModel
from llm_trainer import extract_policy_weights_from_ppo
init_env()

device = 'cpu'
extract_weights = False

policy_model = LlmModel(get_model_config())

policy_weights = extract_policy_weights_from_ppo(
        get_model_config(),
        torch.load('./bin/ppo.bin', weights_only=True)
    )
torch.save(policy_weights, './bin/ppo_policy.bin')