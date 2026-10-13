from pydantic import BaseModel
import json

class Text_generation_param(BaseModel):
    do_sample: bool
    max_length: int


class LLM_Config(BaseModel):
    activation_function: str
    architectures: list[str]
    attn_pdrop: float
    bos_token_id: int
    embd_pdrop: float
    eos_token_id: int
    initializer_range: float
    layer_norm_epsilon: float
    model_type: str
    n_ctx: int
    n_embd: int
    n_head: int
    n_layer: int
    n_positions: int
    resid_pdrop: float
    summary_activation: str | None
    summary_first_dropout: float
    summary_proj_to_labels: bool
    summary_type: str
    summary_use_proj: bool
    task_specific_params: dict[str, Text_generation_param]
    vocab_size: int

def get_config_data(config_path: str) -> LLM_Config:
    """ This fonction upload the config data file and return an LLM_Config class """
    config: LLM_Config
    try:
        with open(config_path, 'r') as f:
            json_file = json.load(f)
            config  = LLM_Config(**json_file)
            return config
    except Exception as e:
        raise Exception(e)
    return None