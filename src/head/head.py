from llm_config import LLM_Config
from enum import Enum
import numpy as np
from safetensors import safe_open
from llm_config import LLM_Config

class WrongHead(Exception):
    def __init__(self, msg = "Wrong head id, choose value between 0 and 11"):
        super().__init__(msg)


class Head_matrix:
    attn_bias: str
    attn_c_attn_bias: str
    attn_c_attn_weight: str
    attn_c_proj_bias: str
    attn_c_proj_weight: str
    ln_1_bias: str
    ln_1_weight: str
    ln_2_bias: str
    ln_2_weight: str
    mlp_c_fc_bias: str
    mlp_c_fc_weight: str
    mlp_c_proj_bias: str
    mlp_c_proj_weight: str

class Head(Head_matrix):
    id: int
    tensor_path: str
    matrices: Head_matrix
    llm_config: LLM_Config

    
    def __init__(self, tensor_path: str, id: int, llm_config: LLM_Config):
        if id < 0 or id > llm_config.n_head - 1:
            raise WrongHead
        self.id = id
        self.tensor_path = tensor_path
        self.attn_bias = f"h.{id}.attn.bias"
        self.attn_c_attn_bias = f"h.{id}.attn.c_attn.bias"
        self.attn_c_attn_weight = f"h.{id}.attn.c_attn.weight"
        self.attn_c_proj_bias = f"h.{id}.attn.c_proj.bias"
        self.attn_c_proj_weight = f"h.{id}.attn.c_proj.weight"
        self.ln_1_bias = f"h.{id}.ln_1.bias"
        self.ln_1_weight = f"h.{id}.ln_1.weight"
        self.ln_2_bias = f"h.{id}.ln_2.bias"
        self.ln_2_weight = f"h.{id}.ln_2.weight"
        self.mlp_c_fc_bias = f"h.{id}.mlp.c_fc.bias"
        self.mlp_c_fc_weight = f"h.{id}.mlp.c_fc.weight"
        self.mlp_c_proj_bias = f"h.{id}.mlp.c_proj.bias"
        self.mlp_c_proj_weight = f"h.{id}.mlp.c_proj.weight"
    
    def get_matrice(self, matrice: str) -> np.ndarray:
        try:
            with safe_open(self.tensor_path, framework="np") as f:
                tensor = f.get_tensor(matrice)
                return tensor
        except Exception as e:
            print(f"Errors: {e}")

    def __str__(self):
        return f"Head: {self.id}\n{self.attn_bias}\n{self.attn_c_attn_bias}\n{self.attn_c_attn_weight}\n\
{self.attn_c_proj_bias}\n{self.attn_c_proj_weight}\n{self.ln_1_bias}\n{self.ln_1_weight}\n\
{self.ln_2_bias}\n{self.ln_2_weight}\n{self.mlp_c_fc_bias}\n{self.mlp_c_fc_weight}\n\
{self.mlp_c_proj_bias}\n{self.mlp_c_proj_weight}" 


