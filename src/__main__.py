import torch
import math
import numpy as np
from enum import Enum
from llm_config import LLM_Config, get_config_data
from head import Head, WrongHead
from embedding import get_embedding_matrix
from attention import get_Q_matrix, get_K_matrix, get_V_matrix, W_qkv
from normalisation import get_variance, get_moyenne, normalisation, get_layer_norm_out, test_layer_norm

TENSOR_PATH: str = "/mnt/data/gpt2/model.safetensors"
CONFIG_PATH: str = "/mnt/data/gpt2/config.json"
NB_OF_HEAD: int = 12

class Tensor_framework(Enum):
    NP = 'np'
    PT= 'pt'
    TF= 'tf'
    FLAX= 'flax'

FRAMWORK : str = Tensor_framework.PT.value


if __name__ == "__main__":

    device_count: int = torch.cuda.device_count()
    if (device_count > 0):
        print(f"GPU INITIALISATION OK -> Nb of device: {device_count} -> Name: {torch.cuda.get_device_name()}")
    else:
        print("Error initialisation torch rocm")
        exit(1)
    
    input: str = "Hello world comment aller vous?"
    try:
        # combined_embeddings is the combinaisaon of token embedding and positional embedding
        combined_embeddings: np.ndarray = get_embedding_matrix(input, TENSOR_PATH, FRAMWORK)
        llm_config: LLM_Config = get_config_data(CONFIG_PATH)
        print(llm_config.model_dump_json(indent=2))
    except Exception as e:
        print(f"LLM config upload error: {e}")
        exit(1)
    try:
        head = Head(TENSOR_PATH, 0, llm_config)
    except WrongHead as e:
        print(f"Error: {e}")
        exit(1)
    attention_matrice = head.get_matrice(head.attn_c_attn_weight, FRAMWORK)


    # layer norm before self attention
    test_layer_norm(head, combined_embeddings, llm_config, FRAMWORK)

    