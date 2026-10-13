from safetensors import safe_open
import numpy as np
from llm_config import LLM_Config, get_config_data
from head import Head, WrongHead
from embedding import get_embedding_matrix
import torch
from enum import Enum

TENSOR_PATH: str = "/mnt/data/gpt2/model.safetensors"
CONFIG_PATH: str = "/mnt/data/gpt2/config.json"
NB_OF_HEAD: int = 12


class Tensor_framework(Enum):
    NP = 'np'
    PT= 'pt'
    TF= 'tf'
    FLAX= 'flax'


if __name__ == "__main__":

    device_count: int = torch.cuda.device_count()
    if (device_count > 0):
        print(f"GPU INITIALISATION OK -> Nb of device: {device_count} -> Name: {torch.cuda.get_device_name()}")
    input: str = "Hello world comment aller vous?"

    try:
        # combined_embeddings is the combinaisaon of token embedding and positional embedding
        
        combined_embeddings: np.ndarray = get_embedding_matrix(input, TENSOR_PATH, Tensor_framework.PT.value)
        llm_config: LLM_Config = get_config_data(CONFIG_PATH)
        ##print(llm_config.model_dump_json(indent=2))
    except Exception as e:
        print(f"LLM config upload error: {e}")
        exit(1)
    try:
        head = Head(TENSOR_PATH, 0, llm_config)
    except WrongHead as e:
        print(f"Error: {e}")
        exit(1)
    
    