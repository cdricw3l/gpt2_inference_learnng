from safetensors import safe_open
import numpy as np
from llm_config import LLM_Config, get_config_data
from head import Head, WrongHead
from embedding import get_embedding_matrix

TENSOR_PATH: str = "/mnt/data/gpt2/model.safetensors"
CONFIG_PATH: str = "/mnt/data/gpt2/config.json"
NB_OF_HEAD: int = 12


if __name__ == "__main__":

    input: str = "Hello world comment aller vous?"

    try:
        # combined_embeddings is the combinaisaon of token embedding and positional embedding
        combined_embeddings: np.ndarray = get_embedding_matrix(input, TENSOR_PATH)
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
    print(head)
    
    