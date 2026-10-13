import numpy as np
import tiktoken
from safetensors import safe_open

def get_embedding_matrix(input: str, tensor_path: str) -> np.ndarray:
    

    try:
        enc = tiktoken.encoding_for_model("gpt2")
        token_list: list[int] = enc.encode(input)
        with safe_open(tensor_path, framework='np') as f:
            token_embedding_tensor = f.get_tensor('wte.weight')
            positional_embeding_tensor = f.get_tensor('wpe.weight')        
    except Exception as e:
        print("Error embedding")
        raise Exception(e)
    
    tokens_embedding: np.ndarray = [token_embedding_tensor[token] for token in token_list]
    positional_embedding: np.ndarray = [positional_embeding_tensor[idx] for idx,token in enumerate(token_list)]
    assert(len(tokens_embedding) == len(positional_embedding) == len(token_list))
    assert(len(tokens_embedding[0]) == len(positional_embedding[0]))
    combined_embeddings: np.ndarray = np.add(tokens_embedding, positional_embedding)
    assert(combined_embeddings[1][1] == (tokens_embedding[1][1] + positional_embedding[1][1]))
    return combined_embeddings