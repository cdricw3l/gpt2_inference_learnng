import numpy as np
import tiktoken
from safetensors import safe_open
import torch

def get_embedding_matrix(input: str, tensor_path: str, framework: str) -> np.ndarray | torch.Tensor:
    """ 
        Return the combined matrice of token embedding and token positional embedding 

        Args:
            input:          the input string to convert as embedding
            tensor_path:    the path to the tensor file
            framework:      The framework you want you tensors in. Supported values: `pt`, `tf`, `flax`, `numpy`.
        Return:
            the return type depend of the framework used
    """

    try:
        enc = tiktoken.encoding_for_model("gpt2")
        token_list: list[int] = enc.encode(input)
        with safe_open(tensor_path, framework=framework) as f:
            token_embedding_tensor: np.ndarray | torch.Tensor = f.get_tensor('wte.weight')
            positional_embeding_tensor: np.ndarray | torch.Tensor = f.get_tensor('wpe.weight')        
    except Exception as e:
        print("Error embedding")
        raise Exception(e)
    
    


    #### error here
    if framework == 'np':
        tokens_embedding: np.ndarray = [token_embedding_tensor[token] for token in token_list]
        positional_embedding: np.ndarray = [positional_embeding_tensor[idx] for idx, token in enumerate(token_list)]
        assert(len(tokens_embedding) == len(positional_embedding) == len(token_list))
        assert(len(tokens_embedding[0]) == len(positional_embedding[0]))
        combined_embeddings: np.ndarray = np.add(tokens_embedding, positional_embedding)
        assert(combined_embeddings[1][1] == (tokens_embedding[1][1] + positional_embedding[1][1]))
    elif framework == 'pt':
        tokens_embedding: torch.Tensor = torch.stack([token_embedding_tensor[token] for token in token_list])
        positional_embedding: torch.Tensor = torch.stack([positional_embeding_tensor[idx] for idx, token in enumerate(token_list)])
        assert(len(tokens_embedding) == len(positional_embedding) == len(token_list))
        assert(len(tokens_embedding[0]) == len(positional_embedding[0]))
        combined_embeddings: torch.Tensor = torch.add(tokens_embedding, positional_embedding)
        assert(combined_embeddings[1][1] == (tokens_embedding[1][1] + positional_embedding[1][1]))
    
    return combined_embeddings