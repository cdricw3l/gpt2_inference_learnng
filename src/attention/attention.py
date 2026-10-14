
import torch
import numpy as np
from llm_config import LLM_Config

#matrice = 768 * 2304

# manual split version off the attention matrice.
# see torch.split in W_qkv class below for better implementation

def get_Q_matrix(matrice: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    Q: torch.Tensor = matrice[0:, 0:768] 
    print(Q.shape)

def get_K_matrix(matrice: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    K: torch.Tensor = matrice[0:, 768:1536] 
    print(K.shape)

def get_V_matrix(matrice: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    V: torch.Tensor = matrice[0:, 1536:] 
    print(V.shape)



class W_qkv:

    """ This class split the global attention matrix to qkv matrice"""
    W_q:torch.Tensor | np.ndarray
    W_k:torch.Tensor | np.ndarray
    W_v:torch.Tensor | np.ndarray

    def __init__(self, matrix: torch.Tensor | np.ndarray, config: LLM_Config):
        self.W_q, self.W_k, self.W_v = torch.split(matrix, config.n_embd, dim=1)