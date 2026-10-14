import torch
import numpy as np 

def get_moyenne(vec: torch.Tensor | np.ndarray, d_model: int):
    return 1 / d_model * sum(vec)

def get_variance(vec: torch.Tensor | np.ndarray, moyenne):
    return  1 / 768 * sum([pow((x - moyenne),2) for x in vec])

def normalisation(vec: torch.Tensor | np.ndarray, moyenne: float, variance: float, layer_norm_epsilon:float) -> torch.Tensor | np.ndarray:
    return torch.Tensor([(x-moyenne)/ torch.sqrt(variance + layer_norm_epsilon)  for x in vec])

def get_layer_norm_out(vec: torch.Tensor | np.ndarray, gamma: torch.Tensor | np.ndarray, beta: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    return vec * gamma + beta


def test_layer_norm(head, combined_embeddings, llm_config, framework) -> None:
    #Normalisation:
        
        # 1. moyenne
        moyenne_x00 = 1 / 768 * sum(combined_embeddings[0])
        moyenne_x01 = get_moyenne(combined_embeddings[0], llm_config.n_embd)
        assert(moyenne_x00 == moyenne_x01)
    
        # 2. Variance
        variance_x00 = 1 / 768 * sum([pow((x - moyenne_x00),2) for x in combined_embeddings[0]])
        variance_x01 = get_variance(combined_embeddings[0], moyenne_x01)
        assert(variance_x00 == variance_x01)
        
        # 3. Normalisation
        norm_0 = torch.Tensor([(x-moyenne_x00)/ torch.sqrt(variance_x00 + llm_config.layer_norm_epsilon)  for x in combined_embeddings[0]])
        norm_1 = normalisation(combined_embeddings[0],moyenne_x01, variance_x01, llm_config.layer_norm_epsilon)
        #assert(norm_0 == norm_1)
    
    
        print(f"moyenne inital: {moyenne_x00} after normalisation : {get_moyenne(norm_1,llm_config.n_embd)}")
        print(f"variance inital: {variance_x00} after normalisation : {get_variance(norm_1, get_moyenne(norm_1,llm_config.n_embd))}")
        # fast version with torch
        #print(f"La moyenne est: {combined_embeddings.mean(dim=1, keepdim=True)}")
    
    
        # creation de la matrice de sortie:
        gamma = head.get_matrice(head.ln_1_weight, framework)
        beta = head.get_matrice(head.ln_1_bias, framework)
        
        layer_norm_out: torch.Tensor = get_layer_norm_out(norm_1, gamma, beta)
    
        print(combined_embeddings[0].device)
        print(gamma.device)
        print(beta.device)
        print(layer_norm_out.device)
    
        import torch.nn.functional as F
        reference = F.layer_norm(
            combined_embeddings[0],
            normalized_shape=(768,),
            weight=gamma,
            bias=beta,
            eps=llm_config.layer_norm_epsilon
        )
        print("Résultats identiques :", torch.allclose(
            layer_norm_out, reference, atol=1e-5
        ))
        print("Différence maximale :", (
            layer_norm_out - reference
        ).abs().max().item())
    

