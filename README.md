## install pytorch for rocm

* uv pip install torch --torch-backend=auto



# Etape 1: Tokenisation de l'entré

# Etape 2: creation de la matrice d'embedding

# Etape 3: creation de la matrice d'embedding positionnelle

# Etape 4: Addition de la matrice d'embedding et de la matrice d'embedding positionnelle

# Etape 5: Layer Normalization (GPT-2)

https://arxiv.org/pdf/1607.06450 (page 4, section 2.3 — Model.)


La normalisation est appliquée indépendamment à chaque token, sur les 768 dimensions de son vecteur d'embedding.

### 1. Calcul de la moyenne

$$
\mu = \frac{1}{768} \sum_{i=1}^{768} x_i
$$

### 2. Calcul de la variance

$$
\sigma^2 = \frac{1}{768} \sum_{i=1}^{768} (x_i - \mu)^2
$$

### 3. Normalisation du vecteur

$$
\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}
$$

Avec :

- $\epsilon = 10^{-5}$ : constante de stabilité numérique.
- $\hat{x}_i$ : composante normalisée du vecteur.

### 4. Application des paramètres appris

$$
y_i = \gamma_i \hat{x}_i + \beta_i
$$

Avec :

- $\gamma_i$ : coefficient de mise à l'échelle (`ln_1.weight`).
- $\beta_i$ : biais de décalage (`ln_1.bias`).

### Équation complète

$$
\boxed{
y_i = \gamma_i \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta_i
}
$$



# Ressources

https://jalammar.github.io/illustrated-gpt2/
https://www.kapilsharma.dev/posts/exploring-gpt2/
