# AE-VAE-GAN

## I. Foundations: KL Divergence & Log-Likelihood

### 1. KL Divergence

* Measures difference between two probability distributions **P(x)** and **Q(x)**.
* Formula:
  $$[
  KL(p(x)\parallel q(x)) = \mathbb{E}_{p(x)} \left[ \log \frac{p(x)}{q(x)} \right]
  ]$$
* Used to compare model distribution **q(x)** to true data distribution **p(x)**.
* Properties:

  * Always ≥ 0 (non-negative)
  * Asymmetric: ( KL(P||Q) \neq KL(Q||P) )
  * Lower KL → better approximation of true distribution.
* Application: Core part of **VAE training objective (ELBO)**.

### 2. Log-Likelihood

* Measures how likely observed data is under model parameters.
* Formula:
  $$[
  L(\theta) = \sum_i \log p_\theta(x_i)
  ]$$
* Maximizing log-likelihood → best-fitting model to data.

### 3. Connection Between KL Divergence & Log-Likelihood

* Minimizing ( KL(P_{data} || P_\theta) ) = Maximizing **expected log-likelihood**.
* Used in **variational inference** to approximate complex distributions.
* In generative models (like VAEs), minimizing KL ensures reconstructed data matches true data distribution.

## II. Generative Models Overview

### Key Idea

Generative models learn the **probability distribution** of data to generate new samples that resemble the original dataset.

### Goals

1. **Modeling the data distribution** ( p(x) )
2. **Learning joint distribution** ( p(x, y) ) (for conditional models)
3. **Sampling latent variables** ( z ) to reconstruct or generate data

### Types

* **Unconditional**: Learns ( p(x) ) (e.g., AE, GAN, VAE)
* **Conditional**: Learns ( p(y|x) ) (e.g., Conditional GAN, Diffusion models)

## III. Autoencoders (AE)

### 1. Concept

* Neural network that compresses and reconstructs data.
* Components:

  * **Encoder (f(x))** → Compress input into latent representation **h**
  * **Decoder (g(h))** → Reconstructs original input from **h**
* Objective:
  $$[
  L = |x - \hat{x}|^2
  ]$$

### **2. Latent Space & Representation Learning**

* Latent space = compressed “essence” of input data.
* Represents **features** the model learns automatically from raw input.
* Enables feature extraction, clustering, or data generation.

### **3. Autoencoder Architecture**

* **Encoder**: compresses → smaller latent code (bottleneck)
* **Decoder**: reconstructs original input
* **Bottleneck**: forces network to learn only essential info.

Roles of Bottleneck Layer:

* Encourages **compactness** (compression)
* Prevents **memorization**
* Enables **dimensionality reduction**

### **4. Preventing Overfitting**

Autoencoders combat overfitting using:

* **Bottleneck layers** → reduced dimensionality
* **Denoising** → learn to remove noise
* **Sparsity regularization (L1)** → forces minimal neuron activations
* **Contractive penalty** → penalizes large encoder gradients (robustness)

### **5. Loss Function**

$$[
\text{Loss} = \text{Reconstruction Error} + \text{Regularization Term}
]$$

* Reconstruction: MSE or Cross-Entropy
* Regularization: L1/L2, sparsity, contractive penalty

### **6. Types of Autoencoders**

#### **(a) Undercomplete Autoencoder**

* Latent dim < Input dim
* Forces compact feature learning (like PCA)
* Prevents memorization.

#### **(b) Overcomplete Autoencoder**

* Latent dim > Input dim
* Needs regularization (sparsity, denoising)
* Learns richer but controlled features.

#### **(c) Denoising Autoencoder (DAE)**

* Adds noise to input → reconstructs clean version.
* Improves generalization and noise removal.

#### **(d) Convolutional Autoencoder (CAE)**

* Uses CNN layers → preserves spatial structure.
* Excellent for image compression and denoising.

#### **(e) Sparse Autoencoder**

* Enforces sparsity on hidden activations.
* Regularization via:

  * **L1 penalty**
  * **KL divergence** between target & actual activations

#### **(f) Recurrent Autoencoder (RAE)**

* Uses LSTM or GRU for sequence data.
* Applications: time series, NLP, speech.

#### **(g) Stacked Autoencoder**

* Multi-layered AE (each trained separately → then fine-tuned).
* Learns hierarchical features.

### **7. Applications**

* Image compression and denoising
* Feature extraction / dimensionality reduction
* Anomaly detection
* Image reconstruction & coloring

## **IV. Variational Autoencoders (VAE)**

### **1. Core Concept**

* Introduced by **Kingma & Welling (2014)**.
* Learns **distribution** of latent variables, not fixed values.
* Models uncertainty: each latent feature = probability distribution.

### **2. Architecture**

1. **Encoder** → outputs mean (μ) and variance (σ²) vectors.
2. **Sampling Layer** → samples latent variable *z ~ N(μ, σ²)*.
3. **Decoder** → reconstructs *x̂* from sampled *z*.

### **3. Training Objective**

VAE optimizes **Evidence Lower Bound (ELBO):**
$$[
\mathcal{L} = \mathbb{E}_{q(z|x)} [\log p(x|z)] - KL(q(z|x) | p(z))
]$$

* **Reconstruction Loss**: measures accuracy of reconstruction.
* **KL Divergence**: regularizes latent space (enforces smooth Gaussian distribution).

### **4. Reparameterization Trick**

$$[
z = \mu + \sigma \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, 1)
]$$

* Allows gradients to pass through stochastic sampling.

### **5. Benefits**

* Continuous, smooth latent space
* Models uncertainty
* Generates new samples by sampling from latent space

### **6. Applications**

* Image synthesis (e.g., CIFAR, CelebA datasets)
* Data compression
* Anomaly detection

(*Pages 38–39 show generated face images and CIFAR-10 samples.*)

## **V. Generative Adversarial Networks (GANs)**

### **1. Core Idea**

Proposed by **Ian Goodfellow (2014)**.
Two networks — **Generator (G)** and **Discriminator (D)** — compete:

| Component         | Function                                |
| -- | --- |
| Generator (G)     | Produces fake samples from random noise |
| Discriminator (D) | Classifies real vs fake samples         |

Objective:
$$[
\min_G \max_D \mathbb{E}*{x \sim p*{data}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]
]$$

### **2. Training Process**

* Alternate updates between D and G:

  1. Train **D** to distinguish real/fake.
  2. Train **G** to fool **D**.
* Minimax competition until **Nash Equilibrium** (D cannot distinguish → D(x)=0.5).

### **3. Loss Functions**

* **Discriminator Loss (LD)**:
  $$[
  L_D = -[\log D(x) + \log(1 - D(G(z)))]
  ]$$
* **Generator Loss (LG)**:
  $$[
  L_G = -\log(D(G(z)))
  ]$$
* The second form helps avoid *vanishing gradients* (page 77).

### **4. Challenges & Solutions**

| Problem              | Solution                                        |
| -- | -- |
| Vanishing gradients  | Use **-log(D(G(z)))**, **Wasserstein Loss**     |
| Mode collapse        | Mini-batch discrimination                       |
| Training instability | Gradient penalty, spectral norm, instance noise |

### **5. GAN Variants**

| Type                          | Key Feature                                        |
| -- | -- |
| **Vanilla GAN**               | Basic minimax setup                                |
| **DCGAN**                     | Uses CNNs for stable image generation              |
| **Conditional GAN (CGAN)**    | Adds label info (y) for controlled generation      |
| **Pix2Pix**                   | Image-to-image translation using paired data       |
| **CycleGAN**                  | Unpaired image translation (Monet ↔ Photo)         |
| **WGAN**                      | Uses Wasserstein distance (Earth Mover’s Distance) |
| **WGAN-GP**                   | Gradient penalty for stability                     |
| **StyleGAN / ProgressiveGAN** | Multi-resolution synthesis for realistic faces     |

### **6. Wasserstein GAN (WGAN)**

* Replaces JS divergence with **Wasserstein distance**:
  $$[
  W(P_r, P_g) = \inf_{\gamma \in \Pi(P_r, P_g)} \mathbb{E}_{(x,y) \sim \gamma}[|x - y|]
  ]$$
* Benefits:

  * Smooth gradients
  * Stable training
* Constraint: **1-Lipschitz continuity** enforced via:

  * Weight clipping (WGAN)
  * Gradient penalty (WGAN-GP)

### **7. Advanced Architectures**

* **DCGAN:** transposed convolutions for upsampling.
* **CycleGAN:** two-way generators, cycle-consistency loss.
* **PatchGAN:** discriminator checks local patches (70×70).
* **ProgressiveGAN:** grows resolution layer-by-layer.
* **StyleGAN:** disentangles style features for high realism.

### **8. GAN Applications**

* Image synthesis (face, art, style transfer)
* Image-to-image translation (Pix2Pix, CycleGAN)
* Super-resolution (SRGAN)
* Domain adaptation
* Data augmentation

(*Page 95 visuals show real-to-painting and summer-to-winter transformations.*)

## **VI. Comparative Summary**

| Model                | Core Idea                      | Loss Function     | Use Case                      |
| --- | --- | --- | --- |
| **Autoencoder (AE)** | Reconstruction                 | MSE, BCE          | Compression, denoising        |
| **Variational AE**   | Probabilistic latent space     | ELBO (Recon + KL) | Generation, compression       |
| **GAN**              | Adversarial training           | Minimax / WGAN    | High-quality image generation |
| **WGAN-GP**          | Wasserstein + Gradient Penalty | Stable training   | Realistic images              |
| **CycleGAN**         | Cycle-consistency              | L1 + GAN          | Unpaired domain transfer      |

(*Page 115 diagram summarizes all four model families.*)

## **VII. Diffusion Models (Preview)**

* **Newer generation of generative models** (OpenAI 2021).
* Replace adversarial training with **progressive denoising**.
* Examples:

  * DDPM (Denoising Diffusion Probabilistic Model)
  * Score-based Generative Modeling
* Gradually adds Gaussian noise → learns to reverse process.
  (*Pages 116–120 show foundational diffusion papers.*)