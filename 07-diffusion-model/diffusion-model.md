# Diffusion Models (Denoising, Implicit, and Score-Based Generative Models)

## **I. Conceptual Foundation**

### **1. Inspiration: Non-Equilibrium Thermodynamics**

* Diffusion models are inspired by **thermodynamic diffusion**:

  * **Forward process:** Adds Gaussian noise → system moves *away from equilibrium* (entropy ↑).
  * **Reverse process:** Gradually removes noise → system moves *toward equilibrium* (entropy ↓).
* The denoising process thus *reverses diffusion*, reconstructing order from randomness. 



### **2. Generative AI Connection**

* **Goal:** Learn to generate realistic data (e.g., images) by reversing noise addition.
* **Forward Diffusion:** Corrupt data step by step (Markov process).
* **Reverse Diffusion:** Neural network learns to remove noise step by step to recover original data.
* **Latent Variable View:** Underlying hidden space captures essential patterns like shape, texture, and style.


## **II. Denoising Diffusion Probabilistic Models (DDPMs)**

### **1. Core Idea**

* Two key processes:

  * **Forward Process:** ( x_0 \to x_1 \to ... \to x_T ) — progressively adds Gaussian noise.
  * **Reverse Process:** Learns ( p_\theta(x_{t-1} | x_t) ) to denoise gradually.
* Introduced by:

  * Sohl-Dickstein et al., *ICML 2015*
  * Ho et al., *NeurIPS 2020*
  * Song et al., *ICLR 2021* 

### **2. Forward Diffusion Process**

$$[
q(x_t | x_{t-1}) = \mathcal{N}(\sqrt{1 - \beta_t} x_{t-1}, \beta_t I)
]$$

* ( x_t ): data with noise at step *t*
* ( \beta_t ): noise variance schedule
* Typically: 1000 steps, ( \beta_1 = 10^{-4} ), ( \beta_T = 0.02 ).
* Markovian: each step depends only on previous state.



### **3. Alpha & Noise Schedule**

* Define:
  [
  \alpha_t = 1 - \beta_t, \quad \bar{\alpha}*t = \prod*{i=1}^t \alpha_i
  ]
  (\bar{\alpha}_t): cumulative product representing how much signal remains after *t* steps.
* Commonly linear or cosine schedules are used for smooth noise growth.



### **4. Reverse (Generative) Process**

* Trained model learns mean and variance:
  $$[
  p_\theta(x_{t-1}|x_t) = \mathcal{N}(\mu_\theta(x_t, t), \Sigma_\theta(x_t, t))
  ]$$
* **Objective:** Minimize KL divergence between true and model posteriors:
  $$[
  KL(q(x_{t-1}|x_t, x_0) | p_\theta(x_{t-1}|x_t))
  ]$$
* Uses **noise prediction network (U-Net)** to predict added noise ( \epsilon_\theta(x_t, t) ).



### **5. Training Objective (ELBO)**

$$[
L = \mathbb{E}*{t, x_0, \epsilon} [| \epsilon - \epsilon*\theta(\sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t}\epsilon, t)|^2]
]$$

* Derived from **variational lower bound** (like in VAEs).
* Training = predict noise at each timestep.



### **6. Key Design Components**

| Component               | Description                                                                  |
| -- | - |
| **Architecture**        | U-Net backbone with ResNet blocks + Self-Attention                           |
| **Time Encoding**       | Sinusoidal or Fourier positional embeddings                                  |
| **Noise Schedule**      | Controls β_t (variance); affects image detail balance                        |
| **Objective Weighting** | Adjust loss per timestep (trade-off between likelihood & perceptual quality) |



### **7. Sampling**

* Start with random noise ( x_T \sim \mathcal{N}(0, I) ).
* Iteratively denoise to obtain ( x_0 ) (final image).
* Each step refines details progressively — coarse → fine.



## **III. Model Improvements and Variants**



### **1. DDIM — Denoising Diffusion Implicit Model**

* **Source:** Song et al., *ICLR 2021*
* **Improvement:** Non-Markovian process allowing **deterministic** sampling.
* **Advantages:**

  * Fewer sampling steps (faster generation).
  * Same trained model as DDPM.
* **Key Idea:**

  * Skip-step denoising (larger time jumps) without losing quality.
  * Maintains continuous latent trajectory.



### **2. Score-Based Diffusion Models (SBDM)**

* Learn **score function** instead of data distribution:
  [
  s_\theta(x, t) \approx \nabla_x \log p_t(x)
  ]
  — the gradient of log-probability (points toward data manifold).
* Continuous-time **stochastic differential equations (SDEs)** model diffusion:

  * **Forward SDE:** adds infinitesimal Gaussian noise.
  * **Reverse SDE:** uses learned score function to denoise.
* Unified with DDPMs → equivalent under different parameterization.
* **Advantages:**

  * Continuous-time formulation = fewer steps.
  * Faster and flexible sampling (adaptive ODE solvers).
  * Up to 100× faster than DDPM. 



### **3. Probability Flow ODE**

* Alternative to stochastic SDE sampling — deterministic variant.
* Uses ODE solvers (Euler/Runge-Kutta) instead of stochastic noise sampling.
* Greatly speeds up generation.



### **4. Connection to Denoising Score Matching**

* Denoising autoencoders (Vincent, 2011) show:
  [
  \nabla_x \log p(x) \propto (x - D(x))
  ]
  — Denoising is a form of *score estimation*.
* Hence, diffusion models can be seen as stacked denoising autoencoders trained progressively across noise scales.



### **5. Advanced Sampling Techniques**

* **Euler-Maruyama** (stochastic Euler method).
* **DPM-Solver** (Lu et al., 2022) — exponential ODE integrators.
* **Runge-Kutta / Heun’s Method** — adaptive multi-step ODE solvers.
* Purpose: improve efficiency and quality in continuous-time diffusion models.



## **IV. Conditional & Guided Diffusion Models**



### **1. Conditional Diffusion**

* Adds conditioning variable *y* (e.g., text prompt, class label) to reverse process:
  [
  p_\theta(x_{t-1}|x_t, y)
  ]
* Enables **text-to-image**, **super-resolution**, or **style-transfer**.



### **2. Classifier Guidance (Dhariwal & Nichol, 2021)**

* Incorporates gradients from an external classifier:
  [
  \nabla_x \log p(x|y) = \nabla_x \log p(x) + \nabla_x \log p(y|x)
  ]
* Combines:

  * Score model (unconditional diffusion)
  * Classifier gradient (external guidance)
* **Effect:** improves fidelity but may reduce diversity. 



### **3. Classifier-Free Guidance (Ho & Salimans, 2021)**

* Avoids separate classifier:

  * Train one model *with* and *without* condition.
  * Randomly drop condition during training (prob. *p_drop*).
  * During inference, combine both:
    [
    s_\theta^{guided}(x_t, y) = s_\theta(x_t, y) + \omega (s_\theta(x_t, y) - s_\theta(x_t))
    ]
  * ( \omega ): guidance scale (trade-off factor).
* **Trade-off:**

  * Large ω → higher quality but lower diversity. (*See page 67 chart: ω = 1 vs. ω = 3*)
* Used in **Stable Diffusion** for prompt-based control.



### **4. Stable Diffusion (2022)**

* Implementation combining:

  * **Latent diffusion** (operating in compressed feature space).
  * **Classifier-free guidance** for controllable text-to-image synthesis.
* Efficient, open-source, and customizable (see *jalammar.github.io/illustrated-stable-diffusion/*).



## **V. Key Equations Summary**

| Concept                  | Equation                                          | Description                                                  |                                   |
|  | - |  |  |
| Forward Process          | ( q(x_t                                           | x_{t-1}) = \mathcal{N}(\sqrt{1-\beta_t}x_{t-1}, \beta_t I) ) | Adds Gaussian noise progressively |
| Reverse Process          | ( p_\theta(x_{t-1}                                | x_t) = \mathcal{N}(\mu_\theta, \Sigma_\theta) )              | Learns to denoise                 |
| Objective                | ( L = | \epsilon - \epsilon_\theta(x_t, t)|^2 )   | Predict noise                                                |                                   |
| Score Function           | ( s_\theta(x, t) = \nabla_x \log p_t(x) )         | Gradient of log-probability                                  |                                   |
| Classifier-Free Guidance | ( s_{guided} = s(x, y) + \omega(s(x, y) - s(x)) ) | Weighted trade-off                                           |                                   |



## **VI. Practical Considerations**

| Factor                   | Importance                                                                           |
|  |  |
| **Noise schedule (β_t)** | Defines how fast data is corrupted; affects balance between global and fine details. |
| **Architecture**         | U-Net + attention → best performance.                                                |
| **Training Weighting**   | Balances early/late timestep loss (critical for image detail).                       |
| **SNR parameterization** | Used for adaptive noise control.                                                     |
| **Sampling Steps**       | Fewer steps = faster generation, but may lose realism.                               |



## **VII. Evolution Summary**

| Model                       | Key Feature                              | Speed               | Type                        |
|  | - | - |  |
| **DDPM (Ho, 2020)**         | Probabilistic, Markovian reverse process | Slow                | Discrete-time               |
| **DDIM (Song, 2021)**       | Deterministic, non-Markovian             | Fast                | Discrete-time               |
| **SBDM (Song, 2021)**       | Continuous-time SDE/ODE                  | Very Fast           | Continuous                  |
| **Classifier-Guided**       | Uses external classifier gradient        | Moderate            | Conditional                 |
| **Classifier-Free (CFG)**   | Joint unconditional/conditional training | Fast & High-quality | Conditional                 |
| **Stable Diffusion (2022)** | Latent + CFG                             | Real-time           | Latent continuous diffusion |

Be able to:

* Explain **forward vs reverse diffusion** mathematically.
* Derive **noise-prediction loss (ELBO)**.
* Differentiate **DDPM**, **DDIM**, and **Score-based models**.
* Explain **SDE → ODE equivalence**.
* Discuss **classifier vs classifier-free guidance**.
* Understand trade-off between **quality & diversity (CFG weight ω)**.
* Outline **architecture choices** and **sampling acceleration** methods.