# GAN

## 1. Introduction

**Introduced by:** Ian Goodfellow et al., 2014
**Goal:** Learn to generate new, realistic data (images, text, audio) that mimics real data.

### **1.1 Architecture**

Two networks trained **adversarially** (like a game):

| Component             | Role                     | Input / Output                                   |
| --------------------- | ------------------------ | ------------------------------------------------ |
| **Generator (G)**     | Produces fake data       | Takes random noise `z` → generates sample `G(z)` |
| **Discriminator (D)** | Classifies real vs. fake | Input: data sample → Output: probability(real)   |

### **1.2 Objective Function**

$$
[
\min_G \max_D V(D, G) = \mathbb{E}*{x \sim p*{data}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]
]
$$

* **D** tries to maximize the probability of correctly classifying real vs. fake.
* **G** tries to fool D (make `D(G(z)) ≈ 1`).
* Training = alternating updates → *minimax game.*

### **1.3 Common Problems**

1. **Unstable Training** — D or G overpowers the other.
2. **Mode Collapse** — Generator produces limited variety of outputs.
3. **Vanishing Gradients** — D becomes too good → G stops learning.
4. **No Explicit Likelihood** — GANs don’t estimate probability of data directly.

## **2. Mode Collapse (a.k.a. Generator Collapse)**

**Definition:**
Generator produces **only a few modes** (types) of samples — e.g., all faces look identical though dataset is diverse.

**Why it happens:**

* D finds one “shortcut” that’s easy to fool → G keeps exploiting it.
* Gradients push G to reproduce similar outputs repeatedly.

**Examples:**

* All generated digits look like “3”.
* All generated faces share similar structure.

**Solutions:**

* **Feature Matching** (Salimans et al., 2016)
* **Mini-batch Discrimination**
* **Unrolled GANs** (Metz et al., 2017)
* **WGAN / WGAN-GP** (see below)

## **3. WGAN — Wasserstein GAN**

**Paper:** *“Wasserstein GAN” (Arjovsky et al., 2017)*
**Motivation:** Fix unstable GAN training and mode collapse by changing the **distance metric**.

### **3.1 Problem with Classic GAN Loss**

* Uses **Jensen–Shannon (JS) divergence**, which becomes saturated when D is perfect → **no gradient** for G.
* Causes **training instability** and **mode collapse**.

### **3.2 Solution — Wasserstein (Earth-Mover) Distance**

Instead of JS-divergence, measure **how much "mass" must move** to transform real → generated distribution.
$$
[
W(P_r, P_g) = \inf_{\gamma \in \Pi (P_r, P_g)} \mathbb{E}_{(x, y) \sim \gamma} [|x - y|]
]
$$
Intuition: Smooth, continuous measure of distance between distributions.

### **3.3 WGAN Objective**
$$
[
\max_D \mathbb{E}*{x \sim P_r}[D(x)] - \mathbb{E}*{z \sim P_z}[D(G(z))]
]
$$

$$
[
\min_G \mathbb{E}_{z \sim P_z}[D(G(z))]
]
$$

* **Discriminator renamed → “Critic”** (since it outputs scores, not probabilities).
* Critic must be **1-Lipschitz continuous**, enforced via:

  * Weight clipping (in original WGAN)
  * Gradient penalty (in **WGAN-GP**)

### **3.4 Key Benefits**

| Property             | GAN               | WGAN                                   |
| -------------------- | ----------------- | -------------------------------------- |
| Distance Metric      | JS Divergence     | Wasserstein Distance                   |
| Gradient Stability   | Unstable          | Smooth gradients                       |
| Mode Collapse        | Common            | Significantly reduced                  |
| Output of D          | Probability (0–1) | Real-valued critic score               |
| Lipschitz Constraint | No                | Yes (via clipping or gradient penalty) |

## **4. Pix2Pix — Conditional GAN (Paired Image-to-Image Translation)**

**Paper:** *Isola et al., 2017 – “Image-to-Image Translation with Conditional Adversarial Networks”*

---

### **4.1 Goal**

Convert **one image domain to another** (where paired data is available).
Examples:

* Sketch → Real photo
* Day → Night
* Segmentation mask → RGB image

### **4.2 Architecture**

* Conditional GAN:

  * **Input = Source image + Random noise (optional)**
  * **Output = Target image**
  * **Discriminator** sees both (input, output) pairs and distinguishes real vs. fake.

**Loss:**
$$
[
L = L_{cGAN}(G, D) + \lambda L_{L1}(G)
]
$$
where

$$
[
L_{cGAN}(G, D) = \mathbb{E}*{x,y}[\log D(x,y)] + \mathbb{E}*{x,z}[\log (1 - D(x,G(x,z)))]
]
$$
and
$$
[
L_{L1}(G) = | y - G(x,z) |_1
]
$$

* **L1 Loss** encourages pixel-level accuracy (reduces blurring).

---

### **4.3 Summary**

| Property         | Pix2Pix                                                 |
| ---------------- | ------------------------------------------------------- |
| Type             | Conditional GAN                                         |
| Data Requirement | **Paired datasets**                                     |
| Example          | Edge → Photo                                            |
| Loss             | Adversarial + L1                                        |
| Advantage        | High-quality results                                    |
| Limitation       | Needs aligned pairs (e.g., day↔night photos same scene) |

## **5. CycleGAN — Unpaired Image-to-Image Translation**

**Paper:** *Zhu et al., 2017 – “Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks”*

### **5.1 Motivation**

Paired data (like Pix2Pix) is often unavailable.
CycleGAN enables **domain translation without paired examples**.
Example: Horse ↔ Zebra, Monet ↔ Photograph.

### **5.2 Architecture**

Two Generators + Two Discriminators:

| Network           | Role                             |
| ----------------- | -------------------------------- |
| **G: X → Y**      | Translate domain X to Y          |
| **F: Y → X**      | Translate back to X              |
| **D<sub>Y</sub>** | Discriminator for real vs fake Y |
| **D<sub>X</sub>** | Discriminator for real vs fake X |

### **5.3 Cycle-Consistency Loss**

Ensures translation is **reversible**:

$$
[
L_{cyc} = \mathbb{E}_{x \sim X}[|F(G(x)) - x|*1] + \mathbb{E}*{y \sim Y}[|G(F(y)) - y|_1]
]
$$

### **5.4 Total Loss**

$$
[
L = L_{GAN}(G, D_Y, X, Y) + L_{GAN}(F, D_X, Y, X) + \lambda L_{cyc}(G, F)
]
$$

### **5.5 Summary**

| Property            | CycleGAN                        |
| ------------------- | ------------------------------- |
| Type                | **Unpaired Conditional GAN**    |
| Uses Paired Data?   | No                            |
| Loss                | Adversarial + Cycle Consistency |
| # of Generators     | 2 (forward + backward)          |
| # of Discriminators | 2                               |
| Advantage           | Works on unpaired domains       |
| Limitation          | Training is slower, less stable |

## **6. Quick Summary Table**

| Aspect            | **GAN**             | **WGAN**                    | **Pix2Pix**               | **CycleGAN**                    |
| ----------------- | ------------------- | --------------------------- | ------------------------- | ------------------------------- |
| **Type**          | Unconditional       | Unconditional               | Conditional (paired)      | Conditional (unpaired)          |
| **Loss**          | JS Divergence       | Wasserstein Distance        | Adversarial + L1          | Adversarial + Cycle Consistency |
| **Architecture**  | G + D               | G + Critic (1-Lipschitz)    | G(x,z) + D(x,y)           | 2G + 2D                         |
| **Goal**          | Generate from noise | Stable realistic generation | Image-to-image (paired)   | Image-to-image (unpaired)       |
| **Mode Collapse** | Frequent            | Rare                        | Less frequent             | Rare                            |
| **Constraint**    | None                | Gradient penalty            | Paired data required      | No paired data                  |
| **Output**        | Generic samples     | Realistic continuous scores | Controlled transformation | Cross-domain transformation     |

## **7. Where These Fit in the Generative Model Timeline**

| Year | Model    | Innovation                            |
| ---- | -------- | ------------------------------------- |
| 2014 | GAN      | Adversarial training for generation   |
| 2017 | WGAN     | Stable training, smooth gradients     |
| 2017 | Pix2Pix  | Conditional, paired image translation |
| 2017 | CycleGAN | Unpaired image translation            |

**“How does WGAN solve mode collapse?”**

> By replacing the JS-divergence loss with Wasserstein distance, WGAN provides smooth, continuous gradients that prevent the generator from getting stuck producing a single mode, thus reducing mode collapse and stabilizing training.
