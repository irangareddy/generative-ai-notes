# Stable Diffusion & Latent Diffusion Models (LDMs)

## **I. Introduction: From Diffusion to Latent Diffusion**

### **1. What Is Stable Diffusion?**

* **Stable Diffusion** = implementation of **Latent Diffusion Models (LDMs)** by **Stability AI**.
* Originated from **Latent Diffusion (Rombach et al., CVPR 2022)** — developed at LMU Munich and Heidelberg University.
* Stable Diffusion = *LDM optimized for open-source, high-fidelity, efficient text-to-image generation*.

### **2. Key Idea**

* Instead of running diffusion directly in **pixel space**, LDMs perform diffusion in a **latent space** — a compressed representation of images.
* This dramatically reduces computational cost while maintaining high perceptual quality.

## **II. Latent Diffusion Model (LDM) Architecture**

### **1. Two-Stage Design**

#### **Stage 1 – Perceptual Compression (Autoencoder)**

* Encoder compresses image ( x ) → latent ( z ).
* Decoder reconstructs ( x ) from ( z ).
* Trained with **perceptual + adversarial losses**.
* Preserves semantic content; discards imperceptible pixel noise.

#### **Stage 2 – Diffusion in Latent Space**

* Perform diffusion (noise addition/removal) on **latent features ( z )**, *not pixels*.
* Inject text or other conditions using **cross-attention**.
* Output → decode back to high-quality image.

**Benefits:**

* Faster & cheaper training
* Higher fidelity
* Model focuses on meaning, not raw pixel textures

### **2. Architecture Summary**

| Component              | Role                        | Typical Settings     |
| --- | --- | --- |
| Encoder                | Compress input image        | CNN + attention      |
| Decoder                | Reconstruct image           | CNN + attention      |
| Diffusion Model        | Denoise latent              | U-Net or Transformer |
| Compression Factor (D) | Spatial downsampling        | 8×                   |
| Channels (C)           | Latent depth                | 16                   |
| Example                | 256×256×3 → 32×32×16 latent |                      |

### **3. Training Procedure**

1. Train **VAE-like encoder–decoder**:

   * Small KL-divergence prior.
   * Add discriminator → sharper outputs (**VAE + GAN hybrid**).
2. Freeze encoder → train **diffusion model** in latent space.
3. During sampling:

   * Start from random latent ( z_T ).
   * Iteratively denoise → ( z_0 ).
   * Decode back to image.

*Modern LDM pipeline = VAE + GAN + Diffusion.*

## **III. VAE, VQ-VAE, and VQ-GAN Foundations**

| Model                            | Year | Key Idea                                            |
| -- | - |  |
| **VAE (Kingma & Welling)**       | 2013 | Continuous latent + KL prior                        |
| **VQ-VAE (van den Oord et al.)** | 2017 | Discrete latent embeddings                          |
| **VQ-GAN (Esser et al.)**        | 2020 | Adds GAN loss for sharp reconstructions             |
| **Used in LDMs**                 | —    | Combines compression + realism for latent diffusion |

**LDM encoder–decoder = VAE trained with GAN-style discriminator.**

## **IV. Conditioning and Cross-Attention**

### **1. Conditioning Inputs**

LDMs can condition generation on multiple inputs:

* **Text (primary mode)**
* **Semantic maps / depth / edge maps**
* **Super-resolution / inpainting guidance**

### **2. Cross-Attention Mechanism**

* Injects conditioning into denoising process.
* Each Transformer block aligns image tokens with text embeddings.
* Enables fine-grained control of spatial and semantic correspondence.

## **V. Text-to-Image Generation**

### **1. Workflow**

**Text Prompt → Text Encoder → Diffusion Model → Decoder → Image**

| Stage                     | Details                                                 |
| - | - |
| **Text Encoder**          | CLIP or T5; converts text to embeddings ( D \times L ). |
| **Diffusion Transformer** | Processes noisy latents + timestep + text embeddings.   |
| **Decoder**               | Converts final latent to RGB image.                     |

**Classifier-Free Guidance (CFG):**

* Controls prompt adherence (typical scale = **7.5**).
* Balances text alignment and image diversity.

### **2. Example Configuration**

* Text Encoder: **T5 + CLIP**
* Encoder/Decoder: **8×8 downsampling**
* Diffusion Model: **12B parameters**
* Patchified: **2×2 → 64×64 = 1024 tokens**
  (*Example: FLUX.1 [dev]*)

### **3. Comparison to DALL·E 2**

| Feature      | DALL·E 2                | Stable Diffusion         |
| --- | --- | --- |
| Architecture | Pixel-space diffusion   | Latent diffusion         |
| Text Encoder | CLIP                    | CLIP / T5                |
| Cost         | High                    | Efficient                |
| Output       | Coherent but less sharp | Realistic & customizable |

## **VI. Diffusion Transformer (DiT)**

### **1. Motivation**

* Replaces U-Net with a **Transformer** backbone for scalability.
* Developed in *Peebles & Xie, ICCV 2023* (“Scalable Diffusion Models with Transformers”).

### **2. Conditioning in DiT**

* **Timestep conditioning:** via scale/shift prediction.
* **Cross-/Joint Attention:** for text or image conditioning.
* DiT architecture scales better for very large models (e.g., video or multimodal diffusion).

## **VII. Text-to-Video Diffusion Models**

### **1. Core Idea**

* Add temporal axis **(T)** to the latent tensor → denoise video frames jointly.
* Each latent = 4D tensor (T × H × W × C).

### **2. Pipeline**

* Text Encoder → 4D DiT → Decoder → Video Frames.
* Example: **Meta MovieGen, OpenAI Sora, Tencent HunyuanVideo.**

**Meta MovieGen example:**

* Text Encoder: UL2, ByT5, MetaCLIP.
* Encoder/Decoder: 8×8×8 downsampling.
* DiT: 30B parameters.
* Patchify 1×2×2 blocks → ~76K latent tokens.
* Output: 1024×576 resolution @ 16 fps for 16 s.

**Applications:** Story videos, generative animation, cinematic rendering.

### **3. Video Model Ecosystem (2024–2025)**

| Model                                           | Developer | Params | Release  |
| --- | --- | --- | --- |
| **Sora**                                        | OpenAI    | —      | 2024 Feb |
| **MovieGen**                                    | Meta      | 14B    | 2024 Oct |
| **HunyuanVideo**                                | Tencent   | 30B    | 2024 Dec |
| **Cosmos**                                      | NVIDIA    | 14B    | 2025 Jan |
| **Wan**                                         | Alibaba   | 30B    | 2025 Mar |
| **Veo 3**                                       | Google    | —      | 2025 Sep |
| (*See page 34 timeline diagram for evolution.*) |           |        |          |

## **VIII. Autoregressive (AR) vs Diffusion Approaches**

### **1. AR Models (e.g., Parti, Imagen)**

* Model image as token sequence.
* Text → Encoder → AR Decoder (predicts next image token).
* ViT-VQGAN converts tokens back to image.
* Converts text-to-image into familiar “next-token prediction” (LLM-style).

### **2. Diffusion Models**

* Iteratively denoise latent → smooth realism and control.
* AR: faster inference, but weaker coherence/physics (esp. for video).
* Diffusion Transformers: state-of-the-art realism (e.g., **Sora 2**).

## **IX. Diffusion Distillation**

### **1. Problem**

* Standard sampling = slow (30–50 denoising steps).

### **2. Solutions**

* **Distillation algorithms** reduce steps (even down to 1).
* Can also *bake in classifier-free guidance (CFG)*.

**Examples:**

| Method                                   | Key Paper       | Year         |
| --- | --- | --- |
| Progressive Distillation                 | Salimans & Ho   | ICLR 2022    |
| Consistency Models                       | Song et al.     | ICML 2023    |
| Adversarial Diffusion Distillation (ADD) | Sauer et al.    | ECCV 2024    |
| Latent ADD (Hi-Res)                      | Sauer et al.    | arXiv 2024   |
| Simplified Consistency                   | Lu & Song       | ICLR 2025    |
| Moment Matching Distillation             | Salimans et al. | NeurIPS 2025 |

**Goal:** Faster, stable, high-resolution diffusion generation.

## **X. Related and Competing Models**

| Model                                | Core Mechanism                           | Domain       | Developer    |
| --- | --- | --- | --- |
| **DALL·E 2 / 3**                     | Diffusion + CLIP guidance                | Text → Image | OpenAI       |
| **Imagen**                           | Pixel-space diffusion + super-resolution | Text → Image | Google       |
| **Parti**                            | Autoregressive                           | Text → Image | Google       |
| **Runway Gen-3, Luma Dream Machine** | Latent video diffusion                   | Video        | Runway, Luma |
| **MusicGen / Suno**                  | Diffusion or AR audio generation         | Audio        | Meta, Suno   |
| **Realtime Voice (OpenAI)**          | Conditional autoregressive               | Speech       | OpenAI       |

(*See pages 46–50 for official links and references.*)

## **XI. Limitations & Challenges**

### **1. Attribute Binding**

* Difficulty handling multiple object properties correctly.
* e.g., “Red cube on blue cube” → may mix colors.

### **2. Text Rendering**

* Struggles to generate accurate text within images (e.g., signs).

### **3. Ethical / Bias Issues**

* Internet-trained models inherit biases.
* Risk of inappropriate or misleading content.

## **XII. Summary Table**

| Stage               | Method                  | Model Type                | Example             |
| - | -- | - | - |
| **Compression**     | VAE / VQ-GAN            | Encoder–Decoder           | LDM                 |
| **Denoising**       | Diffusion / DiT         | Transformer or U-Net      | Stable Diffusion    |
| **Guidance**        | Classifier-Free         | CFG                       | Text prompt control |
| **Speed-up**        | Distillation            | 1–Step generation         | ADD, Consistency    |
| **Video Extension** | 4D Latent Diffusion     | DiT-based                 | Sora, MovieGen      |
| **Future Trends**   | Space-time Transformers | Unified multimodal models | Cosmos, Veo 3       |

You should be able to:

1. **Explain** how LDMs differ from pixel-space diffusion.
2. **Describe** the two-stage training (autoencoder + latent diffusion).
3. **Compare** DALL·E 2, Imagen, and Stable Diffusion.
4. **Explain** Classifier-Free Guidance (CFG) and its role.
5. **Understand** Diffusion Transformers (DiT) and scalability.
6. **Summarize** diffusion distillation methods for faster sampling.
7. **Describe** text-to-video pipelines (MovieGen, Sora).
8. **Discuss** challenges (bias, fidelity, attribute control).
