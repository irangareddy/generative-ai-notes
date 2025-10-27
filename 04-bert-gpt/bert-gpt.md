# BERT & GPT

## **I. Background and Evolution of NLP Pre-Training**

### **1. Traditional NLP (Pre-2018)**

* Early NLP models used **static word embeddings**:

  * **Word2Vec**, **GloVe** → each word = single fixed vector.
  * **Problem:** Same embedding for “bank” (river vs. finance).
* Could not capture **contextual meaning** — lacked sentence-level semantics.

### **2. Transition to Contextual Representations**

* Contextual models like **ELMo (2018)** introduced bidirectional understanding:

  * Two LSTMs (left→right and right→left).
  * Generated **context-aware embeddings** for each word.
  * Inspired BERT (2018) and GPT (2018).

## 🧠 **II. BERT — Bidirectional Encoder Representations from Transformers (Google, 2018)**

**Paper:** *Devlin et al., “BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding” (NAACL-HLT 2019)*

### **1. Motivation**

* Previous models (RNNs, GPT) processed text **unidirectionally**.
* BERT captures **both left and right context simultaneously**, enabling richer understanding.

### **2. Architecture Overview**

| Component                    | Role                                                            |
| ---------------------------- | --------------------------------------------------------------- |
| **Encoder-only Transformer** | Uses *multi-head self-attention* bidirectionally                |
| **Input Embeddings**         | Word + Position + Segment embeddings                            |
| **Pre-training Tasks**       | Masked Language Modeling (MLM) + Next Sentence Prediction (NSP) |

* Output: **Contextual embedding** for each token.
* Used **only the encoder** (unlike GPT’s decoder).

### **3. Pre-training Objectives**

#### **a. Masked Language Model (MLM)**

* Randomly mask **15% of words**, model predicts them using surrounding context.
* Example:
  Input → “He went to the [MASK] to withdraw money.”
  Output → “bank”
* Forces **bidirectional understanding**.

**80–10–10 Strategy:**

* 80% → Replace with `[MASK]`
* 10% → Replace with random word
* 10% → Keep unchanged

#### **b. Next Sentence Prediction (NSP)**

* Task: Predict if **sentence B follows A** in the original text.

  * Helps with tasks like **QA** or **NLI** (entailment).
* Example:
  A: “The sky is blue.”
  B: “It might rain today.” → *Not next sentence.*

### **4. Input Representation**

1. **Token Embeddings:** Word-level encodings (via WordPiece).
2. **Segment Embeddings:** Indicate sentence A (0) or B (1).
3. **Position Embeddings:** Capture token order in the sequence.

### **5. Model Sizes**

| Model          | Layers | Hidden Size | Attention Heads | Parameters |
| -------------- | ------ | ----------- | --------------- | ---------- |
| **BERT-Base**  | 12     | 768         | 12              | 110M       |
| **BERT-Large** | 24     | 1024        | 16              | 340M       |

**Training data:**

* BooksCorpus (800M words) + English Wikipedia (2.5B words).

### **6. Fine-Tuning**

* Add small **task-specific head** on top of pretrained BERT:

  * **Classification (e.g., SST-2)** → [CLS] token output → Softmax.
  * **NER / POS tagging** → Sequence tagging on token outputs.
  * **QA (e.g., SQuAD)** → Predict start and end token positions.
* **Fine-tuning updates all layers**.

### **7. Benchmarks**

* Evaluated on **GLUE** (General Language Understanding Evaluation).

  * Tasks include MNLI, SST-2, QQP, MRPC, STS-B, etc.
* Also tested on **SQuAD** (QA) and **NER (CoNLL 2003)** datasets.

**Results:**

* BERT-Large F1 on NER ≈ 92.8 (near state-of-the-art).

### **8. Why BERT Matters**

* Introduced **transfer learning** for NLP:

  1. Pre-train on massive text (unsupervised).
  2. Fine-tune on small labeled datasets.
* Drastically reduced need for large task-specific data.

## **III. Advancements After BERT**

| Model                         | Key Features                                     | Improvement           |
| ----------------------------- | ------------------------------------------------ | --------------------- |
| **RoBERTa (2019)**            | Removed NSP, 10× data, longer training           | Better performance    |
| **ALBERT (2020)**             | Parameter sharing across layers                  | Smaller, efficient    |
| **ELECTRA (2020)**            | Replaces MLM with **“replaced token detection”** | 4× faster training    |
| **Longformer / BigBird**      | Sparse attention for long texts (4096+ tokens)   | Long document support |
| **SciBERT, BioBERT, FinBERT** | Domain-specific fine-tuning                      | Specialized models    |
| **DistilBERT, TinyBERT**      | Distilled smaller versions                       | Faster inference      |

## **IV. ModernBERT — The Next Generation (2024)**

### **1. Motivation**

* BERT limited to 512 tokens and lacks efficiency.
* ModernBERT improves architecture, scalability, and memory usage.

### **2. Key Innovations**

| Feature                                 | Purpose                                                                   |
| --------------------------------------- | ------------------------------------------------------------------------- |
| **RoPE (Rotary Positional Embeddings)** | Replaces absolute embeddings; handles long-range dependencies efficiently |
| **GeGLU Activations**                   | Gated linear units for faster, smoother training                          |
| **Flash Attention v1/v2**               | Memory-efficient attention for long sequences                             |
| **Hybrid Global + Local Attention**     | Reduces cost for long contexts                                            |
| **Unpadding & Sequence Packing**        | Removes padding overhead, faster batching                                 |
| **8192 Token Context Length**           | 16× longer than original BERT                                             |
| **Hardware Optimized**                  | For GPUs like A100, RTX 4090                                              |
| **Data Mix**                            | Trained on 2T tokens (text + code)                                        |

### **3. ModernBERT Advantages**

* Processes **long documents and code** efficiently.
* Outperforms legacy models on:

  * **GLUE (NLU tasks)**
  * **BEIR (IR tasks)**
  * **CodeSearchNet (code retrieval)**
* Balanced trade-off: Accuracy + Efficiency + Adaptability.

### **4. Flash Attention (V1 → V2)**

| Version                                              | Innovation             | Key Benefit                |
| ---------------------------------------------------- | ---------------------- | -------------------------- |
| **V1 (Tri Dao, 2022)**                               | Tiling + Recomputation | 7× speedup, low memory     |
| **V2 (2023)**                                        | Block-sparse support   | 64K token context handling |
| **Impact:** Enables real-time LLMs, reduces latency. |                        |                            |

**Concept:**
Recomputes intermediate attention scores to save memory bandwidth — critical for training and inference of Transformers at scale.

### **5. Comparison Summary**

| Feature             | BERT       | ModernBERT              |
| ------------------- | ---------- | ----------------------- |
| Context length      | 512 tokens | 8192 tokens             |
| Positional Encoding | Absolute   | Rotary (RoPE)           |
| Activation          | GELU       | GeGLU                   |
| Attention           | Dense      | Hybrid / Sparse (Flash) |
| Data                | Text only  | Text + Code             |
| Efficiency          | Moderate   | High (GPU-optimized)    |
| Training Tokens     | 3B         | 2T                      |

## **V. GPT Family — Generative Pre-trained Transformers (OpenAI, 2018–2024)**

### **1. Core Concept**

* **Architecture:** Transformer **decoder-only** (causal attention).
* Predicts next token:
  [
  P(x_t|x_1,...,x_{t-1})
  ]
* Text generation = sequential token sampling.

### **2. Evolution**

| Model        | Params | Data        | Key Innovation              |
| ------------ | ------ | ----------- | --------------------------- |
| GPT-1 (2018) | 117M   | BooksCorpus | Proof-of-concept            |
| GPT-2 (2019) | 1.5B   | WebText     | Long-context fluency        |
| GPT-3 (2020) | 175B   | 600GB       | Few-shot reasoning          |
| GPT-4 (2023) | 1.8T   | Multimodal  | Vision + Text understanding |

### **3. Training Process**

1. **Pre-training:** Predict next token on massive text.
2. **Fine-tuning (SFT):** Supervised on labeled data.
3. **RLHF:** Aligns responses to human feedback for helpfulness, safety, and truthfulness.

### **4. Fine-Tuning Techniques**

| Technique            | Description                                | Pros          | Cons                  |
| -------------------- | ------------------------------------------ | ------------- | --------------------- |
| **SFT (Supervised)** | Train on labeled data (Q&A, summarization) | High accuracy | Costly labeled data   |
| **RLHF**             | Human feedback reward signal               | Human-aligned | Expensive to evaluate |
| **LoRA / Adapter**   | Adds small trainable matrices              | Efficient     | Limited adaptability  |

### **5. Prompt Engineering (GPT Behavior Control)**

* **Zero-Shot:** Provide a clear instruction (no examples).
  → “Summarize this paragraph.”
* **Few-Shot:** Provide 2–3 input-output examples before the prompt.
  → Helps model infer task style or format.
* **Chain-of-Thought (CoT):** Ask model to reason step-by-step.
* **Self-Consistency / Tree-of-Thought:** Sample multiple reasoning paths, select majority.

### **6. GPT Applications**

* Text generation, code completion, summarization, Q&A.
* Tools like **ChatGPT, Codex, DALL·E** use variants of GPT with domain-specific tuning.

## **VI. LLaMA & Efficient Transformer Innovations**

### **1. LLaMA (Meta, 2023–2024)**

* Open-source GPT-style model.
* Key architectural optimizations:

  * **RMSNorm:** Replaces LayerNorm → lower compute.
  * **SwiGLU:** Solves *dying neuron problem*.
  * **Grouped Multi-Query Attention (G-MQA):** Combines speed (MQA) with quality (MHA).
  * **KV Caching:** Stores computed attention pairs for fast generation.
  * **RoPE Encoding:** Better long-context representation.

**Sizes:** 7B → 70B parameters.

### **2. Dying Neuron Problem (ReLU Limitation)**

* **ReLU(x) = max(0, x)**; if neuron’s inputs are always negative, it stops learning.
* **Solutions:**

  * **Leaky ReLU:** allows small negative gradient.
  * **SwiGLU:** smoother activation used in LLaMA.
  * **Better initialization & learning rate control.**

## **VII. vLLM & Paged Attention (Inference Optimization)**

### **1. What is vLLM?**

* **Virtualized Large Language Model inference engine** (2023).
* Purpose: serve LLMs faster and cheaper.
* Open-source, integrates with HuggingFace models.

### **2. Key Features**

| Feature                   | Description                                                                   |
| ------------------------- | ----------------------------------------------------------------------------- |
| **Paged Attention**       | Splits KV cache into small fixed-size memory “pages”. Prevents fragmentation. |
| **KV Cache Optimization** | Stores and reuses prior attention keys for faster generation.                 |
| **Parallel Batching**     | Enables concurrent users, improving throughput.                               |
| **Compatibility**         | Works with GPT, LLaMA, Mistral, etc.                                          |

### **3. Benefits**

* 2.2× higher throughput.
* 50% lower GPU memory usage.
* Powers **Chatbot Arena** and **FastChat** backend.

**Analogy:**
PagedAttention = virtual memory system for attention — retrieves only needed chunks.

## **VIII. Tokenization — The First Step in NLP Models**

### **1. Purpose**

* Break text into manageable units (tokens) → numeric IDs.
* Input → “Hello world” → [15496, 995].
* Output → Decoded back to text.

### **2. Tokenization Methods**

| Type                      | Description            | Example                           |
| ------------------------- | ---------------------- | --------------------------------- |
| **Word**                  | Splits by whitespace   | “playing”                         |
| **Character**             | Each letter is a token | “p”, “l”, “a”…                    |
| **Subword (Most Common)** | Merges frequent chunks | “apologetic” → [“apolog”, “etic”] |
| **Byte-Level**            | Treats bytes as tokens | ByT5, CANINE                      |

### **3. Algorithms**

| Algorithm                    | Used In      | Mechanism                                        |
| ---------------------------- | ------------ | ------------------------------------------------ |
| **BPE (Byte Pair Encoding)** | GPT          | Merge most frequent character pairs              |
| **WordPiece**                | BERT, ALBERT | Merges pairs by scoring (frequency × likelihood) |
| **SentencePiece**            | T5, XLM      | Language-agnostic, handles spaces                |

**Example (WordPiece):**
“hugging” → [“hug”, “##ging”]

### **4. Vocabulary & Special Tokens**

* `[CLS]`, `[SEP]` → BERT.
* `<|endoftext|>` → GPT.
* Vocabulary size defines token efficiency (e.g., 30K for BERT, 50K for GPT-3).

### **5. Why Subwords Work**

* Handles unknown words.
* Keeps vocabulary small.
* Maintains meaning in multilingual, morphologically rich languages.

## **IX. Key Exam Comparison Table**

| Concept        | **BERT**            | **GPT**               |
| -------------- | ------------------- | --------------------- |
| Type           | Encoder-only        | Decoder-only          |
| Direction      | Bidirectional       | Left-to-right         |
| Objective      | MLM + NSP           | Next-token prediction |
| Task Focus     | Understanding       | Generation            |
| Fine-tuning    | Classifier on [CLS] | Prompt-based tuning   |
| Data           | Wikipedia + Books   | Web-scale datasets    |
| Attention      | Full Self-Attention | Causal Attention      |
| Tokenizer      | WordPiece           | BPE                   |
| Output         | Context embeddings  | Generated text        |
| Example Models | BERT, RoBERTa       | GPT-2, GPT-3, ChatGPT |

## 🎯 **X. Exam Takeaways**

1. **BERT** = Bidirectional understanding; foundation of modern NLP.
2. **GPT** = Generative, left-to-right; foundation of LLMs.
3. **ModernBERT** = Efficient, long-context BERT for 2024 tasks.
4. **FlashAttention / RoPE / GeGLU** = efficiency improvements for Transformers.
5. **vLLM** = optimized inference engine (PagedAttention).
6. **Tokenization (WordPiece, BPE)** = crucial preprocessing for all LLMs.
7. **Fine-tuning + RLHF** = key to task-specific adaptation.
