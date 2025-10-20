# TRANSFORMER

### **1. Motivation & Challenges**

* Traditional models like **RNNs** and **LSTMs** process data sequentially → **slow and hard to parallelize**.
* RNNs suffer from **vanishing gradient problem**.
* **ConvS2S (CNN-based)** models allow parallelism but require increasing computation with distance.
* **Transformers** solve both by using **constant-time self-attention** and **parallel computation** through **Multi-Head Attention**.



### **2. RNNs vs Transformers**

| Feature                 | RNN/LSTM          | Transformer                     |
| -- | -- | - |
| Input Order             | Strict sequential | Uses positional encoding        |
| Processing              | Sequential        | Parallel                        |
| Speed                   | Slow              | Fast                            |
| Memory                  | Low               | High (multiple attention heads) |
| Long-Range Dependencies | Weak              | Strong                          |



### **3. Historical Context**

* Introduced by **Vaswani et al., “Attention Is All You Need” (2017)**.
* Eliminated RNNs/CNNs completely.
* Enabled breakthroughs in **machine translation, summarization, Q&A, text generation**.
* Cited over **100,000 times**, central to the **Generative AI revolution** (as shown in the GTC 2024 session, *page 5*).



### **4. Core Concepts & Mathematical Foundations**

* **Dot Product:** Measures similarity; forms the basis for attention.

  * Scaled Dot Product = divide by √(dimension of keys) → stabilizes gradients.
* **Softmax Function:** Converts scores into probabilities that sum to 1.

  * Used to weigh attention focus across words.



### **5. Self-Attention: The Heart of Transformers**

**Goal:** Determine how much attention each word should pay to every other word.

**Steps:**

1. **Create Query (Q), Key (K), Value (V) vectors** → using learned weights.
2. **Compute Attention Scores:** Dot product of Q and K.
3. **Scale:** Divide by √dₖ.
4. **Softmax:** Normalize into probabilities.
5. **Weight Values:** Multiply probabilities by V vectors.
6. **Sum:** Generate output representation for each word.

**Parallelization:**

* All computations are matrix-based for efficiency (see *page 26 diagrams*).



### **6. Positional Encoding**

* Transformers lack sequence order; positional encoding injects **order information**.
* Uses **sine and cosine functions** at different frequencies.
* Ensures words with similar positions have correlated embeddings.
* Example (pages 31–33): “he hit me with a pie” → word order matters.

Formula:
$$[
PE(pos, 2i) = sin(pos / 10000^{2i/d_{model}}), \quad PE(pos, 2i+1) = cos(pos / 10000^{2i/d_{model}})
]$$

### **7. Multi-Head Attention**

* Uses **multiple Q, K, V matrices** (typically 8 heads).
* Each head captures different relational aspects (syntax, semantics).
* Outputs concatenated → projected → richer final representation.
  (*Pages 35–36 visuals illustrate this clearly.*)



### **8. Feed Forward & Residual Connections**

* Each attention output → **Feed Forward Neural Network** (non-linear).
* **Residual connection** + **Layer Normalization** stabilize deep architectures (page 39).
* Repeated **N times** in both encoder and decoder stacks.



### **9. Encoder-Decoder Architecture**

* **Encoder:** Processes full input sequence → contextual embeddings.
* **Decoder:** Generates output one token at a time using:

  * **Masked Self-Attention** → prevents seeing future words.
  * **Cross-Attention** → connects decoder to encoder outputs.

(*Page 41 explains “cross-attention” as introducing encoder information into decoder layers.*)



### **10. Key Advantages**

* Handles **long-range dependencies** better.
* Enables **parallel computation** → faster training.
* Easier **transfer learning** and **fine-tuning**.
* Forms foundation for **BERT** (encoder) and **GPT** (decoder).
  (*Pages 44–45 list practical advantages.*)



### **11. Variants and Applications**

* **BERT:** Uses **only the encoder** → bidirectional context understanding.
* **GPT:** Uses **only the decoder** → autoregressive text generation.
  (*Page 48 diagram contrasts BERT vs GPT.*)



### **12. GPT-3 Model Architecture (Pages 49–50)**

* **175B parameters**, 96 layers, 96 attention heads.
* Uses embedding dimension 12,288; massive parallel computation.
* Repetition of **MLP layers** per transformer block enhances capacity.



### **13. Mechanistic Interpretability (Fact Recall Study)**

(*Pages 52–56: DeepMind and Alignment Forum findings*)

* Transformers recall facts in **three stages**:

  1. **Token Concatenation:** Early layers combine multi-token entities (e.g., “Michael Jordan”).
  2. **Fact Lookup:** MLPs map tokens to attributes (e.g., plays basketball).
  3. **Attribute Extraction:** Attention heads extract relevant facts (e.g., sport → basketball).
* Fact recall is **distributed**, not localized to specific neurons.



### **14. Hallucination**

* Models may generate **plausible but false facts** (e.g., “Mona Lisa painted in 1815”).
* Highlights need for prompt design and factual grounding.
