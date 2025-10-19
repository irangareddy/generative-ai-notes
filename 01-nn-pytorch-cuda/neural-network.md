# Neural Network

This notes covers quick recap of the neural network, for detailed notes follow the `deep-learning-notes` which will be available shortly.

### 1. **Foundations**

* **Machine Learning (Arthur Samuel, 1959)** – enabling computers to learn without explicit programming.
* **Deep Learning** – subset of ML inspired by the human brain, using hierarchical feature learning.
* **Software 2.0 (Andrej Karpathy)** – paradigm shift where neural networks “write software” by learning from data instead of explicit coding.

### 2. **Biological Inspiration**

* **Neurons, dendrites, axons, synapses** — information transmitted via electrical impulses and neurotransmitters.
* **Learning in biological systems** — adaptation through synaptic strength changes, creation, or elimination.
* Illustrated by humorous examples (pages 17–18) showing "division of work" among neurons (visual, speech, text).

### 3. **Early Artificial Models**

#### ➤ **McCulloch–Pitts Neuron (1943)**

* First mathematical neuron model.
* **Inputs**: binary signals (0/1)
* **Weights**: excitatory (+) or inhibitory (−)
* **Threshold**: neuron fires if weighted sum ≥ threshold
* **Output**: binary (fires or not)

**Logic gates implemented**: AND, OR, NOT.
**Limitations**:

* No learning capability
* Binary only
* Cannot handle XOR (non-linearly separable data)

Example given: “Road Trip Decision” (p. 26) — shows how weighted inputs decide the output (“Go” or “Cancel”).

### 4. **From ANN to DNN**

#### ➤ **Artificial Neural Network (ANN)**

* Typically shallow (1–2 hidden layers).
* Trained using **gradient descent + backpropagation**.
* Suitable for simple classification/regression.

#### ➤ **Deep Neural Network (DNN)**

* Many hidden layers (>3).
* Capable of **hierarchical feature extraction**.
* Needs **GPUs** for efficient training.
* Applications: image/speech/NLP/reinforcement learning.

Diagrams (pp. 30–33) visually compare shallow vs. deep architectures.

### 5. **Historical Milestones**

* **1943** – McCulloch & Pitts neuron
* **1962** – Rosenblatt’s Perceptron
* **1986** – Backpropagation (Rumelhart, Hinton & Williams)
* **2006** – RBM initialization (Hinton, key DL revival)
* **2009 onward** – GPU acceleration → breakthroughs like AlexNet (2012), GANs (2014), Transformers (2017), GPT-3 (2020), ChatGPT (2022).

Timeline charts on pp. 35–36 summarize the evolution from Perceptron → ChatGPT.

### 6. **How DNNs Work**

* **Input Layer** → feature representation.
* **Hidden Layers** → nonlinear transformations.
* **Output Layer** → predictions (classification/regression).
* **Forward propagation** → compute predictions.
* **Backward propagation** → update weights via gradient descent.
* Illustrated flow diagrams (pp. 39–46).

### 7. **Key Mathematical Components**

#### **Activation Functions** (p. 41–42)

* Linear: rarely used.
* Nonlinear:

  * **Sigmoid**: ( \sigma(x) = \frac{1}{1+e^{-x}} )
  * **Tanh**, **ReLU**, **Leaky ReLU**, **Maxout**, **ELU**
* Each adds nonlinearity, enabling complex pattern learning.

#### **Weights and Biases**

* Determine the strength of connections between neurons.
* Randomly initialized, optimized during training.

### 8. **Training Essentials**

* **Loss function** – measures prediction error.
* **Gradient descent** – updates weights to minimize loss.
* **Backpropagation** – applies chain rule to compute gradients.

Forward and backward pass equations illustrated (pp. 45–47).

### 9. **Generative Models (Extension Topic)**

* Transition from discriminative to **statistical generative models** (pp. 48–52).
* Learn **joint probability p(x, y)** or marginal **p(x)**.
* Examples: Naïve Bayes, GMMs, HMMs, LDA, Bayesian Networks.
* Modern variants: **VAE**, **GAN**, **Diffusion Models**.
* Applications: image synthesis, text generation, anomaly detection.

### 10. **Neural & Autoregressive Models**

* **Neural models** generalize logistic regression using nonlinear transformations (p. 54).
* **Autoregressive (AR)** models — predict future values from past observations.
* **Autocorrelation (ACF)** measures the correlation between a time series and a lagged version of itself.
* Extended models: **ARIMA**, **SARIMA**, **VAR**.

### **Summary of Key Neural Network Subtopics**

| Category                | Topics Covered                         |
| -- | -- |
| Fundamentals            | ML, DL, Software 2.0                   |
| Biological Basis        | Neurons, synapses, learning            |
| Early Models            | McCulloch-Pitts neuron, logic gates    |
| Architecture            | ANN vs DNN                             |
| Training Process        | Forward/backward propagation, loss     |
| Mathematical Components | Weights, biases, activations           |
| History                 | Timeline from 1943–2022                |
| Generative Perspective  | Statistical & neural generative models |
| Extensions              | AR, ARIMA, VAR                         |
