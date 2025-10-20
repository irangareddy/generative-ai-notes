# Neural Network Advanced

### 1. Backpropagation & Loss Functions (pp. 3–8)

**Backpropagation (Backward Propagation)**

* Begins at the **output layer** and moves **backward** to the input.
* Purpose → update weights and biases to minimize loss.
* Uses the **chain rule of calculus** to compute gradients of the loss (L) with respect to each weight (w_i).
* Each weight update:
  $$
  [
  w_i \leftarrow w_i - \alpha \frac{\partial L}{\partial w_i}
  ]
  $$
  where (\alpha) = learning rate.

**Loss Function**

* Measures how far predictions deviate from target values.
* Common examples → Mean Squared Error, Cross-Entropy Loss.
* Goal: reduce loss → better model accuracy.

**Gradient Descent** (pp. 5–6)

* Finds minima of convex/concave functions.
* **Gradient Ascent** for maximizing likelihood, **Gradient Descent** for minimizing loss.
* Update rule:
  $$[
  w^{(t+1)} = w^{(t)} - \eta,\nabla_w L(w)
  ]$$
  where (\eta) is the **learning rate**.

**Learning Rate Effect (Fig. p. 6)**

* Too small → slow convergence.
* Too large → overshoot or divergence.

### 2. Forward & Backward Training Steps (pp. 7–9)

**Training Pipeline (Slide 7 diagram):**

1. **Forward Pass:** compute net input → activation → output.
2. **Error Computation:** compare output vs. target.
3. **Backward Pass:** propagate gradients via chain rule.
4. **Weight Update:** apply gradient descent.

**Chain Rule Visualization (pp. 8–21):**

* Fei-Fei Li & Andrej Karpathy example:
  ( f(x,y,z) = (x + y)z )
* Stepwise derivative flow → computes (\frac{\partial f}{\partial x}, \frac{\partial f}{\partial y}, \frac{\partial f}{\partial z}).
* Emphasizes local gradients (\frac{\partial z}{\partial x}) and global gradients (\frac{\partial L}{\partial x}).

The slides (pp. 22–27) show the **local gradient + global gradient** propagation:
[
\frac{\partial L}{\partial x} = \frac{\partial L}{\partial z} \cdot \frac{\partial z}{\partial x}
]
This builds intuition for how **small partials compose** through many layers.

### 3. Backpropagation in Deep Networks (pp. 28–30)

**Diagram (p. 28):** deep feed-forward net with many hidden layers.
**Training Steps (p. 29–30):**

1. Forward propagation → compute activations.
2. Compute error at output.
3. Backward propagation → compute gradients layer-by-layer.
4. Compute ∂L/∂W and ∂L/∂b.
5. Update weights using gradient descent.

### 4. Common Training Challenges (pp. 31–33)

| Problem                             | Description                                                         | Mitigation                                               |
| -- | - | -- |
| **Bad Initialization**              | Poor starting weights → stuck in local minima.                      | Xavier/He initialization                                 |
| **Overfitting**                     | Too many parameters → memorizes training data.                      | Regularization (L1/L2, Dropout, Early Stop)              |
| **Scaling Inputs**                  | Large variance in features → slow training.                         | Normalize inputs (mean 0, std 1)                         |
| **Too few/too many hidden layers**  | Affects capacity and training stability.                            | Cross-validate architecture                              |
| **Vanishing / Exploding Gradients** | Gradients shrink or grow exponentially → no learning / instability. | Proper initialization, ReLU, BatchNorm, skip connections |

**Page 33 Diagram – The Vanishing Gradient Problem:**
When recurrent weights (W_{rec}) are small → gradients vanish; when large → explode.

### 5. Deep Learning Breakthroughs (pp. 34–36)

**Geoffrey Hinton’s Contributions:**

* *Pre-training Deep Belief Networks* (2006).
* Identified why early NNs failed:

  * Labeled datasets too small.
  * Computers too slow.
  * Poor weight initialization.
  * Wrong activation functions.
* Introduced **RBM initialization**, later **Xavier and He initialization** methods (p. 36).
  [
  w \sim U\left[-\frac{\sqrt{6}}{\sqrt{n_i + n_{i+1}}},\ \frac{\sqrt{6}}{\sqrt{n_i + n_{i+1}}}\right]
  ]
  (Xavier) and
  [
  w \sim \mathcal{N}!\left(0,\ \frac{2}{n_i}\right)
  ]
  (Kaiming He).

### 6. Optimizers (pp. 37–38)

**Concept:** refine plain Gradient Descent to accelerate or stabilize training.

| Optimizer        | Memory   | Tunables | Strength                            | Weakness                  |
| - | -- | -- | -- | - |
| **SGD**          | 0 bytes  | 1        | simple & robust                     | slow convergence          |
| **Momentum SGD** | +4 bytes | 2        | faster directional updates          | sensitive to η, β         |
| **AdaGrad**      | +4 bytes | 3        | handles sparse features             | vanishing learning rate   |
| **RMSProp**      | +4 bytes | 3        | good for non-stationary data        | may oscillate             |
| **Adam**         | +8 bytes | 3        | adaptive moments → fast convergence | can overfit               |
| **AdamW**        | –        | 3        | adds weight decay regularization    | needs more memory         |
| **LARS**         | –        | 4        | scales well to large batches        | gradient computation cost |

### 7. Regularization & Overfitting (pp. 39)

**Regularization Strategies:**

1. **Get more data** → reduces variance.
2. **Reduce network capacity** → simpler model.
3. **Weight Regularization** → penalize large weights.

   * L1 ( Lasso ): (\lambda \sum |w_i|) → sparsity.
   * L2 ( Ridge ): (\lambda \sum w_i^2) → weight decay.
4. **Dropout (p ≈ 0.2 – 0.5)** → randomly set neurons to 0 during training.

### **Summary**

| Concept              | Key Takeaway                                           |
| --- | --- |
| **Backpropagation**  | Computes gradients using chain rule to update weights. |
| **Gradient Descent** | Iteratively minimizes loss via learning rate steps.    |
| **Training Loop**    | Forward → Error → Backward → Update.                   |
| **Issues**           | Vanishing gradients, local minima, overfitting.        |
| **Solutions**        | Proper init, normalization, ReLU, regularization.      |
| **Optimizers**       | Adam, RMSProp extend SGD for faster training.          |
| **Regularization**   | Dropout and L1/L2 penalties prevent overfitting.       |
