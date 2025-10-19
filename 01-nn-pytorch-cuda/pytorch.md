# PyTorch

### 1. **GPU Frameworks Overview**

(Page 93)

The lecture first introduces **multiple frameworks** that simplify GPU programming by abstracting CUDA-level details:

| Framework        | Primary Use                   | Notes                                    |
| - | -- | - |
| **PyTorch**      | Deep learning (Facebook/Meta) | Dynamic computation graphs               |
| **TensorFlow**   | Deep learning (Google)        | Static computation graphs                |
| **JAX**          | Research/ML                   | High-performance autodiff + XLA          |
| **Numba**        | CUDA for Python               | JIT compilation for numeric Python       |
| **cuDF, RAPIDS** | GPU dataframes & ML pipelines | NVIDIA ecosystem for GPU data processing |

**Why these frameworks?**

> They *“abstract low-level GPU details and let developers focus on models and algorithms instead of hardware.”*

### 2. **PyTorch GPU Workflow**

(Page 94, repeated again on page 103)

Typical PyTorch GPU pipeline:

1. **Define device:**

   ```python
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   ```

2. **Move tensors/models to GPU:**

   ```python
   model.to(device)
   data = data.to(device)
   ```

3. **Perform computation:**
   PyTorch automatically executes tensor operations on GPU (e.g., `torch.matmul`, `torch.nn.Linear`, etc.)
4. **Transfer results back to CPU:**

   ```python
   output = output.cpu()
   ```

This model-device transparency is central to PyTorch’s design.

### 3. **Introduction to PyTorch**

(Page 95–96)

**PyTorch** is an:

* **Open-source machine learning library**
* Developed by **Facebook AI Research (FAIR)**
* Written in **C++** with **Python bindings**
* Strong **GPU acceleration** and **autograd (automatic differentiation)**

#### Key Characteristics

* **Dynamic computation graph** → updates at runtime (more flexible than TensorFlow 1.x).
* **NumPy-like syntax** → intuitive for researchers.
* **Automatic gradient computation** → via `autograd`.
* **Predefined modules** → layers, losses, optimizers.

> “Makes it easier to test and develop new ideas.” — *p.96*

### 4. **Why PyTorch?**

(Page 97)

Highlights include:

* **Ease of use**
* **Dynamic graphing**
* **GPU acceleration**
* **Large ecosystem**
* **Strong integration with NVIDIA GPUs**

*(Slide shows NVIDIA PyTorch diagram — performance + productivity balance.)*

### 5. **Tensors**

(Page 98–99)

**Core data structure of PyTorch**, analogous to NumPy’s `ndarray` but GPU-capable.

#### **Key properties:**

* Multi-dimensional arrays.
* Can be moved seamlessly between **CPU and GPU**.
* Supports broadcasting, indexing, slicing, and arithmetic operations.

#### **Tensor Creation Functions:**

| Function                            | Description           |
| -- | --- |
| `torch.zeros(size)`                 | Tensor of zeros       |
| `torch.ones(size)`                  | Tensor of ones        |
| `torch.rand(size)`                  | Uniform random        |
| `torch.randn(size)`                 | Normal distribution   |
| `torch.eye(n)`                      | Identity matrix       |
| `torch.tensor(list)`                | From Python list      |
| `torch.from_numpy(np_array)`        | Converts NumPy array  |
| `torch.arange(start, end, step)`    | Like Python `range()` |
| `torch.linspace(start, end, steps)` | Evenly spaced values  |
| `torch.randint(low, high, size)`    | Random integers       |
| `torch.randperm(n)`                 | Random permutation    |

**Device Allocation Example:**

```python
torch.ones(3, device='cuda')  # Tensor directly on GPU
```

### 6. **Tensor Shape Manipulation**

(Page 100)

Common tensor reshaping operations:

| Function                 | Purpose                     |
|  |  |
| `.reshape()` / `.view()` | Change shape                |
| `.unsqueeze()`           | Add dimension               |
| `.squeeze()`             | Remove singleton dimensions |
| `.permute()`             | Reorder dimensions          |
| `.flatten()`             | Convert to 1D               |
| `.transpose()` / `.T`    | Swap dimensions             |
| `.split()` / `.chunk()`  | Split tensors               |
| `torch.cat()`            | Concatenate                 |
| `torch.stack()`          | Stack along new dim         |

Example (p.100):

```python
x = torch.tensor([1, 2])
y = torch.tensor([3, 4])
concat = torch.cat((x, y))
```

### 7. **Managing Data and Devices**

(Page 101)

**Conversions & Device Management**

1. **NumPy → Tensor:**

   ```python
   t = torch.from_numpy(x_train)
   ```

   (creates CPU tensor)

2. **Tensor → NumPy:**

   ```python
   np_arr = t.numpy()
   ```

   (works only if tensor is on CPU)

3. **Move tensor to GPU:**

   ```python
   t = t.to(device)
   ```

4. **Device fallback:**

   ```python
   device = 'cuda' if torch.cuda.is_available() else 'cpu'
   ```

5. **Check data type:**

   ```python
   type(t) or t.type()
   ```

### 8. **Core PyTorch Modules**

(Page 102)

| Module             | Description                                                        |
| --- | --- |
| `torch.nn`         | Building blocks for neural networks (layers, activations, losses). |
| `torch.autograd`   | Automatic differentiation for training.                            |
| `torch.optim`      | Optimizers (SGD, Adam, RMSProp, etc.).                             |
| `torch.utils.data` | Dataset + DataLoader for batching/shuffling.                       |
| `torchvision`      | Pretrained models + transforms for CV.                             |
| `torch.cuda`       | GPU utilities.                                                     |
| `torchmetrics`     | Evaluation metrics (accuracy, precision, recall, etc.).            |

**Autograd Example:**

```python
loss.backward()  # computes gradients automatically
optimizer.step()  # updates weights
```

### ⚡ 9. **PyTorch GPU Workflow (Recap)**

(Page 103)
Reiterated to emphasize device flow:

1. **Device definition**
2. **Model/data transfer**
3. **Computation**
4. **Result transfer**

📊 *(Slide visually shows CPU ↔ GPU tensor flow.)*

### 🧭 10. **Learning Resources**

(Page 106)

Suggested official tutorials and beginner notebooks:

* [PyTorch Beginner Basics](https://pytorch.org/tutorials/beginner/basics/intro.html)
* [DataQuest: PyTorch for Beginners](https://www.dataquest.io/blog/pytorch-for-beginners/)
* [Colab Demo Notebook](https://colab.research.google.com/drive/1-L2LJmiV_rgCtzsIJOMfF_9fwJuxsj09)

### **Summary of the PyTorch Section**

| Concept               | Description                                    |
| --- | --- |
| **Framework Purpose** | High-level GPU abstraction for ML/DL           |
| **Core Idea**         | Dynamic graphs + automatic differentiation     |
| **Device Workflow**   | Define device → Move data → Compute → Retrieve |
| **Tensors**           | GPU-compatible NumPy equivalents               |
| **Modules**           | nn, autograd, optim, utils.data, cuda          |
| **Ease of Use**       | Pythonic API for deep learning                 |
| **Integration**       | Fully CUDA-compatible                          |
| **Learning Path**     | Tutorials and online resources listed          |
