# CUDA and Parallel Computing

### 1. **Introduction to Heterogeneous Parallel Computing (HPC)**

* CUDA stands for **Compute Unified Device Architecture**, NVIDIA’s platform for **heterogeneous computing** — using both **CPUs and GPUs** together for optimal performance.
* This section contrasts **traditional CPU-based computing** with **parallel GPU computing**.

### 2. **Traditional Computing**

* Based on the **Von Neumann architecture**, where:

  * Instructions and data share the same memory path.
  * Execution is **serial** — one instruction after another.
* **Limitations**:

  * Expensive to scale.
  * Limited by **bus speed** (memory bottleneck).
  * Inefficient for data-parallel workloads.

*(See page 62 figure: "Traditional Computing" — serial tasks lined up sequentially before the CPU.)*

### 3. **Parallel Computing**

* A paradigm where **multiple computations are executed simultaneously**.
* **Goal:** Divide large problems into smaller, independent tasks.

**Two main aspects:**

1. **Computer Architecture** (hardware) — multicore CPUs, GPUs, clusters.
2. **Parallel Programming** (software) — writing code that distributes tasks efficiently.

*(Page 63)*

> “Parallel computing solves problems faster by executing tasks simultaneously.”
> — Ref: *Cheng, Grossman, & McKercher (2014), Professional CUDA C Programming.*

### 4. **Harvard vs. Von Neumann Architecture**

* **Harvard Architecture** (page 64):

  * **Separate storage** for data and instructions.
  * Faster access since they don’t share the same bus.
* **Von Neumann Architecture:**

  * Shared memory path → potential bottlenecks.

*(Diagram on page 64)* shows **Instruction Memory** and **Data Memory** as distinct blocks feeding into the **CPU’s ALU** and **Control Unit** — key difference from Von Neumann.

### 5. **Sequential vs. Parallel Programs**

* **Sequential Program** (page 65):
  Tasks execute one by one (linear order).

* **Parallel Program** (page 66):
  Tasks can run concurrently, though some parts remain sequential (Amdahl’s Law).

*(Figures on pp. 65–66)* visualize this contrast — sequential execution line vs. multiple simultaneous task bars.

### 6. **Types of Parallelism**

(page 67)

| Type                 | Description                                                         | Example                                        |
| -- | - | - |
| **Task Parallelism** | Different tasks/functions run independently on separate processors. | Running different simulations at once.         |
| **Data Parallelism** | Same operation applied to multiple data points in parallel.         | Matrix multiplication, pixel-level operations. |

CUDA focuses primarily on **data parallelism** — same operation on many data elements.

### 7. **Data Partitioning in CUDA**

(page 68–69)

CUDA programs **divide data among threads** using:

* **Block Partitioning:** Each thread gets one contiguous data block.
* **Cyclic Partitioning:** Threads handle data in a round-robin manner.

*(Figure on page 69)* shows both 1D and 2D block partitions visually — crucial for balancing GPU workloads.

### 8. **Flynn’s Taxonomy**

(page 70)

Classifies computer architectures by **instruction** and **data streams**:

| Type     | Description                                                   |
| -- | - |
| **SISD** | Single Instruction, Single Data (classic CPU).                |
| **SIMD** | Single Instruction, Multiple Data (GPUs, vector processors).  |
| **MISD** | Multiple Instruction, Single Data (rare).                     |
| **MIMD** | Multiple Instruction, Multiple Data (modern multi-core CPUs). |

CUDA aligns with **SIMD / SIMT (Single Instruction, Multiple Threads)** principles.

### 9. **Performance Metrics**

(page 71)

Goals of parallel architecture:

* **Decrease Latency** – reduce time per operation.
* **Increase Bandwidth** – higher data throughput (MB/s or GB/s).
* **Increase Throughput** – more operations per second (GFLOPS, TFLOPS).

### 10. **Memory Architectures**

(page 72)

Two major types:

1. **Multi-node (Distributed Memory)** – each processor has its own memory.
   Communication via interconnection networks.
2. **Multiprocessor (Shared Memory)** – processors share common memory through a bus.

*(Figure on page 72)* clearly shows both designs — distributed vs. shared memory layouts.

### 11. **Homogeneous vs. Heterogeneous Computing**

(page 73–74)

* **Homogeneous:** Only one type of processor (e.g., CPU cluster).
* **Heterogeneous:** Combination of CPUs and GPUs.

  * GPU = co-processor for data-heavy computations.
  * CPU = host controlling task delegation.

*(Diagram on page 74)* shows CPU and GPU connected by **PCIe Bus**, sharing DRAM and cache.

### 12. **Hardware Acceleration Metrics**

(page 75)

* **Hardware Accelerator:** specialized component (GPU) that offloads computation.
* **Key GPU metrics:**

  * **Number of CUDA cores**
  * **Memory size**
  * **Peak performance (GFLOPS/TFLOPS)**
  * **Memory bandwidth (GB/s)**

### 13. **CPU vs GPU**

(page 76)

| Characteristic   | CPU               | GPU                              |
| - | -- | -- |
| **Task Type**    | Control-intensive | Data-parallel                    |
| **Best For**     | Sequential logic  | Large-scale computation          |
| **Thread Count** | Few heavy threads | Thousands of lightweight threads |

*(Diagram on page 76)* shows how GPU parallelism grows with data size.

### 14. **Threading Model**

(page 78)

* CPU threads are **heavyweight** (managed by OS).
* GPU threads are **lightweight**, allowing **thousands to execute in parallel**.
* GPU rapidly switches between threads to avoid idle cores.

### 15. **CUDA Platform Overview**

(page 79)

* CUDA = NVIDIA’s **parallel computing platform**.
* Provides:

  * **GPU Computing Applications**
  * **Libraries** (cuFFT, cuBLAS, cuDNN, etc.)
  * **Languages** (C, C++, Fortran, Python via Numba or PyTorch)

📘 *(Diagram on page 79)* shows CUDA’s layered stack — from applications to GPU hardware.

### 16. **CUDA Goals**

(page 80)

* Scale across **hundreds of cores** and **thousands of threads**.
* Run GPU tasks **independently from the CPU**.

### 17. **CUDA Execution Model**

(page 81–82)

**Structure:**

* **Threads** grouped into **Thread Blocks**.
* **Blocks** grouped into a **Grid**.
* Each grid executes a **kernel** on the GPU.

**Scalability:**
Blocks map to GPU cores → allows flexible scaling across devices.

📘 *(Figures on pp. 81–82)* visualize 1D–2D grid hierarchy and mapping of blocks to cores.

### 18. **CUDA Compilation Process**

(page 83–84)

* **Host code** (CPU) compiled using a standard C compiler.
* **Device code** (GPU) compiled by **nvcc (NVIDIA CUDA compiler)**.

**Stages:**

1. Split host/device code.
2. Compile CUDA device code → PTX (Parallel Thread Execution).
3. Link host & device binaries.
4. Execute combined code.

📊 *(Page 84 diagram)* shows NVCC generating PTX or CUBIN binaries, then translating to GPU targets.

### 19. **CUDA Program Workflow**

(page 85–86)

1. Allocate GPU memory → `cudaMalloc`
2. Transfer data CPU → GPU → `cudaMemcpy`
3. Launch GPU kernel
4. Transfer results back → `cudaMemcpy`
5. Free GPU memory → `cudaFree`

*(Table on page 86)* maps standard C functions to CUDA equivalents.

### 20. **Kernels and Execution**

(page 87–90)

* **Kernels** are functions executed on the GPU.
* Declared with `__global__` qualifier.
* Must return `void`.

**Thread organization:**

* `blockIdx` → identifies block’s position in the grid.
* `threadIdx` → identifies thread within the block.
* **Kernel launch syntax:**

  ```cpp
  kernel_name<<<grid, block>>>(args);
  ```

*(Diagrams on pages 89–90)* depict thread indexing and grid configuration.

### 21. **Performance Profiling: nvprof**

(page 91–92)

* **`nvprof`** is NVIDIA’s command-line profiling tool.
* Monitors:

  * Kernel execution times
  * Memory transfers (Host ↔ Device)
  * CUDA API calls

Example report (page 92) shows profiling output for `sumArraysOnGPU-timer`, breaking down execution time by function and data movement.

### **Summary of CUDA Section**

| Concept                   | Key Idea                                    |
| - | - |
| **Traditional Computing** | Serial CPU execution limits scalability     |
| **Parallel Computing**    | Divide tasks across multiple processors     |
| **CUDA Platform**         | NVIDIA framework for GPU programming        |
| **Architecture**          | Threads → Blocks → Grid                     |
| **Memory Management**     | `cudaMalloc`, `cudaMemcpy`, `cudaFree`      |
| **Execution Model**       | `__global__` kernels run on GPU             |
| **Profiling**             | `nvprof` identifies performance bottlenecks |
