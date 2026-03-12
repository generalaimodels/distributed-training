

# Infrastructure for Large-Scale Language Model Training: A Comprehensive Technical Report

## On Hardware Architecture, Communication Topology, Memory Hierarchies, Storage Systems, Fault Resilience, and Throughput Optimization for Distributed Deep Learning at Scale

---

## Abstract

Large-scale language model (LLM) training constitutes a systems-level engineering challenge in which the interplay among compute hardware, memory hierarchies, interconnect topologies, storage subsystems, and fault-tolerance mechanisms collectively determines end-to-end training efficiency. Despite the disproportionate research attention directed toward model architecture and data curation, infrastructure remains the dominant source of training bottlenecks, cost overruns, and project failures. This report provides a rigorous, measurement-driven analysis of every layer in the training infrastructure stack—from the internal microarchitecture of modern GPUs through intranode and internode communication fabrics to storage I/O and production resilience engineering. All theoretical specifications are validated against empirical benchmarks conducted on NVIDIA H100 80GB HBM3 GPUs deployed on AWS P5 instances (8 GPUs per node, 48 nodes, 384 GPUs total) during the pretraining of SmolLM3 (3B parameters, 11T tokens, ~24 days). The central thesis is that **identifying and eliminating bottlenecks at every abstraction layer** is the fundamental discipline that separates performant training infrastructure from theoretical hardware specifications.

---

## Table of Contents

1. [Introduction and Motivation](#1-introduction-and-motivation)
2. [GPU Internal Architecture](#2-gpu-internal-architecture)
   - 2.1 Compute Units, Streaming Multiprocessors, and FLOPs
   - 2.2 Precision-Dependent Throughput and Tensor Core Utilization
   - 2.3 Empirical Validation: Achieved vs. Theoretical TFLOPs
   - 2.4 Model FLOPs Utilization (MFU)
   - 2.5 Compute Capability and Kernel Compilation
3. [GPU Memory Hierarchy](#3-gpu-memory-hierarchy)
   - 3.1 Hierarchical Memory Organization
   - 3.2 Memory-Bound vs. Compute-Bound Operations
   - 3.3 Operator Fusion and Flash Attention
   - 3.4 HBM3 Bandwidth Validation
   - 3.5 The Roofline Model
4. [External Communication Architecture](#4-external-communication-architecture)
   - 4.1 CPU-to-GPU Communication via PCIe
   - 4.2 NUMA Affinity and Multi-Socket Performance
   - 4.3 GPU-to-GPU Intranode Communication
   - 4.4 GPU-to-GPU Internode Communication
   - 4.5 Interconnect Troubleshooting
5. [GPU-to-Storage Communication](#5-gpu-to-storage-communication)
   - 5.1 Storage Topology and GPUDirect Storage
   - 5.2 Block Storage Device Configuration
   - 5.3 Storage Bandwidth Benchmarks
6. [Building Resilient Training Systems](#6-building-resilient-training-systems)
   - 6.1 Node Health Monitoring and Diagnostics
   - 6.2 Thermal Throttling and Cascade Failures
   - 6.3 Checkpoint Management
   - 6.4 Automated Evaluation Pipelines
7. [Optimizing Training Throughput](#7-optimizing-training-throughput)
   - 7.1 GPU Sizing and Resource Planning
   - 7.2 Amdahl's Law and Scaling Limits
   - 7.3 Parallelism Configuration Search
   - 7.4 Final Configuration for SmolLM3
8. [Consolidated Bandwidth Summary](#8-consolidated-bandwidth-summary)
9. [Conclusions](#9-conclusions)

---

## 1. Introduction and Motivation

### 1.1 The Infrastructure Knowledge Gap

The development pipeline for large language models encompasses data collection, preprocessing, model architecture design, distributed training, evaluation, and inference deployment. Among these stages, **infrastructure**—the hardware substrate, communication fabric, storage subsystems, and operational tooling—is systematically undervalued despite being the single most consequential determinant of training cost, wall-clock time, and project feasibility. Infrastructure expertise typically resides exclusively with framework developers and cluster engineers and is treated by practitioners as a solved problem: provision GPUs, install PyTorch, commence training.

This characterization is empirically false. During the pretraining of SmolLM3 on 384 NVIDIA H100 GPUs over approximately 24 days processing 11 trillion tokens, the training process encountered:

- **Node failures** requiring hardware replacement
- **Storage I/O bottlenecks** during checkpointing
- **Thermal throttling** events degrading cluster-wide performance
- **Run restarts** necessitating robust checkpoint recovery mechanisms
- **Communication overhead** from suboptimal parallelism configurations

Each of these failure modes is invisible at the model-architecture or data-curation level but directly impacts training throughput, cost efficiency, and the probability of successful completion.

### 1.2 Report Objectives

This report aims to bridge the infrastructure knowledge gap by providing:

1. **Hardware-level understanding**: What constitutes a GPU internally—compute units, memory hierarchy, and their interplay
2. **Communication topology characterization**: How CPUs, GPUs, network interfaces, and storage devices interconnect, with measured bandwidth and latency at every link
3. **Empirical validation methodology**: How to benchmark, diagnose, and validate infrastructure performance using industry-standard tools
4. **Resilience engineering**: How to build training systems that survive hardware failures, thermal events, and operational errors over multi-week training campaigns
5. **Throughput optimization**: How to select parallelism configurations that maximize model FLOPs utilization (MFU) given specific hardware constraints

### 1.3 Guiding Principle

The central principle of this report is **bottleneck identification and elimination**. At every abstraction layer—from register-level data movement within a single GPU to internode collective communication across hundreds of machines—performance is governed by the slowest component in the data path. Understanding how activations flow through multiple levels of cache, each with distinct bandwidth and latency characteristics, naturally motivates design decisions that minimize unnecessary data movement. Recognizing that internode communication bandwidth is orders of magnitude lower than intranode bandwidth directly explains why parallelism strategy selection is critical.

### 1.4 Hardware Platform Specification

All measurements reported herein were conducted on the following hardware platform unless otherwise specified:

| Component | Specification |
|---|---|
| GPU | NVIDIA H100 80GB HBM3 (SXM5) |
| GPUs per Node | 8 |
| Total Nodes | 48 |
| Total GPUs | 384 |
| CPU | AMD EPYC 7R13 (2 sockets, 48 cores each) |
| Intranode Interconnect | NVLink 4.0 (900 GB/s bidirectional per GPU) |
| Internode Interconnect | AWS Elastic Fabric Adapter (EFA), 4 × 100 Gbps per GPU |
| Local Storage | 8 × 3.5 TB NVMe SSDs in RAID 0 |
| Network Storage | WekaFS (393 TB), FSx Lustre (4.5 TB) |
| Cloud Instance | AWS P5.48xlarge |

---

## 2. GPU Internal Architecture

### 2.1 Compute Units, Streaming Multiprocessors, and FLOPs

A GPU is fundamentally a **massively parallel processor** optimized for **throughput** over **latency**. In contrast to CPUs, which excel at executing a small number of complex instruction streams with minimal latency, GPUs achieve performance by executing tens of thousands of simple operations simultaneously. At the highest level of abstraction, a GPU performs two essential tasks:

1. **Move and store data** (the memory system)
2. **Perform useful computation on data** (the compute pipelines)

The performance of any GPU workload is determined by the interplay between these two tasks. A GPU may possess teraflops of theoretical compute capability, but if data cannot reach the compute units at sufficient rate, that computational potential remains unrealized.

#### Streaming Multiprocessors (SMs)

The fundamental building blocks of GPU compute are **Streaming Multiprocessors (SMs)**—independent processing units that execute instructions in parallel. Each SM contains two categories of compute cores:

- **CUDA Cores**: General-purpose floating-point arithmetic units capable of executing standard operations ($a + b$, $a \times b$)
- **Tensor Cores**: Specialized matrix multiply-accumulate (MMA) units optimized for the operation $D = A \times B + C$ where $A$, $B$, $C$, $D$ are matrix tiles—the workhorse operation in deep learning

The NVIDIA H100 SXM5 contains **132 SMs**, each operating independently. Each SM executes groups of 32 threads called **warps** in lockstep, following the **SIMT (Single Instruction, Multiple Threads)** execution model: all threads in a warp execute the same instruction simultaneously on different data elements. **Warp schedulers** within each SM balance instruction dispatch across multiple warps, enabling the SM to hide memory latency by switching between warps when one is stalled waiting for data.

With 132 SMs each executing multiple warps concurrently, a single H100 GPU can sustain tens of thousands of threads simultaneously, enabling the massive parallelism required for the matrix operations that dominate transformer-based deep learning workloads.

### 2.2 Precision-Dependent Throughput and Tensor Core Utilization

GPU compute performance is quantified in **floating-point operations per second (FLOPs)**. A single FLOP represents one arithmetic operation (e.g., $a + b$ or $a \times b$), and modern GPUs execute trillions of these per second (TFLOPs).

Tensor Cores operate at multiple numerical precisions ($\text{FP64}$, $\text{FP32}$, $\text{FP16/BF16}$, $\text{FP8}$, $\text{FP4}$), and **achievable throughput varies by orders of magnitude depending on the data type**. Lower-precision formats enable higher throughput because:

1. They require less data movement per operand (fewer bytes per element)
2. They can pack more operations into the same silicon area per clock cycle
3. They reduce memory bandwidth pressure, allowing compute units to be fed more efficiently

The following table presents theoretical peak performance (TFLOPs) across NVIDIA GPU generations:

| Precision | A100 | H100 | H200 | B100 | B200 |
|---|---|---|---|---|---|
| $\text{FP64}$ | 9.7 | 34 | 34 | 40 | 40 |
| $\text{FP32}$ | 19.5 | 67 | 67 | 80 | 80 |
| $\text{FP16/BF16}$ | 312 | 990 | 990 | 1,750 | 2,250 |
| $\text{FP8}$ | — | 3,960 | 3,960 | 4,500 | 5,000 |
| $\text{FP4}$ | — | — | — | 9,000 | 10,000 |

*Source: NVIDIA, SemiAnalysis*

The H100's 3,960 TFLOPs at $\text{FP8}$ represents a $4\times$ improvement over $\text{FP16/BF16}$, while the B200's 10,000 TFLOPs at $\text{FP4}$ pushes this ratio even further. This progression reflects a fundamental shift toward lower-precision arithmetic, enabling more operations per watt and per second at both training and inference time.

**Critical caveat**: These theoretical peak FLOPs represent the **maximum computational throughput achievable under ideal conditions**—when all compute units are fully utilized and data is readily available in registers. In practice, actual performance depends on (a) how effectively the workload keeps compute units fed with data, and (b) whether the operations can be efficiently mapped to the available hardware units.

### 2.3 Empirical Validation: Achieved vs. Theoretical TFLOPs

To establish realistic throughput expectations for SmolLM3 training, we benchmarked the H100's theoretical specifications against real-world performance using the **SemiAnalysis GEMM benchmark**, which evaluates throughput on matrix multiplication shapes drawn from Meta's Llama 70B training workload.

#### Table: Achieved TFLOPs on H100 80GB GPUs by Precision and Matrix Shape

| Shape $(M, N, K)$ | $\text{FP64}$ `torch.matmul` | $\text{FP32}$ `torch.matmul` | $\text{BF16}$ `torch.matmul` | $\text{FP8}$ `torch._scaled_mm` $(e4m3)$ |
|---|---|---|---|---|
| $(16384, 8192, 1280)$ | 51.5 | 364.5 | 714.5 | 1,209.7 |
| $(16384, 1024, 8192)$ | 56.1 | 396.1 | 757.7 | 1,329.7 |
| $(16384, 8192, 7168)$ | 49.5 | 356.5 | 752.9 | 1,456.6 |
| $(16384, 3584, 8192)$ | 51.0 | 373.3 | 733.0 | 1,370.3 |
| $(8192, 8192, 8192)$ | 51.4 | 372.7 | 729.4 | 1,397.5 |

**Analysis of achieved utilization rates:**

- **FP64 Tensor Core operations**: 49–56 TFLOPs achieved, representing **74–84%** of the theoretical peak (67 TFLOPs). High utilization but rarely used in modern deep learning due to computational cost.

- **TF32 (TensorFloat-32)**: PyTorch uses TF32 by default for $\text{FP32}$ tensors on Tensor Cores. Achieved 356–396 TFLOPs, representing **72–80%** of the theoretical dense peak (~495 TFLOPs). NVIDIA specifications often cite sparse performance (989 TFLOPs for TF32), which assumes 2:4 structured sparsity patterns; dense operations achieve approximately half the sparse peak.

- **BF16 operations**: Consistently achieved **714–758 TFLOPs**, representing approximately **72–77%** of the H100's theoretical 990 TFLOPs peak. This constitutes excellent hardware utilization for a real-world workload and establishes the **upper-bound kernel-level ceiling** for our training setup.

- **FP8 operations**: Using `torch._scaled_mm` with $e4m3$ precision, achieved 1,210–1,457 TFLOPs, representing **31–37%** of the theoretical 3,960 TFLOPs peak. This lower utilization percentage does not indicate poor performance; rather, it reflects that FP8 operations become **increasingly memory-bound** as compute throughput grows. The Tensor Cores can process FP8 data faster than the memory system can deliver it, making **HBM bandwidth the limiting factor**.

A critical empirical finding was that **kernel implementation selection significantly impacts performance**: NVIDIA Transformer Engine's `TE.Linear` achieved 547–1,121 TFLOPs depending on matrix shape, while `torch._scaled_mm` consistently delivered higher throughput. The choice of API can affect performance by **$2\text{–}3\times$** even when targeting identical hardware capabilities.

### 2.4 Model FLOPs Utilization (MFU)

While kernel benchmarks measure raw TFLOPs at the operator level, end-to-end training efficiency is captured by **Model FLOPs Utilization (MFU)**—the ratio of useful model computation to theoretical peak hardware performance:

$$
\text{MFU} = \frac{\text{Observed model FLOPs/sec}}{\text{Theoretical peak FLOPs/sec}}
$$

Our BF16 matmul benchmarks established that the **kernel-level ceiling** is approximately 72–77% of theoretical peak. End-to-end training MFU is necessarily lower due to:

- Non-matmul operations (layer normalization, activation functions, embedding lookups)
- Communication overhead (gradient synchronization, tensor-parallel all-reduce)
- Pipeline bubbles and synchronization barriers
- Memory management and kernel launch overhead
- Data loading and preprocessing

**State-of-the-art MFU benchmarks:**

| System | MFU | Notes |
|---|---|---|
| Meta Llama 3 405B | 38–41% | Dense transformer, large-scale InfiniBand cluster |
| DeepSeek-V3 | ~20–30% | MoE architecture, tighter communication bottlenecks |
| **SmolLM3 (this work)** | **~30%** | 3B dense transformer, 384 H100s, AWS P5 |

Given the kernel-level ceiling of ~77%, the achieved 30% end-to-end MFU represents approximately **39% efficiency relative to achievable matmul performance**. The gap is primarily attributable to internode communication overhead in distributed training. Inference workloads can achieve higher MFU (>70%), though published production deployment data remains scarce.

### 2.5 Compute Capability and Kernel Compilation

**Compute capability (CC)** is NVIDIA's versioning system that abstracts physical GPU details from the PTX instruction set. It determines which instructions, features, and hardware optimizations are available to compiled kernels.

This has direct practical implications: kernels compiled for a specific compute capability may fail on older hardware, and frameworks may silently select suboptimal kernels if not compiled for the target GPU's CC. During our benchmarking, we discovered that **PyTorch selected `sm_75` kernels** (compute capability 7.5, designed for Turing-generation GPUs) on our H100s (compute capability 9.0), causing throughput degradation from approximately 720 TFLOPs to 500 TFLOPs on identical hardware—a **$1.44\times$ performance loss** from a compilation mismatch alone.

**Diagnostic commands:**

```bash
# Query GPU compute capability
nvidia-smi --query-gpu=compute_cap --format=csv

# Verify kernel compilation target (example: sm90 indicates CC 9.0)
# Look for: sm90_xmma_gemm_..._cublas in kernel names
```

**Recommendation**: When using precompiled libraries or custom kernels, always verify that binaries are compiled for the target hardware's compute capability. For H100 GPUs, kernels should target `sm_90` or `sm_90a`.

---

## 3. GPU Memory Hierarchy

### 3.1 Hierarchical Memory Organization

Modern GPUs organize memory in a hierarchy that balances speed, capacity, and cost—a design dictated by fundamental physics and circuit constraints. SRAM (used for caches and registers) is fast but physically large and expensive to fabricate, while DRAM (used for HBM) is dense and cost-effective but slower. The result: fast memory is available in small quantities close to compute units, backed by progressively larger pools of slower memory at greater physical distances.

The H100 (SXM5) memory hierarchy, ordered from slowest/largest to fastest/smallest:

| Memory Tier | Technology | Capacity | Bandwidth | Latency | Scope |
|---|---|---|---|---|---|
| **HBM3 (Global Memory)** | DRAM | 80 GB | 3.35 TB/s (theoretical) | ~hundreds of ns | Shared across all SMs |
| **L2 Cache** | SRAM | 50 MB | ~13 TB/s | ~tens of ns | Shared across all SMs |
| **L1 Cache / Shared Memory (SMEM)** | SRAM | 256 KB per SM | ~31 TB/s per SM | ~few ns | Per-SM |
| **Register Memory (RMEM)** | SRAM | 256 KB per SM | ~100 TB/s per SM | sub-ns | Per-thread |

This hierarchy spans approximately **$30\times$** in bandwidth from HBM to registers and over **$1000\times$** in capacity from registers to HBM. Understanding this hierarchy is not merely academic; it directly determines the performance ceiling for every kernel executed during training.

### 3.2 Memory-Bound vs. Compute-Bound Operations

The critical insight for kernel optimization is that **memory-bound operations are limited by data movement rate, not computation rate**. As formally stated: a kernel that performs `load → multiply → multiply → store` takes essentially the same wall-clock time as `load → multiply → store`, because the computation is "free" relative to the memory access latency.

This distinction determines which optimizations are effective:

- **Memory-bound kernels** (most time spent moving data): Increasing compute throughput provides no benefit. Optimization must target reducing memory traffic through operator fusion, improved access patterns, or increased arithmetic intensity.

- **Compute-bound kernels** (most time spent on FLOPs): Optimizing memory access patterns provides no benefit. Optimization requires more compute power, algorithmic improvements, or leveraging specialized hardware (Tensor Cores).

The transition between these regimes is characterized by **arithmetic intensity** $I$, defined as:

$$
I = \frac{\text{FLOPs performed}}{\text{Bytes transferred to/from memory}}
$$

### 3.3 Operator Fusion and Flash Attention

Operator fusion is a kernel optimization technique that combines multiple sequential operations into a single kernel, keeping intermediate results in fast on-chip SRAM instead of writing them back to slow HBM between operations.

**Flash Attention** is the canonical example of memory-hierarchy-aware kernel design. Standard attention implementations are memory-bound because they materialize the full $N \times N$ attention matrix in HBM:

1. Compute $Q K^\top$ → write $N \times N$ attention scores to HBM
2. Apply softmax → read from HBM, compute, write back to HBM
3. Multiply by $V$ → read attention scores from HBM again

This incurs $O(N^2)$ HBM accesses. Flash Attention eliminates this by **fusing all three operations** and processing attention in tiles that fit in SRAM:

1. Attention is computed in tiles that reside entirely in on-chip shared memory
2. Intermediate attention scores **never leave SRAM**
3. Only the final output is written back to HBM

The result: Flash Attention reduces HBM accesses from $O(N^2)$ to $O(N)$, achieving **$2\text{–}4\times$ speedup** by transforming a memory-bound operation into one that better utilizes compute capabilities. This exemplifies the core principle of efficient kernel design: **minimize slow memory movement, maximize fast computation**.

### 3.4 HBM3 Bandwidth Validation

To validate the H100's HBM3 bandwidth specification, we employed **NVBandwidth**, NVIDIA's open-source benchmarking tool for measuring bandwidth and latency across GPU memory systems. The `device_local_copy` test measures the bandwidth of `cuMemcpyAsync` between device buffers local to the GPU:

```bash
$ ./nvbandwidth -t device_local_copy -b 2048
```

**Results:**

| Transfer Direction | Measured Bandwidth | Theoretical Specification |
|---|---|---|
| Read | ~1,519 GB/s | — |
| Write | ~1,519 GB/s | — |
| **Bidirectional (Read + Write)** | **~3,038 GB/s** | **3,350 GB/s** |

The measured bidirectional bandwidth of ~3.0 TB/s closely validates the H100's theoretical 3.35 TB/s HBM3 specification, representing approximately **91% utilization** for large transfers.

**Important characteristic**: For small message sizes ($< 1$ MB), bandwidth is **latency-bound** rather than bandwidth-bound. The overhead of initiating memory transfers dominates performance, preventing peak bandwidth achievement. Sustained peak bandwidth is only realized for message sizes $\geq 1$ MB.

### 3.5 The Roofline Model

The **roofline model** provides a visual framework for classifying kernel performance as compute-bound or memory-bound and identifying optimization headroom.

The model plots two axes:

- **Vertical axis (Performance)**: Achieved FLOPs (logarithmic scale)
- **Horizontal axis (Arithmetic Intensity)**: Ratio of computation to memory traffic, measured in FLOPs/byte (logarithmic scale)

The roofline consists of two boundaries:

1. **Memory bandwidth boundary** (sloped line): Performance $P = I \times B_{\text{mem}}$, where $B_{\text{mem}}$ is memory bandwidth and $I$ is arithmetic intensity
2. **Peak performance boundary** (flat line): Performance $P = P_{\text{peak}}$, the maximum compute throughput

$$
P_{\text{achievable}}(I) = \min\left(P_{\text{peak}},\; I \times B_{\text{mem}}\right)
$$

The **ridge point** where these boundaries intersect occurs at arithmetic intensity:

$$
I_{\text{ridge}} = \frac{P_{\text{peak}}}{B_{\text{mem}}}
$$

For the H100 at BF16 precision:

$$
I_{\text{ridge}} = \frac{990 \times 10^{12} \text{ FLOPs/s}}{3.35 \times 10^{12} \text{ B/s}} \approx 296 \text{ FLOPs/byte}
$$

Kernels with arithmetic intensity below $I_{\text{ridge}}$ are **memory-bound**; those above are **compute-bound**. The distance from a kernel's plotted point to the roofline boundary represents available **optimization headroom**.

NVIDIA Nsight Compute provides roofline analysis for any profiled kernel:

```bash
ncu --set full --kernel-name "your_kernel_name" --launch-skip 0 --launch-count 1 python your_script.py
```

---

## 4. External Communication Architecture

### 4.1 Overview

A GPU does not operate in isolation. Before any computation occurs, data must be loaded into GPU memory. The CPU must schedule kernels and coordinate work. In distributed training, GPUs must constantly exchange activations, gradients, and model weights. The **external communication infrastructure** therefore determines whether expensive hardware sits idle or operates at high utilization.

Four critical communication links connect a GPU to the outside world:

| Link | Function | Typical Bandwidth |
|---|---|---|
| **CPU ↔ GPU** | Kernel scheduling, data transfer | ~14.2 GB/s (PCIe Gen4 x8) |
| **GPU ↔ GPU (intranode)** | Tensor-parallel communication | ~786 GB/s (NVLink 4.0) |
| **GPU ↔ GPU (internode)** | Data-parallel gradient sync | ~42 GB/s (EFA) |
| **GPU ↔ Storage** | Data loading, checkpointing | ~26.6 GiB/s (local NVMe RAID) |

Each link has distinct bandwidth and latency characteristics spanning approximately **two orders of magnitude**, and understanding these differences is essential for identifying training pipeline bottlenecks.

### 4.2 CPU-to-GPU Communication via PCIe

#### 4.2.1 Physical Topology

The CPU orchestrates GPU computation by launching kernels, managing memory allocations, and coordinating data transfers. The speed of CPU-GPU communication is determined by the **PCIe (Peripheral Component Interconnect Express)** connection.

Using `lstopo`, we characterized the topology of our P5 instances:

- **CPU to PCIe Switch**: PCIe Gen4 x8 links at **15.75 GB/s**
- **PCIe Switch to GPU**: PCIe Gen5 x16 links at **63.02 GB/s**

The CPU-to-GPU data path traverses two PCIe hops. The **bottleneck** is the first hop (CPU to PCIe switch) at 15.75 GB/s, which is $4\times$ slower than the second hop.

**PCIe bandwidth specifications across generations:**

| PCIe Version | Transfer Rate (per Lane) | x16 Throughput (GB/s) |
|---|---|---|
| Gen 3 | 8.0 GT/s | 15.75 |
| Gen 4 | 16.0 GT/s | 31.51 |
| Gen 5 | 32.0 GT/s | 63.02 |
| Gen 6 | 64.0 GT/s | 121.0 |

#### 4.2.2 Bandwidth Validation

Using `nvbandwidth`'s `host_to_device_memcpy_ce` test:

```bash
$ ./nvbandwidth -t host_to_device_memcpy_ce -b <message_size> -i 5
```

**Results**: For large message sizes ($\geq 1$ MB), measured bandwidth converges to **~14.2 GB/s**, representing **90%** of the theoretical 15.75 GB/s for PCIe Gen4 x8. This confirms the CPU-to-PCIe-switch link as the CPU-GPU communication bottleneck.

#### 4.2.3 Latency Measurement

Using `nvbandwidth`'s `host_device_latency_sm` test (round-trip latency via pointer-chase kernel):

**Result**: Approximately **1.4 μs** round-trip latency. This latency manifests as kernel launch overhead of several microseconds per kernel dispatch. For workloads launching many small kernels (e.g., inference with small models or small batches), this overhead can become a significant performance limiter.

**Mitigation strategies:**

- **CUDA Graphs**: Capture a sequence of operations and replay them as a single unit, eliminating per-kernel CPU-GPU round-trip latency. Particularly beneficial for workloads with many small kernels or frequent CPU-GPU synchronization.
- **Kernel fusion**: Combine multiple small kernels into larger ones (e.g., FlashFormer fuses entire transformer layers into single kernel launches).
- **Minimizing CPU-GPU synchronization points**: Some MoE implementations require CPU-GPU synchronization for expert routing at each iteration. MakoGenerate's optimization of DeepSeek MOE kernels reduced synchronization points from 67 to 3 per forward pass, achieving a **97% reduction** in synchronization overhead.

#### 4.2.4 Emerging Architectures: Grace Hopper

NVIDIA's Grace Hopper superchips fundamentally restructure CPU-GPU communication:

| Feature | x86 + Hopper | Grace Hopper |
|---|---|---|
| GPU:CPU ratio | 4:1 | 1:1 |
| CPU-GPU bandwidth | 128 GB/s (PCIe Gen5) | 900 GB/s (NVLink-C2C) |
| Bandwidth improvement | — | $7\times$ |

### 4.3 NUMA Affinity and Multi-Socket Performance

On multi-socket systems (our AMD EPYC 7R13 nodes contain 2 sockets with 48 cores each), **NUMA (Non-Uniform Memory Access) affinity** is critical. Each GPU is physically attached to one CPU socket; running GPU processes on CPUs from the wrong NUMA node forces memory operations to traverse the CPU interconnect (AMD Infinity Fabric), adding significant latency.

```bash
$ numactl --hardware
node distances:
node   0   1
  0:  10  32
  1:  32  10
```

The **$3.2\times$ difference** in memory access latency (distance 32 vs. 10) between local and remote NUMA nodes can significantly degrade GPU performance when processes are pinned to the wrong NUMA node. Each GPU should be bound to CPUs on the same NUMA node as the GPU's PCIe attachment point.

### 4.4 GPU-to-GPU Intranode Communication

#### 4.4.1 Communication Pathways

GPUs within a single node can communicate through three distinct pathways, each with dramatically different performance characteristics:

**Path 1: Through the CPU (SHM)**

Data traverses: GPU₁ → PCIe Switch → CPU → Host Memory → CPU → PCIe Switch → GPU₂

This path involves multiple memory copies and saturates both PCIe and CPU memory buses. With four H100s sharing the same CPU memory buses, congestion is exacerbated by simultaneous multi-GPU communication. The bottleneck is the PCIe Gen4 x8 link at ~16 GB/s.

Configuration: `NCCL_P2P_DISABLE=1`, `FI_PROVIDER=tcp`
Verification: `NCCL_DEBUG=INFO` shows `via SHM/direct/direct`

**Path 2: GPUDirect RDMA over EFA**

**GPUDirect RDMA (GDRDMA)** enables direct GPU-to-GPU memory access without involving the CPU or system memory, achieving up to $10\times$ better performance than CPU-mediated transfers. On our P5 instances, each GPU connects to 4 EFA NICs, each providing 100 Gbps (12.5 GB/s), for an aggregate **50 GB/s** per GPU via EFA.

Configuration: `FI_PROVIDER=efa`, `NCCL_P2P_DISABLE=1`
Verification: `NCCL_DEBUG=INFO` shows `via NET/Libfabric/0/GDRDMA/Shared`

**Path 3: NVLink**

**NVLink 4.0** provides direct GPU-to-GPU interconnect at **900 GB/s bidirectional** per GPU through 18 links, each operating at 50 GB/s bidirectional. In the DGX H100 architecture, four third-generation NVSwitches connect eight GPUs with a constant hop count of one NVSwitch, providing **3.6 TB/s total bidirectional NVLink bandwidth** across the node.

NVLink is the default and highest-priority path in NCCL for intranode communication.
Verification: `NCCL_DEBUG=INFO` shows `via P2P/CUMEM`

**NVLink generations comparison:**

| Generation | Architecture | Bidirectional BW per GPU |
|---|---|---|
| NVLink 2.0 | Volta | 300 GB/s |
| NVLink 3.0 | Ampere | 600 GB/s |
| NVLink 4.0 | Hopper | 900 GB/s |
| NVLink 5.0 | Blackwell | 1,800 GB/s |

#### 4.4.2 Empirical Bandwidth Comparison

Using NCCL's `sendrecv_perf` test between two GPUs on the same node:

| Communication Path | Measured Bandwidth | Relative to NVLink |
|---|---|---|
| Through CPU (SHM) | 3.24 GB/s | $1\times$ (baseline) |
| Through EFA (GDRDMA) | 38.16 GB/s | $11.8\times$ |
| **Through NVLink** | **364.93 GB/s** | **$112.6\times$** |

Using `nvbandwidth`'s bidirectional copy test across all GPU pairs:

$$
B_{\text{NVLink, bidirectional}} \approx 786 \text{ GB/s}
$$

This represents **85%** of NVLink 4.0's theoretical 900 GB/s specification, confirming excellent hardware utilization.

#### 4.4.3 Collective Communication Performance

**All-Reduce (intranode, 8 GPUs):**

Using NCCL's `all_reduce_perf` benchmark:

$$
B_{\text{all-reduce, intranode}} \approx 480 \text{ GB/s (bus bandwidth)}
$$

This **exceeds** the theoretical unidirectional NVLink bandwidth of 450 GB/s (900 GB/s ÷ 2). The explanation is **NVLink SHARP (NVLS)**—NVIDIA's hardware-accelerated collective operations technology implemented in the NVSwitches. NVLS provides approximately **$1.3\times$ speedup** for all-reduce operations by performing reduction operations directly in the NVSwitch hardware rather than requiring data to traverse individual GPU-to-GPU links.

**All-to-All (intranode, 8 GPUs):**

$$
B_{\text{all-to-all, intranode}} \approx 340 \text{ GB/s}
$$

All-to-all operations do **not** benefit from NVLS hardware acceleration, as they require complex point-to-point data exchanges between all GPU pairs rather than reduction operations. This explains the lower bandwidth compared to all-reduce despite using the same NVLink fabric.

#### 4.4.4 Advanced Kernel-Level Communication Optimization

Some optimized kernels (e.g., ThunderKittens) achieve fine-grained overlap of SM compute and NVLink communication by assigning **dedicated warps** to handle NVLink transfers while other warps continue compute operations. This warp-level separation can hide most inter-GPU communication latency, approaching theoretical bandwidth limits more closely than collective-library-based approaches.

### 4.5 GPU-to-GPU Internode Communication

#### 4.5.1 Network Technologies

Three primary networking technologies connect nodes in multi-GPU clusters:

| Technology | Bandwidth | Latency | RDMA Support |
|---|---|---|---|
| Ethernet (standard) | 25–100 Gbps | 10–30 μs | No |
| RoCE (RDMA over Converged Ethernet) | 100 Gbps | ~1 μs | Yes |
| InfiniBand | 400 Gbps | <1 μs | Yes |

On AWS P5 instances, **Elastic Fabric Adapter (EFA)** serves as the network interface. EFA uses the **Scalable Reliable Datagram (SRD)** protocol, an Ethernet-based transport designed for commodity datacenter networks with large numbers of network paths. Each GPU connects to 4 EFA NICs at 100 Gbps each via PCIe Gen5 x16.

Theoretical aggregate bandwidth per node:

$$
B_{\text{node, aggregate}} = 8 \text{ PCIe switches} \times 4 \text{ EFA NICs} \times 100 \text{ Gbps} = 3{,}200 \text{ Gbps} = 400 \text{ GB/s}
$$

When GPUs and EFA NICs are connected to the same PCIe switch, **GPUDirect RDMA** enables their communication to occur solely through that switch, fully utilizing PCIe Gen5 x16 bandwidth without involving other PCIe switches or CPU memory buses.

#### 4.5.2 Bandwidth Scaling Analysis

Systematic benchmarking of collective operations across 1–16 nodes reveals critical scaling characteristics:

**Point-to-Point (SendRecv):**
- 2–4 nodes: ~42–43 GB/s
- 5+ nodes: ~21 GB/s (NCCL reduces channels per peer from 2 to 1)
- Setting `NCCL_NCHANNELS_PER_NET_PEER=2` restores full throughput but may affect other collectives

**All-Reduce:**
- 1 node: 480 GB/s (NVLink + NVLS)
- 2 nodes: 479 GB/s (negligible degradation)
- 3–16 nodes: **320–350 GB/s** (stable scaling)

This near-constant bandwidth beyond 2 nodes is highly favorable for large-scale training. The stable 320–350 GB/s across 3–16 nodes indicates that data-parallel strategies relying on all-reduce can scale to hundreds or thousands of GPUs without significant per-GPU bandwidth degradation. This behavior is characteristic of well-designed multi-tier network topologies using **8-rail optimized fat trees**, where each of the 8 GPUs per node connects to a separate switch rail to maximize bisection bandwidth.

**All-to-All:**
- 1 node: 344 GB/s
- 2 nodes: 81 GB/s ($4.2\times$ degradation)
- 16 nodes: 45–58 GB/s

The steeper degradation for all-to-all reflects the $O(n^2)$ communication pattern where each GPU must exchange data with every other GPU, creating significantly more network congestion than all-reduce's tree-based reduction.

#### 4.5.3 Latency Scaling Analysis

| Operation | 1 Node | 2 Nodes | 16 Nodes | Scaling Characteristic |
|---|---|---|---|---|
| SendRecv | — | 40–53 μs | 40–53 μs | Constant (base RTT) |
| All-Reduce | 12.9 μs | 55.5 μs | 235 μs | Near-linear ($O(n)$) |
| All-to-All | 7.6 μs | 60 μs | 621 μs | Superlinear ($O(n \log n)$ to $O(n^2)$) |

The $4.3\times$ latency increase for all-reduce from 1 to 2 nodes reflects the fundamental cost of crossing node boundaries. The superlinear growth for all-to-all indicates compounding effects of network congestion and coordination overhead.

#### 4.5.4 NVSHMEM for GPU-Initiated Communication

For workloads requiring frequent fine-grained communication (e.g., MoE expert routing), **NVSHMEM** provides a partitioned global address space (PGAS) model that enables **GPU-initiated, asynchronous communication** without CPU involvement. Through GPUDirect Async, GPUs bypass the CPU entirely when issuing internode communication, achieving up to **$9.5\times$ higher throughput** for small messages (<1 KiB) compared to CPU-orchestrated transfers.

### 4.6 Interconnect Troubleshooting

When measured bandwidth falls below expectations, systematic investigation of the following areas is required:

1. **Library versions**: Outdated NCCL, EFA, or CUDA libraries may lack critical performance optimizations. Always verify compatible, up-to-date versions.

2. **CPU affinity configuration**: Improper NUMA binding causes cross-socket traffic. Use `NCCL_IGNORE_CPU_AFFINITY=1` and `--cpu-bind none` when container defaults interfere.

3. **Network topology and placement**: Cloud placement groups do not guarantee minimal network hops. The AWS Instance Topology API reveals actual physical placement; instances sharing the same bottom-layer network node achieve lowest latency.

4. **Environment variables**: Missing or incorrect flags for EFA/NCCL (e.g., adaptive routing, GPU-initiated transfers, buffer sizing) can severely limit bandwidth. AWS provides instance-type-specific recommended configurations.

5. **Container configuration**: Docker containers require explicit configuration for optimal NCCL performance:
   - Shared/pinned memory: `--shm-size=1g --ulimit memlock=-1`
   - NUMA support: `--cap-add SYS_NICE`
   - PCI topology discovery: Proper `/sys` mounting to expose real (not virtual) PCI topology

---

## 5. GPU-to-Storage Communication

### 5.1 Storage Topology and GPUDirect Storage

During training, GPUs continuously read data from storage and periodically write model states to storage (checkpointing). These I/O operations can become significant bottlenecks if not properly optimized.

**GPUDirect Storage (GDS)** enables a direct data path between storage (local NVMe or remote NVMe-oF) and GPU memory, eliminating buffer copies through CPU bounce buffers by allowing the DMA engine near the storage controller to move data directly into/out of GPU memory.

In our P5 instances, each PCIe switch connects to one NVMe SSD via PCIe Gen4 x8 (15.75 GB/s), providing one NVMe drive per GPU. GDS configuration status is verified via:

```bash
$ /usr/local/cuda/gds/tools/gdscheck.py -p
NVMe               : Supported
```

**Note**: GDS was not fully enabled in our cluster configuration, meaning GPU_DIRECT transfer results underperform compared to CPUONLY transfers. With GDS properly configured, direct GPU-to-storage transfers would show significant advantages, particularly for high-performance NVMe arrays.

### 5.2 Block Storage Device Configuration

The P5 instance storage hierarchy:

| Mount Point | Type | Capacity | Filesystem | Purpose |
|---|---|---|---|---|
| `/dev/root` (`/`) | Amazon EBS | 291 GB | ext4 | OS, system files |
| `/scratch` (`/dev/md0`) | 8 × NVMe RAID 0 | 28 TB | XFS | Checkpoints, scratch space |
| `/fsx` | WekaFS (network) | 393 TB | WekaFS | Shared training data |
| `/admin` | FSx Lustre (network) | 4.5 TB | Lustre | Administrative data |

The 8 NVMe instance-store SSDs (3.5 TB each) are configured in **RAID 0** (striping) to maximize sequential read/write throughput, exposed as the `/dev/md0` device mounted at `/scratch`.

### 5.3 Storage Bandwidth Benchmarks

Using `gdsio` with parametric sweeps across thread counts (1–64) and I/O sizes (64K–8M):

| Storage System | Peak Throughput (GiB/s) | Peak IOPS | Optimal Config | Min Latency |
|---|---|---|---|---|
| `/scratch` (NVMe RAID 0) | **26.59** | **337k** | 64 threads, 1M I/O | 190 μs |
| `/fsx` (WekaFS) | 4.21 | 51k | 32 threads, 1M I/O | — |
| `/admin` (FSx Lustre) | 1.13 | 17k | 16 threads, 512K I/O | — |
| `/root` (EBS) | 1.09 | 730 | Any threads, 4M I/O | — |

**Key findings:**

- Local NVMe RAID is **$6.3\times$ faster** than the best network storage (WekaFS) for throughput and **$6.6\times$ better** for IOPS, making it the optimal choice for checkpointing.
- Maximum throughput occurs at **1M I/O sizes** across all storage types, while maximum IOPS occurs at **64K I/O sizes**—the classic throughput-vs-concurrency tradeoff.
- EBS (`/root`) shows poor IOPS performance (730), confirming it is suitable only for large sequential operations.
- For ML training with large checkpoint files, the 1–8M I/O range on `/scratch` provides optimal performance.

---

## 6. Building Resilient Training Systems

### 6.1 Node Health Monitoring and Diagnostics

LLM training runs span weeks to months, during which GPUs that pass initial benchmarks can develop thermal throttling, memory errors, or performance degradation. Proactive monitoring and diagnostics are essential for minimizing training downtime.

#### Pre-Training Diagnostics

Before launching SmolLM3, comprehensive GPU diagnostics were conducted using:

1. **GPU Fryer**: Stress-tests GPUs for thermal throttling, memory errors, and performance anomalies
2. **NVIDIA DCGM Diagnostics**: Validates GPU hardware, monitors performance, and identifies root causes of failures through deep diagnostic tests

DCGM diagnostic levels:

| Test Level | Duration | Key Tests |
|---|---|---|
| `r1` (Short) | Seconds | PCIe/NVLink, GPU Memory, Memory Bandwidth |
| `r2` (Medium) | < 2 min | + Diagnostics, Targeted Stress |
| `r3` (Long) | < 30 min | + Targeted Power, NVBandwidth, Memory Stress |
| `r4` (Extra Long) | 1–2 hours | + Full Input EDPp |

Pre-training `r2` diagnostics caught **two problematic GPUs** that would have caused training failures.

#### Runtime Monitoring

During training, continuous monitoring was maintained using:

- **Prometheus**: Collected DCGM metrics (temperatures, memory usage, compute utilization, error rates) from all GPUs
- **Grafana**: Visualized metrics in real-time dashboards
- **Slack bot**: Automated alerts for suspicious node behavior (thermal throttling, degraded throughput, memory errors)

### 6.2 Thermal Throttling and Cascade Failures

GPUs automatically reduce clock speeds when temperatures exceed thermal limits, a process known as **thermal throttling**. The DCGM metric `DCGM_FI_DEV_CLOCK_THROTTLE_REASONS` reports nonzero values when throttling is active.

**Critical finding**: Thermal throttling on a single GPU cascades across the entire distributed training pipeline. During distributed training, collective communication operations (e.g., all-reduce for gradient synchronization) are **synchronous barriers**—all participating GPUs must complete their portion before any can proceed. A single throttled GPU becomes the **rate-limiting element** for the entire cluster.

Empirical observation during stress testing:

- **1–14 nodes**: All-reduce bandwidth stable at ~350 GB/s
- **15–16 nodes** (with one thermally throttled node): Bandwidth collapsed to **~100 GB/s**, a **$3.5\times$ degradation** caused by a single slow node

$$
B_{\text{effective}} = \min_{i \in \{1, \ldots, N\}} B_i
$$

where $B_i$ is the achievable bandwidth for node $i$. In distributed training, **throughput equals the throughput of the slowest node**.

**Recommendations:**
- Stress-test all hardware before committing to long runs
- Monitor GPU clocks continuously via DCGM telemetry
- Verify GPU clocks are set to maximum performance mode
- Implement automated node replacement when sustained throttling is detected

### 6.3 Checkpoint Management

Checkpoints serve three practical functions: (1) failure recovery, (2) training progress monitoring through evaluation, and (3) sharing intermediate models. The recovery function is paramount: if training fails, the maximum lost computation equals the checkpoint save interval.

#### Design Principles

1. **Background saving**: Checkpoint writes must occur asynchronously without blocking training throughput
2. **Storage management**: Over a 24-day run with 4-hour intervals, ~144 checkpoints are generated. Only the latest local checkpoint is retained; all others are offloaded to S3
3. **Automated resume**: On Slurm, `--requeue` enables automatic restart from the latest checkpoint, eliminating human-in-the-loop delays

#### Checkpoint Pipeline (SmolLM3 Implementation)

1. Save checkpoint locally every 2 hours
2. Immediately upload to S3
3. Delete local copy once S3 upload is confirmed
4. On resume: pull from S3 if latest checkpoint not available locally

#### Failure Mode: Data Loss from Script Errors

During StarCoder-15B training, a residual `rm -rf $CHECKPOINT_PATH` command from old throughput tests deleted the entire checkpoint directory when the Slurm job completed normally (a condition that hadn't occurred during previous restarts). Cost: one day of retraining from the previous day's backup.

**Lessons**: (1) Never include destructive commands in production scripts, (2) Automate checkpoint backups immediately after saving, (3) Maintain off-cluster backups (S3) as primary recovery mechanism.

### 6.4 Automated Evaluation Pipelines

Manual evaluation at scale introduces human bottlenecks, inconsistency, and delays. For SmolLM3, every saved checkpoint automatically triggered an evaluation job using **LightEval** on Nanotron checkpoints, with results pushed directly to **Weights & Biases / Trackio** dashboards.

This automation eliminated:
- Manual benchmark execution overhead
- Inconsistent evaluation configurations across checkpoints
- Delays between checkpoint creation and evaluation result availability

**Recommendation**: If only one aspect of the training pipeline is automated, prioritize automated evaluations. The return on investment in terms of reduced human overhead and improved experimental rigor is the highest among all automatable components.

---

## 7. Optimizing Training Throughput

### 7.1 GPU Sizing and Resource Planning

Determining the required GPU count requires balancing training time, cost, and scaling efficiency. The fundamental sizing equation:

$$
N_{\text{GPU}} = \frac{C_{\text{total}}}{F_{\text{effective}} \times T_{\text{target}}}
$$

where:

- $C_{\text{total}}$ is the total computation required (FLOPs)
- $F_{\text{effective}}$ is the effective per-GPU throughput (FLOPs/s), accounting for MFU
- $T_{\text{target}}$ is the target training wall-clock time (seconds)

**SmolLM3 calculation:**

**Step 1: Total FLOPs** (using the standard transformer approximation of $6N$ FLOPs per token):

$$
C_{\text{total}} = 6 \times N \times D = 6 \times 3 \times 10^9 \times 11 \times 10^{12} = 1.98 \times 10^{23} \text{ FLOPs}
$$

where $N = 3 \times 10^9$ parameters and $D = 11 \times 10^{12}$ training tokens.

**Step 2: Effective per-GPU throughput** (with expected MFU = 30%):

$$
F_{\text{effective}} = F_{\text{BF16 peak}} \times \text{MFU} = 720 \times 10^{12} \times 0.30 = 216 \times 10^{12} \text{ FLOPs/s}
$$

Note: 720 TFLOPs is used as the realistic kernel-level achievable performance rather than the theoretical 990 TFLOPs peak.

**Step 3: GPU count** (with $T_{\text{target}} = 4$ weeks):

$$
N_{\text{GPU}} = \frac{1.98 \times 10^{23}}{216 \times 10^{12} \times 4 \times 604{,}800} = \frac{1.98 \times 10^{23}}{5.23 \times 10^{20}} \approx 379 \text{ GPUs}
$$

This calculation pointed to 375–400 H100s; 384 were secured, aligning with parallelism strategy requirements (384 = $2^7 \times 3$) and providing margin for node failures and restarts.

### 7.2 Amdahl's Law and Scaling Limits

**Amdahl's law** states that speedup from parallelization is fundamentally limited by the serial (non-parallelizable) fraction of the workload:

$$
S(n) = \frac{1}{s + \frac{1 - s}{n}}
$$

where $S(n)$ is the speedup with $n$ processors and $s$ is the serial fraction. The maximum achievable speedup as $n \to \infty$ is:

$$
S_{\max} = \frac{1}{s}
$$

In LLM training, the serial fraction $s$ is primarily **communication overhead**—gradient synchronization, tensor-parallel all-reduce, pipeline flush—that cannot be parallelized. For a 3B parameter model where communication constitutes 10% of each training step ($s = 0.10$), the maximum speedup is capped at $10\times$ regardless of GPU count.

Furthermore, the serial fraction often **increases** with GPU count because:

1. More GPUs imply more all-reduce participants and longer synchronization chains
2. Network latency/bandwidth becomes the bottleneck at scale
3. Small models cannot generate sufficient compute to hide communication behind computation

**Scaling efficiency** at $n$ GPUs:

$$
\eta(n) = \frac{S(n)}{n} = \frac{1}{n \cdot s + (1 - s)}
$$

At $s = 0.05$ and $n = 384$:

$$
\eta(384) = \frac{1}{384 \times 0.05 + 0.95} = \frac{1}{20.15} \approx 4.96\%
$$

This illustrates why **weak scaling** (increasing total work proportionally with GPU count) is preferred over strong scaling (fixed total work) for LLM training: the global batch size scales with GPU count, maintaining a favorable communication-to-computation ratio.

### 7.3 Parallelism Configuration Search

#### Constraint Formulation

Two equations define the parallelism search space:

**Global batch size constraint:**

$$
\text{GBS} = \text{DP} \times \text{MBS} \times \text{GRAD\_ACC} \times \text{SEQLEN} \approx 2 \times 10^6 \text{ tokens}
$$

where:
- $\text{DP}$: data parallelism degree (number of model replicas)
- $\text{MBS}$: micro-batch size (sequences per GPU per micro-step)
- $\text{GRAD\_ACC}$: gradient accumulation steps
- $\text{SEQLEN}$: sequence length (4,096 for pretraining stage 1)

**Hardware constraint:**

$$
\text{DP} \times \text{TP} \times \text{PP} = 384 = 2^7 \times 3
$$

where $\text{TP}$ is tensor parallelism degree and $\text{PP}$ is pipeline parallelism degree.

#### Search Space

Given the constraints, the following parameter ranges were explored:

| Parameter | Range | Rationale |
|---|---|---|
| DP | 1–384 (multiples of 2 and/or 3) | Internode gradient sync via EFA |
| TP | {1, 2, 3, 4, 6, 8} | Keep within single node for NVLink |
| PP | 1–48 | Split model depth across GPUs |
| MBS | {2, 3, 4, 5} | Memory-constrained by parallelism savings |
| ZeRO Level | {0, 1, 3} | Optimizer state sharding |
| Activation Checkpointing | {none, selective, full} | Compute-memory tradeoff |

#### Memory Analysis

Using Nanotron's `predict_memory` tool, the SmolLM3 3B memory footprint was estimated:

| Memory Component | Size (MiB) |
|---|---|
| Model Parameters (BF16) | 5,865 |
| FP32 Parameter Copy | 11,730 |
| FP32 Gradient Buffers | 11,730 |
| Optimizer States (Adam) | 23,460 |
| DDP Gradient Buffers | 104 |
| Activations (peak) | 21,288 |
| **Peak Memory** | **~74,073** |

Peak memory of ~74 GB approaches the H100's 80 GB limit, confirming that some form of parallelism-based memory reduction is required.

#### Elimination of Suboptimal Configurations

Early benchmarking eliminated several parallelism strategies:

- **Pipeline Parallelism**: Poor performance due to frequent pipeline bubble synchronization across nodes. For a relatively small 3B model, communication overhead outweighed potential benefits. Efficient PP schedules (e.g., zero-bubble PP) were not available in Nanotron at the time.
- **ZeRO-1 and ZeRO-3**: Introduced significant all-gather and reduce-scatter operations that degraded throughput more than they improved memory utilization. ZeRO-3 was not yet supported in Nanotron.

### 7.4 Final Configuration for SmolLM3

After systematic benchmarking (5 iterations per configuration, measuring tokens/second/GPU):

| Parameter | Value | Rationale |
|---|---|---|
| **DP** | 192 | Leverages internode EFA for gradient sync |
| **TP** | 2 | Keeps tensor-parallel communication within NVLink |
| **PP** | 1 | Eliminated due to bubble overhead |
| **MBS** | 3 | Memory-throughput balance |
| **GRAD_ACC** | 1 | No accumulation needed |
| **ZeRO Level** | 0 | No optimizer sharding (communication overhead outweighed memory savings) |
| **SEQLEN** | 4,096 | Stage 1 pretraining |

**Verification of global batch size:**

$$
\text{GBS} = 192 \times 3 \times 1 \times 4{,}096 = 2{,}359{,}296 \approx 2.3\text{M tokens}
$$

**Achieved MFU**: ~30%, consistent with the pre-training estimate.

**Topology-aware design**: The configuration exploits the two-tier bandwidth hierarchy:
- **TP = 2** uses NVLink (786 GB/s measured bidirectional) for weight-matrix sharding communication within each layer
- **DP = 192** uses EFA (320–350 GB/s measured all-reduce bandwidth) for gradient synchronization across model replicas

This assignment ensures that the **highest-bandwidth communication** (tensor-parallel all-reduce at every layer) occurs on the **highest-bandwidth interconnect** (NVLink), while the **lower-frequency communication** (data-parallel gradient sync once per step) occurs on the **lower-bandwidth interconnect** (EFA).

---

## 8. Consolidated Bandwidth Summary

The following table synthesizes all empirically measured bandwidths into a unified view, demonstrating the dramatic bandwidth reduction as data moves further from the GPU's compute units:

| Data Path | Theoretical BW | Measured BW | Utilization | Latency |
|---|---|---|---|---|
| **HBM3 (within GPU)** | 3,350 GB/s | ~3,038 GB/s | 91% | — |
| **L2 Cache** | ~13 TB/s | — | — | ~tens of ns |
| **L1/SMEM (per SM)** | ~31 TB/s | — | — | ~few ns |
| **Registers (per SM)** | ~100 TB/s | — | — | sub-ns |
| **NVLink 4.0 (intranode GPU↔GPU)** | 900 GB/s | 786 GB/s bidir. | 87% | — |
| **NVLink All-Reduce (intranode, NVLS)** | — | 480 GB/s bus BW | — | 12.9 μs |
| **PCIe Gen5 x16 (switch↔GPU)** | 63 GB/s | — | — | — |
| **PCIe Gen4 x8 (CPU↔switch)** | 15.75 GB/s | 14.2 GB/s | 90% | 1.4 μs RTT |
| **EFA (internode GPU↔GPU, SendRecv)** | 50 GB/s | 42 GB/s | 84% | 40–53 μs |
| **EFA All-Reduce (internode, 3–16 nodes)** | — | 320–350 GB/s | — | 55–235 μs |
| **NVMe RAID 0 (local storage)** | — | 26.59 GiB/s | — | 190 μs |
| **WekaFS (network storage)** | — | 4.21 GiB/s | — | — |
| **FSx Lustre (network storage)** | — | 1.13 GiB/s | — | — |

**Bandwidth spans approximately three orders of magnitude** from HBM (3 TB/s) to network storage (1 GiB/s), with each tier representing a potential bottleneck depending on the workload's data access patterns.

**Critical qualification**: Raw bandwidth numbers alone do not tell the complete story. Modern training systems employ **computation-communication overlap** techniques to hide communication latency behind compute operations. When properly implemented, internode communication costs can be substantially amortized by concurrent forward/backward computation on subsequent micro-batches.

---

## 9. Conclusions

### 9.1 Principal Findings

1. **The infrastructure-performance gap is quantifiable**: Achieved hardware utilization ranges from 72–87% at the kernel/link level, while end-to-end MFU is approximately 30%, representing ~39% efficiency relative to achievable kernel performance. The gap is attributable to non-matmul computation, communication overhead, and synchronization barriers.

2. **Bandwidth hierarchies dictate parallelism strategy**: The $18\times$ bandwidth difference between NVLink (786 GB/s) and EFA (42 GB/s) directly motivates placing tensor-parallel communication (high-frequency, per-layer) on NVLink and data-parallel communication (low-frequency, per-step) on EFA.

3. **Single-node failures cascade catastrophically**: One thermally throttled GPU degraded cluster-wide all-reduce bandwidth by $3.5\times$, validating the principle that distributed training throughput equals the throughput of the slowest participant.

4. **Kernel compilation details matter**: A CUDA compute-capability mismatch (`sm_75` vs. `sm_90`) caused a $1.44\times$ performance loss on identical hardware—a failure mode invisible at the model-architecture level.

5. **Memory boundedness dominates at low precision**: FP8 kernels achieved only 31–37% of theoretical peak not due to poor implementation but because compute throughput exceeded memory bandwidth, making HBM the bottleneck.

6. **Storage I/O is a hidden bottleneck**: Local NVMe RAID provides $6.3\times$ higher throughput than the best network storage, making storage tier selection critical for checkpoint performance during multi-week training runs.

### 9.2 Operational Recommendations

| Category | Recommendation |
|---|---|
| **Pre-training** | Run DCGM diagnostics (`r2` minimum) on all nodes; stress-test thermals; verify compute capability alignment |
| **Parallelism** | Map highest-bandwidth communication to highest-bandwidth interconnect; benchmark each parallelism dimension independently before combinatorial search |
| **Monitoring** | Deploy continuous DCGM → Prometheus → Grafana telemetry; automate alerting for thermal throttling and throughput degradation |
| **Checkpointing** | Save asynchronously to local NVMe; upload to S3 immediately; automate resume via `--requeue` |
| **Evaluation** | Automate evaluation on every checkpoint; push results to centralized dashboards |
| **Debugging** | Use NVBandwidth for link validation, Nsight Compute for kernel profiling, NCCL tests for collective benchmarking |

### 9.3 Closing Remarks

Infrastructure is not a solved problem. It is the substrate upon which all training efficiency is built, and its optimization requires the same rigor applied to model architecture design and data curation. The central discipline—**systematically measuring, diagnosing, and eliminating bottlenecks at every abstraction layer**—transforms infrastructure from an opaque cost center into a controllable, optimizable system. The measurements, tools, and methodologies presented in this report provide a comprehensive framework for achieving this transformation across any GPU-accelerated training deployment.

---

## References

- NVIDIA Corporation. *NVIDIA H100 Tensor Core GPU Architecture Whitepaper*.
- NVIDIA Corporation. *NVIDIA Grace Hopper Superchip Architecture Whitepaper*.
- NVIDIA Corporation. *NVIDIA DCGM Diagnostics Documentation*.
- NVIDIA Corporation. *NVIDIA Nsight Compute Profiling Guide*.
- NVIDIA Corporation. *NVBandwidth: GPU Memory Bandwidth Benchmark Tool*. GitHub Repository.
- NVIDIA Corporation. *NCCL Tests: Performance Benchmarks for Collective Communications*. GitHub Repository.
- Dao, T., et al. (2022). *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*. NeurIPS 2022.
- He, H. (2024). *Making Deep Learning Go Brrrr from First Principles*.
- Li, S., et al. (2022). *Comparison of High-Performance Interconnect Technologies*. 
- Lindholm, E., et al. (2008). *NVIDIA Tesla: A Unified Graphics and Computing Architecture*. IEEE Micro.
- Nrusimha, A., et al. (2025). *FlashFormer: Fusing Transformer Layers for Accelerated Inference*.
- AWS. *Elastic Fabric Adapter (EFA) Documentation*.
- AWS. *awsome-distributed-training Repository*. GitHub.
- Chen, L. *Harnessing 3200 Gbps Network: A Journey with RDMA, EFA, and libfabric*. Blog Series.
- Gordić, A. *Inside NVIDIA GPUs*. Presentation.
- Hugging Face. *Ultra-Scale Playbook: Distributed Training at Scale*.
- SemiAnalysis. *GEMM Benchmark Suite*.