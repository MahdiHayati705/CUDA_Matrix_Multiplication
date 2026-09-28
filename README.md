# CUDA Matrix Multiplication

Implementation and optimization of **parallel matrix multiplication on the GPU using CUDA**, with performance comparison against a CPU implementation.

This project demonstrates several CUDA programming concepts and optimization techniques.

## What You Will Learn

1. Understand the **hierarchical execution structure** of a CUDA program: **Thread → Block → Grid**.
2. Learn how to allocate and transfer data between **Host (CPU)** and **Device (GPU)**.
3. Write a simple CUDA **kernel** for matrix multiplication.
4. Use **Shared Memory** to optimize performance.
5. Compare **GPU and CPU execution times**.
6. Understand **Memory Coalescing** and implement both **Coalesced** and **Non-Coalesced** versions.

---

## Matrix Multiplication

Given two matrices **A** and **B**, both of size `N × N`, their multiplication produces matrix **C**:

$$
C = A \times B
$$

where each element of **C** is calculated as:

$$
C_{ij} = \sum_{k=0}^{N-1} A_{ik}B_{kj}
$$

or equivalently,

$$
C_{ij} = A_{i0}B_{0j} + A_{i1}B_{1j} + \cdots + A_{i,N-1}B_{N-1,j}
$$

---

# Part 1: CPU Implementation

In this part, I implemented the **baseline CPU version** of the program, which performs matrix multiplication using the CPU.

This implementation is used as a reference for evaluating the performance of the CUDA implementations.

---

# Part 2: GPU Implementation

In this part, a simple CUDA implementation is introduced:

1. Each **thread calculates one element** of matrix `C`.
2. Threads are organized into a **2D grid of 2D blocks**.
3. Matrix elements are accessed through **Global Memory**.

---

# Part 3: Optimization with Shared Memory

In this part, the matrix multiplication is optimized using **Shared Memory**:

1. Matrices are divided into **tiles of size `16 × 16`**.
2. The required matrix elements are loaded into **Shared Memory**.
3. Threads within a block reuse the data stored in Shared Memory, reducing the number of Global Memory accesses.

---

# Part 4: Memory Coalescing

In CUDA, Global Memory performance depends heavily on how threads access memory.

When threads within a **warp** access consecutive memory addresses, memory accesses can be efficiently combined into fewer memory transactions. This behavior is known as **Memory Coalescing**.

In this part, I implemented two versions:

* **Non-Coalesced:** Threads access memory locations that are not consecutive.
* **Coalesced:** Threads within a warp access consecutive memory locations, improving Global Memory access efficiency.

The execution time of both implementations can then be compared to demonstrate the impact of memory access patterns on GPU performance.

---

# Part 5: cuBLAS Library

In this part, matrix multiplication is implemented using the **cuBLAS library**.

The cuBLAS implementation is used to compare the performance of a highly optimized NVIDIA library against the custom CUDA implementations developed in the previous parts.

---

# Performance Results

The execution time of each implementation was measured using the same matrix size and experimental conditions. The results are summarized below.

| Implementation      | Execution Time (ms) |
| ------------------- | ------------------: |
| CPU                 |         `8895.6`    |
| GPU – Global Memory |         `115.56`    |
| GPU – Shared Memory |         `20.51`     |
| GPU – Non-Coalesced |         `29.34`     |
| GPU – Coalesced     |         `9.2`       |
| GPU – cuBLAS        |         `0.619`     |

> **Note:** All measurements were performed using the same matrix size and input data.

### Performance Comparison

The results demonstrate the performance differences between the CPU implementation, the basic CUDA implementation (Global Memory), and the optimized CUDA versions using **Shared Memory**, **Memory Coalescing**, and **cuBLAS**.

