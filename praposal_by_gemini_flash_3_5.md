# Dissertation Topic Proposals: VLSI/RTL/Digital Design

## Background and Goals
*   **Target Career Path:** High-performance, design-heavy roles in premier product-based semiconductor companies (e.g., NVIDIA, AMD, Intel, Qualcomm, Broadcom).
*   **Timeframe:** 3–4 months alongside a full-time job.
*   **Academic Standard:** High feasibility for peer-reviewed publication in reputed conferences/journals (IEEE, ACM, ISCAS, DAC, etc.).
*   **Scope:** Focused on RTL-level design (Verilog/SystemVerilog), standard synthesis, simulation, and PPA (Power, Performance, Area) analysis.

---

## 1. Dissertation Problem Statements

### Topic 1: Dynamically Fracturable Mixed-Precision (FP16 / INT8 / INT4) Processing Element for Edge Transformer Hardware Accelerators
*   **Problem Statement:**  
    Modern Edge Transformer models undergo heavy run-time precision fluctuation (e.g., using INT4 for non-critical token representations and FP16/INT8 for attention matrices and sensitive activations). Dedicated sub-units for each precision type lead to high silicon footprint, poor resource utilization, and static leakage. The challenge is to design a **Dynamically Fracturable Processing Element (PE)** that reconfigures its hardware datapath at run-time to execute either **1x FP16 MAC, 2x INT8 MACs, or 4x INT4 MACs** within the same physical silicon footprint by sharing the underlying multiplier Booth-encoders and adder trees.
*   **Industry Importance:**  
    Highly aligned with NVIDIA's Tensor Core and AMD's CDNA/RDNA architectures. Maximize operations per clock cycle ($TOPS/mm^2$) and minimize power leakage—critical metrics for edge inference (e.g., smartphones, autonomous driving chips like Qualcomm Snapdragon, Tesla FSD).
*   **Technical Approach:**  
    *   Implement a parameterized and deeply pipelined arithmetic execution unit in SystemVerilog. 
    *   Design a fractured multiplier tree utilizing modified Radix-4 Booth multiplication where partial-product generators can be gated, isolated, and redirected based on a 2-bit runtime precision mode. 
    *   Develop a Python model to generate random and edge-case quantized inputs (to verify underflow, overflow, and rounding behavior).
    *   Perform synthesis in AMD Vivado (targeting an UltraScale+ or Versal device) or Synopsys Design Compiler (using generic libraries/PDKs) to measure and prove area savings compared to three independent mathematical lanes.
*   **Expected Outcomes:**  
    *   Functional verification with $100\%$ precision mapping accuracy.
    *   A $>40\%$ reduction in total silicon area compared to discrete math lanes, with comparable maximum clock frequencies ($>300\text{ MHz}$ on FPGA, $>1.5\text{ GHz}$ ASIC targets).
*   **Academic / Publication Angle:**  
    *   *Novelty:* A new, low-mux-overhead interconnect/packing logic for fractured FP16 exponent/mantissa and INT operations.
    *   *Venue:* IEEE Transactions on VLSI Systems (TVLSI) or IEEE ISCAS (International Symposium on Circuits and Systems).

---

### Topic 2: Queue-Prioritized Multi-Channel Virtualized DMA Engine with Adaptive Weighted Round-Robin (WRR) Scheduling for PCIe Gen5/6 Subsystems
*   **Problem Statement:**  
    High-bandwidth, low-latency communication on PCIe Gen5/6 endpoints (up to 64 GT/s) is frequently bottled by structural queuing and scheduling at the DMA engine level. Under mixed-traffic environments (e.g., small low-latency register commands alongside massive network packet streaming), standard Round-Robin arbiters cause Head-of-Line (HoL) blocking and high tail latencies. This research proposes the design of an RTL-level **Virtualized DMA Controller** supporting multiple descriptor queues that implements a hardware-driven, dynamically adaptive **Weighted Round-Robin (WRR)** and Strict Priority scheduler to enforce Quality of Service (QoS) directly in the chip's hard IP.
*   **Industry Importance:**  
    Every modern high-performance SoC (AMD EPYC, Intel Xeon, Broadcom Network chips, NVIDIA BlueField DPU) utilizes high-performance multi-queue DMAs. Knowing how to resolve bus-arbitration bottlenecks in RTL is a premier skill for high-speed interface teams.
*   **Technical Approach:**  
    *   Implement an AXI4-compliant Multi-Queue DMA core in SystemVerilog, consisting of register configurations (AXI-Lite), write-path credit managers, and request queues.
    *   Create a scheduler block using a hardware-based status table that monitors queue depth, timeout counters, and transaction latency to adjust its arbitration priorities dynamically.
    *   Write a complete verification environment using SystemVerilog to model write/read latencies matching real PCIe interconnect bottlenecks.
    *   Analyze link efficiency, throughput fairness, and queue waiting time.
*   **Expected Outcomes:**  
    *   Elimination of Head-of-Line blocking under heavy congestion.
    *   A reduction in peak tail-latency of up to $25\%$ compared to standard static priority schemes, keeping bus utilization above $92\%$.
*   **Academic / Publication Angle:**  
    *   *Novelty:* Hardware-only, low-gate-count adaptive weight-estimation circuit that evaluates congestion without requiring software interrupt involvement. 
    *   *Venue:* IEEE Transactions on Computers or IEEE Symposium on High-Performance Computer Architecture (HPCA) workshops.

---

### Topic 3: Zero-Overhead Hardware Streaming CSC/CSR Decoder for Sparse Matrix Accelerators in Graph Neural Networks (GNNs)
*   **Problem Statement:**  
    Graph Neural Networks and modern recommendation systems feature Sparsity rates $>90\%$. Compressing sparse activation and weight matrices using formats like Compressed Sparse Column (CSC) or Compressed Sparse Row (CSR) reduces SRAM storage and external memory bus requirements. However, traditional hardware decoders introduce multi-cycle stall overhead while fetching indices, resolving pointer offsets, and routing data down to MAC array lanes. This research addresses the design of a **streaming, zero-overhead pipeline-integrated hardware CSC decompressor** that seamlessly decodes indices and non-zero values on-the-fly and aligns them to execution lanes using a custom vector-alignment routing crossbar.
*   **Industry Importance:**  
    Highly relevant to domain-specific AI processing units (Cerebras, Groq, Google TPU, AMD CDNA) where memory-to-compute ratio is the ultimate bottleneck. Exploiting structural sparsity with minimal area overhead is actively researched in commercial design teams.
*   **Technical Approach:**  
    *   Design a streaming decompression state machine in SystemVerilog that ingests a compressed index/value package per cycle.
    *   Implement an optimized spatial router (e.g., dynamic barrel shifters) that routes the sparse values directly to ready multipliers while power-gating idle lanes.
    *   Use sparse representations of open datasets (e.g., the Cora citation network) as stimulus inputs inside the testbench.
    *   Synthesize the architecture, characterizing PPA against a dense matrix-computation array of equivalent numerical capacity.
*   **Expected Outcomes:**  
    *   Up to $2.5\times$ performance increase on $75\%$ sparse datasets.
    *   $>3\times$ reduced SRAM read energy through dynamic gating of zero-value activations.
*   **Academic / Publication Angle:**  
    *   *Novelty:* A pipeline-parallel pointer-tracking mechanism that handles non-contiguous memory accesses without pipeline flushes or multi-cycle branch delays.
    *   *Venue:* Design Automation Conference (DAC) or IEEE Transactions on Parallel and Distributed Systems (TPDS).

---

### Topic 4: Side-Channel Attack (SCA) Resistant Dilithium/Kyber NTT Mathematics Core Utilizing Synthesizable Masked Multipliers
*   **Problem Statement:**  
    The transition to Post-Quantum Cryptography (PQC) standards (NIST ML-KEM/Kyber and ML-DSA/Dilithium) requires high-performance hardware implementations of the Number Theoretic Transform (NTT) or Montgomery modular multiplier. However, standard RTL implementations are vulnerable to first-order Differential Power Analysis (DPA) and Electromagnetic (EMA) attacks, where attackers measure timing or current spikes to reverse-engineer private keys. The goal is to implement a synthesizable **cryptographic NTT arithmetic core featuring hardware-masked multipliers** (additive/multiplicative shares) that decouples power consumption from the secret key data with minimal pipeline-stage impacts.
*   **Industry Importance:**  
    With post-quantum security becoming a government mandate (CNSA 2.0), companies like Qualcomm, Apple (Secure Enclave), Intel, and security-IP manufacturers are redesigning their cryptographic blocks.
*   **Technical Approach:**  
    *   Implement an integer-modular multiplier (Montgomery/Barrett) in RTL.
    *   Secure the multiplication pipeline utilizing a randomized split (masking) into distinct computational shares, implementing a hardware Pseudo-Random Number Generator (PRNG/LFSR) to generate masks per cycle.
    *   Simulate the design to ensure functional results mathematically match unmasked equivalents (validated with standard Python/C libraries for Kyber/Dilithium).
    *   Examine power traces in simulation (using PrimePower or Vivado Power Analyzer) to extract the design's Correlation Power Analysis (CPA) resistance metrics.
*   **Expected Outcomes:**  
    *   First-order CPA/DPA resistance with a strict limit on area propagation ($<40\%$ area overhead for logic masking and PRNG).
    *   Minimal throughput degradation ($<12\%$ latency penalty compared to an unmasked crypto-pipeline).
*   **Academic / Publication Angle:**  
    *   *Novelty:* A unified, low-latency hardware cell capable of executing masked modular addition, subtraction, and multiplication in unified memory space.
    *   *Venue:* IEEE Transactions on Information Forensics and Security (TIFS) or Conference on Cryptographic Hardware and Embedded Systems (CHES).

---

### Topic 5: Hardware-Driven Directory Cache Coherency Controller with Bloom-Filter Directory Compression for Chiplet-Based SoCs
*   **Problem Statement:**  
    As industry shifts toward chiplet-based architectures, core counts increase, raising the complexity of multi-level cache coherency protocols. Broadcast-based (snooping) protocols saturate chiplet interconnects, while standard Directory-based schemes require large on-chip SRAM directories to keep track of sharing states, incurring high area and lookup power penalties. This research proposes the design of an RTL-level **Adaptive Directory Cache Coherency Controller** that compresses sharing states using hardware-based **Multi-Hash Bloom Filters**, dramatically reducing directory storage requirements while keeping directory lookup latency to a single cycle.
*   **Industry Importance:**  
    Central to AMD (Zen Infinity Fabric), Intel (Ultra Path Interconnect), and Apple chiplet development. Optimization of interconnect congestion and cache directory efficiency is a recurring industry issue.
*   **Technical Approach:**  
    *   Design a directory controller module in SystemVerilog along with private L1 caches and a shared directory.
    *   Develop a hardware Bloom Filter featuring 3 configurable hash circuits to represent the sharing list in a compressed bit-vector.
    *   Build a verification environment that feeds the controller realistic write-sharing conflicts, false-sharing tests, and core-to-core transfers.
    *   Perform synthesis and analyze directory spatial savings and false-positive penalty patterns.
*   **Expected Outcomes:**  
    *   A $>50\%$ storage savings in directory SRAM size.
    *   Minimal interconnect overhead ($<4\%$ additional coherency messages) caused by false positives from the compressed Bloom filter representation.
*   **Academic / Publication Angle:**  
    *   *Novelty:* An adaptive Bloom Filter that updates hash selections on-the-fly based on core locality and data-sharing profiles.
    *   *Venue:* IEEE Computer Architecture Letters (CAL) or IEEE Transactions on Multi-Scale Computing Systems.

---

## 2. Comparative Evaluation & Ranking

The table below ranks the proposed topics on three vectors (rated from $1$ to $5$, with $5$ being the best):
1.  **Industry Relevance:** How attractive is this topic to recruiters/hiring managers at high-performance silicon houses?
2.  **Ease of 3-to-4-Month Completion:** Is it achievable for a full-time RTL engineer working evenings/weekends without complex fabrication, reliance on third-party analog platforms, or proprietary design environments?
3.  **Publishability:** Does the topic have a clear path to generating algorithmic or architectural novelty for an IEEE paper?

| Rank | Topic | Industry Relevance | Ease of Completion | Publishability | Overall Score |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **1** | [Topic 1: Fracturable Mixed-Precision PE](#topic-1-dynamically-fracturable-mixed-precision-fp16--int8--int4-processing-element-for-edge-transformer-hardware-accelerators) | 5/5 | 4/5 | 4/5 | **4.33** |
| **2** | [Topic 2: PCIe Virt-DMA QoS Scheduler](#topic-2-queue-prioritized-multi-channel-virtualized-dma-engine-with-adaptive-weighted-round-robin-wrr-scheduling-for-pcie-gen56-subsystems) | 5/5 | 5/5 | 3/5 | **4.33** |
| **3** | [Topic 4: SCA-Resistant PQC Core](#topic-4-side-channel-attack-sca-resistant-dilithiumkyber-nt-mathematics-core-utilizing-synthesizable-masked-multipliers) | 5/5 | 3/5 | 4/5 | **4.00** |
| **4** | [Topic 3: Sparse GNN CSC Decoder](#topic-3-zero-overhead-hardware-streaming-csccsr-decoder-for-sparse-matrix-accelerators-in-graph-neural-networks-gnns) | 4/5 | 3/5 | 5/5 | **4.00** |
| **5** | [Topic 5: Bloom-Filter Cache Coherency](#topic-5-hardware-driven-directory-cache-coherency-controller-with-bloom-filter-directory-compression-for-chiplet-based-socs) | 4/5 | 3/5 | 4/5 | **3.67** |

---

## 3. Top 2 Recommendations

For an RTL Design Engineer working full-time with access to main-line synthesis/simulation tools looking to make a big career impact in 3–4 months, the top recommendations are **Topic 1** and **Topic 2**:

### Recommendation A: Topic 1 (Fracturable Mixed-Precision PE)
*   **Why it aligns with your profile:** AI/ML acceleration is currently the main driver of growth in modern semiconductor logic. High-end product companies (NVIDIA, AMD, Apple, Tenstorrent) consist of large teams dedicated solely to arithmetic datapath architecture.
*   **Feasibility:** You do not have to build the whole accelerator. Designing, implementing, and verifying **one** deeply-engineered processing element is a perfectly bounded task for 3-5 months. 
*   **Impact:** Putting "Booth Multiplier Fracturing for Mixed-Precision (FP16/INT8/INT4)" on your resume instantly catches the eye of design managers working on modern high-performance computational units.

### Recommendation B: Topic 2 (Prioritized Virtual-DMA QoS Scheduler)
*   **Why it aligns with your profile:** High-speed interfaces (PCIe, CXL, NVMe, USB4) are present in every major modern chip. The standard block-level logic involves queuing, priority encoders, handshake engines, and AXI matrix routing.
*   **Feasibility:** This relies heavily on Control Logic and State Machines rather than complex mathematical datapaths. It can be easily developed in standard RTL environments, successfully simulated using high-latency master/slave models, and synthesized cleanly in an FPGA flow.
*   **Impact:** Design-heavy chip roles at storage, networking, and server companies (Broadcom, Mellanox/NVIDIA, Intel, Samsung) require engineers who can construct high-throughput protocol controllers. This thesis demonstrates your mastery over these standard structural interfaces.
