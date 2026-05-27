# Dissertation Topic Proposals: VLSI / RTL / Digital Design

Context: These topics are selected for a Master's student in VLSI Design and Microelectronics who is also working full-time as an RTL Design Engineer. The goals are industry relevance for product-based semiconductor companies, publication potential, and feasibility within 3-4 months using mainly RTL tools and simulation environments.

## 1. Low-Power INT8/INT4 Reconfigurable MAC Array for Edge AI Accelerators

### Problem Statement

Design and evaluate a parameterized RTL architecture for a low-power matrix multiply engine that supports both INT8 and INT4 modes, with operand gating, zero-skipping, and configurable systolic or tiled dataflow.

### Why It Matters in Industry

AI inference accelerators are central to NVIDIA, AMD, Intel, Qualcomm, Apple, and edge-AI startups. Product teams care deeply about TOPS/W, memory bandwidth, utilization, and configurability across model precisions.

### Technical Approach

Implement a small RTL accelerator block, for example an 8x8 or 16x16 MAC array, with:

- INT8 and packed INT4 compute modes
- Clock or operand gating for zero activations or weights
- Configurable accumulation width
- Simple SRAM-like input and output buffers
- Testbench with random matrices and Python/C golden model
- Synthesis comparison across modes using Vivado, Quartus, or open-source tools like Yosys/OpenSTA if needed

### Expected Outcomes

- Area, timing, and switching-activity comparison for INT8-only versus INT8/INT4 reconfigurable design
- Power reduction through operand gating and sparsity-aware execution
- Throughput and utilization analysis for different matrix sizes and sparsity levels

### Publishable Angle

The paper can focus on:

**A configurable low-power RTL microarchitecture for mixed-precision edge-AI matrix multiplication with sparsity-aware operand gating.**

A strong publication contribution would be a clear architecture, measurable power/performance/area tradeoff, and comparison against baseline MAC arrays.

## 2. Runtime-Adaptive Approximate Multiplier for Energy-Efficient AI/Signal Processing Workloads

### Problem Statement

Design an RTL multiplier that can switch between exact and approximate modes at runtime, targeting energy-efficient AI inference or DSP workloads where small numerical error is acceptable.

### Why It Matters in Industry

Modern semiconductor products increasingly use approximate computing for edge AI, image processing, sensor processing, and always-on compute. Companies care about reducing power without visibly hurting output quality.

### Technical Approach

- Implement an exact multiplier baseline
- Implement approximate modes using techniques such as:
  - Truncated partial products
  - Approximate compressors
  - Lower-bit significance skipping
  - Configurable accuracy levels
- Add runtime control input: exact, approximate-low, approximate-high
- Evaluate using:
  - Random arithmetic tests
  - MAC operation tests
  - Small image filter or neural-network layer simulation
- Measure:
  - Error rate
  - Mean error distance
  - Power proxy via toggle count
  - Area/timing through synthesis

### Expected Outcomes

- Area/power reduction compared to exact multiplier
- Tunable accuracy-energy tradeoff
- Better performance or lower critical path delay in approximate modes

### Publishable Angle

The publishable contribution can be:

**A runtime-configurable approximate multiplier architecture with quantified energy-accuracy tradeoffs for AI inference datapaths.**

This is manageable in 3-4 months and can be made academically attractive with good error metrics and RTL synthesis data.

## 3. Lightweight Hardware Security Monitor for RTL-Level Control-Flow Integrity

### Problem Statement

Design a lightweight RTL security monitor that detects illegal finite-state-machine transitions and abnormal control behavior in digital IP blocks.

### Why It Matters in Industry

Hardware security is now a major concern in SoCs, especially for automotive, data center, defense, and mobile chips. Product companies want practical low-overhead mechanisms for detecting Trojans, fault attacks, and unexpected behavior.

### Technical Approach

Select a target RTL block, such as:

- Bus controller
- DMA-like controller
- UART/SPI/I2C controller
- Simple accelerator controller

Build a security monitor that observes:

- FSM state transitions
- Critical control signals
- Illegal command sequences
- Timeout violations

Implement the monitor as a reusable RTL wrapper. Inject faults or malicious behavior in simulation:

- Illegal state jump
- Skipped authentication/configuration state
- Unexpected write enable
- Stuck control signal

Measure:

- Detection coverage
- Latency of detection
- Area/timing overhead

### Expected Outcomes

- A reusable RTL-level security monitor template
- Detection of illegal transitions with low area overhead
- Formalizable or assertion-backed security properties

### Publishable Angle

The paper can be positioned as:

**A low-overhead RTL-level control-flow integrity monitor for detecting illegal FSM behavior in SoC IP blocks.**

This is attractive because it combines RTL design, verification, and hardware security.

## 4. AXI-Stream Traffic Shaper with QoS-Aware Arbitration for Accelerator-Based SoCs

### Problem Statement

Design a QoS-aware AXI-Stream or simplified NoC-style arbiter that dynamically prioritizes traffic from multiple accelerator clients based on latency, bandwidth, and starvation constraints.

### Why It Matters in Industry

Modern SoCs have many accelerators sharing memory and interconnect bandwidth. Companies like AMD, NVIDIA, Qualcomm, and Intel constantly optimize arbitration, QoS, latency, and fairness in SoC fabrics.

### Technical Approach

Implement a multi-client RTL traffic manager:

- 4 or 8 input streams
- Weighted round-robin
- Priority aging
- Latency-sensitive and bandwidth-sensitive modes

Use an AXI-Stream-like valid/ready protocol. Generate synthetic traffic patterns:

- Bursty traffic
- Latency-sensitive traffic
- Mixed traffic
- Saturation conditions

Compare against:

- Round-robin
- Fixed priority
- Weighted round-robin baseline

Measure:

- Average latency
- Worst-case latency
- Throughput
- Starvation rate
- Area/timing

### Expected Outcomes

- Improved fairness and latency under mixed traffic
- Configurable arbitration policy
- RTL implementation suitable for FPGA/ASIC synthesis

### Publishable Angle

The paper could be:

**A lightweight QoS-aware RTL arbitration scheme for accelerator-rich SoC interconnects.**

This is very industry-relevant, though publishability depends on how novel the arbitration policy and evaluation are.

## 5. Error-Resilient SRAM Interface with Lightweight ECC for AI Accelerator Buffers

### Problem Statement

Design and evaluate a lightweight ECC-protected SRAM interface for AI accelerator buffers, optimized for low area and low latency.

### Why It Matters in Industry

Memory reliability is critical in AI accelerators, automotive SoCs, and edge devices. On-chip SRAM buffers are vulnerable to soft errors, voltage scaling effects, and data corruption.

### Technical Approach

- Implement an SRAM-like memory wrapper in RTL
- Add selectable protection modes:
  - Parity
  - SECDED Hamming ECC
  - Lightweight checksum or residue-based coding
- Support write encode and read decode/correction
- Inject single-bit and double-bit errors in testbench
- Evaluate:
  - Correction capability
  - Detection capability
  - Latency overhead
  - Area overhead
  - Impact on accelerator buffer bandwidth

### Expected Outcomes

- Correct single-bit errors
- Detect double-bit errors
- Quantified reliability versus area/latency tradeoff

### Publishable Angle

Paper idea:

**A configurable low-latency ECC memory wrapper for resilient AI accelerator scratchpad buffers.**

This is practical and finishable. It becomes stronger if connected to AI accelerator data reuse and buffer access patterns.

## 6. Power-Aware FSM Encoding and Clock-Gating Methodology for Control-Dominated RTL Blocks

### Problem Statement

Develop and evaluate a practical RTL methodology for reducing switching power in control-heavy modules using FSM encoding selection, clock gating, and activity-aware state assignment.

### Why It Matters in Industry

Not all power is in datapaths. Control logic, always-on blocks, protocol controllers, and low-power domains matter significantly in mobile, automotive, and IoT chips.

### Technical Approach

Choose multiple control blocks:

- Protocol controller
- DMA controller
- Memory controller
- Accelerator scheduler

Implement versions with:

- Binary encoding
- One-hot encoding
- Gray-like low-transition encoding
- Clock enable insertion
- State-dependent datapath gating

Simulate realistic workloads. Extract toggle activity from simulation. Synthesize and compare area/timing/power proxy.

### Expected Outcomes

- Measured reduction in switching activity
- Clear guidelines for FSM encoding choice
- RTL templates for power-aware control design

### Publishable Angle

Possible paper:

**RTL-level activity-aware FSM encoding and gating methodology for low-power digital controllers.**

This is feasible, but the novelty must be sharpened. The risk is that FSM encoding is an older topic, so the work needs a modern angle, such as automated activity profiling or workload-specific encoding.

## 7. Lightweight Runtime Fault Detection for Systolic Array-Based Matrix Multiplication

### Problem Statement

Design a low-overhead fault detection mechanism for a systolic-array matrix multiplier using algorithmic checksums or residue-based verification.

### Why It Matters in Industry

AI accelerators are used in safety-sensitive systems, automotive platforms, and data centers. Silent data corruption is a real concern, especially under voltage scaling, aging, radiation, or hardware faults.

### Technical Approach

- Implement a small systolic array, such as 4x4, 8x8, or parameterized NxN
- Add runtime error detection using:
  - Row/column checksum
  - Residue arithmetic
  - Duplicate lightweight checker for selected outputs
- Inject faults:
  - Bit flips in MAC output
  - Stuck-at partial sum
  - Corrupted input activation
  - Wrong accumulation
- Compare:
  - No protection
  - Duplication
  - Proposed checksum/residue method
- Measure:
  - Fault coverage
  - Detection latency
  - Area overhead
  - Timing impact

### Expected Outcomes

- High detection coverage with less overhead than full duplication
- Practical reliability enhancement for AI datapaths
- Clear tradeoff between coverage and area

### Publishable Angle

Strong paper direction:

**Low-overhead runtime error detection for systolic-array AI accelerators using algorithmic checksum monitoring.**

This is one of the best topics for combining AI hardware, RTL, reliability, and publication potential.

## 8. Configurable Hardware Scheduler for Multi-Mode AI/DSP Accelerator Pipelines

### Problem Statement

Design a configurable RTL scheduler that dynamically maps operations to shared compute resources in a multi-mode accelerator supporting convolution, matrix multiplication, and vector operations.

### Why It Matters in Industry

Many real accelerators reuse hardware across workloads. Efficient scheduling improves utilization, latency, and area efficiency. This is directly relevant to AI SoCs, DSP engines, and domain-specific accelerators.

### Technical Approach

Build a simplified shared datapath:

- MAC units
- Input buffers
- Output accumulator

Support multiple operation modes:

- Matrix-vector multiply
- Small GEMM
- FIR/vector dot product

Implement scheduler FSM or microcoded controller. Compare:

- Static scheduling
- Dynamic ready-based scheduling
- Priority-based scheduling

Evaluate utilization, cycles, and area.

### Expected Outcomes

- Higher compute utilization
- Reduced idle cycles
- Flexible reusable accelerator control architecture

### Publishable Angle

Potential paper:

**A configurable RTL scheduling architecture for resource-shared AI/DSP accelerator pipelines.**

This is product-relevant, but publication strength depends on how clearly the scheduler improves utilization over baselines.

## Ranking

| Rank | Topic | Industry Relevance | 3-4 Month Feasibility | Publishability | Overall |
|---:|---|---|---|---|---|
| 1 | Runtime fault detection for systolic arrays | Very High | High | Very High | Excellent |
| 2 | Low-power INT8/INT4 AI MAC array | Very High | Medium-High | Very High | Excellent |
| 3 | Lightweight RTL security monitor | High | High | High | Excellent |
| 4 | SRAM ECC wrapper for AI buffers | High | High | Medium-High | Very Good |
| 5 | QoS-aware AXI-stream arbiter | Very High | High | Medium | Very Good |
| 6 | Runtime-adaptive approximate multiplier | High | High | Medium-High | Very Good |
| 7 | Configurable AI/DSP scheduler | High | Medium | Medium-High | Good |
| 8 | Power-aware FSM encoding methodology | Medium-High | High | Medium | Good |

## Top 2 Recommendations

### 1. Low-Overhead Runtime Fault Detection for Systolic Array-Based Matrix Multiplication

This is the strongest dissertation topic for your profile. It connects directly to AI accelerators, RTL datapath design, reliability, and product-grade semiconductor concerns. It is also realistic because you can build a small systolic array, inject faults in simulation, measure detection coverage, and synthesize the design for overhead analysis.

Why it suits your profile:

- Shows strong RTL and architecture capability
- Aligns with AI hardware trends
- Does not require fabrication or advanced EDA access
- Has a clear publishable story: reliability with low overhead
- Gives strong interview talking points for NVIDIA/AMD/Intel/Qualcomm-style roles

Best possible title:

**Low-Overhead Runtime Error Detection for Systolic Array AI Accelerators Using Algorithmic Checksum Monitoring.**

### 2. Low-Power INT8/INT4 Reconfigurable MAC Array for Edge AI Accelerators

This is the best topic if the goal is maximum alignment with product-company digital design roles. Mixed precision, sparsity, and power-aware compute are extremely relevant in modern AI accelerators. It lets you demonstrate datapath design, parameterized RTL, timing/area tradeoffs, verification, and architecture thinking.

Why it suits your profile:

- Highly relevant to AI accelerator and high-performance RTL roles
- Strong practical implementation scope
- Good publication potential if multiple modes are compared and power/performance tradeoffs are quantified
- Builds a portfolio-quality RTL project

Best possible title:

**A Mixed-Precision INT8/INT4 Reconfigurable MAC Array with Sparsity-Aware Operand Gating for Edge AI Inference.**

## Final Advice

For the best balance of publication strength and feasibility, choose:

**Runtime Fault Detection for Systolic Arrays.**

For the best alignment with industry RTL/AI accelerator roles, choose:

**Low-Power INT8/INT4 Reconfigurable MAC Array.**

A strong dissertation strategy would be to combine them slightly, without expanding the scope too much:

**Design a small INT8 systolic array and add lightweight runtime error detection.**

That gives one clean architecture, one strong reliability contribution, and one industry-relevant AI accelerator platform. It is focused enough for 3-4 months and strong enough to discuss in both academic and product-company interviews.
