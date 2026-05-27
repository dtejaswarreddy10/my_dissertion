# VLSI/RTL Dissertation Topic Proposals

**Candidate profile**
- M.Tech (WILP) in VLSI Design and Microelectronics — final-semester dissertation.
- Full-time RTL Design Engineer (Verilog/SystemVerilog, synthesis, basic verification).
- Available bandwidth: 3–4 months alongside a full-time job.
- Tooling: RTL + simulation + FPGA (Vivado), no fabrication access.

**Goals**
1. Strengthen profile for product-based semiconductor companies (NVIDIA, AMD, Intel, Qualcomm, Marvell, Tenstorrent).
2. Produce novel work suitable for IEEE / ACM conference or journal publication.
3. Fit realistically within a 3–4 month window alongside full-time work.

---

## 1. Sparsity-Aware Systolic Array for INT8 DNN Inference

**Problem statement.** Conventional systolic arrays (TPU-style) waste cycles and energy on zero-valued activations/weights, which dominate quantized CNN/Transformer workloads (40–80% sparsity). Design an RTL-level *structured-sparsity* systolic array (e.g., 8×8 or 16×16 INT8 MAC grid) that skips zero MACs and clock-gates idle PEs, with a lightweight sparsity-encoding front-end.

**Why it matters in industry.** NVIDIA (Sparse Tensor Cores), AMD XDNA, Qualcomm Hexagon, and every AI-accelerator startup are chasing exactly this. Sparsity exploitation is the #1 perf/W lever in inference silicon today.

**Technical approach.**
- Baseline: dense 8×8 INT8 output-stationary systolic array in SystemVerilog.
- Add 2:4 structured sparsity (NVIDIA Ampere style) decoder + index muxing per PE.
- Add fine-grained clock gating + operand isolation.
- Verify with UVM-lite testbench; drive real CNN layer tensors (ResNet-18 / MobileNet) extracted via PyTorch.
- Synthesize on Vivado for a Zynq/Kintex part; report area, Fmax, dynamic power (vcd → `report_power`).

**Expected outcomes.** 1.5×–2× throughput on sparse layers, 30–45% dynamic power reduction vs dense baseline, <15% area overhead.

**Publication angle.** IEEE TVLSI, IEEE Access, ISVLSI, VLSID, ASP-DAC — *"An RTL Microarchitecture for 2:4 Structured-Sparsity INT8 Systolic Arrays with Per-PE Clock Gating."* Novelty = the combination of structured-sparsity skipping + PE-level power gating + open RTL/methodology.

---

## 2. Reconfigurable IMC (In-Memory Compute) Tile for INT8 GEMM

*(extends existing work in [imc_assignment_2/](../imc_assignment_2/))*

**Problem statement.** Extend the existing IMC tile prototype into a *reconfigurable* tile that supports multiple precisions (INT4 / INT8 / INT16) and tile sizes via mode bits, with a runtime-programmable requantizer.

**Why it matters.** Mixed-precision is the dominant trend (FP8/INT4 for LLM inference). Companies want engineers who understand IMC, quantization, and reconfigurable datapaths.

**Technical approach.**
- Parameterize MAC array width and accumulator depth.
- Add precision-mode FSM that re-packs INT4 pairs into INT8 lanes.
- Build golden Python model (already present: [imc_assignment_2/sw/golden_int8_gemm.py](../imc_assignment_2/sw/golden_int8_gemm.py)).
- Compare PPA across modes on Zedboard/Zynq.

**Expected outcomes.** 2× throughput in INT4 mode vs INT8 with <10% area cost; demonstrable accuracy on a quantized model.

**Publication angle.** ISCAS / VLSID / IEEE Embedded Systems Letters — *"A Reconfigurable Mixed-Precision IMC Tile for Edge Inference."*

**Bonus.** Lowest-risk topic because ~40% of the work already exists in the workspace.

---

## 3. Low-Latency AXI4 ↔ CHI-Lite Bridge with QoS-Aware Arbitration

**Problem statement.** Modern SoCs increasingly use ARM CHI for coherent interconnect, while most IP is AXI4. Design an RTL bridge with credit-based flow control and QoS-aware arbitration, and characterize latency/throughput under synthetic + trace-driven traffic.

**Why it matters.** Interconnect/NoC is a huge hiring area at NVIDIA (NVLink/CMEM), AMD (Infinity Fabric), ARM, Qualcomm. Companies struggle to find engineers who deeply understand AXI/CHI/QoS.

**Technical approach.**
- SystemVerilog bridge with separate AR/AW/R/W/B queues and CHI REQ/RSP/DAT/SNP channel mapping (subset — non-coherent first, then add snoop pass-through).
- UVM (or cocotb) testbench with constrained-random traffic patterns.
- Use open CHI-B subset documentation; no real coherence required for v1.

**Expected outcomes.** Quantified latency vs load curves, head-of-line blocking analysis, area/Fmax on FPGA.

**Publication angle.** IEEE Access / NoCArc workshop / ISVLSI — *"A QoS-Aware AXI-to-CHI Bridge with Credit-Based Flow Control."*

**Caveat.** CHI spec is large — scope tightly to non-coherent subset to fit in 3–4 months.

---

## 4. Hardware Trojan Detection via Side-Channel-Aware RTL Instrumentation

**Problem statement.** Insert lightweight RTL monitors (toggle counters, path-activity sensors) that flag anomalous switching patterns indicative of dormant Trojans, validated against TrustHub benchmarks.

**Why it matters.** Hardware security is exploding (Intel, Qualcomm, NXP, defense). Trojan detection at RTL is publishable and tool-light.

**Technical approach.**
- Pick 3–5 TrustHub benchmarks (AES-T100, RS232-T1000, etc.).
- Add monitor wrappers; train a tiny anomaly model offline (Python).
- Show detection rate, area overhead.

**Expected outcomes.** >90% detection on chosen benchmarks with <5% area overhead.

**Publication angle.** IEEE HOST workshop, IEEE Embedded Systems Letters, ISVLSI — *"Lightweight RTL Instrumentation for Side-Channel-Based Hardware Trojan Detection."*

---

## 5. Approximate Computing MAC Unit Library for Error-Resilient AI Workloads

**Problem statement.** Build a *parameterized* library of approximate INT8/FP16 multipliers and adders (truncation, Mitchell-log, broken-array) and characterize the PPA-vs-accuracy Pareto frontier on a small CNN inference workload.

**Why it matters.** Edge-AI / battery-powered inference (Qualcomm, Apple, Samsung LSI) cares deeply about approximate arithmetic.

**Technical approach.**
- 4–6 multiplier variants in Verilog.
- Plug into a small MAC array; run MNIST/CIFAR inference with PyTorch + bit-accurate Python model.
- Synthesize each, compare area/power/Fmax.

**Expected outcomes.** Pareto chart; identify "sweet spot" approximate MAC giving e.g. 35% power reduction with <1% accuracy loss.

**Publication angle.** IEEE TCAS-II, ISCAS, VLSID — *"An RTL Library and PPA-Accuracy Characterization of Approximate MAC Units for Edge Inference."*

---

## 6. RISC-V Custom ISA Extension for Post-Quantum Cryptography (Kyber/Dilithium)

**Problem statement.** Take an open RISC-V core (CV32E40P / Ibex / VexRiscv) and add custom instructions accelerating NTT (Number-Theoretic Transform) — the kernel of Kyber/Dilithium PQC standards finalized by NIST.

**Why it matters.** Every SoC vendor is racing to add PQC acceleration (Intel, Qualcomm, ARM, Apple). RISC-V custom extensions are hot.

**Technical approach.**
- Identify NTT butterfly hotspot.
- Add `nttbfly` custom instruction (decode + new functional unit).
- Verilator-based sim; measure cycle count vs software baseline.
- Synthesize on FPGA.

**Expected outcomes.** 5–10× speedup on Kyber NTT, modest area cost.

**Publication angle.** IEEE TVLSI / DATE / CARRV workshop — *"An NTT Custom-ISA Extension for PQC on RISC-V."*

**Caveat.** Steepest learning curve (NTT math + RISC-V pipeline + Kyber spec). Higher reward but higher risk.

---

## 7. Clock-Gating + Power-Gating Aware RTL Pass for Low-Power AI Datapaths

**Problem statement.** Develop a methodology (RTL templates + a small Python pass over a Verilog AST or simple coding-style rules) that automatically inserts fine-grained clock and operand-isolation gates into MAC datapaths, and quantify dynamic power savings on a baseline inference kernel.

**Why it matters.** Low-power design is *the* differentiator for mobile / edge silicon. Engineers who can quantify dynamic power are rare.

**Technical approach.**
- Take a baseline 8×8 MAC array.
- Insert ICG cells (FF gating), operand isolators, and observability hooks.
- Measure with Vivado `report_power` from post-impl simulation VCD.

**Expected outcomes.** 25–40% dynamic power reduction with <5% area overhead.

**Publication angle.** ISLPED workshop / IEEE Embedded Systems Letters — *"Fine-Grained RTL Power Gating for INT8 MAC Datapaths."*

---

## Ranking Matrix (1 = poor, 5 = excellent)

| # | Topic | Industry Relevance | Feasibility (3–4 mo + full-time job) | Publishability | **Total** |
|---|---|---|---|---|---|
| 1 | Sparsity-Aware Systolic Array | 5 | 4 | 5 | **14** |
| 2 | Reconfigurable IMC Tile (extends existing work) | 5 | 5 | 4 | **14** |
| 3 | AXI4 ↔ CHI-Lite Bridge | 5 | 3 | 4 | **12** |
| 4 | HW Trojan Detection | 4 | 4 | 4 | **12** |
| 5 | Approximate MAC Library | 4 | 4 | 4 | **12** |
| 6 | RISC-V PQC ISA Extension | 5 | 2 | 5 | **12** |
| 7 | Clock/Power-Gating Methodology | 4 | 5 | 3 | **12** |

---

## Top 2 Recommendations

### #1 — Reconfigurable Mixed-Precision IMC Tile (extends `imc_assignment_2/`)
Lowest risk, highest completion probability. RTL, golden model, Vivado flow, and a working Zedboard build already exist in [imc_assignment_2/](../imc_assignment_2/). Extending to mixed precision + reconfigurability gives a clear novelty story without starting from zero. Strong narrative for AI/edge-inference roles at Qualcomm, AMD, NVIDIA, and IMC startups (Mythic, Rain, EnCharge). Publishable at VLSID or ISCAS.

### #2 — Sparsity-Aware INT8 Systolic Array with Per-PE Clock Gating
Highest profile leverage. "Sparse Tensor Core–style RTL" is the single most directly relevant talking point for product-company AI-hardware roles. Scope is bounded (8×8 array, one sparsity scheme, one workload family), the baseline is well-understood (TPU systolic), and the novelty (RTL-level structured sparsity + power gating + open methodology) is clearly publishable in IEEE TVLSI or ISVLSI.

**Suggested path.** Start with **#2** as the primary topic. If time pressure mounts at month 2, pivot/merge with **#1** — the IMC tile is effectively a systolic compute fabric, so the sparsity work can ride on top of the existing infrastructure. This hybrid plan is the lowest-risk, highest-reward route given the constraints.
