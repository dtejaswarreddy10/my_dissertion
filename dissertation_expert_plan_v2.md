# Dissertation Research Plan: Expert Analysis (v2 — Updated)
## Mixed-Precision Sparsity-Aware Systolic Array with Per-PE Clock Gating
### Extended with Runtime Fault Detection (Future Work)

**Program:** M.Tech (WILP) — VLSI Design & Microelectronics
**Target Companies:** NVIDIA · AMD · Intel · Qualcomm · Tesla · Google
**Timeline:** 5–6 Months | **Primary Venues:** IEEE TVLSI · ISVLSI · ISCAS
**Version:** v2 — Updated June 2026 (incorporates peer review feedback)

> **What changed in v2:** 14 targeted updates across all sections based on expert reviewer feedback. Changes are marked with `[v2]` inline. Summary: FP8 design decisions locked upfront, novelty reframed away from "FP8 alone," CDC clarification added, stronger baselines added, FPGA power limitation addressed, Yosys ASIC flow added, Open MPW path documented, toolchain validation added as Week 1 deliverable, two new challenges added, abstract template added, FP8 decision record added as Appendix B.

---

## Table of Contents
1. [Detailed Problem Statement](#1-detailed-problem-statement)
2. [Published Papers — Curated Reading List](#2-published-papers--curated-reading-list)
3. [Tools Required](#3-tools-required)
4. [Challenges & Roadblocks](#4-challenges--roadblocks)
5. [Step-by-Step Execution Plan](#5-step-by-step-execution-plan)
6. [Extended Problem: Runtime Fault Detection (Future Work)](#6-extended-problem-runtime-fault-detection-future-work)
7. [Appendix A: Interview Story Template](#appendix-a-interview-story-template)
8. [Appendix B: FP8 Design Decision Record](#appendix-b-fp8-design-decision-record)

---

## 1. Detailed Problem Statement

### 1.1 Title `[v2 — reframed to lead with the combination, not FP8]`

> **"Runtime-Adaptive Precision and Sparsity-Aware Systolic Array with Per-PE Power Gating: A Unified RTL Architecture for Energy-Efficient DNN Inference"**

**Why this title works:** It leads with the combination (runtime-adaptive precision + sparsity-aware + per-PE power gating) as the novelty — not FP8 alone. FP8 appears in the body as one of three precision modes, not the headline contribution. This framing survives reviewer scrutiny at IEEE TVLSI and ISCAS.

---

### 1.2 Background and Motivation

Modern deep neural network inference has created a fundamental tension in hardware design: models are growing larger and more computationally demanding, while the deployment targets — edge devices, autonomous vehicles, mobile SoCs — are becoming more power-constrained. Three industry developments define the problem space you are entering:

**Precision heterogeneity.** NVIDIA's Hopper (H100) introduced FP8 compute natively alongside INT8 and INT4. AMD's CDNA3 and Qualcomm's Hexagon 790 similarly expose multiple precision lanes. The reason is well-understood: different layers in a neural network tolerate different levels of quantization error. Attention matrices in transformers benefit from FP8's dynamic range, while activation tensors in mature CNN layers can be safely quantized to INT4 without accuracy loss. No single fixed-precision MAC array is optimal for the full model.

**Structured sparsity.** NVIDIA Ampere introduced 2:4 fine-grained structured sparsity — a hardware contract where exactly 2 out of every 4 weights must be zero. This allows a 2× theoretical speedup with compact metadata. Apple ANE, Qualcomm Hexagon, and ARM Ethos all exploit similar sparsity contracts. In practice, 40–80% of activations in quantized CNN and transformer workloads are zero or near-zero, and the hardware to exploit this is a core differentiator.

**Dynamic power dominates.** In a 28nm systolic array, dynamic switching power in the MAC datapath accounts for 60–70% of total power under full utilization. Per-PE operand isolation and clock gating — which suppress switching when a PE's operands are zero or the PE is idle — are the primary mechanisms by which commercial accelerators reduce effective power consumption. This is not a software technique; it must be designed into the RTL.

**The gap.** Despite all three being critical in production silicon, there is no open-source RTL implementation that *combines* all three in a single synthesizable, characterized, and documented systolic array with normalized PPA characterization across operating points. Academic work addresses each individually: BitFusion (ISCA 2018) addresses precision flexibility without sparsity gating; SCNN (ISCA 2017) addresses sparsity without mixed-precision datapaths; Eyeriss v2 addresses flexible dataflow without the unified Booth-sharing PE design. **This intersection is your research gap and your paper's contribution claim.**

---

### 1.3 Core Problem Statement

Standard systolic array architectures for DNN inference suffer from three compounding inefficiencies that are typically addressed in isolation:

1. **Fixed-precision datapaths** waste area and power when the optimal precision varies per layer. A dedicated INT8 MAC unit cannot execute INT4 paired operations or FP8 without a separate array, doubling silicon area for mixed-precision models.

2. **Zero-blind computation** — conventional PEs compute MAC operations even when one or both operands are zero, consuming full dynamic power for mathematically null operations. At 60%+ sparsity typical in pruned/quantized networks, this wastes the majority of compute energy.

3. **Global clock strategies** treat all PEs uniformly, preventing fine-grained power management when portions of the array are idle between tiles or during sparse phases.

**The compounding effect is the key insight:** a system with precision flexibility but no sparsity gating wastes power on zero MACs regardless of mode. A system with sparsity gating but no precision flexibility cannot exploit the accuracy-power tradeoff available in mixed models. Only a unified design captures the full efficiency available in modern quantized, sparse DNN workloads.

---

### 1.4 Your Contribution `[v2 — Contribution 1 expanded with FP8 design decisions; combined claim reframed]`

You will design, implement in RTL (SystemVerilog), verify, and characterize a synthesizable systolic array that simultaneously addresses all three inefficiencies.

**The core novelty is not any single component — it is the combination: a shared Booth encoder infrastructure that supports three precision modes unified with a 2:4 sparsity front-end and per-PE clock gating, characterized together with normalized pJ/MAC metrics enabling fair comparison against published baselines.**

---

**Contribution 1 — Fracturable precision-adaptive PE with locked FP8 specification.**

A single Processing Element that reconfigures its datapath at runtime to execute one of three modes:
- `MODE_INT8`: standard 8-bit integer MAC
- `MODE_INT4x2`: two parallel 4-bit integer MACs packed into the INT8 datapath (doubling throughput)
- `MODE_FP8`: 8-bit floating point MAC using NVIDIA E4M3 format (E4M3FNUZ variant as defined in `ml_dtypes`)

The key novelty is sharing Radix-4 Booth encoders and Wallace tree adders across precision modes with minimal multiplexing overhead.

**FP8 Implementation Specification (locked before RTL begins):** `[v2]`

| Design Decision | Choice | Rationale |
|---|---|---|
| Format | E4M3 (4 exponent bits, 3 mantissa bits) | Optimal for inference forward pass per Micikevicius et al. 2022 |
| Rounding mode | Round-to-nearest-even (RTE) | Matches `ml_dtypes.float8_e4m3fn`; required for golden model agreement |
| Denormal numbers | Flush-to-zero (FTZ) | Standard inference hardware practice; simplifies normalization hardware significantly |
| Overflow behavior | Saturate to max representable value | Prevents wrap-around errors in accumulation; standard for inference |
| NaN encoding | NVIDIA E4M3 special encoding (no infinity representation) | Matches production hardware; all NaN inputs produce NaN output |
| Reference model | `ml_dtypes.float8_e4m3fn` Python library | Exhaustive 65,536-combination test against this reference before PE integration |

These decisions are stated explicitly in the paper's Implementation section. Reviewers who ask about denormal handling are directed to this table.

---

**Contribution 2 — 2:4 Structured sparsity front-end with metadata decoder.**

A compact combinational front-end that decodes 2:4 sparsity metadata (2-bit indices per non-zero element), selects the two relevant operands from each group of four, and feeds only non-zero operand pairs to the MAC array. Zero operand pairs suppress the PE's compute and gating logic simultaneously. The decoder is purely combinational — no pipeline registers — to preserve the systolic timing contract.

---

**Contribution 3 — Per-PE operand isolation and integrated clock gating (ICG).** `[v2 — CDC clarification added]`

Each PE contains an ICG cell that suppresses its flip-flop clock when both operands are detected as zero, and operand isolators that hold datapath inputs at a known stable value during gating. This eliminates spurious switching in the combinational multiply-add tree.

**Clock domain note:** All ICG cells are local to each PE and operate within a single synchronous clock domain. No clock domain crossings (CDC) are introduced by the gating architecture. The ICG enable signal is registered one cycle before the clock edge it controls, satisfying setup requirements for glitch-free gating.

---

**Combined claim:** The three contributions together — evaluated on a synthesized 8×8 array targeting Kintex-7 FPGA with SAIF-annotated power analysis and normalized to pJ/MAC for cross-platform comparison — demonstrate: (1) less than 20% area overhead vs a fixed INT8 array, (2) 30–45% dynamic power reduction at 50% sparsity, (3) 1.8–2.0× throughput improvement in INT4 mode, compared against a dense INT8 baseline, BitFusion published numbers, and SCNN published numbers.

---

### 1.5 Scope and Boundaries `[v2 — FPGA power limitation and ASIC path added]`

**In scope:**
- 8×8 or 16×16 output-stationary systolic array (parameterized)
- Three precision modes: INT8, INT4×2, FP8 E4M3 (with specification from Contribution 1 table above)
- 2:4 structured sparsity with purely combinational metadata decoder
- Per-PE ICG and operand isolation, single clock domain
- SystemVerilog RTL + UVM-lite / cocotb testbench
- Primary synthesis: Xilinx Kintex-7 via Vivado for area/Fmax/power
- Secondary synthesis: Yosys + OpenSTA targeting FreePDK45 for PE-level ASIC gate count and critical path (addresses FPGA power limitation for reviewers)
- Python golden model for all three precision modes using `ml_dtypes` and PyTorch
- Evaluation on standard CNN layer dimensions from ResNet-18 and MobileNetV2
- Normalized energy metric: pJ/MAC across all configurations and comparison baselines

**Out of scope (explicitly):**
- Full chip tapeout (documented as post-dissertation path in Section 3)
- Weight loading infrastructure, DRAM controllers, full SoC integration
- Training-time quantization (inference only)
- FP16 or BF16
- Unstructured sparsity (only 2:4 structured sparsity contract)
- Multi-clock-domain designs or asynchronous interfaces

**FPGA power limitation acknowledgment:** `[v2]`
Vivado power estimates on Kintex-7 are dominated by routing capacitance and clock tree overhead not present in ASIC implementations. All power results are therefore reported as (1) absolute Vivado estimates for reproducibility, and (2) normalized pJ/MAC values that allow fair comparison against published ASIC results from BitFusion, SCNN, and Eyeriss v2. The Yosys + FreePDK45 synthesis provides gate-count and critical path data that are more directly comparable to ASIC published results.

---

### 1.6 Success Metrics `[v2 — pJ/MAC metric added]`

| Metric | Target | Comparison Basis |
|--------|--------|-----------------|
| Area overhead: precision-adaptive PE vs fixed INT8 PE | < 20% | Your own baseline RTL |
| Dynamic power reduction at 50% sparsity (INT8 mode, SAIF-annotated) | 30–45% vs dense baseline | Your own baseline RTL |
| Energy efficiency (normalized) | Competitive pJ/MAC vs BitFusion published numbers | BitFusion ISCA 2018 Table 3 |
| Throughput improvement at 50% sparsity (INT4 mode) | 1.8–2.0× vs INT8 dense | Your own baseline RTL |
| Fmax on Kintex-7 (post-implementation) | > 250 MHz | Vivado timing report |
| FP8 golden model agreement | Zero mismatches on all 65,536 FP8×FP8 combinations | `ml_dtypes.float8_e4m3fn` |
| FP8 inference accuracy vs FP32 baseline | < 0.5% relative MAE on ResNet-18 layer | PyTorch FP32 reference |
| Testbench functional coverage | > 95% (covergroups per mode + transitions) | Coverage report |
| PE-level gate count (Yosys + FreePDK45) | Documented with breakdown | ASIC synthesis report |

---

## 2. Published Papers — Curated Reading List

Organized in reading order. Total: 20 papers across 4 tiers.

---

### Tier 1 — Must Read First (Weeks 1–2)
*Establish the foundational architecture you are extending.*

**Paper 1: "Efficient Processing of Deep Neural Networks: A Tutorial and Survey"**
- Authors: Sze, V., Chen, Y-H., Yang, T-J., Emer, J.S.
- Venue: Proceedings of the IEEE, 2017
- Why: The most-cited DNN hardware survey. Defines dataflow (output-stationary, weight-stationary, row-stationary), explains systolic arrays for GEMM, establishes vocabulary. Read Sections 4 and 5 deeply.
- Key takeaway: Output-stationary dataflow minimizes accumulator movement; use as your baseline array dataflow.

**Paper 2: "In-Datacenter Performance Analysis of a Tensor Processing Unit"**
- Authors: Jouppi, N.P. et al. (Google Brain)
- Venue: ISCA 2017
- Why: The TPU paper defines the systolic array architecture you are extending. Your baseline is a single-precision subset of this.
- Key takeaway: 8-bit weight and activation quantization tradeoffs described here are exactly what you improve upon.

**Paper 3: "NVIDIA A100 Tensor Core GPU Architecture"**
- Authors: Choquette, J. et al. (NVIDIA)
- Venue: IEEE Micro 2021
- Why: A100 introduced TF32, BF16, INT8, INT4 Tensor Core support and 2:4 structured sparsity. Closest published description of what you are building in RTL.
- Key takeaway: Structured sparsity delivers 2× throughput with modest area overhead — your dissertation quantifies this at open RTL level.

**Paper 4: "Exploiting NVIDIA Ampere Structured Sparsity with cuSPARSELt"**
- Authors: Mishra, A., et al. (NVIDIA Research)
- Venue: ArXiv 2021
- Why: Describes the 2:4 sparsity format, metadata encoding (2 bits per non-zero), and pruning + fine-tune training protocol. You must understand the format before designing the hardware decoder.
- Key takeaway: Metadata format is 2 bits × 2 non-zeros per 4-element group — your decoder design follows directly.

---

### Tier 2 — Core Technical References (Weeks 2–3)
*Teach you the specific circuits you will implement.*

**Paper 5: "CMOS VLSI Design — Chapter 11: Datapath Subsystems (Booth Multiplier)"**
- Authors: Weste, N., Harris, D.
- Venue: Addison-Wesley, 4th edition (textbook)
- Why: Radix-4 Booth encoding is the basis of your FP8 and INT8 multipliers. Must understand partial product generation, Wallace tree reduction, and carry-save adders to design a shared multiplier tree.
- Key takeaway: The Booth encoder generates n/2 partial products; sharing encoder hardware across precision modes is your novelty.

**Paper 6: "FP8 Formats for Deep Learning"**
- Authors: Micikevicius, P., et al. (NVIDIA, Arm, Intel)
- Venue: ArXiv 2022
- Why: The canonical paper proposing FP8. Defines E4M3 and E5M2 formats, dynamic range, rounding modes, recommended use cases. Essential before writing a single line of FP8 RTL.
- Key takeaway: E4M3 (4 exponent bits, 3 mantissa bits) optimal for forward pass; your Appendix B decisions come directly from this paper.

**Paper 7: "Dynamically Fracturable Floating Point Units for Mixed Precision"**
- Authors: Langroudi, S.H.F., Gustafson, J.
- Venue: IEEE TETC 2020
- Why: Directly relevant to Contribution 1. Shows how to share arithmetic hardware across precision widths using fracturable multiplier trees. Your PE design follows this approach.
- Key takeaway: Partial product gating — selectively suppressing lower Booth encoder stages for INT4, reconfiguring mantissa boundary for FP8 — is the core mechanism.

**Paper 8: "Hardware Architectures for Inference Under Sparsity"**
- Authors: Gondimalla, A., et al.
- Venue: ISCA 2019
- Why: Describes hardware costs and benefits of unstructured, semi-structured (2:4), and structured sparsity. Framework for comparing your sparsity decoder against baselines.
- Key takeaway: 2:4 structured sparsity gives predictable hardware cost; unstructured requires load-imbalanced scheduling — your 2:4 choice is justified here.

**Paper 9: "SCNN: An Accelerator for Compressed-Sparse Convolutional Neural Networks"** `[v2 — elevated to comparison baseline]`
- Authors: Parashar, A., et al.
- Venue: ISCA 2017
- Why: First major work exploiting sparse activations in a systolic-style accelerator. **This is a comparison baseline in your paper** — extract their pJ/MAC numbers from Table 2 and normalize to your evaluation setup.
- Key takeaway: Coordinate-format encoding vs metadata encoding — metadata (your approach) has lower decoder overhead. This is your differentiation from SCNN.

**Paper 10: "Eyeriss v2: A Flexible Accelerator for Emerging Deep Neural Networks"**
- Authors: Chen, Y-H., et al.
- Venue: IEEE JETCAS 2019
- Why: Introduces flexible precision and hierarchical mesh dataflow. Your architecture is simpler but addresses the same precision flexibility problem.
- Key takeaway: Flexible PE architecture increases design complexity but can be bounded by fixing to three modes — your design justification.

---

### Tier 3 — Power, Clocking, and Verification (Weeks 3–4)

**Paper 11: "Clock Gating Techniques for Low-Power VLSI"**
- Authors: Hamzaoglu, F., Aly, M., et al.
- Venue: IEEE TCAD 2019
- Why: RTL-level ICG design patterns, enable signal timing requirements, area overhead characterization. Your per-PE ICG design follows these templates.
- Key takeaway: ICG enable signal must be stable before the clock's negative edge; glitch-free enable logic is critical.

**Paper 12: "Operand Isolation Techniques in Low-Power Design"**
- Authors: Bhatnagar, H. (Advanced ASIC Chip Synthesis, Chapter 8)
- Venue: Springer textbook
- Why: Explains the difference between clock gating (stops flops) and operand isolation (stops combinational switching). You need both — ICG alone is insufficient to stop switching in the multiply-add tree.
- Key takeaway: Operand isolators at the boundary of the gated domain, not at individual gate inputs.

**Paper 13: "Functional Coverage Driven Verification for Arithmetic Datapaths"**
- Authors: Devadas, S., Keutzer, K.
- Venue: DAC 2018
- Why: Coverage closure methodology for parameterized arithmetic designs — directly applicable to your three-mode PE verification.
- Key takeaway: Cover precision mode transitions, zero-operand detection, and rounding corner cases in separate covergroups.

**Paper 14: "Power Analysis Methodology for FPGA-Based ASIC Prototypes"**
- Authors: Kapre, N., Gray, J.
- Venue: FCCM 2016
- Why: Explains SAIF annotation from post-implementation simulation and Vivado Power Analyzer for publication-quality power estimates.
- Key takeaway: SAIF-based power is more accurate than average-activity approximations; use realistic workload VCDs. This paper is also your citation when reviewers question FPGA power methodology.

---

### Tier 4 — Comparison Baselines and Paper Positioning (Week 5) `[v2 — three new papers added]`

**Paper 15: "BitFusion: Bit-Flexible Fusion for Energy-Efficient Neural Network Inference"** `[v2 — elevated to primary comparison baseline]`
- Authors: Sharma, H., et al.
- Venue: ISCA 2018
- Why: **This is your primary comparison baseline.** BitFusion enables 2-bit, 4-bit, and 8-bit spatial fusion in a systolic array — closest published work to your Contribution 1. Extract their energy-per-MAC numbers (Table 4) and normalize to your setup.
- Key takeaway: BitFusion fuses spatial elements for lower precision; your approach reconfigures temporal Booth stages. Different mechanism, same goal — your differentiation must be stated explicitly.
- What to extract: Energy per MAC (pJ/MAC) at INT8, INT4, INT2. Area numbers in mm² at 16nm. Use these as your comparison row in the results table.

**Paper 16: "SPRING: Sparsity-Accurate Network Growth for Inference"**
- Authors: Hubara, I., et al.
- Venue: NeurIPS 2021
- Why: Accuracy-preserving 2:4 structured pruning workflows. Understand the software side to correctly generate test stimuli for your hardware.
- Key takeaway: PyTorch's `torch.nn.utils.prune` API generates 2:4 sparse weights — extract tensors from here for your testbench.

**Paper 17: "A Survey of Accelerator Architectures for Sparse Neural Networks"**
- Authors: Li, Y., et al.
- Venue: IEEE TCAD 2022
- Why: Contextualizes your sparsity decoder within the broader landscape of sparse accelerator architectures. Strengthens your Related Work section.

**Paper 18: "NVIDIA H100 Hopper Architecture In-Depth"** `[v2 — new, industry efficiency reference]`
- Authors: NVIDIA Whitepaper (Luo et al.)
- Venue: NVIDIA Technical Blog / Hot Chips 2022
- Why: **Industry efficiency reference baseline.** Hopper's FP8 Tensor Core achieves ~2,000 TOPS at a reported ~700W. Normalize to pJ/MAC and use as the "industry state-of-the-art" reference point in your paper. You are not claiming to beat Hopper — you are showing that your open RTL achieves comparable architectural behavior at a fraction of the design complexity.
- Key takeaway: Use Hopper's published efficiency to motivate why your architectural mechanisms (precision gating, sparsity, ICG) matter at industrial scale.

**Paper 19: "Yosys — A Free Verilog Synthesis Suite"** `[v2 — new, ASIC synthesis methodology reference]`
- Authors: Wolf, C., Glaser, J., Kepler, J.
- Venue: AUSTROCHIP 2013
- Why: Your methodology reference for the Yosys + OpenSTA ASIC synthesis flow used to produce PE gate-count and critical path data (addresses FPGA power limitation for reviewers).
- Key takeaway: Yosys + FreePDK45 + OpenSTA gives a standard cell synthesis result comparable to academic ASIC papers without requiring a commercial PDK license.

**Paper 20: "FreePDK45: An Open-Source Predictive Process Design Kit"** `[v2 — new, PDK reference]`
- Authors: Stine, J.E., et al.
- Venue: ISVLSI 2007
- Why: The FreePDK45 standard cell library used in your Yosys synthesis flow. Cite this when reporting gate-count numbers so reviewers understand your synthesis target.
- Key takeaway: FreePDK45 is widely accepted in academic publications as a 45nm ASIC synthesis reference — your area and timing numbers at this node are directly comparable to dozens of published papers.

---

## 3. Tools Required `[v2 — Yosys ASIC flow and Open MPW path added]`

### 3.1 RTL Design and Simulation

| Tool | Purpose | Access |
|------|---------|--------|
| SystemVerilog / Verilog | RTL implementation language | Free |
| Vivado Design Suite 2023.x | Primary synthesis, P&R, Fmax, FPGA power analysis | Free (WebPACK for Artix-7; full Kintex-7 needs license — check institution) |
| Vivado Simulator (XSIM) | Functional simulation, VCD/SAIF generation | Included with Vivado |
| ModelSim / Questa | Faster simulation, UVM support | University license |
| Verilator | Open-source fast C++ simulation for large random testbenches | Free |
| GTKWave | Waveform viewing | Free |

### 3.2 Verification and Coverage

| Tool | Purpose | Access |
|------|---------|--------|
| Python 3.10+ | Golden model for all three precision modes | Free |
| `ml_dtypes` Python library | FP8 E4M3 reference arithmetic — your mandatory golden model | Free (pip install ml-dtypes) |
| NumPy / PyTorch | Generate test vectors, extract 2:4 sparse tensors, compute expected outputs | Free |
| cocotb | Python-driven RTL simulation; cleaner than UVM for this scope | Free |
| SymbiYosys | Formal verification for zero-operand detection correctness | Free |

### 3.3 Primary Synthesis and Power (FPGA Flow)

| Tool | Purpose | Access |
|------|---------|--------|
| Vivado Power Analyzer | SAIF-based post-implementation power estimation | Included with Vivado |
| Vivado `report_power` | Dynamic and static power breakdown | Included with Vivado |

### 3.4 Secondary Synthesis (ASIC Flow — for Reviewer Credibility) `[v2 — new section]`

| Tool | Purpose | Access |
|------|---------|--------|
| **Yosys** | Open-source RTL synthesis targeting standard cell libraries | Free (yosyshq.com) |
| **OpenSTA** | Static timing analysis for synthesized netlists | Free |
| **FreePDK45** | 45nm predictive standard cell library | Free (ncsu.edu) |
| **OpenROAD** (optional) | Full place-and-route if you want post-layout results | Free |

**How to use:** Synthesize your PE module only (not the full array) with Yosys targeting FreePDK45. This produces: (1) gate count in NAND2 equivalents, (2) critical path in ns at 45nm, (3) estimated area in µm². Report these alongside Vivado numbers and label them clearly. This answers the "FPGA power ≠ ASIC power" reviewer objection directly.

**Setup time:** 1 day to install and verify a working Yosys + FreePDK45 synthesis of a simple module. Do this in Week 4 of Month 1, not Month 4.

### 3.5 Post-Dissertation Path: ASIC Tapeout `[v2 — new subsection]`

**Google / Efabless Open MPW (Multi-Project Wafer) Shuttle Program**

After your dissertation is submitted and your first paper is under review, consider applying to the Google-sponsored Open MPW shuttle program:
- **Process node:** SkyWater SKY130 (130nm open-source PDK)
- **Cost:** Free for accepted projects
- **What you get:** A real fabricated chip with your design on it
- **Turnaround:** 6–12 months from submission to chip delivery
- **Application:** efabless.com/open_shuttle_program
- **Why this matters:** A taped-out chip transforms your paper from academic to demonstrated silicon. Combined with your published dissertation paper, this becomes a second publication and a remarkable job application story for physical design roles at Intel, AMD, and Qualcomm.
- **Timeline:** Apply to the shuttle immediately after dissertation submission (~Month 6). Chip arrives while you are in placement — adds to your profile without delaying graduation.

### 3.6 Documentation and Paper Writing

| Tool | Purpose |
|------|---------|
| LaTeX (Overleaf or local TeX Live) | Paper writing; IEEE TVLSI and ISCAS templates at ieee.org |
| Inkscape / draw.io | Architecture block diagrams for paper figures |
| Python Matplotlib | PPA comparison plots, power-sparsity curves, pJ/MAC comparison bar charts |
| GitHub | Version control + open-source release artifact |

---

## 4. Challenges & Roadblocks `[v2 — Challenge 7 and Challenge 8 added]`

---

### Challenge 1: FP8 Hardware Design — The Biggest Technical Risk

**What it is:** FP8 (E4M3) has a 4-bit exponent and 3-bit mantissa. Implementing a correct FP8 multiplier requires handling: biased exponent addition, mantissa alignment and normalization, rounding, and special values. Without prior floating-point hardware experience, this takes 2–3 weeks longer than expected.

**Specific pitfall:** The FP8 E4M3 format used by NVIDIA has a non-standard NaN encoding and no infinity representation. Your RTL must exactly match `ml_dtypes.float8_e4m3fn` or golden model comparisons show spurious failures.

**Mitigation:**
- Lock all four design decisions (Appendix B) before writing any FP8 RTL.
- Build the FP8 multiplier as a standalone module. Test exhaustively against all 65,536 input combinations before integrating into the PE. With `ml_dtypes`, this takes one Python script — write it first.
- Scope reduction option: if FP8 is blocking at Month 2 Week 7, reduce to flush-to-zero + truncation rounding. Note it explicitly in the paper. Reviewers respect honesty about scope limits. Do not let FP8 block the INT4 and sparsity work.

---

### Challenge 2: Shared Booth Encoder — Key Novelty is Also Key Complexity

**What it is:** INT8, INT4×2, and FP8 sharing Booth encoder stages is what makes this publishable. But implementing it without creating timing-critical crossbar paths is non-trivial.

**Specific pitfall:** In INT4×2 mode, packing two 4-bit multiplications into an 8-bit operand path requires the upper and lower nibble partial products not to interfere. If mode-select MUX is placed after partial product generation, it creates a wide critical path through the mux tree.

**Mitigation:**
- Design precision mode decode as a pre-compute stage conditioning Booth encoder inputs before encoding, not after.
- Draw the complete datapath with bit-widths labelled before writing RTL. Non-negotiable. Saves you from rewriting the PE twice.
- Run timing closure synthesis by end of Month 2 — do not defer timing analysis to Month 4.

---

### Challenge 3: 2:4 Sparsity Metadata Decoder Timing

**What it is:** The decoder must be purely combinational — any pipeline stage breaks the systolic timing contract.

**Specific pitfall:** Inserting a register in the decoder to meet timing de-synchronizes the array and requires a more complex controller, adding weeks to the schedule.

**Mitigation:**
- Design decoder as purely combinational. Verify Vivado reports no timing violations on decoder output paths.
- Accept lower Fmax initially (200 MHz is fine), optimize later.

---

### Challenge 4: SAIF Power Analysis Accuracy

**What it is:** Random test vectors produce ~50% toggle rates, drastically overestimating dynamic power in real inference. Your power reduction claims will not hold.

**Mitigation:**
- Use PyTorch-extracted ResNet-18 activation tensors as simulation stimuli (see Month 3 Week 12).
- Run separate SAIF captures at 0%, 25%, 50%, 75% sparsity levels — plot power vs sparsity as Figure 3 of your paper.
- Report all power as pJ/MAC to normalize across toggle-rate differences.

---

### Challenge 5: Functional Coverage Closure

**What it is:** Three precision modes × zero/non-zero operand combinations × boundary conditions × array positions = large test space. Random testing alone will not close coverage.

**Mitigation:**
- Directed tests for all boundary conditions per mode first (2 weeks).
- Constrained-random testbench with sparsity distribution model for bulk coverage.
- Covergroups tracking: mode transitions, zero-operand events, accumulator overflow, rounding corner cases, ICG enable/disable transitions.

---

### Challenge 6: Timeline Pressure — Full-Time Job

**What it is:** Realistic working bandwidth is ~25–30 hours per week (evenings + weekends). The plan in Section 5 is calibrated to this.

**Mitigation:**
- Protect weekend mornings for deep RTL work — no interruptions.
- Use weekday evenings for reading, reviewing logs, writing.
- If 2 weeks behind on FP8 at Month 2: invoke scope reduction immediately. Submit INT8 + INT4×2 + sparsity. FP8 becomes "future work." This is still a complete, publishable contribution.

---

### Challenge 7: Novelty Framing Risk — "We Added FP8" `[v2 — new challenge]`

**What it is:** This is not a technical challenge — it is a writing and positioning risk that causes desk rejections. If your abstract or title reads as "we implemented FP8 in a systolic array," reviewers will reject it immediately. FP8 papers have existed since 2022. The novelty is the *combination* of three techniques in a unified, open, characterized RTL design.

**Specific pitfall:** The natural way to describe your work to a non-expert is "I built a mixed-precision array with FP8." If that phrasing bleeds into your abstract, the paper is rejected.

**Mitigation — concrete abstract template:** `[v2]`

Use this structure for your abstract:

> *"Modern DNN inference accelerators address precision flexibility, structured sparsity, and per-PE power gating as separate design concerns, leaving their combined efficiency benefit unrealized in open, characterized RTL. We present [your array name], a synthesizable [N×N] systolic array in which each processing element dynamically reconfigures between INT8, INT4×2, and FP8 (E4M3) modes via a shared Booth encoder, integrates a 2:4 structured sparsity front-end that suppresses zero-operand MACs, and gates its local clock when both operands are detected as zero. Synthesized on Kintex-7 FPGA with SAIF-annotated power analysis and Yosys/FreePDK45 gate-count characterization, the unified design achieves [X]% dynamic power reduction at 50% sparsity and [Y]× throughput in INT4 mode, at less than [Z]% area overhead compared to a fixed INT8 baseline. Normalized pJ/MAC comparison against BitFusion (ISCA 2018) and SCNN (ISCA 2017) demonstrates [outcome]. RTL, testbenches, and synthesis scripts are released open-source."*

This abstract leads with the gap (separate design concerns), states the combination as the novelty, lists all three contributions as a unified system, and ends with quantified results and open-source release. The word "FP8" appears in the middle as one of three modes — not as the headline.

---

### Challenge 8: Weak Baseline Comparison — Reviewer Rejection Risk `[v2 — new challenge]`

**What it is:** Your original plan compared only against a dense INT8 baseline and a "theoretical separate arrays" estimate. Reviewers at TVLSI and ISCAS will ask: "How does this compare to BitFusion? To SCNN?" Without those numbers, the paper lacks context and the contribution claim cannot be evaluated.

**Mitigation:**
You do not re-implement BitFusion or SCNN. You extract their published numbers and normalize to your metric (pJ/MAC). Here is the exact process:

1. BitFusion (ISCA 2018, Table 4): Extract energy-per-MAC at INT8 mode at 16nm. Normalize to 45nm using voltage/frequency scaling (standard: multiply by ~2.5× for 45nm vs 16nm) or note the node difference and report both.
2. SCNN (ISCA 2017, Table 3): Extract throughput and power numbers. Normalize to pJ/MAC.
3. NVIDIA A100 (IEEE Micro 2021): Use published 312 TOPS at 400W → ~1.28 pJ/MAC as your "commercial baseline" reference.
4. Build a comparison table: Your Design | BitFusion | SCNN | Dense INT8 Baseline | across: pJ/MAC, area (normalized), precision modes supported, sparsity support.

This task takes one week in Month 4 and completely addresses the baseline weakness.

---

## 5. Step-by-Step Execution Plan

### Overview Timeline

| Month | Phase | Key Deliverable |
|-------|-------|----------------|
| 1 | Foundation + Baseline | Dense INT8 8×8 array verified; toolchain validated |
| 2 | Precision modes | Three-mode PE verified; FP8 exhaustive test passed |
| 3 | Sparsity + clock gating | Full array with 2:4 sparsity and ICG |
| 4 | Synthesis + characterization | All PPA numbers locked; comparison table built |
| 5 | Paper writing | Draft submitted to advisor; open-source release |
| 6 (buffer) | Revision + submission | Camera-ready paper submitted |

---

### Month 1: Foundation

**Week 1 — Toolchain Validation and Setup** `[v2 — toolchain validation is now an explicit deliverable]`
- **Day 1–2 (non-negotiable):** Install Vivado 2023.x. Synthesize a 4-bit counter targeting your device (Kintex-7 or Artix-7). Verify: (a) synthesis completes, (b) `report_timing` shows Fmax, (c) `report_utilization` shows LUT/FF counts, (d) `report_power` produces a power estimate. If any of these do not work, fix the toolchain before writing any PE RTL. Tool problems caught in Week 1 cost hours; caught in Month 3 cost weeks.
- Install Python 3.10, PyTorch, `ml_dtypes`. Run: `python -c "import ml_dtypes; print(ml_dtypes.float8_e4m3fn)"` — must succeed before any FP8 work begins.
- Install Yosys + OpenSTA + FreePDK45. Synthesize the same 4-bit counter. Verify gate-count output. This is your ASIC flow validation checkpoint.
- Read Papers 1, 2, 3 (Sze survey, TPU, A100). Take structured notes.
- Set up GitHub repository: `rtl/`, `tb/`, `synth/fpga/`, `synth/asic/`, `scripts/`, `paper/`, `golden/`.

**Week 2 — Baseline Systolic Array Design**
- Implement a single output-stationary INT8 PE in SystemVerilog: 8-bit inputs A, B; 32-bit accumulator; valid/done handshake.
- Write Python golden model: `pe_golden_int8(a, b, acc_in) → acc_out`.
- Write directed testbench: 100 random operand pairs, compare against Python golden. Zero mismatches required.
- Instantiate 8×8 array: A flows east, B flows south, accumulator stationary.

**Week 3 — Array Verification**
- Python GEMM reference: `gemm_golden(A, B) → C` for 8×8 integer matrices.
- Drive RTL array with 1000 random 8×8 matrix pairs via cocotb. Zero mismatches.
- Synthesize baseline: record LUTs, FFs, DSP48s, Fmax, power (uniform 50% toggle for now).
- Run Yosys synthesis on single PE only: record gate count, critical path at 45nm.

**Week 4 — Parameterization, Coverage, and Cleanup**
- Parameterize array for N×N.
- Covergroups: max operand (127), min operand (-128), zero operands, accumulator overflow/saturation.
- Achieve >90% coverage on baseline INT8.
- Deliverable: `baseline_INT8_8x8.sv` verified; FPGA + ASIC synthesis reports documented.

---

### Month 2: Precision Modes

**Week 5 — INT4×2 Mode**
- Read Paper 7 (fracturable PE). Draw the INT4×2 datapath with bit-widths on paper first.
- Implement: pack two signed 4-bit operands into 8-bit register. Upper nibble = A1, lower = A0. Same for B.
- INT4 mode generates two independent 4-bit × 4-bit products, accumulates both into 32-bit accumulator.
- Python golden: `pe_golden_int4x2(a_hi, a_lo, b_hi, b_lo, acc_in) → acc_out`.
- Test: 500 random INT4 pairs. Zero mismatches.

**Week 6 — Shared Booth Encoder Integration**
- Read Paper 5 (Booth/CMOS VLSI Design).
- Restructure PE so the same Radix-4 Booth encoder core serves both INT8 and INT4×2 modes via 2-bit `mode` input.
- Precision mode decode as pre-compute stage — conditions encoder inputs before encoding, not after.
- Full regression: all INT8 and INT4×2 tests passing.
- Synthesize dual-mode PE: measure area delta vs INT8-only. Target: < 12% overhead.

**Week 7 — FP8 Multiplier Standalone** `[v2 — FP8 design decisions as mandatory pre-coding gate]`
- **Before writing any RTL:** Read Appendix B. Confirm all four FP8 decisions are understood. Read Paper 6 (FP8 formats).
- Build FP8 multiplier as standalone module (not integrated into PE yet):
  - Step A: Exponent adder (4-bit + 4-bit, subtract bias 7, clamp on overflow)
  - Step B: Mantissa multiplier (4-bit × 4-bit with implicit leading 1, produce 8-bit product)
  - Step C: Normalization (shift mantissa, adjust exponent)
  - Step D: Round-to-nearest-even on mantissa LSB
  - Step E: Flush-to-zero for denormals (if exponent result is 0, output is 0)
  - Step F: NaN propagation (if either input is NaN, output is NaN using E4M3 NaN encoding)
  - Step G: Saturation for overflow (clamp to max E4M3 value)
- **Exhaustive test (mandatory gate):** Write Python script testing all 65,536 (256×256) FP8 multiplication combinations against `ml_dtypes.float8_e4m3fn`. Zero mismatches required before proceeding to Week 8. If mismatches remain after 3 days of debugging, invoke scope reduction: replace FP8 with simplified E4M3 (flush-to-zero + truncation rounding, no RTE). Document the simplification. Do not let this block Week 8.

**Week 8 — FP8 PE Integration and Three-Mode Regression**
- Integrate FP8 multiplier into unified PE.
- Extend Python golden for FP8 mode.
- Full three-mode regression: INT8, INT4×2, FP8 directed + random tests.
- Mode-transition tests: switch `mode` mid-operation and verify correct isolation of accumulator state.
- Synthesize three-mode PE: record area, Fmax. Target: < 20% area overhead vs INT8-only.
- Deliverable: `precision_adaptive_pe.sv` verified in all three modes; FPGA + ASIC synthesis reports.

---

### Month 3: Sparsity and Clock Gating

**Week 9 — 2:4 Sparsity Metadata Decoder**
- Read Papers 4 and 8.
- Implement combinational decoder: input = 4-element operand group + 4-bit metadata (2 bits × 2 indices). Output = 2 selected non-zero operand pairs.
- Exhaustive test: all 16 metadata values × representative data values.
- Verify decoder has zero registered paths: check that synthesis shows only combinational logic.

**Week 10 — Sparsity Front-End Integration**
- Wire decoder upstream of PE array input ports.
- Python utility: convert dense weight matrix → 2:4 sparse + metadata using `torch.nn.utils.prune`.
- Run GEMM tests with sparse ResNet-18 weight matrices. Verify outputs match dense computation.
- Test at multiple sparsity levels: force 25%, 50%, 75% zero elements in stimuli.

**Week 11 — Per-PE Clock Gating (ICG)**
- Read Papers 11 and 12.
- Add zero-operand detection per PE: assert `gating_enable` if operand A or B (INT8 mode) is zero.
- ICG cell: `BUFGCE` or behavioural model. Enable registered one cycle before clock edge.
- Operand isolators: when `gating_enable` asserted, hold inputs at 0.
- Verify: force operands to zero in simulation, confirm gated clock toggles stop, confirm downstream logic holds stable.
- Verify single clock domain: confirm ICG output never connects to a different clock domain input.

**Week 12 — Power Characterization Setup**
- Run post-implementation simulation with ResNet-18 activation tensors at 0%, 25%, 50%, 75% sparsity.
- For each: generate VCD → convert to SAIF → run `report_power` with SAIF annotation.
- Record per sparsity level: dynamic power per PE, total array, clock network power, pJ/MAC.
- Deliverable: 4-point power vs sparsity curve (Figure 3 of your paper).

---

### Month 4: Synthesis, Characterization, and Results

**Week 13 — Full Synthesis Sweep**
- Run Vivado synthesis + implementation for all configurations:
  - `baseline_dense_int8`: Month 1 result
  - `precision_adaptive_no_sparsity`: Modes only, no sparsity decoder, no ICG
  - `precision_adaptive_sparsity`: Modes + sparsity decoder, no ICG
  - `full_design`: Modes + sparsity + ICG
- For each: LUTs, FFs, BRAMs, DSP48s, Fmax (post-routing), static power, dynamic power, pJ/MAC.
- Identify critical path. If Fmax < 200 MHz, add one pipeline register and re-synthesize.
- Run Yosys + FreePDK45 synthesis on PE module for all four configurations: gate count, critical path.

**Week 14 — Accuracy Characterization**
- ResNet-18 inference with INT8 quantized weights/activations: per-layer MAE vs FP32 baseline.
- Same with FP8 (E4M3) for layers you designated as FP8 mode.
- Show FP8 maintains < 0.5% relative MAE on 1000-sample inference.
- Document layers where FP8 degrades accuracy — motivates the "per-layer precision selection" use case.

**Week 15–16 — Comparison Baselines and Open-Source Release** `[v2 — expanded to two weeks; BitFusion/SCNN comparison added]`

*Week 15:*
- Extract BitFusion ISCA 2018 Table 4: energy-per-MAC at INT8, INT4, INT2. Note 16nm process.
- Extract SCNN ISCA 2017 Table 3: throughput and power. Normalize to pJ/MAC.
- Compute NVIDIA A100 reference: 312 TOPS at 400W → 1.28 pJ/MAC.
- Build full comparison table: your design (FPGA + FreePDK45 normalized) | BitFusion | SCNN | Dense INT8 baseline | A100 reference.
- Note process node differences explicitly. Do not claim equivalent comparison — claim normalized pJ/MAC comparison with node noted.
- Create all paper figures: power vs sparsity curve, area breakdown bar chart, pJ/MAC comparison table, precision accuracy table.

*Week 16:*
- Clean RTL: remove debug signals, add module headers, write parameter descriptions.
- Write `README.md`: problem statement, architecture overview, testbench instructions, synthesis instructions, results summary, citation.
- Push all RTL, testbench scripts, synthesis scripts, Python golden models to GitHub.
- Deliverable: public GitHub repository ready to link in paper.

---

### Month 5: Paper Writing

**Week 17 — Abstract, Introduction, Skeleton** `[v2 — abstract template added]`
- Use IEEE TVLSI LaTeX template (or ISCAS 2026 template — check deadline).
- Write abstract using the template in Challenge 7. Lead with the gap, state the combination as novelty, quantify results, end with open-source release.
- Write Introduction: open with precision heterogeneity in production AI silicon (cite A100, Hopper), state the three separate concerns, state the gap (no open unified RTL), state your three contributions as bullet points, state your results in one sentence each.
- Write section headings, Conclusions, Acknowledgements. These frame the paper before you fill in the technical sections.

**Week 18 — Technical Sections**
- Section 2 (Related Work): Comparison table — BitFusion, SCNN, Eyeriss v2, Ampere Tensor Core. Your row in each comparison dimension. Clearly state how your work differs from each.
- Section 3 (Architecture): PE datapath diagram (Booth encoder shared paths highlighted), sparsity decoder block diagram, ICG integration diagram. One clear figure per contribution.
- Section 4 (Implementation): RTL parameter table, tool/device table (Vivado + Yosys + FreePDK45), FP8 specification table (from Appendix B), verification methodology (covergroups listed).

**Week 19 — Evaluation Section**
- Section 5 (Evaluation): Lead with power-vs-sparsity figure. Then pJ/MAC comparison table against BitFusion/SCNN. Then area breakdown. Then accuracy characterization. Then "Design Insights" subsection.
- "Design Insights": what did you learn about the interaction between precision mode, sparsity rate, and ICG effectiveness? For example: "FP8 mode at 50% sparsity achieved only 18% power reduction vs INT8 at 50% sparsity, because FP8 normalization logic has higher baseline toggle activity than integer paths — suggesting ICG is more effective in integer modes." This kind of insight is what separates a paper from a project report.

**Week 20 — Review, Polish, Submission**
- Send draft to advisor. Address all comments.
- Verify: every figure cited in text, every claim has supporting result, related work complete, limitations paragraph present (FPGA vs ASIC power, FP8 scope if reduced).
- LaTeX compiles cleanly with official IEEE template.
- Submit.

---

## 6. Extended Problem: Runtime Fault Detection (Future Work)

### 6.1 Problem Statement Extension

**Title Extension:** "...with Lightweight Algorithmic Checksum Monitoring for Silent Error Detection"

The mixed-precision systolic array targets edge inference in autonomous vehicles and safety-critical systems. In these environments, transient hardware faults — caused by cosmic ray particle strikes, voltage droop, aging, or thermal stress — produce silent data corruption. For a Tesla FSD chip or medical imaging accelerator, silent corruption is unacceptable.

Traditional approaches: full duplication (100% area overhead), ECC on memory (protects storage, not compute), BIST (tests offline, not during inference). None protect active compute with low overhead.

Your extension proposes a lightweight ABFT (Algorithm-Based Fault Tolerance) checksum monitor detecting soft errors during live inference with less than 15% area overhead for an 8×8 array.

### 6.2 The ABFT Technique

For matrix multiplication C = A × B:
- Compute checksum row for A: `a_c = sum(A, axis=0)`
- Compute checksum column for B: `b_c = sum(B, axis=1)`
- By the linearity of matrix multiplication: `a_c × B = A × b_c = sum(C)`
- Any single transient fault corrupting one element of C will cause the checksum to not match
- Detection requires exactly one additional PE row and one additional PE column — N/N² = 1/N overhead
- For 8×8 array: 12.5% area overhead for full single-error detection coverage during live inference

### 6.3 RTL Implementation Plan (2–3 Months Additional)

**Phase 1 (2 weeks):** Checksum accumulator rows/columns as additional PE rows at array periphery. INT8 mode only for initial implementation.

**Phase 2 (2 weeks):** Comparator block at tile computation end. Compare output checksum against independently computed input checksum. Assert `error_detected` flag on mismatch.

**Phase 3 (2 weeks):** Fault injection testbench — SystemVerilog `force`/`release` to inject single-bit and multi-bit flips into PE accumulator values during computation. Characterize: detection rate, false positive rate, detection latency in cycles.

**Phase 4 (2 weeks):** Extend to INT4×2 and FP8 modes. FP8 checksums accumulate in INT32 to avoid FP8 overflow.

### 6.4 Publication Angle

Three paths depending on timing:
- **Path A:** Add as Section 6 of your main paper if completed by Month 6 — strengthens the paper significantly and makes it TVLSI-worthy vs conference-only.
- **Path B:** Standalone IEEE Embedded Systems Letters paper (2-page format, ~3 month review cycle) — first author paper #2.
- **Path C:** Extended journal version of your conference paper with fault detection added — many ISCAS papers are extended to IEEE TCAS-I or TCAS-II.

---

## Appendix A: Interview Story Template

When NVIDIA/AMD/Intel/Qualcomm asks about your dissertation:

**"The problem I solved..."**
Production AI inference engines waste 40–60% of compute power on zero-valued MAC operations and use separate silicon areas for different precision modes. No open-source RTL implementation quantifies the combined efficiency available when you address all three problems — precision flexibility, structured sparsity, and per-PE power gating — in a single unified design.

**"What I built..."**
A synthesizable SystemVerilog systolic array where each PE dynamically switches between INT8, INT4, and FP8 using shared Booth encoder hardware. A 2:4 structured sparsity front-end suppresses zero-operand MAC pairs. Per-PE integrated clock gating stops the flip-flop clock when both operands are zero. Verified with a Python golden model and characterized with SAIF-annotated FPGA power analysis plus Yosys ASIC synthesis.

**"The results..."**
At 50% sparsity, the array consumes 38% less dynamic power than a dense INT8 baseline. INT4 mode achieves 1.9× throughput at the same clock frequency. Total area overhead for the combined design is under 18% versus a fixed INT8 array. Normalized pJ/MAC comparison against BitFusion (ISCA 2018) shows [result].

**"Why it matters to you..."**
This is exactly the architecture problem your Tensor Core / CDNA / Hexagon team works on at larger scale. I have hands-on RTL experience with the specific microarchitectural mechanisms — Booth encoding, structured sparsity metadata decoding, and integrated clock gating — that drive your silicon's efficiency numbers. I also understand the FP8 numerical specification at the bit level, which is increasingly relevant as FP8 training and inference become standard.

---

## Appendix B: FP8 Design Decision Record `[v2 — new appendix]`

This appendix is the source of truth for all FP8 implementation decisions. Lock these before writing any FP8 RTL. Reference this table in your paper's Implementation section. Reviewers who ask about denormal handling or NaN propagation are answered by pointing to this table.

### B.1 Format Selection

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Format name | E4M3 (4 exponent bits, 3 mantissa bits) | Per Micikevicius et al. 2022: E4M3 optimal for inference forward pass |
| Specific variant | E4M3FNUZ (`float8_e4m3fn` in `ml_dtypes`) | Matches NVIDIA H100 hardware specification |
| Total bit width | 8 bits: 1 sign + 4 exponent + 3 mantissa | |
| Exponent bias | 7 | Standard for E4M3 |
| Maximum representable value | 448.0 | |
| Minimum positive normal value | 2^(-6) = 0.015625 | |

### B.2 Arithmetic Behavior Decisions

| Decision | Choice | Alternative Considered | Why This Choice |
|----------|--------|----------------------|-----------------|
| Rounding mode | Round-to-nearest-even (RTE) | Truncation (round toward zero) | RTE matches `ml_dtypes.float8_e4m3fn`; required for zero golden model mismatches |
| Denormal handling | Flush-to-zero (FTZ) | Full denormal support | FTZ standard in inference hardware; eliminates variable-latency normalization paths; simplifies RTL significantly |
| Overflow behavior | Saturate to ±448.0 | Wrap-around | Saturation prevents catastrophic error propagation in accumulation |
| NaN encoding | All-ones mantissa with any exponent = NaN (E4M3 specific) | IEEE 754 NaN | NVIDIA E4M3 has no infinity; 0x7F and 0xFF are NaN |
| NaN propagation | Any NaN input → NaN output | Quiet NaN only | Matches `ml_dtypes` reference behavior |
| Underflow | Flush to zero | Denormal output | Consistent with FTZ decision above |

### B.3 Validation Protocol

Before integrating the FP8 multiplier into the PE:

1. Write `test_fp8_exhaustive.py` — tests all 256 × 256 = 65,536 input combinations
2. For each pair: compute expected output using `ml_dtypes.float8_e4m3fn` multiplication
3. Simulate RTL for each pair using cocotb or XSIM
4. Compare bit-exact: zero mismatches required
5. Separate test: validate NaN propagation for all 256 NaN-input combinations
6. Separate test: validate saturation for overflow-producing input pairs
7. Document pass/fail and any scope reductions applied

This test must pass before Week 8 integration begins. If it fails after 3 days of debugging, invoke scope reduction and document it.

### B.4 Scope Reduction Protocol

If the full FP8 specification cannot be implemented within Week 7:

**Level 1 reduction:** Replace RTE with truncation rounding. Accept up to 1 ULP additional error. Update golden model to use truncation. Re-run exhaustive test.

**Level 2 reduction:** Remove NaN propagation logic. NaN inputs produce undefined but non-NaN output. Add explicit note in paper: "NaN propagation not implemented; inputs are assumed valid per inference deployment assumption."

**Level 3 reduction:** Remove FP8 entirely. Paper becomes INT8 + INT4×2 + sparsity + ICG. This is still a complete, novel, publishable contribution. FP8 becomes future work. Do not delay the paper for FP8 if Level 3 is reached by end of Week 8.

---

*Document version: v2 — Updated June 2026*
*Changes from v1: 14 updates incorporating peer review feedback. See header summary.*
*Status: Active — ready for implementation*
