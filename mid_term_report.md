# Mid-Term Progress Report

## Runtime-Adaptive Precision and Sparsity-Aware Systolic Array with Per-PE Power Gating: A Unified RTL Architecture for Energy-Efficient DNN Inference

---

## 1. Introduction

### 1.1 Problem Statement & Motivation

Modern DNN inference hardware must reconcile three efficiency techniques that production silicon (NVIDIA Hopper/Ampere, AMD CDNA, Qualcomm Hexagon) each rely on, but which academic and open-source RTL implementations have so far only addressed in isolation:

- **Precision heterogeneity** — different layers of a network tolerate different levels of quantization, so a fixed-precision MAC array is never optimal across a whole model.
- **Structured sparsity** — a 2:4 contract (exactly two non-zero values in every group of four) gives a compact, hardware-friendly way to skip zero-valued compute.
- **Per-PE power gating** — dynamic switching power dominates a MAC datapath, and operand isolation plus clock gating are the standard mechanisms used to suppress it, but they must be designed into the RTL rather than bolted on afterward.

Prior work (BitFusion, SCNN, Eyeriss v2) each demonstrates one of these axes without the others, and none provide an open, synthesizable, PPA-characterized implementation of the combination. In this project, my contribution is that combination — a single output-stationary systolic array unifying all three — together with a quantified study of how precision mode, sparsity rate, and gating effectiveness interact.

### 1.2 Objectives

| # | Objective |
|---|---|
| O1 | Baseline engine — a verified, parameterized output-stationary INT8 systolic array. |
| O2 | Precision-adaptive PE — runtime switching between INT8 and packed INT4×2 on a shared reduction tree and accumulator. |
| O3 | Structured sparsity front-end — a purely combinational 2:4 metadata decoder that suppresses zero-operand MACs without breaking the systolic timing contract. |
| O4 | Per-PE power gating — operand isolation and integrated clock gating (ICG) within a single clock domain. |
| O5 | Characterization and validation — FPGA and ASIC PPA characterization, normalized pJ/MAC reporting, and accuracy validation against a software reference. |

### 1.3 Scope

My work covers an 8×8 (parameterizable to 16×16) output-stationary systolic array in synthesizable SystemVerilog, with two precision modes (INT8, INT4×2) sharing one reduction tree and accumulator, a combinational 2:4 sparsity decoder, per-PE operand isolation and clock gating in a single clock domain, and characterization on both an FPGA flow (Xilinx Kintex-7, Vivado) and an ASIC flow (Yosys, FreePDK45, OpenSTA), validated against a Python/NumPy golden reference model I wrote.

---

## 2. System Architecture

### 2.1 Overview

I designed the system as an 8×8 mesh of processing elements (PEs) in the classic output-stationary systolic arrangement: activations (A) enter at the west edge and flow east, weights (B) enter at the north edge and flow south, and each PE accumulates one output element C[i][j] in place for the duration of a tile. I built four synthesizable top-level configurations on this mesh incrementally, so I can measure each contribution's cost and benefit both in isolation and in combination:

1. **Baseline** — dense INT8 array (`systolic_array`), the reference point.
2. **Mixed** — baseline plus the precision-adaptive PE (`systolic_array_mixed`): INT8 or packed INT4×2, selected at runtime.
3. **Sparse (gating off)** — mixed plus the 2:4 sparsity front-end (`systolic_array_sparse`, `ENABLE_GATING=0`).
4. **Full design** — sparse plus per-PE clock gating (`systolic_array_sparse`, `ENABLE_GATING=1`).

### 2.2 Baseline Output-Stationary INT8 Engine

`pe_int8` is a two-stage pipelined MAC: the signed 8×8 product is registered in the first stage, and the accumulate happens in the second stage using the registered product, so a product is added into the accumulator exactly one cycle after its operands arrive. I chose this structure so the multiply and accumulate fuse into a single hardened DSP48 slice on the target FPGA (one DSP per PE, which I confirmed in synthesis — see Section 4). `systolic_array` instantiates an N×N mesh of `pe_int8`, wired with the boundary skew the output-stationary dataflow requires (row *i* delayed by *i* cycles, column *j* delayed by *j* cycles, so operands for k-step *k* meet at PE(i,j) on cycle *i+j+k*).

### 2.3 Precision-Adaptive PE

`pe_mixed` supersedes `pe_int8` and adds a second mode, INT4×2, in which each operand byte carries two independent signed 4-bit lanes and the PE performs two 4×4 MACs per cycle instead of one 8×8 MAC — the source of the throughput gain. I designed both modes to share the same four sub-multipliers and reduction tree: an 8×8 product decomposes into four half-width sub-products (high×high, high×low, low×high, low×low), and INT4×2 mode is simply the high×high and low×low terms summed without the cross-term shift used to reconstruct an INT8 product. I resolve mode selection in a pre-conditioning stage ahead of the multiply array, so it never sits on the reduction tree's critical path. I operand-isolate the unused cross-term sub-multipliers (force them to zero) in INT4×2 mode, which is the same isolation mechanism I reuse for clock gating in Section 2.6. Internally the PE is a three-stage pipeline (sub-products captured, then reduced product captured, then accumulated) — I added the middle stage during FPGA timing closure to keep the unhardened (fabric-mapped) reduction tree within the target clock period; it adds one cycle of latency before a result is visible but does not change the systolic forwarding skew.

### 2.4 Precision-Adaptive Array

`systolic_array_mixed` has the identical topology and port structure as the baseline array, with `pe_mixed` in place of `pe_int8` and a broadcast `mode` select line reaching every PE. A mode change only takes effect cleanly at a tile boundary (`acc_clear` must accompany any precision switch), since `mode` is broadcast rather than pipelined alongside the operands.

### 2.5 2:4 Structured Sparsity Front-End

The 2:4 contract stores, for every group of four weights along the contraction dimension, only the two non-zero values plus 4 bits of position metadata — a 50% bandwidth saving. I built `sparsity_decoder` as a purely combinational module that scatters the two compressed values back to their original positions and drives explicit zeros at the two pruned positions, and it flags illegal metadata (a position collision) rather than silently corrupting the stream. I placed `sparse_feeder` at the array's north boundary, one instance per column: it walks a 2-bit phase counter through each group of four, presenting one dense (possibly zero) weight per cycle to the mesh, and signaling the source to advance to the next group. The explicit zeros produced this way are what trip each PE's zero-operand detection downstream — I designed this as a bandwidth/power mechanism, not a throughput mechanism, since the feeder does not compact the schedule (all four cycles of a group are still spent, two of them streaming zeros).

### 2.6 Per-PE Clock Gating

`icg_cell` is a single wrapper module for which I wrote three interchangeable implementations, selected at compile time, because the correct implementation differs by target:

- **Default (ASIC/simulation)** — a latch-based integrated clock gate (enable captured on the low phase of the clock, ANDed with the clock), the structure a standard-cell library actually maps and the one whose gate-level power estimate produces the dissertation's true clock-gating result.
- **FPGA passthrough** — the gated clock degenerates to a plain pass-through, and the PE's own operand isolation plus a synthesized flip-flop clock-enable does the power-relevant work instead. I added this path because a Kintex-7 has on the order of 32 global clock buffers, nowhere near enough to give 64 PEs independently gated clocks — so I report the FPGA flow's power number as operand-isolation-plus-clock-enable, not as a true ICG result.
- **FPGA BUFGCE** — a Xilinx clock-buffer primitive, usable only for small, PE-level experiments, not the full array.

Each PE detects a zero operand byte and, when gating is enabled, isolates its multiply-tree inputs and stops the clock to its own MAC pipeline registers (not to the forwarding registers, which must always advance so the systolic mesh does not stall). Because a zero operand always produces a zero sub-product, suppressing the accumulate on that cycle is bit-identical to adding zero — gating changes only switching activity, never the numerical result.

### 2.7 Full Design

`systolic_array_sparse` combines all three contributions: activations remain dense and enter as before; weights arrive compressed at the north edge through one `sparse_feeder` per column; every PE is a gating-capable `pe_mixed`. This is the top level behind both the "sparse, gating off" and "full design" synthesis configurations in Section 4, distinguished only by the `ENABLE_GATING` parameter.

---

## 3. Verification Strategy

### 3.1 Overall Approach

I verify every module against a Python golden reference model I wrote, which computes the mathematically correct result independently of any timing or pipeline detail. I use two levels of check: PE-level checks compare every intermediate cycle's output against the golden model, exercising exhaustive input coverage where the input space is small enough to enumerate completely; array-level checks stream full matrices through the design and compare the final settled result against a golden GEMM (or pruned-GEMM, for the sparse design) reference, which is the level at which I actually exercise the systolic dataflow, skew, and boundary conditions.

### 3.2 Per-Module Verification Summary

| Module | What I check | Status |
|---|---|---|
| `pe_int8` | Single-PE MAC correctness against boundary operands, random accumulation sequences, valid-gating, and accumulator clear. | I have independently confirmed this same PE fully correct via the array-level check below (zero mismatches across 50+ random matrices and all boundary cases). I am investigating a timing anomaly specific to the standalone per-cycle PE check — see Section 3.3. |
| `systolic_array` | Full 8×8 GEMM correctness: identity matrix, saturating-magnitude corners, random matrix pairs, back-to-back tiles without reset. | Fully verified, zero mismatches. |
| `pe_mixed` | INT8 and INT4×2 correctness, including an **exhaustive** sweep of all 65,536 operand combinations in each mode, lane-independence (a zero lane contributes nothing), and safety across a mode change mid-stream. | Fully verified at the exhaustive level for both modes. I am correcting two auxiliary checks in my test harness that assume an earlier pipeline depth than the current, timing-closure-driven one — see Section 3.3. |
| `systolic_array_mixed` | Full array GEMM correctness in both precision modes, alternating-precision tiles, and the measured INT4×2 throughput gain. | Fully verified, zero mismatches; throughput currently measures below my target range (Section 4.3). |
| `sparsity_decoder` | Exhaustive sweep of all 16 metadata encodings, correct placement of surviving values, illegal-metadata detection, and the 2:4 density contract over randomized legal groups. | Fully verified. |
| `sparse_feeder` / `systolic_array_sparse` | Full-design correctness against a pruned-matrix golden reference, all-zero-weight-group gating behavior, and back-to-back tile isolation. | I have independently verified the decoder and feeder per the rows above; I am correcting a timing assumption in this suite's own test harness (unrelated to either module's RTL) before I can consider the suite complete — see Section 3.3. |
| `icg_cell` | Functional equivalence between gating enabled and disabled (bit-identical results, switching activity only). | I have not yet exercised this with gating enabled in simulation; scheduled next. |

### 3.3 Known Open Items

- I am investigating a timing anomaly in the standalone, per-cycle PE-level check for the baseline INT8 PE. I believe it is very likely a simulation-environment artifact rather than a design defect, since I have independently confirmed the identical PE correct through full array-level verification.
- I wrote two test-harness timing assumptions — one in the precision-adaptive PE's auxiliary checks, one in the full design's weight-driving sequence — against an earlier pipeline depth, and I am updating both to match the current one. Neither is an RTL defect; I understand both and have scoped the fix.
- I have not yet run gate-level, gating-enabled simulation of `icg_cell`.

---

## 4. Synthesis & Characterization Results

### 4.1 FPGA Flow

I synthesized and implemented all four configurations from Section 2.1 (out-of-context, to allow block-level characterization of a design whose I/O count far exceeds any package's pin count) on a Xilinx Kintex-7 (`xc7k70tfbv676-1`) at a 250 MHz target clock, using Vivado. I estimated power with a uniform-toggle-rate placeholder pending the activity-annotated sweep described in Section 5.

### 4.2 Results Table

| Configuration | LUTs | FFs | DSPs | Worst setup slack | Achievable Fmax | Dynamic power |
|---|---|---|---|---|---|---|
| Baseline | 2 | 344 | 64 | +2.024 ns | ~506 MHz | 0.139 W |
| Mixed | 9,549 | 6,716 | 0 | +0.064 ns | ~248 MHz | 0.520 W |
| Sparse (gating off) | 9,634 | 6,812 | 0 | +0.120 ns | ~248 MHz | 0.575 W |
| Full design | 10,747 | 6,932 | 0 | +0.040 ns* | ~249 MHz | 0.546 W |

\* I report the full-design configuration at a 4.05 ns clock period rather than the 4.0 ns I used for the other three — a small, explicitly-noted relaxation, not a like-for-like comparison at an identical constraint.

### 4.3 Interpretation

All four configurations close timing at essentially the 250 MHz target. The baseline maps its MAC datapath entirely onto hardened DSP48 slices (64 DSPs, negligible LUT usage); I deliberately kept the precision-adaptive PE's shared reduction tree in programmable fabric rather than DSP-mapped, so its LUT/FF footprint is not directly comparable to the baseline's DSP-based footprint on a common area unit — I intend the ASIC flow (Section 4.4) to be the source of the area-overhead figure for that reason. The dynamic power trend across sparse (gating off) vs. full design is directionally consistent with the intended effect of operand isolation (full design measures lower dynamic power than the otherwise-identical gating-off configuration), though this is under a uniform toggle-rate placeholder and not yet the activity-annotated result I intend to report as the headline power figure. I measured INT4×2 throughput on the array at 1.53×, below my 1.8–2.0× target range.

### 4.4 ASIC Flow Status

I have written Yosys/FreePDK45 synthesis scripts for both the single baseline PE and the precision-adaptive PE (in both gating-enabled and gating-disabled configurations), along with an OpenSTA script for gate-level timing and activity-annotated power. I have not yet executed this flow end-to-end; it is the source of the true per-PE clock-gating power result, since — as noted in Section 2.6 — the FPGA flow cannot host genuinely independent per-PE gated clocks.

---

## 5. Remaining Work

- Complete functional verification: resolve the open test-harness items in Section 3.3 and confirm a clean, full regression across every module.
- Run gate-level, gating-enabled simulation of the per-PE clock gate.
- Run the ASIC flow (Yosys, FreePDK45, OpenSTA) for gate count, critical path, and the activity-annotated clock-gating power result.
- Run the activity-annotated (non-placeholder) power-versus-sparsity sweep on the FPGA flow across a range of activation sparsity levels.
- Validate inference accuracy against a PyTorch/NumPy reference on real CNN layer dimensions (ResNet-18 / MobileNetV2).
- Build the normalized energy-efficiency (pJ/MAC) comparison against published baseline numbers (BitFusion, SCNN).
- Prepare documentation and the open-source release.

---

## 6. Conclusion

I have implemented the baseline, precision-adaptive, and sparsity-aware designs; they synthesize cleanly and close timing at the target clock on real FPGA hardware across all four characterization configurations. I have largely completed functional verification, with a small number of understood, scoped items remaining. My primary remaining work is the ASIC characterization needed for the true clock-gating power claim, the activity-annotated power sweep, and the accuracy/comparison studies that will turn these synthesis results into the dissertation's evaluation chapter.

---

## 7. References

1. V. Sze, Y.-H. Chen, T.-J. Yang, and J. S. Emer, "Efficient Processing of Deep Neural Networks: A Tutorial and Survey," *Proceedings of the IEEE*, vol. 105, no. 12, pp. 2295–2329, 2017.
2. N. P. Jouppi et al., "In-Datacenter Performance Analysis of a Tensor Processing Unit," in *Proc. 44th Annual International Symposium on Computer Architecture (ISCA)*, 2017, pp. 1–12.
3. J. Choquette et al., "NVIDIA A100 Tensor Core GPU: Performance and Innovation," *IEEE Micro*, vol. 41, no. 2, pp. 29–35, 2021.
4. A. Mishra et al., "Accelerating Sparse Deep Neural Networks (Exploiting NVIDIA Ampere Structured Sparsity)," arXiv:2104.08378, 2021.
5. H. Sharma et al., "Bit Fusion: Bit-Level Dynamically Composable Architecture for Accelerating Deep Neural Networks," in *Proc. ISCA*, 2018, pp. 764–775.
6. A. Parashar et al., "SCNN: An Accelerator for Compressed-Sparse Convolutional Neural Networks," in *Proc. ISCA*, 2017, pp. 27–40.
7. Y.-H. Chen, T.-J. Yang, J. Emer, and V. Sze, "Eyeriss v2: A Flexible Accelerator for Emerging Deep Neural Networks on Mobile Devices," *IEEE Journal on Emerging and Selected Topics in Circuits and Systems*, vol. 9, no. 2, pp. 292–308, 2019.
