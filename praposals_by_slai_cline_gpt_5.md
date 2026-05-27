# Dissertation Problem Statements in VLSI/RTL/Digital Design (3–4 Months Feasible)

Author: [Your Name]
Program: M.Tech (WILP) – VLSI Design and Microelectronics
Profile context: Full-time RTL Design Engineer (Digital design, Verilog/SystemVerilog, synthesis, basic verification)

Objective:
- Pick an industry-relevant, design-heavy RTL topic aligned with modern trends (AI hardware, low power, memory systems, interconnects, security).
- Scope must be achievable in 3–4 months alongside a full-time job.
- Aim for publishable novelty (IEEE/ACM, DVCon/DesignCon class venues).

Table of Contents
- 1) Precision-Adaptive INT8/FP8 MAC Array with Dynamic Sparsity Skipping
- 2) Banked Scratchpad + Streaming DMA Prefetcher for ML/GEMM Datapaths
- 3) Low-Latency Adaptive NoC Router with QoS and Link Power Gating
- 4) AXI4-Stream ⇄ AXI4-MM Bridge with Assertion Mining and Formal Coverage
- 5) Side-Channel-Resilient AES-128 with RTL-Level Power Equalization
- 6) High-Throughput Iterative Divider with Operand Isolation and Gating
- 7) Lightweight Chiplet (Die-to-Die) Link RTL Prototype with CRC/Replay

Notes on Tools and Flow
- RTL: SystemVerilog/Verilog
- Simulation: Vivado XSIM/Questa/Verilator (as available)
- Synthesis/Timing: Vivado, Yosys+OpenSTA (or equivalent accessible)
- Power: SAIF/VCD-based estimation (Vivado Power Analyzer, or vendor-equivalent)
- Verification: SVA assertions, basic UVM-lite or directed/self-checking TBs, optional formal (SymbiYosys or vendor tools if available)

--------------------------------------------------------------------------------

1) Precision-Adaptive INT8/FP8 MAC Array with Dynamic Sparsity Skipping

Problem Statement
Design and evaluate a parameterizable MAC array that supports mixed-precision INT8/FP8 compute with:
- Per-channel precision control (INT8/FP8) and dynamic operand gating.
- Runtime sparsity skipping (zero-skipping) at operand granularity.
- Fine-grained clock/operand isolation to reduce unnecessary switching.
Demonstrate end-to-end throughput/W/mm² and energy benefits on GEMM/Conv microbenchmarks.

Industry Importance
- AI accelerators (NVIDIA, AMD, Qualcomm) increasingly exploit mixed precision and sparsity.
- RTL-proven micro-architectural knobs (precision gating, zero-skipping) translate directly to PPA wins in production silicon.

Technical Approach
- MAC Tile: Parameterizable array (e.g., 8x8 to 32x32), with selectable INT8/FP8 datapaths.
- Precision Control: Per-channel mode registers drive mantissa/exponent/logical truncation or bypass logic paths.
- Sparsity Skipping: Detect zero (or near-zero for FP8) operands at input FIFO and gate multiplier/adder enable.
- Accumulation Strategy: Configurable K dimension depth; saturation/rounding logic; optional stochastic rounding.
- Memory Interface: Simple scratchpad ports (AXI-Stream/AXI-MM) for tiles, weight/activation buffers.
- Verification: Directed tests (GEMM, Conv1x1/3x3), random tests, scoreboarding vs. Python/NumPy golden.
- Power/Perf: SAIF-based toggling comparison (w/ and w/o precision gating and sparsity skipping), post-synth frequency.

Expected Outcomes
- 15–35% dynamic power reduction at similar throughput under realistic sparsity (e.g., 30–60%).
- 1.1–1.3x throughput at iso-power by reallocating saved power to higher frequency/parallelism.
- RTL micro-architecture and knobs packaged as reusable IP.

Publishability Angle
- Novelty: Combined per-channel precision control + runtime sparsity gating at MAC-array granularity with measured PPA.
- Paper: Introduce architecture, gating schemes, evaluation on standard GEMM kernels; compare to fixed-precision baselines.
- Target: IEEE Access, IEEE ICECS/ISCAS, VDAT; workshop/demo at VLSI/ML accelerators venues; DVCon for methodology.

Feasible 3–4 Month Plan
- Weeks 1–2: Baseline MAC tile, scratchpad adapters, testbench w/ golden model.
- Weeks 3–4: Add precision control, FP8 datapath, functional validation.
- Weeks 5–6: Zero-skipping and operand isolation; synth/timing for baseline vs. optimized.
- Weeks 7–8: Campaign of runs: throughput/frequency/power across precisions/sparsity; draft paper.

Risks/Mitigations
- FP8 numeric stability: add saturation/rounding; keep reference model for correctness.
- Timing closure: pipeline multiplier/adder; tune array dimensions.

--------------------------------------------------------------------------------

2) Banked Scratchpad + Streaming DMA Prefetcher for ML/GEMM Datapaths

Problem Statement
Design a banked, multiported scratchpad with address interleaving and a lightweight streaming DMA prefetcher to feed a MAC array. Focus on hiding memory latency, maximizing bank-level parallelism, and minimizing stalls under realistic GEMM/Conv access patterns.

Industry Importance
- Memory bandwidth is the limiter in AI SoCs. Efficient scratchpads and prefetchers directly improve TOPS/W.
- These techniques are core in NPUs, GPUs, and edge AI accelerators.

Technical Approach
- Scratchpad: N banks (e.g., 8–16) with address interleaving; single/dual-port banks; conflict detector and arbiter.
- DMA Prefetcher: Circular buffers, strided/tiled access support, lookahead queues, programmable depth, back-pressure handling.
- QoS/Arbitration: Age-based or round-robin between compute and DMA to prevent starvation.
- Interfaces: AXI-MM master for DMA; AXI-Stream for compute ports.
- Verification: Directed tiling sequences; randomized strides; scoreboard against a software dataflow model.
- Metrics: Bank conflicts, sustained GB/s, burst efficiency, stall cycles.

Expected Outcomes
- 1.3–1.7x sustained bandwidth to compute vs. naive scratchpad.
- >80–90% bank utilization on tiled GEMM; 20–35% reduction in compute stalls.
- Configurable RTL scratchpad+DMA block suitable for ML kernels.

Publishability Angle
- Novelty: Combining bank-interleaved scratchpad and programmable prefetcher with quantitative conflict/stall modeling at RTL.
- Paper: Model, hardware design, perf analysis across tile sizes/strides; methodology portable to varied ML kernels.
- Target: IEEE Access, Embedded Systems Letters, VDAT; DVCon for verification methodology.

Feasible 3–4 Month Plan
- Weeks 1–2: Banked scratchpad RTL + simple arbiter; TB and perf counters.
- Weeks 3–4: DMA prefetcher with lookahead; golden model for ordering/correctness.
- Weeks 5–6: Integration and stress tests; synth/timing and bandwidth/stall metrics.
- Weeks 7–8: Sensitivity study (banks, tile sizes); write-up.

Risks/Mitigations
- Arbitration complexity: keep policies simple; add param hooks for future work.
- Verification scope: prioritize realistic ML access traces.

--------------------------------------------------------------------------------

3) Low-Latency Adaptive NoC Router with QoS and Link Power Gating

Problem Statement
Implement a parameterized 2D mesh NoC router with:
- Virtual channels (2–4), credit-based flow control.
- Age/QoS-aware arbitration for ML/latency-sensitive traffic.
- Idle link power gating (clock gating) with rapid wakeup.
Quantify latency/throughput/energy on synthetic and ML-inspired workloads.

Industry Importance
- Interconnects dominate SoC scalability. QoS and energy-aware links are widely used in GPU/CPU/AI designs.

Technical Approach
- Topology: 2D mesh router RTL with five ports (N/E/S/W/Local), flit width configurable.
- Flow Control: Credits, elastic buffers, VC allocator, switch allocator.
- QoS: Simple priority classes; age-based boost to avoid starvation.
- Power Gating: Idle detection counters; clock-gate crossbar and link FIFOs; wakeup handshake.
- Verification: Random traffic, hotspot, transpose; trace-based traffic from ML layers (approximated).
- Metrics: Zero-load and saturation latency, throughput, link toggle counts, wakeup penalties.

Expected Outcomes
- 10–20% latency reduction under mixed QoS vs. round-robin.
- 10–25% link dynamic power reduction with idle gating at modest load.
- Clean, synthesizable NoC router RTL.

Publishability Angle
- Novelty: Simple but effective QoS+gating scheme analyzed at RTL with traffic mixes representative of ML workloads.
- Paper: Router architecture, arbiters, gating behavior, and PPA on synthetic + ML-like traces.
- Target: NoC/VLSI symposia, IEEE Access; DVCon for testbench methodology.

Feasible 3–4 Month Plan
- Weeks 1–2: Baseline router, buffers, allocators, TB.
- Weeks 3–4: QoS/age arbitration, traffic patterns.
- Weeks 5–6: Power gating + wakeup; measurement infra.
- Weeks 7–8: Param sweep; paper draft.

Risks/Mitigations
- Deadlock/livelock: use escape VC or dimension-ordered routing; formal checks on routing constraints.

--------------------------------------------------------------------------------

4) AXI4-Stream ⇄ AXI4-MM Bridge with Assertion Mining and Formal Coverage

Problem Statement
Build an AXI4-Stream to AXI4-MM bridge with:
- Burst assembly/disassembly, width conversion, backpressure handling.
- Protocol compliance verified via SVA libraries and property checking.
- Assertion mining (derive assertions from TB traces) to boost coverage.
Assess functional robustness and coverage closure benefits.

Industry Importance
- AXI bridges are ubiquitous in SoCs; verification complexity and coverage closure are major schedule drivers.
- Strong methodology adds value for product teams.

Technical Approach
- Bridge RTL: Parameterized data widths; burst generator; response handling; skid buffers.
- SVA: AXI property set; check channel ordering, handshake, bursts; bind assertions for reuse.
- Formal: Bounded proofs for protocol invariants (ready/valid safety, no data loss).
- Assertion Mining: Generate candidate properties from clean TB traces; feed back into SVA for regressions.
- Metrics: Functional coverage (covergroups or SVA cover), bug find rate, proof depth, sim time overhead.

Expected Outcomes
- Robust, reusable AXI bridge with measured coverage and formal proofs of key invariants.
- Demonstrated uplift in coverage using mined assertions.

Publishability Angle
- Novelty: Practical assertion-mining loop applied to AXI bridging, quantified coverage/proof outcomes.
- Paper: Methodology + results; contribution is verification technique more than new hardware function.
- Target: DVCon India/US, IEEE Access (methodology), SNUG papers.

Feasible 3–4 Month Plan
- Weeks 1–2: AXI bridge MVP and TB.
- Weeks 3–4: SVA property integration; regressions.
- Weeks 5–6: Formal checks; assertion mining pipeline; re-run coverage.
- Weeks 7–8: Results and write-up.

Risks/Mitigations
- Tool availability for formal: if limited, emphasize simulation assertions and constrained random.

--------------------------------------------------------------------------------

5) Side-Channel-Resilient AES-128 with RTL-Level Power Equalization

Problem Statement
Implement an AES-128 core with countermeasures against power analysis:
- Masked or composite-field S-Box with register balancing.
- Constant-time/constant-control-flow rounds.
- Toggle equalization using dual-rail-style balancing at RTL (approximate).
Evaluate leakage reduction using toggle-count proxies and TVLA-like statistical tests on switching activity.

Industry Importance
- Security is a first-order requirement; mitigations at RTL translate to real products in secure elements and SoCs.
- Demonstrable RTL countermeasures are valuable to product teams.

Technical Approach
- AES-128 Iterative Core: Round-based pipeline for area efficiency.
- Countermeasures: Random mask injection for S-Box or bitslicing; balanced toggling nets; dummy operations insertion.
- TB: NIST vectors for correctness; Monte-Carlo runs with random keys/plaintexts.
- Leakage Estimation: Capture VCD/SAIF; perform basic TVLA (t-test) on toggles across keys/data groups.
- Synthesis: Compare overheads vs. unprotected core.

Expected Outcomes
- Significant reduction in detectable correlation (proxy leakage) with acceptable area/frequency penalty.
- Reference RTL with parameterized countermeasures.

Publishability Angle
- Novelty: Practical RTL balancing + mask strategy evaluated with TVLA-like analysis using switching proxies.
- Paper: Architecture, countermeasure choice, leakage metrics, area/freq trade-offs.
- Target: IEEE Transactions/Letters (applied), CHES workshops (method-style), VDAT.

Feasible 3–4 Month Plan
- Weeks 1–2: Baseline AES; correctness.
- Weeks 3–4: Integrate masked S-Box/balancing; functional + performance checks.
- Weeks 5–6: Leakage proxy analysis; parameter sweeps.
- Weeks 7–8: Write-up.

Risks/Mitigations
- Perfect balancing is hard without custom cells; scope as RTL-level approximations and discuss limitations.

--------------------------------------------------------------------------------

6) High-Throughput Iterative Divider with Operand Isolation and Gating

Problem Statement
Design a parameterizable iterative divider (e.g., Radix-4 SRT or Newton-Raphson for FP/Fixed) with:
- Early-out and operand-range detection to reduce iterations.
- Operand isolation and clock gating to cut dynamic power when quotient bits stabilize.
- Interface compatible with simple in-order RISC-V pipeline stage.
Quantify throughput, cycles/op, and energy vs. a baseline.

Industry Importance
- Arithmetic units are critical IP; low-power dividers are important for CPUs/MCUs and DSPs.
- Demonstrates micro-architectural skills beyond multipliers/MACs.

Technical Approach
- Algorithm: Fixed-point first (e.g., Radix-4 SRT) with configurable width; optional FP wrapper.
- Micro-architecture: Pipelined compare-subtract, on-the-fly quotient selection; early-out detection for small operands.
- Gating: Disable datapath slices when MSBs/MIN bits are resolved.
- TB: Random operands, corner cases, scoreboard against Python/MPFR.
- Metrics: Average cycles/op, Fmax, SAIF-based dynamic power.

Expected Outcomes
- 20–40% energy/op reduction on real operand distributions.
- Comparable or slightly improved throughput at iso-frequency.

Publishability Angle
- Novelty: Practical gating/early-exit policies quantified for divider workloads; reusable RTL block.
- Paper: Architecture, operand profiling, energy/perf gains; comparison to textbook iterative divider.
- Target: IEEE Access, Embedded Systems Letters, DVCon (arithmetic IP).

Feasible 3–4 Month Plan
- Weeks 1–2: Baseline divider; TB and golden model.
- Weeks 3–4: Early-out + gating; validation and perf.
- Weeks 5–6: Synthesis/power estimation; operand distribution study.
- Weeks 7–8: Write-up.

Risks/Mitigations
- Verification of edge cases (divide-by-zero, min/max): add assertions and targeted tests.

--------------------------------------------------------------------------------

7) Lightweight Chiplet (Die-to-Die) Link RTL Prototype with CRC/Replay

Problem Statement
Prototype a simplified chiplet interconnect inspired by UCIe-like concepts:
- Credit-based flow control, lane striping, elastic buffers.
- CRC per flit and replay/NACK mechanism on error.
- Optional training/deskew model (abstracted at RTL).
Measure latency, throughput, and overheads under error injection.

Industry Importance
- Chiplets are mainstream; understanding flow control, integrity, and PHY-agnostic link layers is valuable to product teams.

Technical Approach
- Link Layer: Parameterized flit width (e.g., 128–256b), credits, Tx/Rx elastic FIFOs.
- Reliability: CRC-16/32 on flits, NACK-based selective replay buffer.
- Training: Simple FSM model for ready state; link-up sequencing.
- TB: Back-to-back traffic, bursty, and error injection (bit flips).
- Metrics: Goodput vs. BER, latency distribution, retransmission rate, buffer sizing.

Expected Outcomes
- Quantified reliability-overhead trade-offs and sizing guidelines.
- Clean, synthesizable link-layer IP usable for architectural studies.

Publishability Angle
- Novelty: Minimal yet practical replay/CRC scheme with quantified error-injection study at RTL.
- Paper: Architecture, protocol, performance under BER; synthesis results.
- Target: DesignCon, IEEE Access, ETS.

Feasible 3–4 Month Plan
- Weeks 1–2: Basic Tx/Rx + credits; TB.
- Weeks 3–4: CRC and replay buffer; error injection infra.
- Weeks 5–6: Param sweeps (flit size, credits, buffers) and metrics.
- Weeks 7–8: Write-up.

Risks/Mitigations
- Scope creep into full PHY: keep PHY abstract; focus on link layer and verification.

--------------------------------------------------------------------------------

Ranking (Higher is better; 1–5 scale)

1) Precision-Adaptive INT8/FP8 MAC Array
- Industry relevance: 5
- Ease in 3–4 months: 4
- Publishability: 5
Overall: 14/15

2) Banked Scratchpad + DMA Prefetcher
- Industry relevance: 5
- Ease in 3–4 months: 4
- Publishability: 4
Overall: 13/15

3) AXI Stream⇄MM Bridge + Formal Coverage
- Industry relevance: 4
- Ease in 3–4 months: 5
- Publishability: 4
Overall: 13/15

4) Adaptive NoC Router with QoS & Link Gating
- Industry relevance: 4
- Ease in 3–4 months: 3
- Publishability: 4
Overall: 11/15

5) Side-Channel-Resilient AES-128
- Industry relevance: 4
- Ease in 3–4 months: 3
- Publishability: 4
Overall: 11/15

6) Iterative Divider with Gating
- Industry relevance: 3
- Ease in 3–4 months: 4
- Publishability: 3
Overall: 10/15

7) Chiplet Link with CRC/Replay
- Industry relevance: 4
- Ease in 3–4 months: 3
- Publishability: 3
Overall: 10/15

Recommended TOP 2 Topics

A) Precision-Adaptive INT8/FP8 MAC Array with Dynamic Sparsity Skipping
- Why: Directly aligned with AI accelerator trends at NVIDIA/AMD/Qualcomm; strong design-heavy RTL focus; clear, quantifiable PPA gains; high novelty (precision + sparsity + gating).
- Fit: Leverages your RTL and synthesis skills; publishable with convincing PPA plots on GEMM/Conv.

B) Banked Scratchpad + Streaming DMA Prefetcher for ML/GEMM
- Why: Memory subsystems are a bottleneck in modern SoCs; design is highly reusable and performance-critical; realistic scope with strong evaluation story (bandwidth, stalls, utilization).
- Fit: Demonstrates end-to-end micro-architecture thinking and verification; valued by product teams.

Appendix: Suggested Evaluation Checklist (for any chosen topic)
- Correctness: Self-checking TB with golden model; assertions for key invariants.
- Performance: Throughput/latency counters, backpressure stats, stalls.
- Power: SAIF/VCD capture over representative workloads; compare variants.
- Area/Timing: Post-synth Fmax and resource usage across parameters.
- Sensitivity: Parameter sweeps (e.g., tile size, banks, flit width) to show robustness.
- Packaging: Clean README, parameter docs, scripts for reproducible runs.
