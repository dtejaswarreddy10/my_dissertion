# Mixed-Precision PE --- Dissertation Experimental Results

## 1. Scope

This report consolidates the available gate-level switching-power
estimates, synthesis-area results, and static-timing results for the
PE-level mixed-precision processing element (`pe_mixed`).

The PE supports INT8 and INT4x2 operation. The power experiments
summarized here use an INT8 workload and compare a no-gating
implementation against a clock-gated implementation at A-input
zero-sparsity targets of 0%, 25%, 50%, and 75%. The B input is not
sparsified.

The clock-gated implementation uses three integrated clock-gating (ICG)
cells for the internal sub-multiplier, MAC, and accumulator pipeline
clock domains. Forwarding outputs (`a_out`, `b_out`, and `valid_out`)
remain ungated.

**Frequency distinction:** SAIF power simulations use a 100 MHz clock.
Static timing analysis uses a 4.0 ns clock target (250 MHz). These are
separate experiments. These results are PE-level, not
systolic-array-level measurements.

## 2. Power methodology

Gate-level simulations were performed with QuestaSim and Cocotb. Matched
SAIF activity captures were generated for no-gating and clock-gated
implementations at each sparsity point. The custom estimator
`synth/asic/gl_power.py` combines SAIF toggle counts with pin
capacitances from the FreePDK45 Liberty file.

The estimator uses:

\[ P\_{dyn}=`\frac{1}{2}`{=tex}CV\^2`\frac{TC}{T}`{=tex} \]

where `C` is mapped Liberty pin capacitance, `V` is 1.1 V, `TC` is the
SAIF toggle count, and `T` is the observation window.

The estimate includes load-capacitance-based switching power. It
excludes wire capacitance, internal cell power, and short-circuit power.
Absolute mW values therefore do not represent complete silicon power;
percentage comparisons are the main result, assuming matched stimulus
and capture windows.

### 75% gated capture validation

The regenerated capture used for the final 75% result has:

-   SAIF: `activity_gated_sp75.saif`
-   Observation window: 40,015 ns
-   Nets matched: 1,448 / 1,448
-   Stimulus: 4,000 valid cycles
-   Measured zero-operand sparsity: 75.3% (75% target)

The earlier gated 75% SAIF had zero duration and no meaningful activity.
It was backed up as `activity_gated_sp75_old_invalid.saif`; the
associated old JSON was also backed up. The regenerated measurement is
used below.

## 3. Gate-level dynamic-power results

### 3.1 Total dynamic power

  -----------------------------------------------------------------------
       A-input zero    No-gating (mW)  Clock-gated (mW)         Reduction
           sparsity                                     
  ----------------- ----------------- ----------------- -----------------
                 0%          0.298988          0.331524           -10.88%

                25%          0.274504          0.276512            -0.73%

                50%          0.240045          0.209330            12.80%

                75%          0.191148          0.123908        **35.18%**
  -----------------------------------------------------------------------

Positive reduction means lower estimated power with clock gating.
Negative reduction means the gated implementation has higher estimated
power.

### 3.2 Clock-net dynamic power

  -----------------------------------------------------------------------
       A-input zero    No-gating (mW)  Clock-gated (mW)         Clock-net
           sparsity                                             reduction
  ----------------- ----------------- ----------------- -----------------
                 0%          0.106470          0.108083            -1.52%

                25%          0.106470          0.086018            19.21%

                50%          0.106470          0.064683            39.25%

                75%          0.106470          0.042089        **60.47%**
  -----------------------------------------------------------------------

### 3.3 Interpretation

-   **0% sparsity:** estimated total dynamic power increases by 10.88%.
    With no zero operands to suppress, clock-gating overhead is not
    offset by sufficient clock-activity savings.
-   **25% sparsity:** near break-even, with a 0.73% increase.
-   **50% sparsity:** estimated total dynamic power decreases by 12.80%.
-   **75% sparsity:** estimated total dynamic power decreases by 35.18%,
    while clock-net power decreases by 60.47%.

For this INT8 workload, clock gating becomes beneficial at the tested
50% and 75% sparsity points. These four points do not establish the
exact break-even sparsity between 25% and 50%; additional intermediate
points would be needed.

## 4. Synthesis area

  Metric                                     No-gating             Clock-gated
  --------------------------------------- ------------ -----------------------
  Mapped cell count                              1,312   1,427 + 3 ICG latches
  Corrected count including ICG latches          1,312                   1,430
  Yosys-reported area (µm²)                 4,438.6394              4,807.5092
  Added ICG latch area (µm²)                       ---                 15.4869
  Corrected total area (µm²)                4,438.6394              4,822.9961

The three ICG latches were not included in the original Yosys area
total. Their combined area was added using 5.1623 µm² per latch.

\[
`\mathrm{Area\ overhead}`{=tex}=`\frac{4822.9961-4438.6394}{4438.6394}`{=tex}`\times100`{=tex}=8.66%
\]

The clock-gated implementation therefore has approximately **8.66% area
overhead** over the no-gating implementation.

## 5. Static-timing analysis

OpenSTA analysis was performed with the FreePDK45 Liberty library and a
4.0 ns clock target (250 MHz).

### 5.1 Available path results

  ------------------------------------------------------------------------
  Implementation / path                        Slack Status
  --------------------- ---------------------------- ---------------------
  No-gating worst                         +0.8363 ns Met
  reported setup path                                

  No-gating worst                         +0.1216 ns Met
  reported hold path                                 

  Clock-gated                             +1.9314 ns Met
  clock-gating setup                                 
  path                                               

  Clock-gated `b_in[4]`                   +0.9374 ns Met
  to `u_icg_sub_latch`                               
  setup path                                         

  Clock-gated                             +0.0447 ns Met
  `u_icg_acc_latch` to                               
  `_2439_` hold path                                 

  Clock-gated `_2707_`                    +0.0833 ns Met
  to `u_icg_mac_latch`                               
  hold path                                          
  ------------------------------------------------------------------------

All listed paths have positive slack under their respective constraints.

### 5.2 Constraint mismatch

The no-gating worst setup result (+0.8363 ns) and the clock-gated input
setup result (+0.9374 ns) were generated with different `b_in` input
delays:

-   No-gating report: 2.1 ns.
-   Clock-gated report: 0.8 ns.

These setup-slack values are **not a like-for-like comparison**. To
compare setup margin directly, rerun both implementations with identical
clock, input-delay, output-delay, uncertainty, and other timing
constraints.

The clock-gated hold margins listed above are positive but relatively
small (+0.0447 ns and +0.0833 ns). They meet the reported constraints,
but should be monitored if constraints or implementation details change.

### 5.3 Gate-level simulation qualification

A separate 250 MHz gate-level simulation encountered a setup violation
at a mapped DFF input. This issue was not resolved as part of the 100
MHz power experiment. The 100 MHz SAIF results do not prove that the
gate-level design passes 250 MHz timing. The OpenSTA results and
gate-level simulation are separate evidence and should be reported with
this qualification.

## 6. Consolidated findings

  -----------------------------------------------------------------------
  Evaluation item                     Result
  ----------------------------------- -----------------------------------
  Power workload                      INT8

  Power simulation frequency          100 MHz

  Tested A-input zero sparsity        0%, 25%, 50%, 75%

  Maximum measured total              35.18% at 75% sparsity
  dynamic-power reduction             

  Maximum measured clock-net power    60.47% at 75% sparsity
  reduction                           

  Clock-gating area overhead          8.66%

  STA target period                   4.0 ns (250 MHz)

  Listed OpenSTA paths                Positive slack / met

  Setup-slack comparison              Not directly comparable; input
                                      delays differ

  250 MHz gate-level simulation       Setup violation observed;
                                      unresolved
  -----------------------------------------------------------------------

### Main conclusion

For the evaluated PE and INT8 workload, operand-activity-driven clock
gating trades approximately 8.66% additional mapped area for reduced
clock-net switching. Its total estimated dynamic-power impact depends on
input sparsity: estimated power increases at 0% and 25% sparsity, but
decreases at 50% and 75%. The largest measured benefit occurs at 75%
sparsity, with a 35.18% reduction in estimated total dynamic switching
power and a 60.47% reduction in clock-net power.

The results demonstrate a workload-dependent benefit, not a universal
power reduction. Timing evidence is positive for the listed OpenSTA
paths, but the constraint mismatch prevents a direct setup-slack
comparison, and the separate 250 MHz gate-level setup violation remains
an open validation item.

## 7. Reproducibility artifacts

-   RTL: `src/pe_mixed.sv`
-   Power estimator: `synth/asic/gl_power.py`
-   No-gating netlist: `synth/asic/reports/pe_mixed_nogate_abcfix.v`
-   Clock-gated netlist: `synth/asic/reports/pe_mixed_gated_abcfix.v`
-   SAIF captures:
    `synth/asic/reports/activity_{nogate,gated}_sp{00,25,50,75}.saif`
-   Power JSON reports:
    `synth/asic/reports/power_{nogate,gated}_sp{00,25,50,75}.json`
-   No-gating STA reports:
    `synth/asic/reports/sta_pe_mixed_nogate_abcfix_*`
-   Clock-gated STA reports:
    `synth/asic/reports/sta_pe_mixed_gated_abcfix_sta_*`

For reproducibility, retain the exact RTL, netlists, Liberty file,
simulation configuration, SAIF files, and estimator JSON reports used to
generate these values.
