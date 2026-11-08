from pathlib import Path
import re

src = Path("synth/asic/reports/pe_mixed_gated_abcfix.v")
dst = Path("synth/asic/reports/pe_mixed_gated_abcfix_sta.v")

text = src.read_text()

latches = ["acc", "mac", "sub"]

for name in latches:
    prefix = rf"\g_icg.u_icg_{name}"

    # Remove behavioral latch declaration.
    declaration = f"  reg {prefix}.en_latched ;"
    assert text.count(declaration) == 1, (
        f"Expected exactly one declaration for {name}"
    )
    text = text.replace(declaration, "", 1)

    # Match the behavioral latch process.
    process = (
        r"  always @\*\s*"
        + r"if \(!clk\)\s*"
        + re.escape(prefix)
        + r"\.en_latched\s*=\s*"
        + re.escape(prefix)
        + r"\.enable\s*;"
    )

    # Use a callback so backslashes in Verilog are not interpreted
    # as Python regular-expression replacement escapes.
    replacement = (
        f"  LATCH u_icg_{name}_latch (\n"
        f"    .D({prefix}.enable ),\n"
        f"    .CLK(clk_n),\n"
        f"    .Q({prefix}.en_latched )\n"
        f"  );"
    )

    text, count = re.subn(
        process,
        lambda match: replacement,
        text
    )

    assert count == 1, (
        f"Expected exactly one behavioral latch process for {name}, "
        f"got {count}"
    )

# Add one shared clock inverter for all three latches.
marker = "module pe_mixed"
assert marker in text

# Insert after the module declaration.
module_end = text.index("\n", text.index(marker))
text = (
    text[:module_end + 1]
    + "\n  wire clk_n;\n"
    + "  INVX1 u_icg_clk_inv (.A(clk), .Y(clk_n));\n"
    + text[module_end + 1:]
)

dst.write_text(text)

print(f"Created: {dst}")
print("Replaced three behavioral latches with library LATCH instances.")
print("Used inverted clock to preserve transparent-low RTL behavior.")
