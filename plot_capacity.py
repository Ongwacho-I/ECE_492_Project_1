import glob
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


# ============================================================
# 1. READ THE COARSE CAPACITY SWEEP
# ============================================================

coarse = pd.read_csv("capacity_coarse_summary.csv")

# Sort coarse measurements by working-set size
coarse = coarse.sort_values("actual_bytes")


# ============================================================
# 2. READ ALL DENSE CAPACITY MEASUREMENTS
# ============================================================

dense_files = glob.glob("dense_*_summary.csv")

dense_data = []

for file in dense_files:
    df = pd.read_csv(file)
    dense_data.append(df)

dense = pd.concat(dense_data, ignore_index=True)

# Sort dense measurements by working-set size
dense = dense.sort_values("actual_bytes")


# ============================================================
# 3. COMBINE COARSE + DENSE DATA FOR THE CONNECTING LINE
# ============================================================

all_data = pd.concat(
    [coarse, dense],
    ignore_index=True
)

all_data = all_data.sort_values("actual_bytes")


# ============================================================
# 4. PRINT FILES AND DATA FOR VERIFICATION
# ============================================================

print("Coarse file:")
print("  capacity_coarse_summary.csv")

print("\nDense files:")
for file in sorted(dense_files):
    print(" ", file)

print("\nData being plotted:")

print(
    all_data[
        [
            "actual_bytes",
            "median_tsc_ticks_per_access",
            "mean_tsc_ticks_per_access"
        ]
    ].to_string(index=False)
)


# ============================================================
# 5. FORMAT BYTES AS KiB / MiB
# ============================================================

def format_bytes(x, pos):

    if x >= 1024**2:
        return f"{x / 1024**2:g} MiB"

    elif x >= 1024:
        return f"{x / 1024:g} KiB"

    else:
        return f"{x:g} B"


# ============================================================
# 6. CREATE FIGURE
# ============================================================

plt.figure(figsize=(11, 6.5))


# ============================================================
# 7. DRAW OVERALL CONNECTING LINE
# ============================================================

plt.plot(
    all_data["actual_bytes"],
    all_data["median_tsc_ticks_per_access"],
    linewidth=1.5
)


# ============================================================
# 8. PLOT COARSE MEASUREMENTS AS CIRCLES
# ============================================================

plt.scatter(
    coarse["actual_bytes"],
    coarse["median_tsc_ticks_per_access"],
    marker="o",
    s=45,
    label="Coarse sweep"
)


# ============================================================
# 9. PLOT DENSE MEASUREMENTS AS X's
# ============================================================

plt.scatter(
    dense["actual_bytes"],
    dense["median_tsc_ticks_per_access"],
    marker="x",
    s=55,
    linewidths=1.5,
    label="Dense measurements"
)


# ============================================================
# 10. USE LOG BASE-2 X-AXIS
# ============================================================

plt.xscale("log", base=2)

ax = plt.gca()

ax.xaxis.set_major_formatter(
    FuncFormatter(format_bytes)
)


# ============================================================
# 11. AXIS LABELS AND TITLE
# ============================================================

plt.xlabel(
    "Working-Set Size",
    fontsize=11
)

plt.ylabel(
    "Median Latency (TSC ticks/access)",
    fontsize=11
)

plt.title(
    "Sunbird: Pointer-Chase Latency vs. Working-Set Size",
    fontsize=13
)


# ============================================================
# 12. LEGEND
# ============================================================

plt.legend(
    loc="upper left",
    frameon=True
)


# ============================================================
# 13. CLEAN UP FIGURE SPACING
# ============================================================

plt.tight_layout()


# ============================================================
# 14. SAVE HIGH-QUALITY PNG
# ============================================================

plt.savefig(
    "sunbird_capacity_latency.png",
    dpi=300,
    bbox_inches="tight"
)


# ============================================================
# 15. SAVE VECTOR PDF FOR REPORT
# ============================================================

plt.savefig(
    "sunbird_capacity_latency.pdf",
    bbox_inches="tight"
)


# ============================================================
# 16. DISPLAY GRAPH
# ============================================================

plt.show()