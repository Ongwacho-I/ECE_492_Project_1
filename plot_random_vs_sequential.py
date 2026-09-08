import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------
# Files
# --------------------------------------------------

random_files = [
    "dense_49152_summary.csv",
    "dense_393216_summary.csv",
    "dense_25165824_summary.csv"
]

sequential_files = [
    "prefetch_48k_summary.csv",
    "prefetch_384k_summary.csv",
    "prefetch_24m_summary.csv"
]

sizes = ["48 KiB", "384 KiB", "24 MiB"]


# --------------------------------------------------
# Read median latency from each CSV
# --------------------------------------------------

random_medians = []
sequential_medians = []

for filename in random_files:
    df = pd.read_csv(filename)
    median = df["median_tsc_ticks_per_access"].iloc[0]
    random_medians.append(median)

for filename in sequential_files:
    df = pd.read_csv(filename)
    median = df["median_tsc_ticks_per_access"].iloc[0]
    sequential_medians.append(median)


# --------------------------------------------------
# Print values so we can verify them
# --------------------------------------------------

print("Randomized medians:")
for size, value in zip(sizes, random_medians):
    print(f"{size}: {value:.4f}")

print("\nSequential medians:")
for size, value in zip(sizes, sequential_medians):
    print(f"{size}: {value:.4f}")


# --------------------------------------------------
# Plot
# --------------------------------------------------

x = range(len(sizes))

plt.figure(figsize=(8, 6))

plt.plot(
    x,
    random_medians,
    marker="o",
    linewidth=2,
    markersize=7,
    label="Randomized traversal"
)

plt.plot(
    x,
    sequential_medians,
    marker="x",
    linewidth=2,
    markersize=8,
    label="Sequential traversal"
)

plt.xticks(x, sizes)

plt.xlabel("Working-Set Size")
plt.ylabel("Median Latency (TSC ticks/access)")

plt.title(
    "Sunbird: Randomized vs. Sequential Pointer-Chase Latency"
)

plt.legend()

plt.tight_layout()


# --------------------------------------------------
# Save
# --------------------------------------------------

plt.savefig(
    "sunbird_capacity_random_vs_sequential.png",
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    "sunbird_capacity_random_vs_sequential.pdf",
    bbox_inches="tight"
)

plt.show()