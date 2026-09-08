import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------
# Files for the first candidate boundary
# --------------------------------------------------

first_boundary_files = {
    "32 KiB": "capacity_coarse_32768_random_raw.csv",
    "48 KiB": "dense_49152_49152_random_raw.csv",
    "64 KiB": "capacity_coarse_65536_random_raw.csv"
}


# --------------------------------------------------
# Files for the second candidate boundary
# --------------------------------------------------

second_boundary_files = {
    "256 KiB": "capacity_coarse_262144_random_raw.csv",
    "384 KiB": "dense_393216_393216_random_raw.csv",
    "512 KiB": "capacity_coarse_524288_random_raw.csv"
}


# --------------------------------------------------
# Files for the third candidate boundary
# --------------------------------------------------

third_boundary_files = {
    "18 MiB": "dense_18874368_18874368_random_raw.csv",
    "24 MiB": "dense_25165824_25165824_random_raw.csv",
    "32 MiB": "capacity_coarse_33554432_random_raw.csv"
}


# --------------------------------------------------
# Function for making one box plot
# --------------------------------------------------

def make_boxplot(file_dictionary, title, output_name):

    data = []
    labels = []

    for label, filename in file_dictionary.items():

        df = pd.read_csv(filename)

        latency = df["latency_tsc_ticks_per_access"]

        data.append(latency)
        labels.append(label)

        print(
            f"{label}: "
            f"{len(latency)} samples, "
            f"median = {latency.median():.4f}"
        )

    plt.figure(figsize=(8, 6))

    plt.boxplot(
        data,
        tick_labels=labels,
        showfliers=False
    )

    plt.xlabel("Working-Set Size")
    plt.ylabel("Latency (TSC ticks/access)")
    plt.title(title)

    plt.tight_layout()

    plt.savefig(
        output_name + ".png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.savefig(
        output_name + ".pdf",
        bbox_inches="tight"
    )

    plt.show()


# --------------------------------------------------
# First candidate boundary
# --------------------------------------------------

make_boxplot(
    first_boundary_files,
    "Sunbird: Latency Distribution Around First Candidate Boundary",
    "sunbird_boundary1_boxplot"
)


# --------------------------------------------------
# Second candidate boundary
# --------------------------------------------------

make_boxplot(
    second_boundary_files,
    "Sunbird: Latency Distribution Around Second Candidate Boundary",
    "sunbird_boundary2_boxplot"
)


# --------------------------------------------------
# Third candidate boundary
# --------------------------------------------------

make_boxplot(
    third_boundary_files,
    "Sunbird: Latency Distribution Around Third Candidate Boundary",
    "sunbird_boundary3_boxplot"
)