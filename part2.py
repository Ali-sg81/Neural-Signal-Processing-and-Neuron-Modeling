import numpy as np
import matplotlib.pyplot as plt
import os

from scipy.signal import butter
from scipy.signal import sosfiltfilt
from scipy.signal import welch


# ============================================================
# 1. SETTINGS
# ============================================================

# Path to dataset
DATA_PATH = "recording.npy"

# Sampling frequency
# IMPORTANT:
# This must match the real sampling rate of the dataset.
FS = 30000  # Hz

# High-pass filter cutoff frequency
HIGH_PASS_CUTOFF = 300  # Hz

# Butterworth filter order
FILTER_ORDER = 4

# Duration of signal shown in detailed plots
DISPLAY_DURATION = 0.1  # seconds

# Output directory
OUTPUT_DIR = "Results/Preprocessing"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 70)
print("STEP 2: SIGNAL PREPROCESSING")
print("=" * 70)

print("\nLoading dataset...")

raw_signal = np.load(
    DATA_PATH,
    mmap_mode="r"
)

# Convert to 1D
raw_signal = np.asarray(
    raw_signal
).squeeze()

# Convert to float64
# This prevents integer overflow during filtering
raw_signal = raw_signal.astype(
    np.float64
)

print("Dataset loaded successfully.")


# ============================================================
# 3. BASIC INFORMATION
# ============================================================

n_samples = len(
    raw_signal
)

duration = n_samples / FS

print("\nDataset Information")
print("-" * 70)

print(
    f"Number of Samples : {n_samples:,}"
)

print(
    f"Data Type         : {raw_signal.dtype}"
)

print(
    f"Minimum           : {np.min(raw_signal):.6f}"
)

print(
    f"Maximum           : {np.max(raw_signal):.6f}"
)

print(
    f"Mean              : {np.mean(raw_signal):.6f}"
)

print(
    f"Median            : {np.median(raw_signal):.6f}"
)

print(
    f"Standard Deviation: {np.std(raw_signal):.6f}"
)

print(
    f"RMS               : {np.sqrt(np.mean(raw_signal**2)):.6f}"
)

print("\nSignal Duration:")

print(
    f"{duration:.3f} seconds"
)

print(
    f"{duration / 60:.3f} minutes"
)


# ============================================================
# 4. STEP 1 - DC COMPONENT REMOVAL
# ============================================================

print("\n")
print("=" * 70)
print("STEP 1: DC COMPONENT REMOVAL")
print("=" * 70)


# Estimate DC component
dc_component = np.mean(
    raw_signal
)

print(
    f"DC Component: {dc_component:.6f}"
)


# Remove DC
signal_no_dc = (

    raw_signal -
    dc_component

)


print("\nAfter DC Removal:")

print(
    f"Mean   : {np.mean(signal_no_dc):.6e}"
)

print(
    f"Minimum: {np.min(signal_no_dc):.6f}"
)

print(
    f"Maximum: {np.max(signal_no_dc):.6f}"
)


# ============================================================
# 5. STEP 2 - HIGH-PASS FILTER
# ============================================================

print("\n")
print("=" * 70)
print("STEP 2: HIGH-PASS FILTERING")
print("=" * 70)

print(
    f"High-pass cutoff frequency: {HIGH_PASS_CUTOFF} Hz"
)

print(
    f"Filter order: {FILTER_ORDER}"
)


# Design Butterworth High-Pass Filter
sos = butter(

    FILTER_ORDER,

    HIGH_PASS_CUTOFF,

    btype="highpass",

    fs=FS,

    output="sos"

)


# Zero-phase filtering
filtered_signal = sosfiltfilt(

    sos,

    signal_no_dc

)


print(
    "\nHigh-pass filtering completed."
)


# ============================================================
# 6. STEP 3 - NOISE ESTIMATION
# ============================================================

print("\n")
print("=" * 70)
print("STEP 3: NOISE ESTIMATION")
print("=" * 70)


# Robust noise estimation
# Median Absolute Deviation

signal_median = np.median(

    filtered_signal

)


mad = np.median(

    np.abs(

        filtered_signal -
        signal_median

    )

)


# Robust estimate of Gaussian noise standard deviation
noise_std = (

    mad /
    0.6745

)


# Standard deviation of filtered signal
filtered_std = np.std(

    filtered_signal

)


print(
    f"Filtered Signal STD : {filtered_std:.6f}"
)

print(
    f"MAD                 : {mad:.6f}"
)

print(
    f"Estimated Noise STD : {noise_std:.6f}"
)


# ============================================================
# 7. STEP 4 - BASELINE CORRECTION
# ============================================================

print("\n")
print("=" * 70)
print("STEP 4: BASELINE CORRECTION")
print("=" * 70)


# Estimate residual baseline using median
baseline = np.median(

    filtered_signal

)


print(
    f"Estimated Residual Baseline: {baseline:.6f}"
)


# Remove residual baseline
preprocessed_signal = (

    filtered_signal -
    baseline

)


print("\nAfter Baseline Correction:")

print(
    f"Mean   : {np.mean(preprocessed_signal):.6f}"
)

print(
    f"Median : {np.median(preprocessed_signal):.6f}"
)

print(
    f"Minimum: {np.min(preprocessed_signal):.6f}"
)

print(
    f"Maximum: {np.max(preprocessed_signal):.6f}"
)

print(
    f"STD    : {np.std(preprocessed_signal):.6f}"
)


# ============================================================
# 8. SIGNAL STATISTICS
# ============================================================

print("\n")
print("=" * 70)
print("RAW VS PREPROCESSED SIGNAL")
print("=" * 70)


raw_mean = np.mean(
    raw_signal
)

raw_std = np.std(
    raw_signal
)

raw_min = np.min(
    raw_signal
)

raw_max = np.max(
    raw_signal
)


processed_mean = np.mean(
    preprocessed_signal
)

processed_median = np.median(
    preprocessed_signal
)

processed_std = np.std(
    preprocessed_signal
)

processed_min = np.min(
    preprocessed_signal
)

processed_max = np.max(
    preprocessed_signal
)


print("\nRAW SIGNAL")

print(
    f"Mean: {raw_mean:.6f}"
)

print(
    f"STD : {raw_std:.6f}"
)

print(
    f"Min : {raw_min:.6f}"
)

print(
    f"Max : {raw_max:.6f}"
)


print("\nPREPROCESSED SIGNAL")

print(
    f"Mean   : {processed_mean:.6f}"
)

print(
    f"Median : {processed_median:.6f}"
)

print(
    f"STD    : {processed_std:.6f}"
)

print(
    f"Min    : {processed_min:.6f}"
)

print(
    f"Max    : {processed_max:.6f}"
)


# ============================================================
# 9. SELECT SHORT SEGMENT FOR VISUALIZATION
# ============================================================

display_samples = int(

    DISPLAY_DURATION *
    FS

)


display_samples = min(

    display_samples,

    n_samples

)


time_short = (

    np.arange(
        display_samples
    ) /
    FS

)


raw_short = raw_signal[

    :display_samples

]


processed_short = preprocessed_signal[

    :display_samples

]


# ============================================================
# 10. RAW SIGNAL PLOT
# ============================================================

plt.figure(

    figsize=(14, 5)

)


plt.plot(

    time_short,

    raw_short,

    linewidth=0.7

)


plt.title(

    "Raw Extracellular Neural Signal"

)


plt.xlabel(

    "Time (seconds)"

)


plt.ylabel(

    "Amplitude"

)


plt.grid(

    True,

    alpha=0.3

)


plt.tight_layout()


plt.savefig(

    os.path.join(

        OUTPUT_DIR,

        "01_raw_signal.png"

    ),

    dpi=300

)


plt.show()


# ============================================================
# 11. PREPROCESSED SIGNAL PLOT
# ============================================================

plt.figure(

    figsize=(14, 5)

)


plt.plot(

    time_short,

    processed_short,

    linewidth=0.7

)


plt.title(

    "Preprocessed Neural Signal"

)


plt.xlabel(

    "Time (seconds)"

)


plt.ylabel(

    "Amplitude"

)


plt.grid(

    True,

    alpha=0.3

)


plt.tight_layout()


plt.savefig(

    os.path.join(

        OUTPUT_DIR,

        "02_preprocessed_signal.png"

    ),

    dpi=300

)


plt.show()


# ============================================================
# 12. RAW VS PREPROCESSED
# ============================================================

fig, axes = plt.subplots(

    2,

    1,

    figsize=(14, 8),

    sharex=True

)


axes[0].plot(

    time_short,

    raw_short,

    linewidth=0.7

)


axes[0].set_title(

    "Raw Extracellular Neural Signal"

)


axes[0].set_ylabel(

    "Amplitude"

)


axes[0].grid(

    True,

    alpha=0.3

)


axes[1].plot(

    time_short,

    processed_short,

    linewidth=0.7

)


axes[1].set_title(

    "Preprocessed Neural Signal"

)


axes[1].set_xlabel(

    "Time (seconds)"

)


axes[1].set_ylabel(

    "Amplitude"

)


axes[1].grid(

    True,

    alpha=0.3

)


plt.tight_layout()


plt.savefig(

    os.path.join(

        OUTPUT_DIR,

        "03_raw_vs_preprocessed.png"

    ),

    dpi=300

)


plt.show()


# ============================================================
# 13. HISTOGRAM COMPARISON
# ============================================================

plt.figure(

    figsize=(12, 6)

)


plt.hist(

    raw_signal,

    bins=150,

    alpha=0.5,

    label="Raw Signal"

)


plt.hist(

    preprocessed_signal,

    bins=150,

    alpha=0.5,

    label="Preprocessed Signal"

)


plt.title(

    "Amplitude Distribution: Raw vs Preprocessed"

)


plt.xlabel(

    "Amplitude"

)


plt.ylabel(

    "Number of Samples"

)


plt.legend()


plt.grid(

    True,

    alpha=0.3

)


plt.tight_layout()


plt.savefig(

    os.path.join(

        OUTPUT_DIR,

        "04_histogram_comparison.png"

    ),

    dpi=300

)


plt.show()


# ============================================================
# 14. POWER SPECTRAL DENSITY
# ============================================================

print("\nCalculating Power Spectral Density...")


# Welch PSD
# nperseg=4096 gives frequency resolution:
#
# Frequency Resolution = FS / nperseg
#
# = 30000 / 4096
# ≈ 7.32 Hz

f_raw, psd_raw = welch(

    raw_signal,

    fs=FS,

    window="hann",

    nperseg=4096,

    noverlap=2048,

    detrend="constant",

    scaling="density"

)


f_processed, psd_processed = welch(

    preprocessed_signal,

    fs=FS,

    window="hann",

    nperseg=4096,

    noverlap=2048,

    detrend="constant",

    scaling="density"

)


# ============================================================
# 15. PSD COMPARISON
# ============================================================

plt.figure(

    figsize=(12, 6)

)


plt.semilogy(

    f_raw,

    psd_raw,

    label="Raw Signal"

)


plt.semilogy(

    f_processed,

    psd_processed,

    label="Preprocessed Signal"

)


plt.title(

    "Power Spectral Density: Raw vs Preprocessed"

)


plt.xlabel(

    "Frequency (Hz)"

)


plt.ylabel(

    "Power Spectral Density"

)


plt.xlim(

    0,

    5000

)


plt.legend()


plt.grid(

    True,

    alpha=0.3

)


plt.tight_layout()


plt.savefig(

    os.path.join(

        OUTPUT_DIR,

        "05_PSD_comparison.png"

    ),

    dpi=300

)


plt.show()


# ============================================================
# 16. SAVE PREPROCESSED SIGNAL
# ============================================================

processed_path = os.path.join(

    OUTPUT_DIR,

    "preprocessed_signal.npy"

)


np.save(

    processed_path,

    preprocessed_signal

)


print("\nPreprocessed signal saved to:")

print(

    processed_path

)


# ============================================================
# 17. SAVE NOISE INFORMATION
# ============================================================

noise_info = {

    "Sampling_Frequency_Hz":
        FS,

    "High_Pass_Cutoff_Hz":
        HIGH_PASS_CUTOFF,

    "Filter_Order":
        FILTER_ORDER,

    "DC_Component":
        dc_component,

    "Residual_Baseline":
        baseline,

    "MAD":
        mad,

    "Estimated_Noise_STD":
        noise_std,

    "Filtered_Signal_STD":
        filtered_std,

    "Raw_Mean":
        raw_mean,

    "Raw_STD":
        raw_std,

    "Raw_Min":
        raw_min,

    "Raw_Max":
        raw_max,

    "Processed_Mean":
        processed_mean,

    "Processed_Median":
        processed_median,

    "Processed_STD":
        processed_std,

    "Processed_Min":
        processed_min,

    "Processed_Max":
        processed_max

}


info_path = os.path.join(

    OUTPUT_DIR,

    "processing_info.txt"

)


with open(

    info_path,

    "w"

) as file:

    for key, value in noise_info.items():

        file.write(

            f"{key}: {value}\n"

        )


# ============================================================
# 18. FINAL MESSAGE
# ============================================================

print("\n")
print("=" * 70)

print(
    "SIGNAL PREPROCESSING COMPLETED SUCCESSFULLY"
)

print("=" * 70)


print("\nGenerated Results:")

print(
    "01_raw_signal.png"
)

print(
    "02_preprocessed_signal.png"
)

print(
    "03_raw_vs_preprocessed.png"
)

print(
    "04_histogram_comparison.png"
)

print(
    "05_PSD_comparison.png"
)

print(
    "preprocessed_signal.npy"
)

print(
    "processing_info.txt"
)


print("\nAll files saved in:")

print(

    OUTPUT_DIR

)

print("=" * 70)