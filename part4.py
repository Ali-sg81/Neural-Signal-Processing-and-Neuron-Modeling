import os
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import find_peaks
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


BASE_DIR = r"D:\ex\python\machin_vision\norosience"

PREPROCESSED_PATH = os.path.join(
    BASE_DIR,
    "Results",
    "Preprocessing",
    "preprocessed_signal.npy"
)

SPIKE_INDEX_PATH = os.path.join(
    BASE_DIR,
    "Results",
    "SpikeDetection",
    "spike_indices.npy"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "Results",
    "WaveformAnalysis"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

FS = 30000  # Hz
PRE_SAMPLES = 30
POST_SAMPLES = 60
WAVEFORM_LENGTH = PRE_SAMPLES + POST_SAMPLES
MAX_WAVEFORMS_TO_PLOT = 500

MAX_WAVEFORMS_FOR_ML = 10000
N_CLUSTERS = 2

print("=" * 70)
print("STEP 4: SPIKE WAVEFORM EXTRACTION")
print("=" * 70)
print("\nLoading preprocessed signal...")

signal = np.load(
    PREPROCESSED_PATH,
    mmap_mode="r"
)

print("Preprocessed signal loaded successfully.")

print("\nSignal Information")
print("-" * 70)

print(f"Number of Samples : {len(signal):,}")
print(f"Sampling Frequency: {FS:,} Hz")
print(f"Signal Duration   : {len(signal) / FS:.3f} seconds")
print("\nLoading detected spike indices...")

if not os.path.exists(SPIKE_INDEX_PATH):

    raise FileNotFoundError(
        "\nSpike index file was not found!\n"
        f"Expected path:\n{SPIKE_INDEX_PATH}\n\n"
        "Make sure part3.py saves spike indices using:\n"
        "np.save('spike_indices.npy', spike_indices)"
    )

spike_indices = np.load(
    SPIKE_INDEX_PATH
)

spike_indices = np.asarray(
    spike_indices,
    dtype=np.int64
)

print("Spike indices loaded successfully.")

print(f"Detected Spikes: {len(spike_indices):,}")

# حذف اسپایک‌هایی که به ابتدای یا انتهای سیگنال خیلی نزدیک هستند
valid_mask = (
    (spike_indices >= PRE_SAMPLES) &
    (spike_indices < len(signal) - POST_SAMPLES)
)

spike_indices = spike_indices[valid_mask]

print(
    f"Valid spikes for waveform extraction: "
    f"{len(spike_indices):,}"
)


# ================================================================
# 6. EXTRACT SPIKE WAVEFORMS
# ================================================================

print("\n" + "=" * 70)
print("EXTRACTING SPIKE WAVEFORMS")
print("=" * 70)

waveforms = np.empty(
    (
        len(spike_indices),
        WAVEFORM_LENGTH
    ),
    dtype=np.float32
)

for i, index in enumerate(spike_indices):

    start = index - PRE_SAMPLES
    end = index + POST_SAMPLES

    waveforms[i] = signal[start:end]


print("Waveform extraction completed.")

print(f"Number of Waveforms : {len(waveforms):,}")
print(f"Waveform Length     : {WAVEFORM_LENGTH} samples")
print(
    f"Waveform Duration   : "
    f"{WAVEFORM_LENGTH / FS * 1000:.2f} ms"
)


# ================================================================
# 7. SAVE WAVEFORMS
# ================================================================

waveform_path = os.path.join(
    OUTPUT_DIR,
    "spike_waveforms.npy"
)

indices_path = os.path.join(
    OUTPUT_DIR,
    "waveform_spike_indices.npy"
)

np.save(
    waveform_path,
    waveforms
)

np.save(
    indices_path,
    spike_indices
)

print("\nWaveforms saved to:")
print(waveform_path)


# ================================================================
# 8. TIME AXIS
# ================================================================

time_ms = (
    np.arange(
        -PRE_SAMPLES,
        POST_SAMPLES
    ) / FS * 1000
)


# ================================================================
# 9. PLOT INDIVIDUAL SPIKE WAVEFORMS
# ================================================================

print("\nGenerating individual waveform plot...")

rng = np.random.default_rng(42)

if len(waveforms) > MAX_WAVEFORMS_TO_PLOT:

    plot_indices = rng.choice(
        len(waveforms),
        size=MAX_WAVEFORMS_TO_PLOT,
        replace=False
    )

else:

    plot_indices = np.arange(
        len(waveforms)
    )


plt.figure(
    figsize=(12, 7)
)

for idx in plot_indices:

    plt.plot(
        time_ms,
        waveforms[idx],
        alpha=0.05
    )


plt.axvline(
    0,
    linestyle="--",
    linewidth=1.5,
    label="Spike Detection Point"
)

plt.xlabel(
    "Time (ms)"
)

plt.ylabel(
    "Amplitude"
)

plt.title(
    "Individual Extracellular Spike Waveforms"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "01_individual_spike_waveforms.png"
    ),
    dpi=300
)

plt.close()


# ================================================================
# 10. MEAN SPIKE WAVEFORM
# ================================================================

print("Calculating mean spike waveform...")

mean_waveform = np.mean(
    waveforms,
    axis=0
)

std_waveform = np.std(
    waveforms,
    axis=0
)

sem_waveform = (
    std_waveform /
    np.sqrt(len(waveforms))
)


# ================================================================
# 11. PLOT MEAN WAVEFORM ± STD
# ================================================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    time_ms,
    mean_waveform,
    linewidth=2,
    label="Mean Spike Waveform"
)

plt.fill_between(
    time_ms,
    mean_waveform - std_waveform,
    mean_waveform + std_waveform,
    alpha=0.25,
    label="±1 STD"
)

plt.axvline(
    0,
    linestyle="--",
    linewidth=1.5
)

plt.xlabel(
    "Time (ms)"
)

plt.ylabel(
    "Amplitude"
)

plt.title(
    "Mean Extracellular Spike Waveform ± STD"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "02_mean_spike_waveform.png"
    ),
    dpi=300
)

plt.close()


# ================================================================
# 12. SPIKE MORPHOLOGY ANALYSIS
# ================================================================

print("\n" + "=" * 70)
print("SPIKE MORPHOLOGY ANALYSIS")
print("=" * 70)


# Peak
peak_index = np.argmax(
    mean_waveform
)

peak_amplitude = mean_waveform[
    peak_index
]


# Trough
trough_index = np.argmin(
    mean_waveform
)

trough_amplitude = mean_waveform[
    trough_index
]


# Peak-to-Peak amplitude
peak_to_peak = (
    peak_amplitude -
    trough_amplitude
)


# Peak time
peak_time_ms = time_ms[
    peak_index
]


# Trough time
trough_time_ms = time_ms[
    trough_index
]


# Peak-to-Trough duration
peak_to_trough_ms = abs(
    trough_time_ms -
    peak_time_ms
)


print(f"Mean Peak Amplitude       : {peak_amplitude:.4f}")

print(f"Mean Trough Amplitude     : {trough_amplitude:.4f}")

print(f"Peak-to-Peak Amplitude    : {peak_to_peak:.4f}")

print(f"Peak Time                 : {peak_time_ms:.4f} ms")

print(f"Trough Time               : {trough_time_ms:.4f} ms")

print(
    f"Peak-to-Trough Duration   : "
    f"{peak_to_trough_ms:.4f} ms"
)


# ================================================================
# 13. PLOT MORPHOLOGY
# ================================================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    time_ms,
    mean_waveform,
    linewidth=2,
    label="Mean Waveform"
)

plt.scatter(
    peak_time_ms,
    peak_amplitude,
    s=80,
    label="Peak"
)

plt.scatter(
    trough_time_ms,
    trough_amplitude,
    s=80,
    label="Trough"
)

plt.axvline(
    0,
    linestyle="--",
    linewidth=1
)

plt.xlabel(
    "Time (ms)"
)

plt.ylabel(
    "Amplitude"
)

plt.title(
    "Spike Morphology Analysis"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "03_spike_morphology.png"
    ),
    dpi=300
)

plt.close()


# ================================================================
# 14. WAVEFORM AMPLITUDE ANALYSIS
# ================================================================

print("\nCalculating waveform amplitude statistics...")


waveform_max = np.max(
    waveforms,
    axis=1
)

waveform_min = np.min(
    waveforms,
    axis=1
)

waveform_ptp = (
    waveform_max -
    waveform_min
)


print(
    f"Mean Peak-to-Peak : "
    f"{np.mean(waveform_ptp):.4f}"
)

print(
    f"STD Peak-to-Peak  : "
    f"{np.std(waveform_ptp):.4f}"
)

print(
    f"Median Peak-to-Peak: "
    f"{np.median(waveform_ptp):.4f}"
)


# ================================================================
# 15. COEFFICIENT OF VARIATION
# ================================================================

cv_amplitude = (
    np.std(waveform_ptp) /
    np.mean(waveform_ptp)
)

print(
    f"Amplitude CV       : "
    f"{cv_amplitude:.4f}"
)


# ================================================================
# 16. AMPLITUDE DISTRIBUTION
# ================================================================

plt.figure(
    figsize=(10, 6)
)

plt.hist(
    waveform_ptp,
    bins=100
)

plt.xlabel(
    "Peak-to-Peak Amplitude"
)

plt.ylabel(
    "Number of Spikes"
)

plt.title(
    "Spike Peak-to-Peak Amplitude Distribution"
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "04_amplitude_distribution.png"
    ),
    dpi=300
)

plt.close()


# ================================================================
# 17. PCA ANALYSIS
# ================================================================

print("\n" + "=" * 70)
print("PCA ANALYSIS")
print("=" * 70)


# انتخاب subset برای کاهش مصرف RAM
n_ml = min(
    MAX_WAVEFORMS_FOR_ML,
    len(waveforms)
)

ml_indices = rng.choice(
    len(waveforms),
    size=n_ml,
    replace=False
)

waveforms_ml = waveforms[
    ml_indices
]


# Standardization
scaler = StandardScaler()

waveforms_scaled = scaler.fit_transform(
    waveforms_ml
)


# PCA
pca = PCA(
    n_components=3
)

pca_result = pca.fit_transform(
    waveforms_scaled
)


print(
    f"PC1 Explained Variance: "
    f"{pca.explained_variance_ratio_[0] * 100:.2f}%"
)

print(
    f"PC2 Explained Variance: "
    f"{pca.explained_variance_ratio_[1] * 100:.2f}%"
)

print(
    f"PC3 Explained Variance: "
    f"{pca.explained_variance_ratio_[2] * 100:.2f}%"
)


# ================================================================
# 18. PCA PLOT
# ================================================================

plt.figure(
    figsize=(10, 7)
)

plt.scatter(
    pca_result[:, 0],
    pca_result[:, 1],
    s=5,
    alpha=0.3
)

plt.xlabel(
    f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)"
)

plt.ylabel(
    f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)"
)

plt.title(
    "PCA Analysis of Spike Waveforms"
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "05_PCA_waveform_analysis.png"
    ),
    dpi=300
)

plt.close()


# ================================================================
# 19. K-MEANS CLUSTERING
# ================================================================

print("\n" + "=" * 70)
print("CLUSTERING ANALYSIS")
print("=" * 70)


kmeans = KMeans(
    n_clusters=N_CLUSTERS,
    random_state=42,
    n_init=10
)

cluster_labels = kmeans.fit_predict(
    pca_result[:, :2]
)


for cluster_id in range(
    N_CLUSTERS
):

    count = np.sum(
        cluster_labels == cluster_id
    )

    percentage = (
        count /
        len(cluster_labels)
        * 100
    )

    print(
        f"Cluster {cluster_id + 1}: "
        f"{count:,} spikes "
        f"({percentage:.2f}%)"
    )


# ================================================================
# 20. PCA + CLUSTER PLOT
# ================================================================

plt.figure(
    figsize=(10, 7)
)

for cluster_id in range(
    N_CLUSTERS
):

    mask = (
        cluster_labels ==
        cluster_id
    )

    plt.scatter(
        pca_result[mask, 0],
        pca_result[mask, 1],
        s=5,
        alpha=0.3,
        label=f"Cluster {cluster_id + 1}"
    )


plt.xlabel(
    "PC1"
)

plt.ylabel(
    "PC2"
)

plt.title(
    "Spike Waveform Clustering Using PCA + K-Means"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "06_PCA_KMeans_clustering.png"
    ),
    dpi=300
)

plt.close()


# ================================================================
# 21. CLUSTER MEAN WAVEFORMS
# ================================================================

print("\nCalculating cluster mean waveforms...")


plt.figure(
    figsize=(10, 6)
)


for cluster_id in range(
    N_CLUSTERS
):

    mask = (
        cluster_labels ==
        cluster_id
    )

    cluster_waveforms = waveforms_ml[
        mask
    ]

    cluster_mean = np.mean(
        cluster_waveforms,
        axis=0
    )

    plt.plot(
        time_ms,
        cluster_mean,
        linewidth=2,
        label=f"Cluster {cluster_id + 1}"
    )


plt.axvline(
    0,
    linestyle="--",
    linewidth=1
)

plt.xlabel(
    "Time (ms)"
)

plt.ylabel(
    "Amplitude"
)

plt.title(
    "Mean Spike Waveform of Each Cluster"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "07_cluster_mean_waveforms.png"
    ),
    dpi=300
)

plt.close()


# ================================================================
# 22. SINGLE vs MULTIPLE NEURON INTERPRETATION
# ================================================================

print("\n" + "=" * 70)
print("SINGLE VS MULTIPLE NEURON ANALYSIS")
print("=" * 70)


# معیار ساده بر اساس CV
if cv_amplitude < 0.25:

    amplitude_result = (
        "Low waveform amplitude variability"
    )

else:

    amplitude_result = (
        "High waveform amplitude variability"
    )


# معیار پراکندگی PCA
pc1_variance = (
    pca.explained_variance_ratio_[0]
)

pc2_variance = (
    pca.explained_variance_ratio_[1]
)


print("\nWaveform Variability:")
print(amplitude_result)

print(
    f"Amplitude CV: "
    f"{cv_amplitude:.3f}"
)

print(
    f"PC1 variance: "
    f"{pc1_variance * 100:.2f}%"
)

print(
    f"PC2 variance: "
    f"{pc2_variance * 100:.2f}%"
)


print("\nInterpretation:")

if cv_amplitude < 0.25:

    print(
        "The spike waveforms show relatively low "
        "amplitude variability."
    )

else:

    print(
        "The spike waveforms show considerable "
        "amplitude variability."
    )


print(
    "\nPCA and clustering should be interpreted "
    "together with waveform morphology."
)

print(
    "Clear separated clusters with distinct mean "
    "waveforms may indicate the presence of multiple "
    "neuronal units."
)

print(
    "A single compact cluster with similar waveform "
    "shapes is more consistent with a single dominant "
    "neural unit."
)


# ================================================================
# 23. SAVE ANALYSIS RESULTS
# ================================================================

results_path = os.path.join(
    OUTPUT_DIR,
    "waveform_analysis_results.txt"
)

with open(
    results_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "SPIKE WAVEFORM ANALYSIS RESULTS\n"
    )

    f.write(
        "=" * 60 + "\n\n"
    )

    f.write(
        f"Number of Detected Spikes: "
        f"{len(spike_indices):,}\n"
    )

    f.write(
        f"Sampling Frequency: "
        f"{FS} Hz\n"
    )

    f.write(
        f"Waveform Length: "
        f"{WAVEFORM_LENGTH} samples\n"
    )

    f.write(
        f"Waveform Duration: "
        f"{WAVEFORM_LENGTH / FS * 1000:.3f} ms\n\n"
    )

    f.write(
        f"Mean Peak Amplitude: "
        f"{peak_amplitude:.6f}\n"
    )

    f.write(
        f"Mean Trough Amplitude: "
        f"{trough_amplitude:.6f}\n"
    )

    f.write(
        f"Mean Peak-to-Peak Amplitude: "
        f"{peak_to_peak:.6f}\n"
    )

    f.write(
        f"Peak Time: "
        f"{peak_time_ms:.6f} ms\n"
    )

    f.write(
        f"Trough Time: "
        f"{trough_time_ms:.6f} ms\n"
    )

    f.write(
        f"Peak-to-Trough Duration: "
        f"{peak_to_trough_ms:.6f} ms\n"
    )

    f.write(
        f"Amplitude CV: "
        f"{cv_amplitude:.6f}\n\n"
    )

    f.write(
        f"PC1 Explained Variance: "
        f"{pc1_variance * 100:.3f}%\n"
    )

    f.write(
        f"PC2 Explained Variance: "
        f"{pc2_variance * 100:.3f}%\n"
    )

    f.write(
        "\nInterpretation:\n"
    )

    f.write(
        "Waveform variability and PCA clustering "
        "should be considered together when determining "
        "whether the recording contains one or multiple "
        "neuronal units.\n"
    )


# ================================================================
# 24. FINAL OUTPUT
# ================================================================

print("\n" + "=" * 70)
print("STEP 4 COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nGenerated Results:")

print("01_individual_spike_waveforms.png")
print("02_mean_spike_waveform.png")
print("03_spike_morphology.png")
print("04_amplitude_distribution.png")
print("05_PCA_waveform_analysis.png")
print("06_PCA_KMeans_clustering.png")
print("07_cluster_mean_waveforms.png")

print("spike_waveforms.npy")
print("waveform_spike_indices.npy")
print("waveform_analysis_results.txt")

print("\nAll files saved in:")
print(OUTPUT_DIR)

print("=" * 70)