import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. SETTINGS
# ============================================================

BASE_DIR = r"D:\ex\python\machin_vision\norosience"

WAVEFORM_PATH = os.path.join(
    BASE_DIR, "Results", "WaveformAnalysis", "spike_waveforms.npy"
)

SPIKE_INDEX_PATH = os.path.join(
    BASE_DIR, "Results", "WaveformAnalysis", "waveform_spike_indices.npy"
)

OUTPUT_DIR = os.path.join(BASE_DIR, "Results", "FeatureExtraction")
os.makedirs(OUTPUT_DIR, exist_ok=True)

FS = 30000
PRE_SAMPLES = 30
POST_SAMPLES = 60

# --- تشخیص Outlier (روش Modified Z-score بر مبنای میانه/MAD) ---
# این ستون‌ها برای تشخیص outlier بررسی می‌شوند (همان‌هایی که در نمودارها
# دم بلند/پرت‌ترین مقادیر را نشان می‌دادند)
OUTLIER_FEATURES = ["Peak_to_Peak", "Energy_au", "AUC_au"]
# آستانه‌ی استاندارد پیشنهادی Iglewicz & Hoaglin برای Modified Z-score
OUTLIER_Z_THRESHOLD = 3.5


print("=" * 70)
print("STEP 5: SPIKE FEATURE EXTRACTION")
print("=" * 70)


# ============================================================
# 2. LOAD WAVEFORMS
# ============================================================

for path in (WAVEFORM_PATH, SPIKE_INDEX_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\nفایل موردنیاز پیدا نشد:\n{path}\n"
            "لطفاً ابتدا part4.py (استخراج موج اسپایک) را اجرا کنید.\n"
        )

print("\nLoading spike waveforms...")

waveforms = np.load(WAVEFORM_PATH)
spike_indices = np.load(SPIKE_INDEX_PATH)

n_spikes, waveform_length = waveforms.shape

print("Waveforms loaded successfully.")
print(f"Number of Spikes : {n_spikes:,}")
print(f"Waveform Length  : {waveform_length} samples")

if waveform_length != PRE_SAMPLES + POST_SAMPLES:
    raise ValueError(
        "طول موج بارگذاری‌شده با PRE_SAMPLES + POST_SAMPLES تعریف‌شده در این "
        "اسکریپت همخوانی ندارد. مقادیر PRE_SAMPLES/POST_SAMPLES را با "
        "part4.py هماهنگ کنید."
    )

time_ms = np.arange(-PRE_SAMPLES, POST_SAMPLES) / FS * 1000


# ============================================================
# 3. FEATURE FUNCTIONS
# ============================================================

def dominant_peak(waveforms):
    idx = np.argmax(np.abs(waveforms), axis=1)
    rows = np.arange(len(waveforms))
    return waveforms[rows, idx]


def peak_to_peak(waveforms):
    return np.max(waveforms, axis=1) - np.min(waveforms, axis=1)


def peak_trough_width_ms(waveforms, time_ms):
    max_idx = np.argmax(waveforms, axis=1)
    min_idx = np.argmin(waveforms, axis=1)
    return np.abs(time_ms[max_idx] - time_ms[min_idx])


def signal_energy(waveforms):
    return np.sum(waveforms ** 2, axis=1)


def area_under_curve(waveforms, fs):
    dt = 1.0 / fs
    return np.trapz(np.abs(waveforms), dx=dt, axis=1)


def half_amplitude_width_ms(waveforms, fs):
    widths = np.full(len(waveforms), np.nan)
    for i, wf in enumerate(waveforms):
        dom_idx = np.argmax(np.abs(wf))
        dom_val = wf[dom_idx]
        half_level = dom_val / 2.0
        if dom_val < 0:
            crossing = np.where(wf <= half_level)[0]
        else:
            crossing = np.where(wf >= half_level)[0]
        if len(crossing) >= 2:
            widths[i] = (crossing[-1] - crossing[0]) / fs * 1000
    return widths


def robust_outlier_flags(df, columns, threshold):
    """
    تشخیص outlier با Modified Z-score (بر مبنای میانه و MAD، نه میانگین/STD).
    برخلاف z-score معمولی، این روش خودش تحت تأثیر outlierها منحرف نمی‌شود،
    پس برای داده‌هایی مثل Energy/Peak-to-Peak که چند مقدار بسیار بزرگ دارند
    مناسب‌تر است. اگر یک اسپایک در *هر یک* از ستون‌های داده‌شده از آستانه
    عبور کند، outlier علامت می‌خورد.
    """
    z_scores = pd.DataFrame(index=df.index)
    for col in columns:
        x = df[col].to_numpy(dtype=float)
        median = np.median(x)
        mad = np.median(np.abs(x - median))
        mad = mad if mad > 0 else 1e-9
        z_scores[col] = 0.6745 * (x - median) / mad

    max_abs_z = z_scores.abs().max(axis=1)
    is_outlier = max_abs_z > threshold
    return max_abs_z, is_outlier


# ============================================================
# 4. COMPUTE FEATURES
# ============================================================

print("\nExtracting features...")

peak_amp = dominant_peak(waveforms)
ptp_amp = peak_to_peak(waveforms)
width_ms = peak_trough_width_ms(waveforms, time_ms)
energy = signal_energy(waveforms)
auc = area_under_curve(waveforms, FS)
half_width = half_amplitude_width_ms(waveforms, FS)

print("Feature extraction completed.")


# ============================================================
# 5. BUILD FEATURE TABLE (+ OUTLIER COLUMN)
# ============================================================

feature_table = pd.DataFrame({
    "Spike": np.arange(1, n_spikes + 1),
    "Spike_Index": spike_indices,
    "Spike_Time_s": np.round(spike_indices / FS, 6),
    "Peak_Amplitude": np.round(peak_amp, 3),
    "Peak_to_Peak": np.round(ptp_amp, 3),
    "Width_ms": np.round(width_ms, 3),
    "Energy_au": np.round(energy, 3),
    "AUC_au": np.round(auc, 3),
    "Half_Width_ms": np.round(half_width, 3),
})

outlier_score, is_outlier = robust_outlier_flags(
    feature_table, OUTLIER_FEATURES, OUTLIER_Z_THRESHOLD
)
feature_table["Outlier_Score"] = np.round(outlier_score, 3)
feature_table["Is_Outlier"] = is_outlier

n_outliers = int(is_outlier.sum())
print(f"\nOutlier Detection (Modified Z-score, threshold={OUTLIER_Z_THRESHOLD}):")
print(f"Columns checked : {OUTLIER_FEATURES}")
print(f"Outliers flagged: {n_outliers:,} / {n_spikes:,} ({100 * n_outliers / n_spikes:.2f}%)")

print("\nFeature Table Preview:")
print(
    feature_table[["Spike", "Peak_Amplitude", "Width_ms", "Energy_au", "Is_Outlier"]]
    .head(10)
    .to_string(index=False)
)

csv_path = os.path.join(OUTPUT_DIR, "spike_features.csv")
feature_table.to_csv(csv_path, index=False)
print(f"\nFull feature table (including outliers) saved to:\n{csv_path}")

feature_table_clean = feature_table.loc[~is_outlier].reset_index(drop=True)
clean_csv_path = os.path.join(OUTPUT_DIR, "spike_features_clean.csv")
feature_table_clean.to_csv(clean_csv_path, index=False)
print(f"Outlier-free feature table saved to:\n{clean_csv_path}")


# ============================================================
# 6. SUMMARY STATISTICS (RAW vs CLEAN)
# ============================================================

feature_cols = [
    "Peak_Amplitude", "Peak_to_Peak", "Width_ms",
    "Energy_au", "AUC_au", "Half_Width_ms",
]

summary_raw = feature_table[feature_cols].describe()
print("\nFeature Summary Statistics (all spikes, includes outliers):")
print(summary_raw.to_string())

summary_path = os.path.join(OUTPUT_DIR, "feature_summary_statistics.csv")
summary_raw.to_csv(summary_path)

summary_clean = feature_table_clean[feature_cols].describe()
print("\nFeature Summary Statistics (outliers excluded):")
print(summary_clean.to_string())

summary_clean_path = os.path.join(OUTPUT_DIR, "feature_summary_statistics_clean.csv")
summary_clean.to_csv(summary_clean_path)


# ============================================================
# 7. FEATURE HISTOGRAMS (روی داده‌ی تمیز، برای خوانا بودن)
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(12, 9))

hist_features = [
    ("Peak_Amplitude", "Peak Amplitude"),
    ("Peak_to_Peak", "Peak-to-Peak Amplitude"),
    ("Width_ms", "Spike Width (Peak-to-Trough, ms)"),
    ("Energy_au", "Energy (a.u.)"),
]

for ax, (col, title) in zip(axes.flat, hist_features):
    ax.hist(
        feature_table_clean[col].dropna(),
        bins=60,
        color="steelblue",
        edgecolor="white",
    )
    ax.set_title(title)
    ax.set_xlabel(title)
    ax.set_ylabel("Number of Spikes")
    ax.grid(alpha=0.3)

fig.suptitle(
    f"Feature Histograms — outliers excluded from view "
    f"({n_outliers} of {n_spikes} spikes; see spike_features.csv for full data)",
    fontsize=10,
)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "01_feature_histograms.png"), dpi=300)
plt.show()


# ============================================================
# 8. FEATURE SPACE SCATTER (WIDTH vs AMPLITUDE, outliers highlighted)
# ============================================================

plt.figure(figsize=(9, 7))

inliers = feature_table.loc[~is_outlier]
outliers = feature_table.loc[is_outlier]

plt.scatter(
    inliers["Width_ms"], inliers["Peak_to_Peak"],
    s=8, alpha=0.35, color="darkorange", label="Normal spikes",
)
if len(outliers) > 0:
    plt.scatter(
        outliers["Width_ms"], outliers["Peak_to_Peak"],
        s=30, marker="x", color="red", label="Outliers", zorder=3,
    )

plt.xlabel("Spike Width (ms)")
plt.ylabel("Peak-to-Peak Amplitude")
plt.title("Spike Feature Space: Width vs. Amplitude")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "02_width_vs_amplitude.png"), dpi=300)
plt.show()

# نمای زوم‌شده روی جمعیت اصلی (بدون outlierها) برای دیدن جزئیات داخل توده
if len(outliers) > 0:
    plt.figure(figsize=(9, 7))
    plt.scatter(
        inliers["Width_ms"], inliers["Peak_to_Peak"],
        s=8, alpha=0.4, color="darkorange",
    )
    plt.xlabel("Spike Width (ms)")
    plt.ylabel("Peak-to-Peak Amplitude")
    plt.title("Spike Feature Space (zoomed, outliers excluded)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "02b_width_vs_amplitude_zoomed.png"), dpi=300)
    plt.show()


# ============================================================
# 9. FEATURE CORRELATION MATRIX (روی داده‌ی تمیز)
# ============================================================

corr = feature_table_clean[feature_cols].corr()

plt.figure(figsize=(7, 6))
im = plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(im, label="Correlation")
plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
plt.yticks(range(len(corr.columns)), corr.columns)
plt.title("Feature Correlation Matrix (outliers excluded)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "03_feature_correlation.png"), dpi=300)
plt.show()


# ============================================================
# 10. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("STEP 5 COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nGenerated Files:")
print("spike_features.csv               (all spikes + Is_Outlier column)")
print("spike_features_clean.csv         (outliers removed)")
print("feature_summary_statistics.csv   (all spikes)")
print("feature_summary_statistics_clean.csv (outliers excluded)")
print("01_feature_histograms.png        (outliers excluded, readable range)")
print("02_width_vs_amplitude.png        (outliers marked with red x)")
print("02b_width_vs_amplitude_zoomed.png (only if outliers were found)")
print("03_feature_correlation.png       (computed on outlier-free data)")

print("\nAll files saved in:")
print(OUTPUT_DIR)
print("=" * 70)