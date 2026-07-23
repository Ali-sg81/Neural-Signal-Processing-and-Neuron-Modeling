import numpy as np
import matplotlib.pyplot as plt
import os

from scipy.signal import find_peaks

INPUT_PATH = (
    r"D:\ex\python\machin_vision\norosience\Results\Preprocessing\preprocessed_signal.npy"
)

FS = 30000  # Hz

NOISE_FACTOR = 0.6745

THRESHOLD_FACTOR = 4.0


REFRACTORY_PERIOD_MS = 1.5

REFRACTORY_SAMPLES = int(

    REFRACTORY_PERIOD_MS
    * 1e-3
    * FS

)

PRE_SPIKE_MS = 1.0

POST_SPIKE_MS = 2.0


PRE_SPIKE_SAMPLES = int(

    PRE_SPIKE_MS
    * 1e-3
    * FS

)


POST_SPIKE_SAMPLES = int(

    POST_SPIKE_MS
    * 1e-3
    * FS

)


DISPLAY_DURATION = 0.1

OUTPUT_DIR = (
    "Results/SpikeDetection"
)


os.makedirs(

    OUTPUT_DIR,

    exist_ok=True

)

print("=" * 70)

print(
    "STEP 3: SPIKE DETECTION"
)

print("=" * 70)


print(
    "\nLoading preprocessed signal..."
)


signal = np.load(

    INPUT_PATH

)


signal = np.asarray(

    signal

).squeeze()


signal = signal.astype(

    np.float64

)


N = len(

    signal

)


duration = N / FS


print(
    "Signal loaded successfully."
)


print(
    f"\nNumber of samples: {N:,}"
)


print(
    f"Duration: {duration:.3f} seconds"
)


print(
    f"Duration: {duration / 60:.3f} minutes"
)


# ============================================================
# 3. ESTIMATE BACKGROUND NOISE
# ============================================================

print("\n")

print("=" * 70)

print(
    "STEP 1: BACKGROUND NOISE ESTIMATION"
)

print("=" * 70)


# Median of signal

signal_median = np.median(

    signal

)


# Median Absolute Deviation

MAD = np.median(

    np.abs(

        signal -
        signal_median

    )

)



noise_std = (

    MAD /
    NOISE_FACTOR

)


print(
    f"Signal Median : {signal_median:.6f}"
)


print(
    f"MAD           : {MAD:.6f}"
)


print(
    f"Noise STD     : {noise_std:.6f}"
)



print("\n")

print("=" * 70)

print(
    "STEP 2: ADAPTIVE THRESHOLD CALCULATION"
)

print("=" * 70)


# Positive threshold

positive_threshold = (

    THRESHOLD_FACTOR
    *
    noise_std

)


# Negative threshold

negative_threshold = (

    -THRESHOLD_FACTOR
    *
    noise_std

)


print(
    f"Threshold Factor: "
    f"{THRESHOLD_FACTOR} sigma"
)


print(
    f"Positive Threshold: "
    f"{positive_threshold:.6f}"
)


print(
    f"Negative Threshold: "
    f"{negative_threshold:.6f}"
)


#
abs_signal = np.abs(

    signal

)


print("\n")

print("=" * 70)

print(
    "STEP 3: CANDIDATE SPIKE DETECTION"
)

print("=" * 70)


# Detect peaks above threshold

candidate_peaks, properties = find_peaks(

    abs_signal,

    height=positive_threshold,

    distance=1

)


print(
    f"Candidate spikes detected: "
    f"{len(candidate_peaks):,}"
)



print("\n")

print("=" * 70)

print(
    "STEP 4: REFRACTORY PERIOD"
)

print("=" * 70)


print(
    f"Refractory Period: "
    f"{REFRACTORY_PERIOD_MS} ms"
)


print(
    f"Refractory Samples: "
    f"{REFRACTORY_SAMPLES}"
)


detected_spikes = []


for peak in candidate_peaks:

    # First spike

    if len(detected_spikes) == 0:

        detected_spikes.append(

            peak

        )

        continue


    # Distance from previous accepted spike

    distance = (

        peak -
        detected_spikes[-1]

    )


    # Check refractory period

    if distance >= REFRACTORY_SAMPLES:

        detected_spikes.append(

            peak

        )


    else:

        # If two spikes are inside refractory period,
        # keep the spike with larger amplitude

        previous_peak = (

            detected_spikes[-1]

        )


        if (

            abs_signal[peak]
            >
            abs_signal[previous_peak]

        ):

            detected_spikes[-1] = peak


# Convert to numpy array

detected_spikes = np.array(

    detected_spikes,

    dtype=np.int64

)

number_of_spikes = len(

    detected_spikes

)


firing_rate = (

    number_of_spikes /
    duration

)


print("\n")

print("=" * 70)

print(
    "STEP 5: SPIKE DETECTION RESULTS"
)

print("=" * 70)


print(
    f"Total detected spikes: "
    f"{number_of_spikes:,}"
)


print(
    f"Signal duration: "
    f"{duration:.3f} seconds"
)


print(
    f"Firing Rate: "
    f"{firing_rate:.4f} Hz"
)

spike_times = (

    detected_spikes /
    FS

)

spike_amplitudes = signal[

    detected_spikes

]


spike_polarity = np.where(

    spike_amplitudes >= 0,

    "Positive",

    "Negative"

)


positive_spikes = np.sum(

    spike_amplitudes > 0

)


negative_spikes = np.sum(

    spike_amplitudes < 0

)


print("\nSpike Polarity:")

print(

    f"Positive Spikes: "
    f"{positive_spikes:,}"

)


print(

    f"Negative Spikes: "
    f"{negative_spikes:,}"

)


print("\n")

print("=" * 70)

print(
    "STEP 6: SPIKE WAVEFORM EXTRACTION"
)

print("=" * 70)


waveforms = []


valid_spike_indices = []


for spike_index in detected_spikes:

    start = (

        spike_index -
        PRE_SPIKE_SAMPLES

    )


    end = (

        spike_index +
        POST_SPIKE_SAMPLES

    )


    # Check boundaries

    if (

        start >= 0
        and
        end < N

    ):

        waveform = signal[

            start:end

        ]


        waveforms.append(

            waveform

        )


        valid_spike_indices.append(

            spike_index

        )


waveforms = np.array(

    waveforms

)


valid_spike_indices = np.array(

    valid_spike_indices

)


print(

    f"Extracted waveforms: "
    f"{len(waveforms):,}"

)


print(

    f"Waveform length: "
    f"{waveforms.shape[1]} samples"

)


print(

    f"Waveform duration: "
    f"{PRE_SPIKE_MS + POST_SPIKE_MS} ms"

)


print("\n")

print("=" * 70)

print(
    "STEP 7: INTER-SPIKE INTERVAL ANALYSIS"
)

print("=" * 70)


if len(spike_times) > 1:

    ISI = np.diff(

        spike_times

    )

else:

    ISI = np.array([])


if len(ISI) > 0:

    print(

        f"Mean ISI: "
        f"{np.mean(ISI) * 1000:.3f} ms"

    )


    print(

        f"Median ISI: "
        f"{np.median(ISI) * 1000:.3f} ms"

    )


    print(

        f"Minimum ISI: "
        f"{np.min(ISI) * 1000:.3f} ms"

    )


    print(

        f"Maximum ISI: "
        f"{np.max(ISI) * 1000:.3f} ms"

    )


# ============================================================
# 13. SAVE RESULTS
# ============================================================

print("\n")

print("=" * 70)

print(
    "STEP 8: SAVING RESULTS"
)

print("=" * 70)

np.save(

    os.path.join(

        OUTPUT_DIR,

        "spike_indices.npy"

    ),

    detected_spikes

)


np.save(

    os.path.join(

        OUTPUT_DIR,

        "spike_times.npy"

    ),

    spike_times

)


np.save(

    os.path.join(

        OUTPUT_DIR,

        "spike_waveforms.npy"

    ),

    waveforms

)


np.save(

    os.path.join(

        OUTPUT_DIR,

        "ISI.npy"

    ),

    ISI

)


display_samples = int(

    DISPLAY_DURATION *
    FS

)


display_samples = min(

    display_samples,

    N

)


time_display = (

    np.arange(

        display_samples

    )
    /
    FS

)


signal_display = signal[

    :display_samples

]


# Select spikes inside displayed section

display_spikes = detected_spikes[

    detected_spikes
    <
    display_samples

]


display_spike_times = (

    display_spikes /
    FS

)


display_spike_amplitudes = signal[

    display_spikes

]


plt.figure(

    figsize=(14, 6)

)


plt.plot(

    time_display,

    signal_display,

    linewidth=0.7,

    label="Preprocessed Signal"

)


plt.scatter(

    display_spike_times,

    display_spike_amplitudes,

    color="red",

    s=25,

    label="Detected Spikes",

    zorder=3

)


plt.axhline(

    positive_threshold,

    linestyle="--",

    linewidth=1,

    label="Positive Threshold"

)


plt.axhline(

    negative_threshold,

    linestyle="--",

    linewidth=1,

    label="Negative Threshold"

)


plt.title(

    "Threshold-Based Spike Detection"

)


plt.xlabel(

    "Time (seconds)"

)


plt.ylabel(

    "Amplitude"

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

        "01_spike_detection.png"

    ),

    dpi=300

)


plt.show()


# Display first 20 detected waveforms

num_waveforms_to_plot = min(

    20,

    len(waveforms)

)


if num_waveforms_to_plot > 0:

    waveform_time = np.linspace(

        -PRE_SPIKE_MS,

        POST_SPIKE_MS,

        waveforms.shape[1]

    )


    plt.figure(

        figsize=(12, 6)

    )


    for i in range(

        num_waveforms_to_plot

    ):

        plt.plot(

            waveform_time,

            waveforms[i],

            alpha=0.5

        )


    plt.axvline(

        0,

        linestyle="--",

        linewidth=1

    )


    plt.title(

        "Detected Spike Waveforms"

    )


    plt.xlabel(

        "Time relative to spike peak (ms)"

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

            "02_spike_waveforms.png"

        ),

        dpi=300

    )


    plt.show()


if len(waveforms) > 0:

    mean_waveform = np.mean(

        waveforms,

        axis=0

    )


    std_waveform = np.std(

        waveforms,

        axis=0

    )


    waveform_time = np.linspace(

        -PRE_SPIKE_MS,

        POST_SPIKE_MS,

        waveforms.shape[1]

    )


    plt.figure(

        figsize=(10, 6)

    )


    plt.plot(

        waveform_time,

        mean_waveform,

        linewidth=2,

        label="Mean Spike Waveform"

    )


    plt.fill_between(

        waveform_time,

        mean_waveform - std_waveform,

        mean_waveform + std_waveform,

        alpha=0.3,

        label="±1 STD"

    )


    plt.axvline(

        0,

        linestyle="--",

        linewidth=1

    )


    plt.title(

        "Average Detected Spike Waveform"

    )


    plt.xlabel(

        "Time relative to spike peak (ms)"

    )


    plt.ylabel(

        "Amplitude"

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

            "03_mean_spike_waveform.png"

        ),

        dpi=300

    )


    plt.show()


if len(ISI) > 0:

    plt.figure(

        figsize=(10, 6)

    )


    plt.hist(

        ISI * 1000,

        bins=100

    )


    plt.axvline(

        REFRACTORY_PERIOD_MS,

        linestyle="--",

        linewidth=2,

        label="Refractory Period"

    )


    plt.title(

        "Inter-Spike Interval (ISI) Distribution"

    )


    plt.xlabel(

        "ISI (ms)"

    )


    plt.ylabel(

        "Number of Intervals"

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

            "04_ISI_histogram.png"

        ),

        dpi=300

    )


    plt.show()


plt.figure(

    figsize=(10, 6)

)


plt.hist(

    spike_amplitudes,

    bins=100

)


plt.title(

    "Detected Spike Amplitude Distribution"

)


plt.xlabel(

    "Spike Amplitude"

)


plt.ylabel(

    "Number of Spikes"

)


plt.grid(

    True,

    alpha=0.3

)


plt.tight_layout()


plt.savefig(

    os.path.join(

        OUTPUT_DIR,

        "05_spike_amplitude_histogram.png"

    ),

    dpi=300

)


plt.show()


info_path = os.path.join(

    OUTPUT_DIR,

    "spike_detection_info.txt"

)


with open(

    info_path,

    "w"

) as file:

    file.write(

        "SPIKE DETECTION RESULTS\n"

    )

    file.write(

        "=" * 60 +
        "\n\n"

    )


    file.write(

        f"Sampling Frequency: "
        f"{FS} Hz\n"

    )


    file.write(

        f"Signal Duration: "
        f"{duration:.6f} seconds\n"

    )


    file.write(

        f"Number of Samples: "
        f"{N}\n"

    )


    file.write(

        f"\nNoise Estimation\n"

    )


    file.write(

        f"MAD: "
        f"{MAD:.6f}\n"

    )


    file.write(

        f"Estimated Noise STD: "
        f"{noise_std:.6f}\n"

    )


    file.write(

        f"\nThreshold Parameters\n"

    )


    file.write(

        f"Threshold Factor: "
        f"{THRESHOLD_FACTOR} sigma\n"

    )


    file.write(

        f"Positive Threshold: "
        f"{positive_threshold:.6f}\n"

    )


    file.write(

        f"Negative Threshold: "
        f"{negative_threshold:.6f}\n"

    )


    file.write(

        f"\nRefractory Period\n"

    )


    file.write(

        f"Refractory Period: "
        f"{REFRACTORY_PERIOD_MS} ms\n"

    )


    file.write(

        f"Refractory Samples: "
        f"{REFRACTORY_SAMPLES}\n"

    )


    file.write(

        f"\nDetection Results\n"

    )


    file.write(

        f"Candidate Spikes: "
        f"{len(candidate_peaks)}\n"

    )


    file.write(

        f"Detected Spikes: "
        f"{number_of_spikes}\n"

    )


    file.write(

        f"Positive Spikes: "
        f"{positive_spikes}\n"

    )


    file.write(

        f"Negative Spikes: "
        f"{negative_spikes}\n"

    )


    file.write(

        f"Firing Rate: "
        f"{firing_rate:.6f} Hz\n"

    )


    if len(ISI) > 0:

        file.write(

            f"\nISI Statistics\n"

        )


        file.write(

            f"Mean ISI: "
            f"{np.mean(ISI) * 1000:.6f} ms\n"

        )


        file.write(

            f"Median ISI: "
            f"{np.median(ISI) * 1000:.6f} ms\n"

        )


        file.write(

            f"Minimum ISI: "
            f"{np.min(ISI) * 1000:.6f} ms\n"

        )


        file.write(

            f"Maximum ISI: "
            f"{np.max(ISI) * 1000:.6f} ms\n"

        )


# ============================================================
# 20. FINAL OUTPUT
# ============================================================

print("\n")

print("=" * 70)

print(
    "SPIKE DETECTION COMPLETED SUCCESSFULLY"
)

print("=" * 70)


print("\nFinal Results:")

print(

    f"Detected Spikes: "
    f"{number_of_spikes:,}"

)


print(

    f"Firing Rate: "
    f"{firing_rate:.4f} Hz"

)


print(

    f"Noise STD: "
    f"{noise_std:.4f}"

)


print(

    f"Adaptive Threshold: "
    f"±{positive_threshold:.4f}"

)


print("\nGenerated Files:")

print(
    "01_spike_detection.png"
)

print(
    "02_spike_waveforms.png"
)

print(
    "03_mean_spike_waveform.png"
)

print(
    "04_ISI_histogram.png"
)

print(
    "05_spike_amplitude_histogram.png"
)

print(
    "spike_indices.npy"
)

print(
    "spike_times.npy"
)

print(
    "spike_waveforms.npy"
)

print(
    "ISI.npy"
)

print(
    "spike_detection_info.txt"
)


print("\nAll files saved in:")

print(

    OUTPUT_DIR

)

print("=" * 70)