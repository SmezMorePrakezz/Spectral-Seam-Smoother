# Spectral Seam Smoother 🎛️

Spectral Seam Smoother is a professional, GUI-based digital signal processing (DSP) tool designed to surgically repair visible "seams" or artifacts left behind by audio spectral expansion algorithms.

Instead of relying on standard parametric EQs that require manual node placement, this tool analyzes the spectral trajectory (momentum) leading into and out of a damaged frequency band. It then generates a perfectly smooth, zero-phase Cubic Hermite Spline (Bezier curve) to bridge the gap, effectively erasing the seam while keeping original transients and rhythms flawlessly intact.

# ✨ Features

Zero-Phase STFT Processing: Applies the EQ curve strictly to the magnitude domain and recombines it with the original untouched phase, guaranteeing no phase-shift smearing or transient loss.

Real-time Spectrogram & Analyzer: See the impact of your DSP adjustments instantly on a targeted slice of your audio without waiting for full-file renders.

Smart Interpolation: Calculates entry and exit slopes using contextual frequency bands to ensure a mathematically $C^1$ continuous bridge.

Resolution Quantization: Simulate analog graphic EQs. Group frequencies into discrete bands and melt them together using Gaussian blur, preventing the curve from sounding too "synthetic" or smooth.

Drag & Drop UI: Fast workflow native to Windows, macOS, and Linux.

# ⚙️ Installation

Make sure you have Python 3.8 or newer installed.

Clone or download this repository.

Install the required dependencies using pip:
```
pip install librosa numpy soundfile scipy matplotlib tkinterdnd2
```

(Note: tkinterdnd2 enables the Drag & Drop functionality. If you skip it, the app will still run perfectly using the standard "Load Audio File" button).

# 🚀 Usage

Run the program from your terminal:
```
python spectral_smoother_gui.py
```

Parameter Guide

Preview Settings

* Instant Preview: Updates the graphs continuously as you drag sliders. Uncheck if your PC lags while sliding.

* Show Smoothed (After): Toggle to compare the BEFORE and AFTER states of the spectrogram/analyzer.

* Length & Position: Controls the size and location of the preview window within the track.

DSP Parameters

* Start / End Freq (Hz): The exact boundaries of the "seam" or gap you want to reconstruct.

* Context (Hz): The width of the frequency band outside the gap analyzed to find the natural slope/trajectory of the audio entering the seam.

* Tension: Controls how strictly the curve follows the Context slopes. 0.0 is a rigid straight line. 1.0 curves aggressively. ~0.3 to 0.5 is usually the sweet spot.

* Resolution (Bands): Acts like a graphic EQ. Groups frequencies into discrete bands instead of adjusting every single FFT bin. 0 means infinite resolution.

* Feather (Hz): Crossfades the edges of the EQ curve into the original audio over this many Hz to prevent harsh blocky cuts at the start/end frequencies.

Fingerprint Source

* Preview Region (Local): Calculates the target EQ curve based ONLY on the visible 3-second (or custom length) preview segment.

* Whole Audio (Global): Calculates the target EQ curve based on the average spectrum of the ENTIRE track. Recommended for consistent master-bus processing.

# 🧠 How it Works (Under the Hood)

STFT Conversion: The audio is converted into the frequency domain (n_fft = 4096).

1. Trajectory Analysis: Linear regression (np.polyfit) calculates the slope of the spectral energy just before fmin and just after fmax.

2. Spline Generation: scipy.interpolate.CubicHermiteSpline generates a curve connecting fmin and fmax that mathematically respects the entry and exit slopes.

3. Quantization & Blurring: If a resolution is set, the curve is stepped and then blurred using scipy.ndimage.gaussian_filter1d to simulate overlapping Q-factors.

4. Dynamic EQ: A gain multiplier matrix is calculated by dividing the Target Spline by the Original Spectral Fingerprint, which is then multiplied against the original magnitude spectrogram.

5. Reconstruction: Inverse STFT brings the repaired audio back to the time domain.

# 🧑‍💻 Credits

Made By Smez Moré Prakezz (SMPTHEHEDGEHOG).
