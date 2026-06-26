import tkinter as tk
from tkinter import filedialog, messagebox
import librosa
import numpy as np
import soundfile as sf
from scipy.interpolate import CubicHermiteSpline
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import sys
import os

# Try to import Drag and Drop support
try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    HAS_DND = True
except ImportError:
    HAS_DND = False

class SpectralSmootherApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Spectral Seam Smoother")
        self.root.geometry("1150x750")
        
        # Intercept the window close event
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Setup Drag and Drop if available
        if HAS_DND:
            self.root.drop_target_register(DND_FILES)
            self.root.dnd_bind('<<Drop>>', self.on_drop)
        
        # Audio State
        self.audio_path = None
        self.full_y = None
        self.sr = None
        self.whole_audio_fingerprints = None
        
        # Preview State
        self.preview_S_mag = None  # Cached magnitude
        self.preview_freqs = None
        self.preview_times = None
        self.preview_fingerprints = None
        self.n_fft = 4096
        
        # Threading/Debouncing controls
        self.update_timer = None
        self.stft_timer = None
        self.stft_request_id = 0

        self.create_widgets()

    def on_closing(self):
        """Handle the window close event to ensure a clean exit."""
        if getattr(self, 'update_timer', None) is not None:
            self.root.after_cancel(self.update_timer)
        if getattr(self, 'stft_timer', None) is not None:
            self.root.after_cancel(self.stft_timer)
        
        if hasattr(self, 'fig'):
            plt.close(self.fig)
            
        self.root.quit()
        self.root.destroy()
        sys.exit(0)

    def create_widgets(self):
        # Top Bar
        top_frame = tk.Frame(self.root, pady=10, padx=10)
        top_frame.pack(fill=tk.X)
        
        self.btn_load = tk.Button(top_frame, text="Load Audio File", command=self.load_audio, font=('Arial', 10, 'bold'))
        self.btn_load.pack(side=tk.LEFT, padx=5)
        
        lbl_text = "Drag & Drop an audio file here or click Load" if HAS_DND else "No file loaded (Install tkinterdnd2 for Drag & Drop)"
        self.lbl_file = tk.Label(top_frame, text=lbl_text, fg="gray")
        self.lbl_file.pack(side=tk.LEFT, padx=10)

        self.btn_help = tk.Button(top_frame, text="Usage Manual", command=self.show_manual, font=('Arial', 10, 'bold'), bg="#2196F3", fg="white")
        self.btn_help.pack(side=tk.RIGHT, padx=5)

        # Main Body
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Left Panel (Controls)
        left_frame = tk.Frame(main_frame, width=320, padx=10, pady=10)
        left_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        # --- PREVIEW SETTINGS ---
        tk.Label(left_frame, text="Preview Settings", font=('Arial', 13, 'bold')).pack(pady=(0, 5), anchor="w")
        
        self.instant_preview_var = tk.BooleanVar(value=True)
        chk_instant = tk.Checkbutton(left_frame, text="Instant Preview (Update while sliding)", variable=self.instant_preview_var)
        chk_instant.pack(pady=(0, 5), anchor="w")
        
        self.show_after_var = tk.BooleanVar(value=True)
        chk_after = tk.Checkbutton(left_frame, text="Show Smoothed (After)", variable=self.show_after_var, command=lambda: self.schedule_update(0))
        chk_after.pack(pady=(0, 10), anchor="w")

        # Preview Length
        frame_len = tk.Frame(left_frame)
        frame_len.pack(fill=tk.X, pady=2)
        tk.Label(frame_len, text="Length (s)", width=15, anchor="w").pack(side=tk.LEFT)
        self.var_len = tk.IntVar(value=3)
        spin_len = tk.Spinbox(frame_len, from_=3, to=60, increment=1, textvariable=self.var_len, width=6, command=self.on_preview_slide)
        spin_len.bind('<Return>', lambda e: self.schedule_preview_recalc(10))
        spin_len.pack(side=tk.RIGHT)
        scale_len = tk.Scale(frame_len, from_=3, to=60, resolution=1, variable=self.var_len, orient=tk.HORIZONTAL, showvalue=0, command=self.on_preview_slide)
        scale_len.bind('<ButtonRelease-1>', self.on_preview_release)
        scale_len.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=5)

        # Preview Position
        frame_pos = tk.Frame(left_frame)
        frame_pos.pack(fill=tk.X, pady=2)
        tk.Label(frame_pos, text="Position (%)", width=15, anchor="w").pack(side=tk.LEFT)
        self.var_pos = tk.DoubleVar(value=50.0)
        spin_pos = tk.Spinbox(frame_pos, from_=0, to=100, increment=1, textvariable=self.var_pos, width=6, command=self.on_preview_slide)
        spin_pos.bind('<Return>', lambda e: self.schedule_preview_recalc(10))
        spin_pos.pack(side=tk.RIGHT)
        scale_pos = tk.Scale(frame_pos, from_=0, to=100, resolution=0.1, variable=self.var_pos, orient=tk.HORIZONTAL, showvalue=0, command=self.on_preview_slide)
        scale_pos.bind('<ButtonRelease-1>', self.on_preview_release)
        scale_pos.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=5)

        # --- DSP PARAMETERS ---
        tk.Label(left_frame, text="DSP Parameters", font=('Arial', 13, 'bold')).pack(pady=(20, 5), anchor="w")

        self.vars = {}
        controls = [
            ("fmin", "Start Freq (Hz)", 0, 20000, 3200, 10),
            ("fmax", "End Freq (Hz)", 0, 20000, 4200, 10),
            ("context", "Context (Hz)", 50, 2000, 400, 10),
            ("tension", "Tension", 0.0, 1.0, 0.3, 0.01),
            ("resolution", "Resolution (Bands)", 0, 200, 40, 1),
            ("feather", "Feather (Hz)", 0, 1000, 50, 5)
        ]

        for key, label, vmin, vmax, vdef, res in controls:
            frame = tk.Frame(left_frame)
            frame.pack(fill=tk.X, pady=5)
            tk.Label(frame, text=label, width=15, anchor="w").pack(side=tk.LEFT)
            
            var = tk.DoubleVar(value=vdef) if isinstance(vdef, float) else tk.IntVar(value=vdef)
            self.vars[key] = var
            
            spin = tk.Spinbox(frame, from_=vmin, to=vmax, increment=res, textvariable=var, width=8, command=self.on_dsp_slide)
            spin.pack(side=tk.RIGHT)
            spin.bind('<Return>', lambda e: self.schedule_update(10))
            
            scale = tk.Scale(frame, from_=vmin, to=vmax, resolution=res, variable=var, orient=tk.HORIZONTAL, showvalue=0, command=self.on_dsp_slide)
            scale.bind('<ButtonRelease-1>', self.on_dsp_release)
            scale.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=5)

        # --- BOTTOM CONTROLS (Mode & Save) ---
        bottom_frame = tk.Frame(left_frame)
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=10)

        self.fingerprint_mode_var = tk.StringVar(value="preview")
        tk.Label(bottom_frame, text="Fingerprint Source:", font=('Arial', 10, 'bold')).pack(anchor="w", pady=(0, 2))
        tk.Radiobutton(bottom_frame, text="Preview Region (Local)", variable=self.fingerprint_mode_var, value="preview", command=lambda: self.schedule_update(10)).pack(anchor="w")
        tk.Radiobutton(bottom_frame, text="Whole Audio (Global)", variable=self.fingerprint_mode_var, value="whole", command=lambda: self.schedule_update(10)).pack(anchor="w")

        self.btn_save = tk.Button(bottom_frame, text="Apply & Save Audio", command=self.save_audio, state=tk.DISABLED, font=('Arial', 11, 'bold'), bg="#4CAF50", fg="white")
        self.btn_save.pack(fill=tk.X, pady=(15, 0))

        tk.Label(bottom_frame, text="Made By Smez Moré Prakezz (SMPTHEHEDGEHOG)", font=('Arial', 8, 'italic'), fg="gray").pack(pady=(10, 0))

        # Right Panel (Spectrogram & Analyzer)
        right_frame = tk.Frame(main_frame, bg="black")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.lbl_cursor = tk.Label(right_frame, text="Cursor Freq: -- Hz", bg="black", fg="cyan", font=('Arial', 10, 'bold'))
        self.lbl_cursor.pack(side=tk.TOP, anchor="e", padx=20, pady=(5, 0))

        self.fig, (self.ax_spec, self.ax_line) = plt.subplots(1, 2, figsize=(10, 6), gridspec_kw={'width_ratios': [2.5, 1]})
        self.fig.patch.set_facecolor('#2b2b2b')
        
        for ax in (self.ax_spec, self.ax_line):
            ax.set_facecolor('#1a1a1a')
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.title.set_color('white')
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Matplotlib event bindings for cursor hover
        self.canvas.mpl_connect("motion_notify_event", self.on_mouse_move)
        self.canvas.mpl_connect("axes_leave_event", self.on_mouse_leave)

    # --- UI Event Handlers ---

    def show_manual(self):
        manual_window = tk.Toplevel(self.root)
        manual_window.title("Usage Manual")
        manual_window.geometry("580x520")
        
        text = tk.Text(manual_window, wrap=tk.WORD, padx=15, pady=15, font=('Arial', 10), bg="#f5f5f5")
        text.pack(fill=tk.BOTH, expand=True)
        
        manual_content = """Spectral Seam Smoother - Usage Manual

1. PREVIEW SETTINGS
• Instant Preview: Updates the graphs continuously as you drag sliders. Uncheck if your PC lags while sliding.
• Show Smoothed (After): Toggle to compare the BEFORE and AFTER states of the spectrogram.
• Length & Position: Controls the size and location of the preview window within the track.

2. DSP PARAMETERS
• Start / End Freq (Hz): The exact boundaries of the "seam" or gap you want to reconstruct.
• Context (Hz): The width of the frequency band outside the gap analyzed to find the natural slope/trajectory of the audio.
• Tension: Controls how strictly the curve follows the Context slopes. 0.0 is a rigid straight line. 1.0 curves aggressively. ~0.3 to 0.5 is usually best.
• Resolution (Bands): Acts like a graphic EQ. Groups frequencies into discrete bands instead of adjusting every single FFT bin. 0 means infinite resolution. Smoothly interpolated via Gaussian blur.
• Feather (Hz): Crossfades the edges of the EQ curve into the original audio over this many Hz to prevent harsh blocky cuts at the start/end frequencies.

3. FINGERPRINT SOURCE
• Preview Region (Local): Calculates the target EQ curve based ONLY on the visible preview segment.
• Whole Audio (Global): Calculates the target EQ curve based on the average spectrum of the ENTIRE track (recommended for consistent master-bus processing).
"""
        text.insert(tk.END, manual_content)
        text.config(state=tk.DISABLED)

    def on_mouse_move(self, event):
        # Both graphs share the Y-axis as frequency
        if event.inaxes in (self.ax_spec, self.ax_line) and event.ydata is not None:
            self.lbl_cursor.config(text=f"Cursor Freq: {event.ydata:.1f} Hz")
        else:
            self.lbl_cursor.config(text="Cursor Freq: -- Hz")

    def on_mouse_leave(self, event):
        self.lbl_cursor.config(text="Cursor Freq: -- Hz")

    def on_dsp_slide(self, *args):
        if self.instant_preview_var.get():
            self.schedule_update(100)

    def on_dsp_release(self, *args):
        if not self.instant_preview_var.get():
            self.schedule_update(10)

    def on_preview_slide(self, *args):
        if self.instant_preview_var.get():
            self.schedule_preview_recalc(300)

    def on_preview_release(self, *args):
        if not self.instant_preview_var.get():
            self.schedule_preview_recalc(10)

    def schedule_update(self, delay=200):
        if self.update_timer is not None:
            self.root.after_cancel(self.update_timer)
        self.update_timer = self.root.after(delay, self.update_plot)

    def schedule_preview_recalc(self, delay=300):
        if self.stft_timer is not None:
            self.root.after_cancel(self.stft_timer)
        self.stft_timer = self.root.after(delay, self.start_preview_recalc)

    # --- Audio Loading & Parsing ---

    def on_drop(self, event):
        file_path = event.data
        if file_path.startswith('{') and file_path.endswith('}'):
            file_path = file_path[1:-1]
        
        if file_path:
            self._start_load_process(file_path)

    def load_audio(self):
        file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav *.flac *.mp3 *.ogg")])
        if file_path:
            self._start_load_process(file_path)

    def _start_load_process(self, file_path):
        self.btn_load.config(state=tk.DISABLED)
        self.lbl_file.config(text="Loading...", fg="blue")
        self.root.update()

        threading.Thread(target=self._process_load, args=(file_path,), daemon=True).start()

    def _process_load(self, file_path):
        try:
            y, sr = librosa.load(file_path, sr=None, mono=False)
            
            if y.ndim == 1:
                y = np.expand_dims(y, axis=0)
            
            # Pre-calculate Whole Audio Fingerprint
            # We do this in 30-second chunks so it doesn't cause Out-Of-Memory errors on long tracks
            whole_fingerprints = []
            chunk_samples = sr * 30 
            for ch in range(y.shape[0]):
                fp_sum = np.zeros(self.n_fft // 2 + 1)
                total_frames = 0
                for start in range(0, y.shape[1], chunk_samples):
                    end = min(y.shape[1], start + chunk_samples)
                    D_chunk = librosa.stft(y[ch, start:end], n_fft=self.n_fft)
                    S_chunk = np.abs(D_chunk)
                    fp_sum += np.sum(S_chunk, axis=1)
                    total_frames += S_chunk.shape[1]
                whole_fingerprints.append(fp_sum / total_frames)
            
            self.audio_path = file_path
            self.full_y = y
            self.sr = sr
            self.whole_audio_fingerprints = whole_fingerprints
            
            self.root.after(0, self._load_complete, file_path)
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to load audio:\n{str(e)}"))
            self.root.after(0, lambda: self.btn_load.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.lbl_file.config(text="Load failed", fg="red"))

    def _load_complete(self, file_path):
        display_name = file_path.split("/")[-1]
        if '\\' in display_name:
            display_name = display_name.split("\\")[-1]

        self.lbl_file.config(text=display_name, fg="black")
        self.btn_load.config(state=tk.NORMAL)
        self.btn_save.config(state=tk.NORMAL)
        self.start_preview_recalc()

    # --- STFT Recalculation (Threaded) ---

    def start_preview_recalc(self):
        if self.full_y is None:
            return
        
        self.stft_request_id += 1
        current_id = self.stft_request_id
        
        self.ax_spec.set_title("Recalculating STFT segment...", color='#FFD700')
        self.ax_line.set_title("Recalculating...", color='#FFD700')
        self.canvas.draw_idle()
        
        threading.Thread(target=self._calc_stft_thread, args=(current_id,), daemon=True).start()

    def _calc_stft_thread(self, request_id):
        try:
            length_s = self.var_len.get()
            pos_pct = self.var_pos.get() / 100.0

            total_samples = self.full_y.shape[1]
            length_samples = int(length_s * self.sr)
            
            center_idx = int(total_samples * pos_pct)
            start_idx = max(0, center_idx - length_samples // 2)
            end_idx = min(total_samples, start_idx + length_samples)
            
            if end_idx - start_idx < length_samples:
                start_idx = max(0, end_idx - length_samples)

            # We calculate local fingerprints for all channels to make sure saving 
            # with "Preview Region" mode works flawlessly across stereo.
            preview_fingerprints = []
            S_mag_display = None
            
            for ch in range(self.full_y.shape[0]):
                y_prev_ch = self.full_y[ch, start_idx:end_idx]
                D_ch = librosa.stft(y_prev_ch, n_fft=self.n_fft)
                S_mag_ch = np.abs(D_ch)
                preview_fingerprints.append(np.mean(S_mag_ch, axis=1))
                
                # Channel 0 is always our visual display
                if ch == 0:
                    S_mag_display = S_mag_ch

            freqs = librosa.fft_frequencies(sr=self.sr, n_fft=self.n_fft)
            
            start_time = start_idx / self.sr
            times = librosa.frames_to_time(np.arange(S_mag_display.shape[1]), sr=self.sr, n_fft=self.n_fft) + start_time

            self.root.after(0, self._apply_recalc, request_id, S_mag_display, freqs, times, length_s, pos_pct, preview_fingerprints)
        except Exception as e:
            print(f"STFT Thread Error: {e}")

    def _apply_recalc(self, request_id, S_mag, freqs, times, length_s, pos_pct, preview_fingerprints):
        if request_id != self.stft_request_id:
            return
            
        self.preview_S_mag = S_mag
        self.preview_freqs = freqs
        self.preview_times = times
        self.preview_fingerprints = preview_fingerprints
        
        self._current_title = f"Preview: {length_s}s at {pos_pct*100:.1f}%"
        self.update_plot()

    # --- DSP Application & Rendering ---

    def apply_dsp_to_magnitude(self, S_mag, full_fingerprint):
        fmin = self.vars["fmin"].get()
        fmax = self.vars["fmax"].get()
        context_hz = self.vars["context"].get()
        tension = self.vars["tension"].get()
        resolution = self.vars["resolution"].get()
        feather_hz = self.vars["feather"].get()

        if fmin >= fmax:
            return S_mag

        idx_start = np.argmin(np.abs(self.preview_freqs - fmin))
        idx_end = np.argmin(np.abs(self.preview_freqs - fmax))
        
        bin_resolution = self.sr / self.n_fft
        context_bins = int(context_hz / bin_resolution)

        context_bins = max(1, min(idx_start, context_bins))
        if idx_end + context_bins >= len(self.preview_freqs):
            context_bins = len(self.preview_freqs) - 1 - idx_end

        if idx_start >= idx_end or context_bins <= 0:
            return S_mag

        S_mag_processed = S_mag.copy()

        # Trajectories
        x_pre = np.arange(idx_start - context_bins, idx_start)
        if len(x_pre) > 1:
            poly_pre = np.polyfit(x_pre, full_fingerprint[x_pre], 1)
            val_start = np.polyval(poly_pre, idx_start)
            slope_start = poly_pre[0] * tension
        else:
            val_start = full_fingerprint[idx_start]
            slope_start = 0

        x_post = np.arange(idx_end, idx_end + context_bins)
        if len(x_post) > 1:
            poly_post = np.polyfit(x_post, full_fingerprint[x_post], 1)
            val_end = np.polyval(poly_post, idx_end)
            slope_end = poly_post[0] * tension
        else:
            val_end = full_fingerprint[idx_end]
            slope_end = 0

        # Spline
        try:
            spline = CubicHermiteSpline([idx_start, idx_end], [val_start, val_end], [slope_start, slope_end])
            x_target = np.arange(idx_start, idx_end)
            target_fingerprint = np.maximum(spline(x_target), 1e-10)
        except ValueError:
            return S_mag

        original_gap_fingerprint = full_fingerprint[idx_start:idx_end]

        # Resolution quantization
        if resolution > 0:
            num_bins = len(target_fingerprint)
            actual_res = min(resolution, num_bins)
            chunks = np.array_split(np.arange(num_bins), actual_res)
            
            stepped_gain = np.zeros(num_bins)
            
            for chunk_indices in chunks:
                if len(chunk_indices) > 0:
                    mean_target = np.mean(target_fingerprint[chunk_indices])
                    mean_original = np.mean(original_gap_fingerprint[chunk_indices])
                    stepped_gain[chunk_indices] = mean_target / (mean_original + 1e-10)
            
            # Use a Gaussian filter to smoothly blur the hard steps. 
            # This perfectly simulates the overlapping bell-curves (Q-factor) of an analog graphic EQ.
            chunk_size = num_bins / actual_res
            gain_curve = gaussian_filter1d(stepped_gain, sigma=chunk_size * 0.75, mode='nearest')
        else:
            gain_curve = target_fingerprint / (original_gap_fingerprint + 1e-10)

        # --- FEATHERING LOGIC ---
        feather_bins = int(feather_hz / bin_resolution)
        if feather_bins > 0:
            # Prevent feathering from crossing over the middle
            feather_bins = min(feather_bins, len(gain_curve) // 2)
            
            if feather_bins > 0:
                fade_in = np.linspace(0.0, 1.0, feather_bins)
                fade_out = np.linspace(1.0, 0.0, feather_bins)
                
                blend_mask = np.ones(len(gain_curve))
                blend_mask[:feather_bins] = fade_in
                blend_mask[-feather_bins:] = fade_out
                
                # Blend between 1.0 (no change) at the edges to the target gain_curve internally
                gain_curve = gain_curve * blend_mask + 1.0 * (1.0 - blend_mask)

        S_mag_processed[idx_start:idx_end, :] *= gain_curve[:, np.newaxis]
        return S_mag_processed

    def update_plot(self):
        if self.preview_S_mag is None or self.preview_fingerprints is None:
            return

        # Figure out which fingerprint source to base our EQ curve on
        fingerprint_mode = self.fingerprint_mode_var.get()
        if fingerprint_mode == "whole" and getattr(self, 'whole_audio_fingerprints', None) is not None:
            fp = self.whole_audio_fingerprints[0]
        else:
            fp = self.preview_fingerprints[0]

        # 1. Apply DSP using the chosen fingerprint
        S_processed = self.apply_dsp_to_magnitude(self.preview_S_mag, fp)
        
        # 2. Calculate Viewport Bounds
        fmin = self.vars["fmin"].get()
        fmax = self.vars["fmax"].get()
        context = self.vars["context"].get()
        
        y_bottom = max(0, fmin - (context * 2))
        y_top = min(self.sr / 2, fmax + (context * 2))

        # OPTIMIZATION: Find the exact array indices for our visible viewport
        idx_bot = np.argmin(np.abs(self.preview_freqs - y_bottom))
        idx_top = np.argmin(np.abs(self.preview_freqs - y_top))
        if idx_bot >= idx_top:
            idx_top = idx_bot + 1 # Prevent 0-height arrays

        # Crop the data BEFORE converting to Decibels to massively speed up rendering
        active_S = S_processed if self.show_after_var.get() else self.preview_S_mag
        active_S_crop = active_S[idx_bot:idx_top, :]
        S_db_crop = librosa.amplitude_to_db(active_S_crop, ref=np.max)

        self.ax_spec.clear()
        self.ax_line.clear()
        
        # --- 1. Draw Spectrogram (Optimized) ---
        # Using imshow instead of pcolormesh is dramatically faster for uniform grids
        extent = [self.preview_times[0], self.preview_times[-1], self.preview_freqs[idx_bot], self.preview_freqs[idx_top]]
        self.ax_spec.imshow(S_db_crop, origin='lower', aspect='auto', cmap='magma', extent=extent)
        
        self.ax_spec.set_ylim(y_bottom, y_top)
        
        base_title = getattr(self, '_current_title', "Preview Segment")
        state_text = "AFTER Smoothing" if self.show_after_var.get() else "BEFORE Smoothing"
        self.ax_spec.set_title(f"{base_title} | {state_text}", color='white')
        
        self.ax_spec.set_ylabel("Frequency (Hz)")
        self.ax_spec.set_xlabel("Time (s)")

        self.ax_spec.axhline(fmin, color='cyan', linestyle='--', alpha=0.5)
        self.ax_spec.axhline(fmax, color='cyan', linestyle='--', alpha=0.5)
        
        # --- 2. Draw Spectral Analyzer ---
        # Crop data before calculating means to save performance
        orig_mean = np.mean(self.preview_S_mag[idx_bot:idx_top, :], axis=1)
        proc_mean = np.mean(S_processed[idx_bot:idx_top, :], axis=1)
        
        # Keep the reference identical so the dB scales match perfectly
        ref_val = np.max(orig_mean)
        orig_db = librosa.amplitude_to_db(orig_mean, ref=ref_val)
        proc_db = librosa.amplitude_to_db(proc_mean, ref=ref_val)
        
        freqs_crop = self.preview_freqs[idx_bot:idx_top]
        
        # Swap colors depending on what we are currently viewing
        if self.show_after_var.get():
            c_orig, w_orig, a_orig = 'gray', 1, 0.5
            c_proc, w_proc, a_proc = '#32CD32', 2, 1.0
        else:
            c_orig, w_orig, a_orig = '#32CD32', 2, 1.0
            c_proc, w_proc, a_proc = 'gray', 1, 0.5

        self.ax_line.plot(orig_db, freqs_crop, color=c_orig, linewidth=w_orig, alpha=a_orig, label='Original')
        self.ax_line.plot(proc_db, freqs_crop, color=c_proc, linewidth=w_proc, alpha=a_proc, label='Smoothed')
        
        self.ax_line.set_ylim(y_bottom, y_top)
        
        # Dynamically scale X-axis
        all_db = np.concatenate([orig_db, proc_db])
        self.ax_line.set_xlim(np.min(all_db) - 2, np.max(all_db) + 5)
            
        self.ax_line.set_title("Average Spectrum", color='white')
        self.ax_line.set_xlabel("Magnitude (dB)")
        
        self.ax_line.axhline(fmin, color='cyan', linestyle='--', alpha=0.5)
        self.ax_line.axhline(fmax, color='cyan', linestyle='--', alpha=0.5)
        
        self.ax_line.legend(loc='upper right', facecolor='#2b2b2b', edgecolor='none', labelcolor='white')

        self.canvas.draw_idle()

    # --- Save Execution ---

    def save_audio(self):
        if self.full_y is None or not self.audio_path:
            return

        init_dir, init_file = os.path.split(self.audio_path)
        base_name, ext = os.path.splitext(init_file)
        
        # soundfile cannot write mp3/m4a natively, fallback to wav for unsupported formats
        if ext.lower() not in ['.wav', '.flac', '.ogg']:
            ext = '.wav'
            
        default_file = f"{base_name}-smoothen{ext}"

        output_path = filedialog.asksaveasfilename(
            initialdir=init_dir,
            initialfile=default_file,
            defaultextension=ext,
            filetypes=[(f"Audio File", f"*{ext}"), ("WAV files", "*.wav")],
            title="Save Processed Audio"
        )
        if not output_path:
            return

        self.btn_save.config(state=tk.DISABLED, text="Processing...")
        self.root.update()

        threading.Thread(target=self._process_and_save, args=(output_path,), daemon=True).start()

    def _process_and_save(self, output_path):
        try:
            processed_channels = []
            fingerprint_mode = self.fingerprint_mode_var.get()
            
            for ch_idx in range(self.full_y.shape[0]):
                y_ch = self.full_y[ch_idx]
                
                D = librosa.stft(y_ch, n_fft=self.n_fft)
                S_mag = np.abs(D)
                S_phase = np.angle(D)

                # Fetch the correct fingerprint context for the current channel
                if fingerprint_mode == "whole" and getattr(self, 'whole_audio_fingerprints', None) is not None:
                    fp = self.whole_audio_fingerprints[ch_idx]
                else:
                    fp = self.preview_fingerprints[ch_idx]

                S_mag_processed = self.apply_dsp_to_magnitude(S_mag, fp)
                
                D_smoothed = S_mag_processed * np.exp(1j * S_phase)
                y_ch_smoothed = librosa.istft(D_smoothed)
                processed_channels.append(y_ch_smoothed)

            y_smoothed = np.vstack(processed_channels)
            sf.write(output_path, y_smoothed.T, self.sr)
            
            self.root.after(0, lambda: messagebox.showinfo("Success", f"File saved successfully to:\n{output_path}"))

        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to save audio:\n{str(e)}"))
        finally:
            self.root.after(0, lambda: self.btn_save.config(state=tk.NORMAL, text="Apply & Save Audio"))

if __name__ == "__main__":
    if HAS_DND:
        root = TkinterDnD.Tk()
    else:
        root = tk.Tk()
        print("Note: To enable drag-and-drop support, please run: pip install tkinterdnd2")
        
    app = SpectralSmootherApp(root)
    root.mainloop()
