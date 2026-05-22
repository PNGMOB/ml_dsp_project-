"""
features.py - The Ears of the System
=====================================
Raw audio (a list of numbers representing air pressure changes) is 
not directly useful for a neural network. We first need to extract
FEATURES - a compact, information-rich representation.

This module implements the same feature pipeline used by SwiftF0:
  1. Apply a window function to reduce spectral leakage
  2. Compute the Fast Fourier Transform (FFT)
  3. Take the magnitude spectrum (throw away phase)
  4. Normalise to [0, 1] so the model isn't thrown off by volume

Concept - Why FFT?
------------------
The FFT answers the question: "Which frequencies are present in this
short chunk of sound, and how loud are they?"
  
  Time domain: a messy squggle of air pressure samples
  Frequency domain: a bar chart showing loudness at frequency

  440 Hz sine wave (A4):
  Time: ~~~~sinusoidal~~~~
  FFT:  0 ... 440Hz ...
            ↑ one tall spike

  A4 + E5 chord:
  FFT:  0 ... 440Hz ... 659Hz ...
            ↑ spike   ↑ spike

The neural network learns to read this bar chart and predict pitch
"""
import numpy as np
from scipy.signal import get_window
!git clone https://github.com/PNGMOB/ml_dsp_project-.git
import sys
sys.path.insert(0, '/content/ml_dsp_project-')
import config
from config import AUDIO

# ---------------------------
# WINDOW FUNCTIONS
# ---------------------------
def make_hann_window(length: int) -> np.ndarray:
  """
  Create a Hann (Hanning) window of the given length.

  Problem: When we cut out a short frame of audio, we get sharp 
           edges at the start and end. These sharp edges create 
           artificial high frequencies in the FFT (spectral leakage).

  Solution: Multiply the frame by a smooth bell-shaped window that
            fades to zero at both ends. This reduces leakage.

  The Hann window formula:
    w[n] = 0.5 * (1 - cos(2 * pi * n / (N - 1)))

  Args:
    Length: Number of samples in the window

  Returns:
    1-D numpy array of window cofficients (values 0 -> 1 -> 0)
  """
  return get_window("hann", length, fftbins=True).astype(np.float32)
  
# Pre-build the window once so we don't recreate it on every frame call
_HANN_WINDOW = make_hann_window(AUDIO.frame_length)

# ------------------------------------
# FRAME EXTRACTION
# ------------------------------------
def extract_frames(audio: np.ndarray, frame_length:int, hop_length: int) -> np.ndarray:
  """
  Slice a 1-D audio array into overlapping frames.

  This is the "sliding window" operation described in the config.

  Args:
    audio: 1-D float32 array of audio samples
    frame_length: Number of samples per frame
    hop_length: Step size between consecutive frames

  Returns:
    2-D array of shape (n_frames, frame_length)
    Each row is one analysis frame.

  Example:
    audio = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], frame=4, hop=2
    frame 0: [1, 2, 3, 4]
    frame 1: [3, 4, 5, 6]
    frame 2: [5, 6, 7, 8]
    frame 3: [7, 8, 9, 10]
  """
  if len(audio) <= frame_length:
   # Pad short audio with zeros
   audio = np.pad(audio, (0, frame_length - len(audio)))

  # Use stride tricks for memory-efficient framing
  n_frames = 1 + (len(audio) - frame_length) // hop_length
  shape = (n_frames, frame_length)
  strides = (audio.strides[0] * hop_length, audio.strides[0])
  return np.lib.stride_tricks.as_strided(audio, shape=shape, strides=strides).copy()

# -------------------------------
# FFT MAGNITUDE SPECTRUM
# -------------------------------
def frame_to_spectrum(frame: np.ndarray) -> np.ndarray:
  """
  Convert a single audio frame -> magnitude spectrum

  Steps:
    1. Apply Hann window  (reduce spectral leakage)
    2. Zero-pad to n_fft  (optional - already same size here)
    3. Compute FFT        (frequency decomposition)
    4. Take magnitude     |FFT| = √(real^2 + imag^2)
    5. Keep only the first half (FFT is symmetric for real signals)

  Args:
    frame: 1-D array of length frame_length

  Returns:
    1-D array of length (n_fft // 2 + 1) = 1025 magnitude values

  Note:
    The frequencies represented by the output bins are:
      bin k -> freq = k x sample_rate / n_fft
      bin 0 -> 0 Hz (DC)
      bin 512 -> 5512.5 Hz
      bin 1024 -> 11_025 Hz (Nyquist limit)
  """
  # 1. Apply window to reduce edge effects
  windowed = frame[:AUDIO.frame_length] * _HANN_WINDOW

  # 2. FFT - produces complex numbers
  fft_complex = np.fft.rfft(windowed, n=AUDIO.n_fft)

  # 3. Magnitude = distance from origin in complex plane
  magnitued = np.abs(fft_complex).astype(np.float32)

  return magnitued  # shape: (n_fft // 2 + 1,) = (1025,)

def normalise_spectrum(spectrum: np.ndarray) -> np.ndarray:
  """
  Normalise a magnitude spectrum to range [0, 1].

  Neural networks work best when inputs are small numbers around 0-1.
  A raw FFT spectrum can have values ranging from 0 to 10 000+,
  which makes training unstable.

  We use log-magnitude normalisation (inspired by SwiftF0):
    1. Add a tiny constant to avoid log(0)
    2. Take log - this compresses the large dynamic range
    3. Shift and scale to [0, 1]

  Args:
    spectrum: 1-D magnitude spectrum array

  Returns:
    Normalised 1-D float32 array with values in [0, 1]
  """
  # Log compression - makes quiet harmonics visible
  log_spec = np.log1p(spectrum) # log(1 + x)  avoids log(0)

  # Min-max normalise to [0, 1]
  spec_min = log_spec.min()
  spec_max = log_spec.max()

  if spec_max - spec_min < 1e-8:
    # Nearly silent frame - return zeros
    return np.zeros_like(log_spec)

  return (log_spec - spec_min) / (spec_max - spec_min)

# ---------------------------------
# VOICE ACTIVITY DETECTION
# ---------------------------------
def compute_rms(frame: np.ndarray) -> float:
  """
  Compute Root Mean Square (RMS) energy of an audio frame.

  RMS is the "effective volume" of the frame.
  Low RMS = silence or near-silence

  Args:
    frame: 1-D audio frame

  Returns:
    RMS energy as a float (0.0 = silence)
  """
  return float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)))


def is_voiced(frame: np.ndarray, threshold: float = None) -> bool:
  """
  Return True if the frame contains enough energy to contain a pitch.

  We don't want to run pitch detection on silence - it would output 
  garbage. This simple check filters those frames out.

  Args:
    frame:     Audio fram array
    threshold: Minimum RMS to be considered voiced
               (defaults to AUDIO.silence_threshold)

  Returns:
    True if the frame is likely to contain a pitched sound
  """
  if threshold is None:
    threshold = AUDIO.silence_threshold
  return compute_rms(frame) >= threshold

# --------------------------
# COMBINED PIPELINE
# --------------------------
def audio_to_features(audio_frame: np.ndarray) -> np.ndarray | None:
  """
  Full pipeline: raw audio frame -> normalised feature vector.

  This is the single funtion called by both the data generator
  (training) and the real-time detector (inference).

  Args:
    audio_frame: 1-D float32 array of 'frame_length' samples

  Returns:
    Normalised feature vector of shape (1025,),
    OR None if the frame is too quiet (silence)
  """
  if not is_voiced(audio_frame):
    return None

  spectrum = frame_to_spectrum(audio_frame)
  normalised = normalise_spectrum(spectrum)
  return normalised

def compute_spectral_centroid(spectrum: np.ndarray) -> float:
  """
  Compute the spectral centroid of a magnitude spectrum.

  Higher centroid -> brighter, more treble-heavy sound.
  Lower centroid -> darker, more bass-heavy sound.

  Args:
    spectrum: Magnitude spectrum array

  Returns:
    Centroid frequency in Hz
  """
  n_bins = len(spectrum)
  freqs = np.linspace(0, AUDIO.sample_rate / 2, n_bins)
  total = spectrum.sum()
  if total < 1e-8:
    return 0.0
  return float(np.dot(freqs, spectrum) / total)


# -----------------------
# QUICK SELF-TEST
# -----------------------
if __name__ == "__main__":
  print("=" * 50)
  print(" Feature Extractor - Self Test")
  print("=" * 50)

  sr = AUDIO.sample_rate
  dur = AUDIO.frame_length / sr
  t = np.linspace(0, dur, AUDIO.frame_length, endpoint=False)
  
  # Create a pure 440 Hz sine wave (A4)
  test_signal = np.sin(2 * np.pi * 440 * t).astype(np.float32)
  features = audio_to_features(test_signal)

  print(f"  Input shape:  {test_signal.shape}")
  print(f"  Output shape: {features.shape}")
  print(f"  Output min:   {features.min():.4f}")
  print(f"  Output max:   {features.max():.4f}")

  # Find the peak frequency bin
  spectrum = frame_to_spectrum(test_signal)
  peak_bin = np.argmax(spectrum)
  peak_hz = peak_bin * sr / AUDIO.n_fft
  print(f"\n  A4 = 440 Hz sine wave")
  print(f"  Detected peak bin: {peak_bin} -> {peak_hz:.1f} Hz (expected ~440 Hz)")
  
  # Test silence detection
  silence = np.zeros(AUDIO.frame_length, dtype=np.float32)
  print(f"\n Silence voiced? {is_voiced(silence)} (expected False)")
  print(f" Tone voiced? {is_voiced(test_signal)} (expected True)")
  print("=" * 50)
