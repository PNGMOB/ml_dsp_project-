"""
data_generator.py - The Training Data Factory
=============================================
ML models need examples to learn from. For pitch detection we
don't need to record thousands of hours of music - we can generate
perfectly labelled synthetic tones mathematically.

This is inspired by how SwiftF0 and CREPE generate training data:
  - Pure sine waves at every piano key frequency
  - Harmonic tones (realistic instrument-like sounds)
  - Added noise and pitch drift (data augmentation)
  - Labelled with the exact piano key class

Why synthetic data works here:
------------------------------
Pitch detection is fundamentally a mathematical problem. If we give
the model enough varied examples of each frequency, it will generalise
to real instruments too - because real instruments are just a mix of
the same sine waves (harmonics) at integer multiples of the fundamental.

Data pipeline overview:
  For each of the 88 piano keys:
    1. Get the exact frequency (Hz)
    2. Generate N synthetic audio frames with variations:
        - Pure sine wave
        - Harmoni tone (sine + overtones)
        - With random noise
        - With slight pitch wobble (vibrato)
    3. Extract features (FFT spectrum) from each frame
    4. Assign label = piano key index

  Total: 88 keys x 300 samples = 26 400 labelled examples
"""

import numpy as np
from pathlib import Path
from tqdm import tqdm # Progress bars - great for neurodivergent focus feedback

import os
repo_dir = 'ml_dsp_project-'
repo_path = f'/content/{repo_dir}'

# Clone repository only if it doesn't already exist
if not os.path.exists(repo_path):
    print(f"Cloning repository into '{repo_dir}'...")
    !git clone https://github.com/PNGMOB/ml_dsp_project-.git
else:
    print(f"Directory '{repo_dir}' already exists. Skipping git clone.")

# Always ensure problematic lines are removed from the files for clean import
# This handles cases where the repo was cloned but cleaning was skipped in a previous run.
print("Ensuring problematic lines are removed from features.py and note_converter.py...")

# Remove the problematic line (line 35) from features.py
if os.path.exists(f'{repo_path}/features.py'):
    !sed -i '35d' {repo_path}/features.py
else:
    print(f"Warning: {repo_path}/features.py not found. Skipping sed for features.py.")

# Remove the problematic line (line 29) from note_converter.py
if os.path.exists(f'{repo_path}/note_converter.py'):
    !sed -i '29d' {repo_path}/note_converter.py
else:
    print(f"Warning: {repo_path}/note_converter.py not found. Skipping sed for note_converter.py.")


import sys
sys.path.insert(0, repo_path)

import config
import features
import note_converter

from config import AUDIO, TRAINING, PATHS
from features import audio_to_features
from note_converter import PIANO_FREQUENCIES

# --------------------------------
# TONE SYNTHESIS
# --------------------------------
def synthesise_pure_tone(freq_hz: float, n_samples: int, sample_rate: int) -> np.ndarray:
  """
  Generate a pure sine wave at the given frequency.

  A pure tone is the simplest possible pitced sound -
  just one frequency, nothing else.

    signal[t] = sin(2 * pi * freq * t)

  Args:
    freq_hz:     Fundamental frequency in Hz
    n_samples:   Number of samples to generate
    sample_rate: Sample rate in Hz
  Returns:
    1-D float32 array of audio samples in range [-1, 1]
  """
  t = np.linspace(0, n_samples / sample_rate, n_samples, endpoint=False)
  return np.sin(2 * np.pi * freq_hz * t).astype(np.float32)

def synthesise_harmonic_tone(
    freq_hz: float,
    n_samples: int,
    sample_rate: int,
    n_harmonics: int = 8,
    decay: float = 0.65
) -> np.ndarray:
  """
  Generate a harmonic tone - a realistic instrument-like sound.

  Real instruments don't produce pure sine waves. A guitar string
  vibrates at 440 Hz also vibrates at 880, 1320, 1760 Hz, etc.
  These are called Harmonics or Overtones.

  The fundamental (lowest) frequency defines the note name.
  The mix of harmonics defines the timbre ("the color" of the sound
  - why a guitar and a piano sound different playing the same note).

  Formula:
    signal = sum (decay^k) * sin(2 * pi * freq * k * t)
          k = 1 to n_harmonics

  Args:
    freq_hz:     Fundamental frequency in Hz
    n_samples:   Number of samples to generate
    sample_rate: Sample rate in Hz
    n_harmonics: Number of harmonics to include
    decay:       Each harmonic is this fraction as loud as the previous
                (0.65 -> natural instrument-like decay)
  Returns:
    1-D float32 normalised array of audio samples in range [-1, 1]
  """
  t = np.linspace(0, n_samples / sample_rate, n_samples, endpoint=False)
  signal = np.zeros(n_samples, dtype=np.float64)

  for k in range(1, n_harmonics + 1):
    harmonic_freq = freq_hz * k
    # Stop adding harmonics beyond Nyquist limit (would alias)
    if harmonic_freq >= sample_rate / 2:
      break
    amplitude = decay ** (k - 1)
    signal += amplitude * np.sin(2.0 * np.pi * harmonic_freq * t)

  # Normalise to prevent clipping
  max_val = np.abs(signal).max()
  if max_val > 1e-8:
    signal /= max_val

  return signal.astype(np.float32)


def add_noise(signal: np.ndarray, snr_db: float = 20.0) -> np.ndarray:
  """
  Add Gaussian white noise at a given Signal-to-Noise Ratio (SNR).

  SNR controls how "clean" the signal is:
    30 dB -> very clean (barely any noise)
    20 dB -> slightly noisy (like a quiet room)
    10dB -> noticeably noisy (like a busy street)

  Args:
    signal: Clean audio signal
    snr_db: Target SNR in decibels

  Returns:
    Noisy signal (same shape as input)
  """
  signal_power = np.mean(signal ** 2)
  # SNR in linear scale
  snr_linear = 10.0 ** (snr_db / 10.0)
  # Noise power
  noise_power = signal_power / snr_linear
  noise = np.random.normal(0, np.sqrt(noise_power), size=signal.shape).astype(np.float32)
  return signal + noise

def add_pitch_jitter(freq_hz: float, max_cents: float = 15.0) -> float:
  """
  Randomly shift a frequency by a small amount (pitch jitter).

  Real singers and instruments are never perfectly in tune.
  This augmentation teaches the model to handle slight detuning.

  1 semitone = 100 cents, so 15 cents = one-sixth of a semitone.

  Args:
    freq_hz:    Original frequency
    max_cents:  Maximum deviation in cents

  Returns:
    Slightly shifted frequency in Hz
  """
  cents_shift = np.random.uniform(-max_cents, max_cents)
  return freq_hz * (2.0 ** (cents_shift / 1200.0))

def apply_amplitude_envelope(signal: np.ndarray) -> np.ndarray:
  """
  Apply a simple attack-sustain-release (ASR) amplitude envelope.

  Instead of a flat amplitude throughout, this makes the tone fade
  in and out, like a real note being played.

  ASR shape:
          ┌──── sustain ────┐
       /                   \\
      / attack       release\\
  """
  n = len(signal)
  attack = int(n * 0.08)  #8% fade in
  release = int (n * 0.12)  #12% fade out

  envelope = np.ones(n, dtype=np.float32)
  if attack > 0:
    envelope[:attack] = np.linspace(0, 1, attack)
  if release > 0:
    envelope[-release:] = np.linspace(1, 0, release)

  return signal * envelope

import numpy as np
from pathlib import Path
from tqdm import tqdm # Progress bars - great for neurodivergent focus feedback

import os

# -----------------------------------
# DATASET GENERATION
# -----------------------------------
def generate_sample_for_key(
    key_index: int,
    sample_rate: int = AUDIO.sample_rate,
    frame_length: int = AUDIO.frame_length,
) -> tuple[np.ndarray, int] | tuple[None, int]:
  """
  Generate ONE random training sample for a given piano key.

  Each call creates a slightly different version of the tone:
  different synthesis method, different noise level, slight pitch jitter.
  This variety prevents the model from over-fitting to one exact sound.

  Args:
    key_index:    Piano key index 0-87
    sample_rate:  Audio sample rate in Hz
    frame_length: No. of audio samples per frame

  Returns:
    Tuple of (feature_vector, label) where:
      feature_vector: np.ndarray of shape (1025,)
      label:          int (= key_index)
    OR (None, key_index) if feature extraction failed
  """
  base_freq = PIANO_FREQUENCIES[key_index]

  # Random pitch jitter: +/- 15 cents (subtle)
  freq = add_pitch_jitter(base_freq, max_cents=15.0)

  # Randomly choose synthesis method
  method = np.random.choice(["pure", "harmonic", "harmonic_rich"])

  if method == "pure":
    signal = synthesise_pure_tone(freq, frame_length, sample_rate)

  elif method == "harmonic":
    signal = synthesise_harmonic_tone(freq, frame_length, sample_rate,
                                      n_harmonics=5, decay=0.7)
  else: # harmonic_rich
    signal = synthesise_harmonic_tone(freq, frame_length, sample_rate,
                                      n_harmonics=10, decay=0.55)

    # Apply amplitude envelope -50% of the time
    if np.random.random() > 0.5:
      signal = apply_amplitude_envelope(signal)

    # Add noise at a random SNR between 15 and 35 dB
    snr = np.random.uniform(15.0, 35.0)
    signal = add_noise(signal, snr_db=snr)

  # Extract features (FFT spectrum) for all methods
  features = audio_to_features(signal)

  return features, key_index

def generate_dataset(
    samples_per_key: int = TRAINING.samples_per_note,
    save_path: Path = PATHS.data_path,
    force_regenerate: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
  """
  Generate the full synthetic training dataset.

  Creates 'samples_per_key' samples for each of the 88 piano keys.
  The result is cached to disk so you don't need to regenerate every run.

  Args:
    samples_per_key:  Training samples per piano key (default: 300)
    save_path:        Where to save/load the cached dataset
    force_regenerate: If True, regenerate even if cache exists

  Returns:
    Tuple of:
      X: float32 array of shape (n_samples, 1025)  - features
      y: int64 array of shape (n_samples,)         - labels (0-87)

  Progress bar example:
    Generating data: 100%|████████████| 88/88 [00:45<00:00]
        ✓ Dataset: 26 400 samples, 88 classes
  """
  save_path = Path(save_path)

  # --- Load from cache if available ---------------------
  if save_path.exists() and not force_regenerate:
    print(f"📂 Loading cached dataset from {save_path} ...")
    data = np.load(save_path)
    X, y = data["X"], data["y"]
    print(f"✓  Loaded {len(X):,} samples ({len(np.unique(y))} classes)\n")
    return X, y

  # -- Generate fresh ---------------------------------------
  print(f"\n🎵 Generating synthetic training data")
  print(f"  {samples_per_key} samples x 88 keys = "
        f"{samples_per_key * 88:,} total examples\n")

  all_features: list[np.ndarray] = []
  all_labels:   list[int]        = []
  skipped = 0

  for key_idx in tqdm(range(88), desc='Generating data', unit="key", ncols=70):
    for _ in range(samples_per_key):
      features, label = generate_sample_for_key(key_idx)
      if features is not None:
        all_features.append(features)
        all_labels.append(label)
      else:
        skipped += 1

  X = np.array(all_features, dtype=np.float32)
  y = np.array(all_labels, dtype=np.int64)

  # -- Shuffle the dataset -----------------------------------------
  # Mix all keys together so the model doesn't just memorise the order
  shuffle_idx = np.random.permutation(len(X))
  X, y = X[shuffle_idx], y[shuffle_idx]

  # -- Save to disk -------------------------------------------------
  save_path.parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(save_path, X=X, y=y)

  print(f"\nn✓  Generated {len(X):,} samples (skipped{skipped} silent frames)")
  print(f"✓ Saved to {save_path}\n")
  return X, y

# ─────────────────────────────────────────────
#  QUICK SELF-TEST
# ─────────────────────────────────────────────
if __name__ == "__main__":
  print("=" * 50)
  print(" Data Generator - Self Test")
  print("=" * 50)

  # Test single sample generation
  feat, label = generate_sample_for_key(48) # A4
  print(f"\n Single sample for A4 (key 48):")
  print(f" Feature shape: {feat.shape}")
  print(f" Feature range: [{feat.min():.3f}, {feat.max():.3f}]")
  print(f" Label: {label}")

  # Test small dataset generation
  print("\n Generating small test dataset (10 per key) ...")
  X_test, y_test = generate_dataset(
      samples_per_key=10,
      save_path=Path("data/test_data.npz"),
      force_regenerate=True,
  )
  print(f"  X shape: {X_test.shape}")
  print(f"  y shape: {y_test.shape}")
  print(f"  Unique classes: {len(np.unique(y_test))}")
  print("=" * 50)
