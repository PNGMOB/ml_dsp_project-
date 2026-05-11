""" 
config.py - Control room

If you want to experiment, THIS is the first file to edit.
Changing a value here changes behaviour everywhere else automatically.
This is more like the settings menu 
"""

from dataclasses import dataclass, field
from pathlib import Path

# Audio Settings
@dataclass
class AudioConfig:
  """Controls how audio is captured and sliced into frames.""" 

  sample_rate: int = 22_050
  frame_length: int = 2048
  hop_length: int = 512
  n_fft: int = 2048
  n_mels: int = 128
  silence_threshold: float = 0.01

# Model Settings
@dataclass

class ModelConfig:
  """
  Controls the shape of the neural network.
  Our network is a Multi-Layer Perceptron (MLP):

    Input (1025 freq bins)
      |
    [Linear -> BatchNorm -> ReLU -> Dropout]  <- Layer 1
      |
    [Linear -> BatchNorm -> ReLU -> Dropout]  <- Layer 2
      |
    [Linear -> BatchNorm -> ReLU -> Dropout]  <- Layer 3
      |
    Output (88 piano-key probabilities)

  Each layer squashes the data through fewer and fewer neurons,
  forcing it to learn the most important patterns of the data.
  """

  input_size: int = 1025
  hidden_sizes: tuple = (512, 256, 128) # No. of neurons in each hidded layer
  n_classes: int = 88 # One output per key (A0 -> C8)
  dropout_rate: float = 0.25 # Prevents model from overfitting

# Training Settings
@dataclass
class TrainingConfig:
  """Controls how the model learns from data."""

  epochs: int = 60 
  # full passes through dataset. 
  #More epochs = more learning (to a point)
  batch_size: int = 128
  # Samples processed before updating weights.
  #Larger = more stable gradients, more memory needed.
  learning_rate: float = 1e-3
  # How big each learning step is (0.001)
  # Too high -> overshoots; too low -> learns painfully slow
  weight_decay: float = 1e-4
  # L2 regularisation - penalises very large weights.
  # Another tool to reduce overfitting.
  lr_patience: int = 8
  # If validation loss doesn't improve for this many epochs,
  # automatically reduce the learning rate.
  early_stop_patience: int = 15
  # Stops training early if no improvement for this many epochs.
  # Saves time and prevents overfitting.
  val_split: float = 0.15
  # Percentage of data to use for validation.
  samples_per_note: int = 300
  # Synthetic training examples generated per key.
  # 300 x 88 keys = 26,400 total training samples.

# File Paths
@dataclass
class PathConfig:
  """Where files are saved and loaded from."""

  model_dir: Path = Path("models")
  data_dir: Path = Path("data")

  @property
  def model_path(self) -> Path:
    return self.model_dir / "pitch_model.pth"

  @property
  def data_path(self) -> Path:
    return self.data_dir / "synthetic_data.npz"

# Musical Theory Constants
# Piano key frequencies: A0 = 27.5 Hz, each semitone = x 2^(1/12)
# Key index 0 -> A0 (27.5 Hz), index 87 -> C8 (4186 Hz)
MIN_FREQ_HZ: float = 27.5     # A0
MAX_FREQ_HZ: float = 4186.0   # C8

# The 12 semitone names in one octave, starting from A
# Index 0 = A, 1 = A#/B♭, 2 = B, 3 = C, ...
NOTE_NAMES_FROM_A = [
    "A", "A#/B♭", "B", "C", "C#/D♭", "D", 
    "D#/E♭", "E", "F", "F#/G♭", "G", "G#/A♭" 
]

#Solfege syllables - fixed-Do system (C is always Do)
# Index matches semitone offset from C:
# C=0, C#/D♭=2, D=2, D#/E♭=3, E=4, F=5,
# F#/G♭=6, G=7, G#/A♭=8, A=9, A#=10, B=11
SOLFEGE_FROM_C = [
    "Do", "Do#/Re♭", "Re", "Re#/Mi♭", "Mi", "Fa",
    "Fa#/Sol♭", "Sol", "Sol#/La♭", "La", "La#/Si♭", "Si"
]

#Emoji to make terminal output friendlier
NOTE_EMOJI = {
    "C": "🎹", "D": "🎸", "E": "🎺", "F": "🎻",
    "G": "🥁", "A": "🎷", "B": "🪗",
}

