"""
note_converter.py - The Music Theory Helper
--------------------------------------------
Translates numbers into human readable musical information:
note name, octave number and solfege syllable

This module has ZERO machine learning - it is pure maths and music theory.
Even if you have never studied music, the functions below are the only
nusic knowledge the project needs.


How pitch works
-------------------------------------------
Sound is vibrations. Pitch = vibration frequency (Hz).
The musical note A4 (concert pitch) vibrates at exactly 440Hz.

Notes repeat every octave (x2):
A3 = 220 Hz,  A4 = 440Hz,   A5 = 880Hz

Within each octave there are 12 equal semitones:
A   A#  B   C   C#  D   D#  E   F   F#  G   G# (then back to A)
0   1   2   3   4   5   6   7   8   9   10  11

Going up one semitone multiplies the frequenct by 2^(1/12) = 1.05946309
"""

import math
from dataclasses import dataclass
!git clone https://github.com/PNGMOB/ml_dsp_project-.git
import sys
sys.path.insert(0, '/content/ml_dsp_project-')
import config
from config import NOTE_NAMES_FROM_A, SOLFEGE_FROM_C, MIN_FREQ_HZ, NOTE_EMOJI

@dataclass
class NoteInfo:
  """
  A neat container for all the information about one musical note.

  Example:
    NoteInfo(
      freq_hz   = 440.0,
      note_name = "A",
      octave    = 4,
      full_name = "A4",
      solfege   = "La",
      midi_note = 69,
      cents_off = +3.2,
      piano_key = 48
    )
  """
  freq_hz   : float # raw frequency in Hz
  note_name : str
  octave    : int   # octave number
  full_name : str   # note_name + octave
  solfege   : str
  midi_note : int
  cents_off : float # tuning deviation in cents (±50 = half semitone)
  piano_key : int   # index into 88-key piano (0-87, A0=0)
  emoji     : str   # fun emoji for the note

from IPython.core.compilerop import code_name
# -------------------------------
# CORE CONVERSION
# -------------------------------

def hz_to_midi(freq_hz: float) -> float:
  """
  Convert a frequency in Hz to a continuous MIDI note number.

  MIDI 69 = A4 = 440 Hz exactly
  non-integer results mean the pitch is between two semitones.

  Formula derived from equal-tempered tuning:
    midi = 69 + 12 x log2(freq/440)

  Args:
    freq_hz: frequency in Hz (must be > 0)

  Returns:
    float MIDI note number (e.g. 69.0 for A4, 69.3 for slightly sharp A4)

  Example:
    >>> hz_to_midi(440.0)
    69.0
    >>> hz_to_midi(441.0)
    69.3
  """
  if freq_hz <= 0:
    raise ValueError(f"freq_hz must be positive, got {freq_hz}")
  return 69.0 + 12.0 * math.log2(freq_hz / 440.0)

def midi_to_hz(midi_note: float) -> float:
  """
  Convert a continuous MIDI note number to a frequency in Hz.
  This is the inverse of hz_to_midi().

  Args:
    midi_note: MIDI note number (can be fractional)

  Returns:
    float frequency in Hz (e.g. 440.0 for A4, 441.0 for slightly sharp A4)

  Example:
    >>> midi_to_hz(69.0)
    440.0
    >>> midi_to_hz(69.3)
    441.0

  Formula derived from equal-tempered tuning:
  freq = 440 x 2^(midi/12)
  """
  return 440.0 * 2.0 ** ((midi_note - 69.0) / 12.0)

def piano_key_to_hz(key_index: int) -> float:
  """
  Convert a piano key index (0-87) to a frequency in Hz..

  The piano starts at A0 = 27.5 Hz, so:
    key 0 -> MIDI 21 -> A0 -> 27.5 Hz
    key 48 -> MIDI 69 -> A4 -> 440.0 Hz
    key 87 -> MIDI 108 -> C8 -> 4186.0 Hz

  Args:
    key_index: Integer 0 to 87

  Returns:
    float frequency in Hz
  """
  midi = key_index + 21
  return midi_to_hz(midi)

def hz_to_note_info(freq_hz: float) -> NoteInfo:
  """
  The main funtion: turn a raw frequency into a NoteInfo object.

  This is what the rest of the code calls every time the model
  outputs a detected pitch.

  Steps:
    1. Convert Hz -> continuous MIDI number
    2. Round the nearest semitone -> integer MIDI
    3. Look up the note name and octave from MIDI
    4. Look up solfege from note name
    5. Calculate how far off-tune the note is (cents)
    6. Bundle everything into a NoteInfo object

  Args:
    freq_hz: detected frequency in Hz

  Returns:
    NoteInfo object with all musical metadata

  Example:
    >>> info = hz_to_note_info(442.0)
    >>> info.full_name
    'A4'
    >>> info.solfege
    'La'
    >>> info.cents_off  #sharp by -7.85
    7.85
  """
  # Step 1 - continuous MIDI position
  midi_float = hz_to_midi(freq_hz)

  # Step 2 - round to nearest semitone
  midi_int = round(midi_float)

  #Step 3 - derive note name and octave from MIDI
  # MIDI 21 = A0 (octave 0), MIDI 33 = A1 (octave 1), etc.
  # Octave calculation: MIDI 0 = C-1, so octave = (midi // 12) - 1
  octave = (midi_int // 12) - 1

  # Semitone within the octave (0=C, 1=C#, ... 11=B)
  semitone_from_c = midi_int % 12

  #Convert semitone_from_c to semitone_from_a for our NOTES_NAMES list
  # NOTE_NAMES_FROM_A starts at A, so we shift by 9 semitones
  semitone_from_a = (semitone_from_c - 9) % 12 # Corrected: Changed +9 to -9
  # Get note name from semitone_from_a
  note_name = NOTE_NAMES_FROM_A[semitone_from_a]

  # Step 4 - solfege uses semitone_from_c
  solfege = SOLFEGE_FROM_C[semitone_from_c]

  # Step 5 - how many cents off the perfect pitch?
  # 1 semitone = 100 cents, so 1 cent = 0.01 semitones
  cents_off = 100.0 * (midi_float - midi_int)

  # Step 6 - piano key index (A0 = key 0 = MIDI 21)
  piano_key = max(0, min(87, midi_int - 21))

  # Pick an emoji - strip sharps/flats to get the base letter
  base_letter = note_name[0]
  emoji = NOTE_EMOJI.get(base_letter, "🎵")

  return NoteInfo(
      freq_hz= round(freq_hz, 2),
      note_name = note_name,
      octave = octave,
      full_name= f"{note_name}{octave}",
      solfege = solfege,
      midi_note = midi_int,
      cents_off = round(cents_off, 1),
      piano_key = piano_key,
      emoji = emoji,
  )

# ----------------------------------
# Piano Key Frequency Table
# ----------------------------------
# Moved from cell T9UVR_dzRQYa and fixed indentation
def piano_key_to_hz(key_index: int) -> float:
  """
  Convert a piano key index (0-87) to a frequency in Hz..

  The piano starts at A0 = 27.5 Hz, so:
    key 0 -> MIDI 21 -> A0 -> 27.5 Hz
    key 48 -> MIDI 69 -> A4 -> 440.0 Hz
    key 87 -> MIDI 108 -> C8 -> 4186.0 Hz

  Args:
    key_index: Integer 0 to 87

  Returns:
    float frequency in Hz
  """
  midi = key_index + 21
  return midi_to_hz(midi)

def build_piano_frequency_table() -> list[float]:
  """
  Build a list of the exact frequency for all 88 piano keys.

  Returns:
    List of 88 floats, index 0 = A0 = 27.5 Hz, index 87 = C8

  This table is used during training to label synthetic audio samples
  and during interference to convert model output classes -> Hz.
  """
  return [piano_key_to_hz(k) for k in range(88)]

# Pre-built at module load - used everywhere
PIANO_FREQUENCIES: list[float] = build_piano_frequency_table()

def class_index_to_hz(class_idx: int) -> float:
  """
  Convert a model output class index (0-87) to a frequency in Hz.

  Args:
    class_idx: Integer 0 to 87

  Returns:
    float frequency in Hz
  """
  return PIANO_FREQUENCIES[class_idx]

def weighted_class_to_hz(class_probs: "list[float]") -> float:
  """
  Convert a probability distribution over 88 classes to a smooth Hz value.

  Instead of just picking the highest class(argmax), we take a
  weighted average of nearby class frequencies. This gives sub-semitone
  precision and smoother real-time output.

  Think of it like:
    If the model says 70% it's A4 and 30% it's A#4,
    the true pitch is probably slightly sharp of A4.

  Args:
    class_probs: List of 88 probabilities (should sum to 1)

  Returns:
    Estimated fundamental frequency in Hz
  """
  total_weight = sum(class_probs)
  if total_weight == 0:
    return 0.0

  weighted_freq = sum(
      prob * freq
      for prob, freq in zip(class_probs, PIANO_FREQUENCIES)
  )
  return weighted_freq / total_weight

#------------------------------------------------
# QUICK SELF-TEST (run: python note_converter.py)
#------------------------------------------------
if __name__ == "__main__":
  print("=" * 50)
  print(" Note Converter - Self Test")
  print("=" * 50)

  test_cases = [
      (27.5, "A0"),
      (261.63, "C4"), # Middle C
      (440.0, "A4"),  # Concert A
      (880.0, "A5"),
      (4186.0, "C8"),
  ]

  for freq_hz, expected_name in test_cases:
    info = hz_to_note_info(freq_hz) # Fixed: changed freq to freq_hz
    # Fixed: Removed call to full_name as it's a string attribute, not a method
    status = "✓" if info.full_name == expected_name else "x"
    print(f"  {status} {freq_hz:8.2f} Hz -> {info.full_name:5s} " # Fixed: changed freq to freq_hz
          f"({info.solfege:5s}) [{info.cents_off:+.1f} cents] {info.emoji}")
  
  print("\n Piano key 0 (A0):", PIANO_FREQUENCIES[0], "Hz")
  print("\n Piano key 48 (A4):", PIANO_FREQUENCIES[48], "Hz")
  print("\n Piano key 87 (C8):", round(PIANO_FREQUENCIES[87],1), "Hz")
  print("=" * 50)
