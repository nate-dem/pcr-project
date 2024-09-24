import torch
from transformers import pipeline
from datasets import load_dataset

import os

# Ensure ffmpeg is accessible
os.environ["PATH"] += os.pathsep + "/opt/homebrew/bin"

device = "cuda:0" if torch.cuda.is_available() else "cpu"

pipe = pipeline(
  "automatic-speech-recognition",
  model="openai/whisper-base",
  chunk_length_s=30,
  device=device,
)

# ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
# sample = ds[0]["audio"]

# prediction = pipe(sample.copy(), batch_size=8)["text"]
audio = "pcrtest1.wav"
transcript = pipe(audio)["text"]

print(f"Trascription: {transcript}")