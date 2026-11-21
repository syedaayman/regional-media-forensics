from app.modules.voice_detection.dataset_loader import iter_audio_samples

samples = iter_audio_samples(
    manifest_path="datasets/voice_detection/train.csv",
    dataset_path="../IndicTTS-Download/data",
    batch_size=32
)

print("Checking dataset loader...")

sample = next(samples)

print("Audio loaded successfully!")
print("Language:", sample["language"])
print("Label:", sample["label"])
print("Audio size:", len(sample["audio_bytes"]), "bytes")