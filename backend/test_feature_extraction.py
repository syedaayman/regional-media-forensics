from app.modules.voice_detection.dataset_loader import iter_audio_samples
from app.modules.voice_detection.feature_extraction import extract_audio_features

samples = iter_audio_samples(
    manifest_path="datasets/voice_detection/train.csv",
    dataset_path="../IndicTTS-Download/data",
    batch_size=32
)

sample = next(samples)

features = extract_audio_features(sample["audio_bytes"])

print("FEATURE EXTRACTION TEST")
print("=" * 35)

for name, value in features.items():
    print(f"{name}: {value}")

print("\nTotal features:", len(features))