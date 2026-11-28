import numpy as np

from app.modules.voice_detection.dataset_loader import (
    iter_audio_samples
)

from app.modules.voice_detection.feature_extraction import (
    extract_audio_features
)


EXPECTED_FEATURES = [
    "rms",
    "zero_crossing_rate",
    "spectral_centroid",
    "spectral_bandwidth",
] + [f"mfcc_{i}" for i in range(1, 14)]


def test_feature_extraction():

    # Load one audio recording
    samples = iter_audio_samples(
        manifest_path="datasets/voice_detection/train.csv",
        dataset_path="../IndicTTS-Download/data",
        batch_size=32
    )

    sample = next(samples)

    # Extract features
    features = extract_audio_features(
        sample["audio_bytes"]
    )

    # Check feature count
    assert len(features) == 17, (
        f"Expected 17 features, received {len(features)}."
    )

    # Check feature names and order
    assert list(features.keys()) == EXPECTED_FEATURES, (
        "Feature names or order do not match."
    )

    # Check numeric values
    for name, value in features.items():

        assert isinstance(value, (int, float)), (
            f"{name} is not numeric."
        )

        assert np.isfinite(value), (
            f"{name} contains an invalid value: {value}"
        )

    # Check repeatability
    repeated_features = extract_audio_features(
        sample["audio_bytes"]
    )

    for name in EXPECTED_FEATURES:

        assert np.isclose(
            features[name],
            repeated_features[name],
            rtol=1e-5,
            atol=1e-7
        ), f"Feature {name} is not repeatable."

    print("\nFEATURE EXTRACTION TEST")
    print("=" * 40)

    for name, value in features.items():
        print(f"{name}: {value}")

    print(f"\nTotal features: {len(features)}")
    print("All feature extraction checks passed!")


if __name__ == "__main__":
    test_feature_extraction()