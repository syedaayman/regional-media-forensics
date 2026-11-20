import io
import librosa
import numpy as np


def extract_audio_features(audio_bytes: bytes) -> dict:
    """
    Extract acoustic features from an audio file.
    """

    # Load audio from memory
    audio, sample_rate = librosa.load(
        io.BytesIO(audio_bytes),
        sr=16000,
        mono=True
    )

    if len(audio) == 0:
        raise ValueError("Audio file is empty.")

    # Choose FFT size based on audio length
    n_fft = min(2048, len(audio))

    # Keep FFT size at least 256
    if n_fft < 256:
        raise ValueError("Audio recording is too short.")

    hop_length = max(1, n_fft // 4)

    # Extract basic audio features
    features = {
        "rms": float(np.mean(
            librosa.feature.rms(
                y=audio,
                frame_length=n_fft,
                hop_length=hop_length
            )
        )),

        "zero_crossing_rate": float(np.mean(
            librosa.feature.zero_crossing_rate(
                y=audio,
                frame_length=n_fft,
                hop_length=hop_length
            )
        )),

        "spectral_centroid": float(np.mean(
            librosa.feature.spectral_centroid(
                y=audio,
                sr=sample_rate,
                n_fft=n_fft,
                hop_length=hop_length
            )
        )),

        "spectral_bandwidth": float(np.mean(
            librosa.feature.spectral_bandwidth(
                y=audio,
                sr=sample_rate,
                n_fft=n_fft,
                hop_length=hop_length
            )
        ))
    }

    # Extract 13 MFCC features
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=13,
        n_fft=n_fft,
        hop_length=hop_length
    )

    for i in range(13):
        features[f"mfcc_{i + 1}"] = float(
            np.mean(mfcc[i])
        )

    return features