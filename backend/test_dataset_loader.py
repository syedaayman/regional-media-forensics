from pathlib import Path
from collections import Counter

import pandas as pd
import pyarrow.parquet as pq

from app.modules.voice_detection.dataset_loader import (
    iter_audio_samples
)


# Project paths
backend_path = Path(__file__).resolve().parent

manifest_path = (
    backend_path
    / "datasets"
    / "voice_detection"
    / "train.csv"
)

dataset_path = (
    backend_path.parent
    / "IndicTTS-Download"
    / "data"
)


def test_dataset_loader():

    assert manifest_path.exists(), "Training manifest not found."
    assert dataset_path.exists(), "Dataset directory not found."

    manifest = pd.read_csv(manifest_path)

    # Stream payloads instead of retaining every recording in memory.
    sample_ids = set()
    label_counts = Counter()
    loaded_count = 0

    for sample in iter_audio_samples(
        manifest_path=manifest_path,
        dataset_path=dataset_path,
        batch_size=32
    ):

        assert sample["audio_bytes"] is not None
        assert len(sample["audio_bytes"]) > 0

        assert sample["language"] is not None
        assert sample["label"] in [0, 1]

        assert sample["source_file"] is not None
        assert sample["row_index"] >= 0

        sample_ids.add((sample["source_file"], sample["row_index"]))
        label_counts[sample["label"]] += 1
        loaded_count += 1

    # Check that all manifest recordings were loaded.
    assert loaded_count == len(manifest), (
        f"Expected {len(manifest)} samples, "
        f"but loaded {loaded_count}."
    )
    assert loaded_count > 0

    # Check for duplicate recordings.

    assert len(sample_ids) == len(set(sample_ids)), (
        "Duplicate audio samples detected."
    )

    # Verify CSV metadata against original Parquet records
    checked = 0

    for source_file, group in manifest.groupby("source_file"):

        parquet_file = dataset_path / source_file
        parquet = pq.ParquetFile(parquet_file)

        selected_indices = set(
            group["row_index"].astype(int)
        )

        row_offset = 0

        for batch in parquet.iter_batches(
            columns=["language", "is_tts"],
            batch_size=32
        ):

            languages = batch.column(
                batch.schema.get_field_index("language")
            ).to_pylist()

            labels = batch.column(
                batch.schema.get_field_index("is_tts")
            ).to_pylist()

            for position, language in enumerate(languages):

                row_index = row_offset + position

                if row_index not in selected_indices:
                    continue

                original = group[
                    group["row_index"] == row_index
                ].iloc[0]

                assert language == original["language"], (
                    f"Language mismatch in {source_file}, "
                    f"row {row_index}."
                )

                assert labels[position] == original["is_tts"], (
                    f"Label mismatch in {source_file}, "
                    f"row {row_index}."
                )

                checked += 1

            row_offset += len(languages)

    assert checked == len(manifest), (
        "Not all manifest rows were checked against Parquet."
    )

    print("\nDATASET LOADER TEST RESULTS")
    print("=" * 40)
    print(f"Manifest samples: {len(manifest)}")
    print(f"Loaded samples:   {loaded_count}")
    print(f"Verified metadata: {checked}")
    print(f"Genuine samples:  {label_counts[0]}")
    print(f"AI samples:       {label_counts[1]}")
    print("\nAll dataset loader tests passed!")
