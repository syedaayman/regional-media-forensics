from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


REQUIRED_COLUMNS = {
    "source_file",
    "row_index",
    "language",
    "is_tts"
}


def iter_audio_samples(
    manifest_path,
    dataset_path,
    batch_size=32
):
    """
    Read selected audio samples and reference transcripts
    from Parquet files in small batches.
    """

    manifest_path = Path(manifest_path)
    dataset_path = Path(dataset_path)

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}"
        )

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {dataset_path}"
        )

    if batch_size <= 0:
        raise ValueError("Batch size must be greater than zero.")

    manifest = pd.read_csv(manifest_path)

    missing_columns = REQUIRED_COLUMNS - set(manifest.columns)

    if missing_columns:
        raise ValueError(
            f"Manifest is missing columns: {sorted(missing_columns)}"
        )

    if manifest.empty:
        raise ValueError("Manifest contains no samples.")

    if manifest["row_index"].isna().any():
        raise ValueError("Manifest contains missing row indices.")

    if manifest["source_file"].isna().any():
        raise ValueError("Manifest contains missing source filenames.")

    if manifest.duplicated(["source_file", "row_index"]).any():
        raise ValueError(
            "Manifest contains duplicate Parquet row references."
        )

    manifest["row_index"] = manifest["row_index"].astype(int)

    if (manifest["row_index"] < 0).any():
        raise ValueError("Manifest contains negative row indices.")

    if not manifest["is_tts"].isin([0, 1]).all():
        raise ValueError("Labels must be either 0 or 1.")

    loaded_count = 0
    skipped_count = 0

    for source_file, group in manifest.groupby("source_file"):

        parquet_file = dataset_path / source_file

        if not parquet_file.exists():
            raise FileNotFoundError(
                f"Parquet file not found: {parquet_file}"
            )

        parquet = pq.ParquetFile(parquet_file)

        selected_rows = (
            group.set_index("row_index").to_dict("index")
        )

        total_rows = parquet.metadata.num_rows

        invalid_indices = [
            index for index in selected_rows
            if index >= total_rows
        ]

        if invalid_indices:
            raise ValueError(
                f"Invalid row indices in {source_file}: "
                f"{invalid_indices[:10]}"
            )

        row_offset = 0

        for batch in parquet.iter_batches(
            columns=["audio", "text"],
            batch_size=batch_size
        ):

            audio_values = batch.column(
                batch.schema.get_field_index("audio")
            ).to_pylist()

            text_values = batch.column(
                batch.schema.get_field_index("text")
            ).to_pylist()

            for position, audio_value in enumerate(audio_values):

                row_index = row_offset + position

                if row_index not in selected_rows:
                    continue

                if isinstance(audio_value, dict):
                    audio_bytes = audio_value.get("bytes")
                else:
                    audio_bytes = audio_value

                if not audio_bytes:
                    skipped_count += 1
                    continue

                metadata = selected_rows[row_index]

                loaded_count += 1

                yield {
                    "audio_bytes": audio_bytes,
                    "reference_text": text_values[position],
                    "language": metadata["language"],
                    "label": int(metadata["is_tts"]),
                    "source_file": source_file,
                    "row_index": row_index
                }

            row_offset += len(audio_values)

    print(
        f"\nDataset loader completed. "
        f"Loaded: {loaded_count}, "
        f"Missing audio: {skipped_count}"
    )