from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq


def iter_audio_samples(
    manifest_path,
    dataset_path,
    batch_size=32
):
    """
    Read selected audio samples and reference transcripts
    from Parquet files in small batches.
    """

    manifest = pd.read_csv(manifest_path)

    for source_file, group in manifest.groupby("source_file"):

        parquet_file = Path(dataset_path) / source_file
        parquet = pq.ParquetFile(parquet_file)

        selected_rows = (
            group.set_index("row_index").to_dict("index")
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
                    continue

                metadata = selected_rows[row_index]

                yield {
                    "audio_bytes": audio_bytes,
                    "reference_text": text_values[position],
                    "language": metadata["language"],
                    "label": int(metadata["is_tts"]),
                    "source_file": source_file,
                    "row_index": row_index
                }

            row_offset += len(audio_values)