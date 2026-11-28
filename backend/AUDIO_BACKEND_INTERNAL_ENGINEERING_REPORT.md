# Audio Backend Internal Engineering Report

Audit date: 2026-11-28

## Improvements Implemented

### ASR and long audio

- Preserved the existing specialized Kannada checkpoint and multilingual Whisper
  medium checkpoint.
- Kept 16 kHz mono decoding and 120-second chunk processing.
- Removed boundary filtering that discarded the first five seconds of later
  chunks.
- Added overlap-aware segment merging that removes duplicated suffix/prefix
  words while preserving chronological segment order.
- Added overlap and boundary-deduplication unit coverage.
- Kept the existing 1,800-second upload limit.

### Tampering detection

- Preserved existing clipping, RMS, silence, spectral-flatness, and mean-frame
  indicators.
- Added robust spectral-centroid discontinuity evidence using the same
  median/MAD-style approach already used for other frame measurements.
- Added controlled spectral-boundary test coverage.
- No existing threshold was changed without evidence. The added centroid
  threshold is constrained by an 250 Hz minimum and an 8-MAD robust difference.

### Legacy database and API

- Existing malformed analysis payload handling returns an evidence-only result
  instead of failing retrieval.
- Evidence lookup now tolerates malformed or missing metadata JSON.
- Original-path validation remains in the integrity path.
- Complete-analysis processing now has an explicit generic 500 response for
  unexpected server failures.
- Existing API response contracts remain unchanged.

## Test Results

| Validation | Result |
|---|---|
| Focused Audio tests before latest overlap changes | 35 passed |
| Focused Audio tests after latest changes | 37 passed |
| Complete `backend/tests` suite | 108 passed |
| Standalone Audio unit/evaluation selection | 2 passed, 2 sklearn-version warnings |
| Existing voice test manifest | 4,666 samples processed, 0 missing |

Existing voice-model evaluation metrics:

| Metric | Result |
|---|---|
| Accuracy | 89.35% |
| Precision | 90.15% |
| Recall | 88.33% |
| F1-score | 89.23% |

The voice-model metrics were unchanged because no model or feature contract was
replaced. They are based on the existing test manifest.

## Known Engineering Tasks

- Complete independent reference-based WER/CER evaluation for all five primary
  target languages: Kannada, Hindi, Telugu, Tamil, and English.
- Add source-partitioned and synthesis-family-partitioned voice evaluation.
- Add controlled splicing/copy-paste labeled audio to calculate tampering
  precision, recall, confusion matrices, and F1-score.
- Perform concurrent production-style ASR and complete-analysis load tests.
- Add an external PDF reader check for report rendering.
- Monitor the scipy/scikit-learn serialization warning and align checkpoint
  environments before deployment.
- Verify cancellation behavior for underlying audio decoders under stalled or
  malformed high-duration inputs.

## README Verification

The README was rewritten to describe:

- Actual Audio Backend architecture and supported endpoints.
- The nine currently supported ASR language codes.
- 16 kHz mono processing and 30-minute upload support.
- Existing Faster-Whisper and Random Forest model usage.
- Evidence, integrity, report, visualization, and database behavior.
- Local installation, run, test, and API usage commands.

No paid API, model replacement, external transcription service, or frontend,
video, or visual-forensics implementation change was introduced.
