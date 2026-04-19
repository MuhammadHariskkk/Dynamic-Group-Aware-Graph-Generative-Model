# Data Directory

This project targets **ETH/UCY** pedestrian trajectory datasets.

## Expected structure (planned)

```text
data/
  raw/
    eth/
    hotel/
    univ/
    zara1/
    zara2/
  processed/
```

## Notes

- Raw parsing assumptions will be documented in `groupaware/datasets/preprocessing.py` in Phase 2.
- Processed artifacts should be reproducible from raw data using `scripts/preprocess_data.py` (Phase 2).
- This repository does not bundle ETH/UCY raw files.
