# ODELIA Classifier

This repository now uses a flattened top-level module layout.

## Structure

- `cli.py`: command-line interface
- `model.py`: 3D CNN model
- `data.py`: dataset metadata, NIfTI loading, and MRI preprocessing
- `pipeline.py`: training and prediction workflows

## Install

```bash
pip install -r requirements.txt
```

## Usage

Train:

```bash
python cli.py train --data-root ODELIA2025/data --artifact model_cnn.pt
```

Predict:

```bash
python cli.py predict --artifact model_cnn.pt --cases-root ODELIA2025/data --output-file predictions_pytorch.csv
```

Run the CLI directly with `python cli.py`.

