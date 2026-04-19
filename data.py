import csv
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from scipy import ndimage

SUPPORTED_SEQUENCE_NAMES = ("Pre", "Sub_1", "Post_1", "Post_2", "T2")

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = PROJECT_ROOT / "ODELIA2025" / "data"
DEFAULT_ARTIFACT_PATH = PROJECT_ROOT / "model_cnn.pt"
DEFAULT_PREDICTIONS_PATH = PROJECT_ROOT / "predictions_pytorch.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass(frozen=True)
class BreastRecord:
    institution: str
    study_id: str
    side: str
    breast_dir: Path
    label: int | None = None
    split: str | None = None
    fold: int | None = None


def _infer_study_id(uid: str) -> str:
    return uid.rsplit("_", 1)[0] if "_" in uid else uid


def _infer_side(uid: str) -> str:
    tail = uid.rsplit("_", 1)[-1].lower()
    return tail if tail in ("left", "right") else "left"


def _load_split_lookup(split_path: Path) -> dict[str, dict[str, str]]:
    lookup: dict[str, dict[str, str]] = {}
    if not split_path.exists():
        return lookup

    with open(split_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            uid = (row.get("UID") or "").strip()
            if uid:
                lookup[uid] = row
    return lookup


def load_breast_records(
    data_root: Path,
    *,
    splits: tuple[str, ...] | None = None,
    max_breasts: int | None = None,
) -> list[BreastRecord]:
    data_root = Path(data_root)
    records: list[BreastRecord] = []
    split_set = set(splits) if splits is not None else None

    for institution_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        annotation_path = institution_dir / "metadata_unilateral" / "annotation.csv"
        split_path = institution_dir / "metadata_unilateral" / "split.csv"
        data_dir = institution_dir / "data_unilateral"
        if not annotation_path.exists() or not data_dir.exists():
            continue

        split_lookup = _load_split_lookup(split_path)
        with open(annotation_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                uid = (row.get("UID") or "").strip()
                if not uid:
                    continue

                split_info = split_lookup.get(uid, {})
                split_value = split_info.get("Split")
                if split_set is not None and split_value not in split_set:
                    continue

                breast_dir = data_dir / uid
                if not breast_dir.is_dir():
                    continue

                label_value = row.get("Lesion")
                label = int(label_value) if label_value not in (None, "") else None
                records.append(
                    BreastRecord(
                        institution=institution_dir.name,
                        study_id=_infer_study_id(uid),
                        side=_infer_side(uid),
                        breast_dir=breast_dir,
                        label=label,
                        split=split_value,
                    )
                )

                if max_breasts is not None and len(records) >= max_breasts:
                    return records

    return records


def discover_breast_cases(cases_root: Path) -> list[BreastRecord]:
    cases_root = Path(cases_root)
    case_dirs: set[Path] = set()
    for pattern in ("Pre.nii.gz", "Pre.nii"):
        case_dirs.update(p.parent for p in cases_root.rglob(pattern))

    records: list[BreastRecord] = []
    for breast_dir in sorted(case_dirs):
        name = breast_dir.name
        if name.endswith("_left"):
            study_id, side = name[:-5], "left"
        elif name.endswith("_right"):
            study_id, side = name[:-6], "right"
        else:
            study_id, side = name, "left"

        records.append(
            BreastRecord(
                institution=breast_dir.parent.name,
                study_id=study_id,
                side=side,
                breast_dir=breast_dir,
            )
        )
    return records


def read_nifti(path: Path) -> tuple[np.ndarray, None]:
    img = nib.load(str(path))
    data = np.asarray(img.get_fdata(dtype=np.float32), dtype=np.float32)
    return data, None


def _resize_volume(volume: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray:
    current_shape = volume.shape[1:]
    zoom_factors = tuple(t / c for t, c in zip(target_shape, current_shape))

    resized = np.zeros((volume.shape[0], *target_shape), dtype=np.float32)
    for i in range(volume.shape[0]):
        resized[i] = ndimage.zoom(volume[i], zoom_factors, order=1)
    return resized


def augment_volume(volume: np.ndarray) -> np.ndarray:
    if np.random.rand() > 0.5:
        angle = np.random.uniform(-15, 15)
        volume = np.stack([ndimage.rotate(v, angle, reshape=False, order=1) for v in volume])

    if np.random.rand() > 0.5:
        scale = np.random.uniform(0.9, 1.1)
        volume = volume * scale

    return volume


def load_mri_volume(case_dir: Path) -> np.ndarray:
    volumes: dict[str, np.ndarray] = {}
    for seq_name in SUPPORTED_SEQUENCE_NAMES:
        for candidate in (case_dir / f"{seq_name}.nii.gz", case_dir / f"{seq_name}.nii"):
            if candidate.exists():
                volume, _ = read_nifti(candidate)
                volumes[seq_name] = volume
                break

    if not volumes:
        raise FileNotFoundError(f"No NIfTI volumes found in {case_dir}")

    reference = next(iter(volumes.values()))
    stacked = [volumes.get(seq, np.zeros_like(reference)) for seq in SUPPORTED_SEQUENCE_NAMES]
    volume_5ch = np.stack(stacked, axis=0).astype(np.float32)

    for i in range(volume_5ch.shape[0]):
        ch = volume_5ch[i]
        ch_nonzero = ch[ch > 0]
        if len(ch_nonzero) > 0:
            mean = np.mean(ch_nonzero)
            std = np.std(ch_nonzero)
            if std > 0:
                volume_5ch[i] = (ch - mean) / (std + 1e-6)

    return _resize_volume(volume_5ch, target_shape=(96, 96, 96))

