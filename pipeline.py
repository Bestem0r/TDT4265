import csv
from pathlib import Path
import time
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from data import DEVICE, BreastRecord, discover_breast_cases, load_breast_records, load_mri_volume, augment_volume
from model import BreastCNN


class BreastMRIDataset(Dataset):
	"""PyTorch dataset for breast MRI volumes."""

	def __init__(self, records: list[BreastRecord], augment: bool = False):
		self.records = records
		self.augment = augment

	def __len__(self) -> int:
		return len(self.records)

	def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
		record = self.records[idx]
		volume = load_mri_volume(record.breast_dir)

		if self.augment and np.random.rand() > 0.5:
			volume = augment_volume(volume)

		volume_tensor = torch.from_numpy(volume).float()
		label_tensor = torch.tensor(record.label, dtype=torch.long)
		return volume_tensor, label_tensor


def train_model_pytorch(
	*,
	data_root: Path,
	fit_splits: Iterable[str] = ("train",),
	val_splits: Iterable[str] | None = ("val",),
	max_breasts: int | None = None,
	epochs: int = 50,
	batch_size: int = 4,
	learning_rate: float = 1e-3,
	monitor_every: int = 20,
	checkpoint_path: Path | None = None,
	best_checkpoint_path: Path | None = None,
	monitor_csv_path: Path | None = None,
	target_specificity: float = 0.90,
	target_sensitivity: float = 0.90,
	selection_metric: str = "aggregate",
	max_batches: int | None = None,
) -> BreastCNN:
	train_records = load_breast_records(data_root, splits=tuple(fit_splits), max_breasts=max_breasts)
	if not train_records:
		raise RuntimeError(f"No labeled breast records found under {data_root}")

	val_records: list[BreastRecord] = []
	if val_splits:
		val_records = load_breast_records(data_root, splits=tuple(val_splits), max_breasts=max_breasts)

	print(f"Loaded {len(train_records)} training samples")
	if val_splits:
		print(f"Loaded {len(val_records)} validation samples")

	train_dataset = BreastMRIDataset(train_records, augment=False)
	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
	val_loader = None
	if val_records:
		val_dataset = BreastMRIDataset(val_records, augment=False)
		val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

	model = BreastCNN(in_channels=5, num_classes=3).to(DEVICE)
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
	criterion = nn.CrossEntropyLoss()

	print(f"Training on {DEVICE}")
	if selection_metric not in {"aggregate", "val_loss"}:
		raise ValueError("selection_metric must be one of: aggregate, val_loss")

	selection_key = "aggregate_score" if selection_metric == "aggregate" else selection_metric
	best_metric = float("-inf") if selection_key == "aggregate_score" else float("inf")
	if monitor_csv_path is not None:
		monitor_csv_path.parent.mkdir(parents=True, exist_ok=True)
		with open(monitor_csv_path, "w", newline="", encoding="utf-8") as f:
			writer = csv.writer(f)
			writer.writerow([
				"epoch",
				"train_loss",
				"val_loss",
				"val_auc",
				"sensitivity_at_target_specificity",
				"specificity_at_target_sensitivity",
				"aggregate_score",
				"val_acc",
			])

	for epoch in range(epochs):
		epoch_start = time.time()
		model.train()
		total_loss = 0.0
		n_batches = len(train_loader)
		if max_batches is not None:
			n_batches = min(n_batches, max_batches)

		for batch_idx, (volumes, labels) in enumerate(train_loader, start=1):
			volumes, labels = volumes.to(DEVICE), labels.to(DEVICE)
			optimizer.zero_grad()
			outputs = model(volumes)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()
			total_loss += loss.item()

			if monitor_every > 0 and (batch_idx % monitor_every == 0 or batch_idx == n_batches):
				elapsed = time.time() - epoch_start
				avg_so_far = total_loss / batch_idx
				eta_sec = (elapsed / batch_idx) * (n_batches - batch_idx)
				print(
					f"  epoch {epoch + 1}/{epochs} "
					f"batch {batch_idx}/{n_batches} "
					f"loss={loss.item():.4f} avg={avg_so_far:.4f} eta={eta_sec:.0f}s"
				)

			if max_batches is not None and batch_idx >= max_batches:
				break

		train_loss = total_loss / max(n_batches, 1)
		metric_for_best = train_loss
		metrics = {
			"val_loss": float("nan"),
			"val_acc": float("nan"),
			"val_auc": float("nan"),
			"sensitivity_at_target_specificity": float("nan"),
			"specificity_at_target_sensitivity": float("nan"),
			"aggregate_score": float("nan"),
		}
		summary = f"epoch={epoch + 1:3d}/{epochs} train_loss={train_loss:.5f}"

		if val_loader is not None:
			metrics = _evaluate_model(
				model,
				val_loader,
				criterion,
				target_specificity=target_specificity,
				target_sensitivity=target_sensitivity,
			)
			metric_for_best = metrics[selection_key]
			summary += (
				f" val_loss={metrics['val_loss']:.5f}"
				f" auc={metrics['val_auc']:.4f}"
				f" sens@spec{target_specificity:.2f}={metrics['sensitivity_at_target_specificity']:.4f}"
				f" spec@sens{target_sensitivity:.2f}={metrics['specificity_at_target_sensitivity']:.4f}"
				f" agg={metrics['aggregate_score']:.4f}"
			)

		is_better = False
		if np.isfinite(metric_for_best):
			if selection_key == "aggregate_score":
				is_better = metric_for_best > best_metric
			else:
				is_better = metric_for_best < best_metric

		if is_better:
			best_metric = metric_for_best
			if best_checkpoint_path is not None:
				best_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
				torch.save(model.state_dict(), best_checkpoint_path)

		if monitor_csv_path is not None:
			with open(monitor_csv_path, "a", newline="", encoding="utf-8") as f:
				writer = csv.writer(f)
				writer.writerow([
					epoch + 1,
					f"{train_loss:.6f}",
					f"{metrics['val_loss']:.6f}" if np.isfinite(metrics["val_loss"]) else "nan",
					f"{metrics['val_auc']:.6f}" if np.isfinite(metrics["val_auc"]) else "nan",
					f"{metrics['sensitivity_at_target_specificity']:.6f}" if np.isfinite(metrics["sensitivity_at_target_specificity"]) else "nan",
					f"{metrics['specificity_at_target_sensitivity']:.6f}" if np.isfinite(metrics["specificity_at_target_sensitivity"]) else "nan",
					f"{metrics['aggregate_score']:.6f}" if np.isfinite(metrics["aggregate_score"]) else "nan",
					f"{metrics['val_acc']:.6f}" if np.isfinite(metrics["val_acc"]) else "nan",
				])

		if checkpoint_path is not None:
			checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
			torch.save(model.state_dict(), checkpoint_path)

		summary += f" epoch_time={(time.time() - epoch_start):.1f}s"
		print(summary)

	if best_checkpoint_path is not None and best_checkpoint_path.exists():
		model.load_state_dict(torch.load(best_checkpoint_path, map_location=DEVICE))
		print(f"Loaded best model from {best_checkpoint_path}")

	return model


def _evaluate_model(
	model: BreastCNN,
	loader: DataLoader,
	criterion: nn.Module,
	*,
	target_specificity: float,
	target_sensitivity: float,
) -> dict[str, float]:
	model.eval()
	total_loss = 0.0
	total_correct = 0
	total_samples = 0
	y_true: list[int] = []
	y_score: list[float] = []

	with torch.no_grad():
		for volumes, labels in loader:
			volumes, labels = volumes.to(DEVICE), labels.to(DEVICE)
			outputs = model(volumes)
			loss = criterion(outputs, labels)
			total_loss += float(loss.item())
			probs = F.softmax(outputs, dim=1)
			malignant_prob = probs[:, 2]
			y_score.extend(malignant_prob.detach().cpu().tolist())
			y_true.extend((labels == 2).detach().cpu().int().tolist())
			preds = torch.argmax(outputs, dim=1)
			total_correct += int((preds == labels).sum().item())
			total_samples += int(labels.numel())

	avg_loss = total_loss / max(len(loader), 1)
	accuracy = total_correct / max(total_samples, 1)
	auc = _compute_auc(np.asarray(y_true, dtype=bool), np.asarray(y_score, dtype=np.float64))
	sens_at_spec = _sensitivity_at_specificity(
		np.asarray(y_true, dtype=bool),
		np.asarray(y_score, dtype=np.float64),
		target_specificity,
	)
	spec_at_sens = _specificity_at_sensitivity(
		np.asarray(y_true, dtype=bool),
		np.asarray(y_score, dtype=np.float64),
		target_sensitivity,
	)
	aggregate = float(np.mean([auc, sens_at_spec, spec_at_sens])) if np.all(np.isfinite([auc, sens_at_spec, spec_at_sens])) else float("nan")
	model.train()
	return {
		"val_loss": float(avg_loss),
		"val_acc": float(accuracy),
		"val_auc": float(auc),
		"sensitivity_at_target_specificity": float(sens_at_spec),
		"specificity_at_target_sensitivity": float(spec_at_sens),
		"aggregate_score": float(aggregate),
	}


def _compute_auc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
	y_true = np.asarray(y_true, dtype=bool)
	y_scores = np.asarray(y_scores, dtype=np.float64)

	n_pos = int(y_true.sum())
	n_neg = int(y_true.size - n_pos)
	if n_pos == 0 or n_neg == 0:
		return float("nan")

	order = np.argsort(y_scores)
	ranks = np.empty_like(order, dtype=np.float64)
	ranks[order] = np.arange(1, y_scores.size + 1, dtype=np.float64)
	rank_sum = float(ranks[y_true].sum())
	auc = (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
	return float(auc)


def _roc_points(y_true: np.ndarray, y_scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
	thresholds = np.unique(y_scores)[::-1]
	thresholds = np.concatenate(([np.inf], thresholds, [-np.inf]))

	tpr_list: list[float] = []
	fpr_list: list[float] = []
	for thr in thresholds:
		pred = y_scores >= thr
		tp = float(np.sum(pred & y_true))
		fp = float(np.sum(pred & ~y_true))
		tn = float(np.sum(~pred & ~y_true))
		fn = float(np.sum(~pred & y_true))
		tpr = tp / max(tp + fn, 1.0)
		fpr = fp / max(fp + tn, 1.0)
		tpr_list.append(tpr)
		fpr_list.append(fpr)

	return np.asarray(tpr_list, dtype=np.float64), np.asarray(fpr_list, dtype=np.float64)


def _sensitivity_at_specificity(y_true: np.ndarray, y_scores: np.ndarray, target_specificity: float) -> float:
	if not np.any(y_true) or np.all(y_true):
		return float("nan")
	tpr, fpr = _roc_points(y_true, y_scores)
	specificity = 1.0 - fpr
	valid = specificity >= target_specificity
	if not np.any(valid):
		return float("nan")
	return float(np.max(tpr[valid]))


def _specificity_at_sensitivity(y_true: np.ndarray, y_scores: np.ndarray, target_sensitivity: float) -> float:
	if not np.any(y_true) or np.all(y_true):
		return float("nan")
	tpr, fpr = _roc_points(y_true, y_scores)
	specificity = 1.0 - fpr
	valid = tpr >= target_sensitivity
	if not np.any(valid):
		return float("nan")
	return float(np.max(specificity[valid]))


def predict_cases_pytorch(
	*,
	model: BreastCNN,
	cases_root: Path,
	output_file: Path,
	max_breasts: int | None = None,
) -> Path:
	records = discover_breast_cases(cases_root)
	if max_breasts is not None:
		records = records[:max_breasts]

	grouped: dict[str, dict[str, BreastRecord]] = {}
	for record in records:
		grouped.setdefault(record.study_id, {})[record.side] = record

	output_file.parent.mkdir(parents=True, exist_ok=True)
	rows: list[tuple[str, float, float, float]] = []

	model.eval()
	with torch.no_grad():
		for study_id in sorted(grouped):
			side_records = grouped[study_id]
			for side in ("left", "right"):
				record = side_records.get(side)
				if record is None:
					probs = np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float32)
				else:
					try:
						volume = load_mri_volume(record.breast_dir)
						volume_tensor = torch.from_numpy(volume).unsqueeze(0).to(DEVICE)
						logits = model(volume_tensor)
						probs = F.softmax(logits, dim=1)[0].cpu().numpy().astype(np.float32)
					except Exception as exc:
						print(f"Warning: failed to process {record.breast_dir}: {exc}")
						probs = np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float32)

				row_id = f"{study_id}_{side}"
				rows.append((row_id, float(probs[0]), float(probs[1]), float(probs[2])))

	with open(output_file, "w", newline="", encoding="utf-8") as f:
		writer = csv.writer(f)
		writer.writerow(["ID", "normal", "benign", "malignant"])
		for row_id, normal, benign, malignant in rows:
			writer.writerow([row_id, f"{normal:.4f}", f"{benign:.4f}", f"{malignant:.4f}"])

	return output_file
