"""Fundus and OCT dataset readers. Ref: Sec. III-F and Table 2."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageOps

from halluciguard_eye.data.annotations import ClinicalAnnotation
from halluciguard_eye.data.manifest import DatasetManifest, ManifestRecord


@dataclass(frozen=True)
class OphthalmicSample:
    sample_id: str
    image: np.ndarray
    label: int
    patient_id: str
    modality: str
    annotation: ClinicalAnnotation | None = None

    def __post_init__(self) -> None:
        if self.image.ndim != 3 or self.image.shape[-1] != 3:
            raise ValueError("images must have shape H x W x 3")
        if self.image.dtype != np.float32:
            raise ValueError("images must use float32")
        if not np.isfinite(self.image).all():
            raise ValueError("images must contain finite values")


ImageTransform = Callable[[np.ndarray], np.ndarray]


class OphthalmicDataset:
    def __init__(
        self,
        root: str | Path,
        manifest: DatasetManifest,
        image_size: int = 336,
        annotations: Sequence[ClinicalAnnotation] = (),
        transform: ImageTransform | None = None,
    ) -> None:
        self.root = Path(root)
        self.manifest = manifest
        self.image_size = image_size
        self.transform = transform
        self.annotations = {annotation.image_id: annotation for annotation in annotations}
        if image_size <= 0:
            raise ValueError("image size must be positive")

    def __len__(self) -> int:
        return len(self.manifest)

    def __iter__(self) -> Iterator[OphthalmicSample]:
        for index in range(len(self)):
            yield self[index]

    def __getitem__(self, index: int) -> OphthalmicSample:
        record = self.manifest[index]
        image = load_ophthalmic_image(
            self.root / record.relative_path, self.image_size, record.modality
        )
        if self.transform is not None:
            image = self.transform(image)
        return OphthalmicSample(
            sample_id=record.sample_id,
            image=np.asarray(image, dtype=np.float32),
            label=record.label,
            patient_id=record.patient_id,
            modality=record.modality,
            annotation=self.annotations.get(record.sample_id),
        )

    def batch(self, indices: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
        samples = [self[index] for index in indices]
        if not samples:
            raise ValueError("batch cannot be empty")
        images = np.stack([sample.image for sample in samples])
        labels = np.asarray([sample.label for sample in samples], dtype=np.int64)
        return images, labels


def load_ophthalmic_image(path: str | Path, size: int, modality: str) -> np.ndarray:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        if modality == "fundus":
            image = crop_fundus_field(image)
        elif modality == "oct":
            image = normalize_oct_contrast(image)
        else:
            raise ValueError("modality must be fundus or oct")
        image = letterbox_resize(image, size)
        array = np.asarray(image, dtype=np.float32) / 255.0
    return normalize_imagenet(array)


def crop_fundus_field(image: Image.Image, threshold: int = 8) -> Image.Image:
    gray = np.asarray(image.convert("L"))
    coordinates = np.argwhere(gray > threshold)
    if not len(coordinates):
        return image.copy()
    y_min, x_min = coordinates.min(axis=0)
    y_max, x_max = coordinates.max(axis=0) + 1
    width = x_max - x_min
    height = y_max - y_min
    side = max(width, height)
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    left = max(0, int(round(center_x - side / 2)))
    upper = max(0, int(round(center_y - side / 2)))
    right = min(image.width, left + side)
    lower = min(image.height, upper + side)
    return image.crop((left, upper, right, lower))


def normalize_oct_contrast(image: Image.Image) -> Image.Image:
    gray = ImageOps.autocontrast(image.convert("L"), cutoff=0.5)
    enhanced = ImageEnhance.Contrast(gray).enhance(1.1)
    return enhanced.convert("RGB")


def letterbox_resize(
    image: Image.Image, size: int, fill: tuple[int, int, int] = (0, 0, 0)
) -> Image.Image:
    if size <= 0:
        raise ValueError("target size must be positive")
    scale = min(size / image.width, size / image.height)
    width = max(1, int(round(image.width * scale)))
    height = max(1, int(round(image.height * scale)))
    resized = image.resize((width, height), Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (size, size), fill)
    canvas.paste(resized, ((size - width) // 2, (size - height) // 2))
    return canvas


def normalize_imagenet(image: np.ndarray) -> np.ndarray:
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("image must have shape H x W x 3")
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    return np.asarray((image - mean) / std, dtype=np.float32)


def random_horizontal_flip(
    image: np.ndarray, rng: np.random.Generator, probability: float = 0.5
) -> np.ndarray:
    if not 0 <= probability <= 1:
        raise ValueError("probability must be in [0, 1]")
    return np.ascontiguousarray(image[:, ::-1]) if rng.random() < probability else image.copy()


def random_brightness(
    image: np.ndarray, rng: np.random.Generator, limit: float = 0.1
) -> np.ndarray:
    if limit < 0:
        raise ValueError("brightness limit must be nonnegative")
    factor = rng.uniform(1 - limit, 1 + limit)
    return np.asarray(image * factor, dtype=np.float32)


def discover_classification_records(
    root: str | Path,
    class_names: Sequence[str],
    modality: str,
    split: str,
) -> tuple[ManifestRecord, ...]:
    base = Path(root)
    records: list[ManifestRecord] = []
    suffixes = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    for label, class_name in enumerate(class_names):
        class_root = base / class_name
        for path in sorted(
            item for item in class_root.rglob("*") if item.suffix.casefold() in suffixes
        ):
            relative = path.relative_to(base).as_posix()
            sample_id = path.stem
            patient_id = infer_patient_id(sample_id)
            with Image.open(path) as image:
                width, height = image.size
            records.append(
                ManifestRecord(
                    sample_id=sample_id,
                    relative_path=relative,
                    label=label,
                    patient_id=patient_id,
                    modality=modality,
                    split=split,
                    width=width,
                    height=height,
                )
            )
    return tuple(records)


def infer_patient_id(sample_id: str) -> str:
    pieces = sample_id.replace("_", "-").split("-")
    if len(pieces) >= 3 and pieces[-1].isdigit():
        return pieces[-2]
    if len(pieces) >= 2:
        return pieces[0]
    return sample_id
