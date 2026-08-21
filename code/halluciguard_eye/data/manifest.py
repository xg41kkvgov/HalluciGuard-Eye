"""Dataset manifests and content checks. Ref: Sec. III-F and Table 2."""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ManifestRecord:
    sample_id: str
    relative_path: str
    label: int
    patient_id: str
    modality: str
    split: str
    sha256: str = ""
    width: int = 0
    height: int = 0

    def __post_init__(self) -> None:
        if not self.sample_id.strip() or not self.relative_path.strip():
            raise ValueError("manifest identifiers cannot be empty")
        path = Path(self.relative_path)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("manifest paths must be safe relative paths")
        if self.label < 0:
            raise ValueError("label must be nonnegative")
        if self.modality not in {"fundus", "oct"}:
            raise ValueError("modality must be fundus or oct")
        if self.split not in {"train", "validation", "test", "external"}:
            raise ValueError("invalid split")
        if self.sha256 and (
            len(self.sha256) != 64 or any(c not in "0123456789abcdef" for c in self.sha256)
        ):
            raise ValueError("sha256 must be lowercase hexadecimal")
        if self.width < 0 or self.height < 0:
            raise ValueError("image dimensions cannot be negative")

    def as_dict(self) -> dict[str, str | int]:
        return {
            "sample_id": self.sample_id,
            "relative_path": self.relative_path,
            "label": self.label,
            "patient_id": self.patient_id,
            "modality": self.modality,
            "split": self.split,
            "sha256": self.sha256,
            "width": self.width,
            "height": self.height,
        }

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> ManifestRecord:
        return cls(
            sample_id=str(raw["sample_id"]),
            relative_path=str(raw["relative_path"]),
            label=int(raw["label"]),
            patient_id=str(raw.get("patient_id", "")),
            modality=str(raw["modality"]),
            split=str(raw["split"]),
            sha256=str(raw.get("sha256", "")),
            width=int(raw.get("width", 0)),
            height=int(raw.get("height", 0)),
        )


class DatasetManifest:
    def __init__(self, name: str, records: Iterable[ManifestRecord]) -> None:
        self.name = name
        self._records = tuple(records)
        self._validate()

    def _validate(self) -> None:
        if not self.name.strip():
            raise ValueError("manifest name cannot be empty")
        identifiers = [record.sample_id for record in self._records]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("sample identifiers must be unique")
        paths = [record.relative_path for record in self._records]
        if len(paths) != len(set(paths)):
            raise ValueError("sample paths must be unique")

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self) -> Iterator[ManifestRecord]:
        return iter(self._records)

    def __getitem__(self, index: int) -> ManifestRecord:
        return self._records[index]

    def split(self, name: str) -> DatasetManifest:
        return DatasetManifest(self.name, (record for record in self if record.split == name))

    def classes(self) -> tuple[int, ...]:
        return tuple(sorted({record.label for record in self}))

    def modalities(self) -> tuple[str, ...]:
        return tuple(sorted({record.modality for record in self}))

    def patient_count(self) -> int:
        return len({record.patient_id for record in self if record.patient_id})

    def class_counts(self) -> dict[int, int]:
        counts: dict[int, int] = {}
        for record in self:
            counts[record.label] = counts.get(record.label, 0) + 1
        return counts

    def split_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for record in self:
            counts[record.split] = counts.get(record.split, 0) + 1
        return counts

    def digest(self) -> str:
        payload = json.dumps(
            [record.as_dict() for record in self],
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def verify_files(self, root: str | Path, require_hash: bool = True) -> dict[str, str]:
        base = Path(root)
        results: dict[str, str] = {}
        for record in self:
            path = base / record.relative_path
            if not path.is_file():
                results[record.sample_id] = "missing"
                continue
            if require_hash and not record.sha256:
                results[record.sample_id] = "hash_absent"
                continue
            if record.sha256 and file_sha256(path) != record.sha256:
                results[record.sample_id] = "hash_mismatch"
                continue
            results[record.sample_id] = "valid"
        return results

    def to_csv(self, path: str | Path) -> None:
        fieldnames = list(ManifestRecord("x", "x.png", 0, "", "fundus", "train").as_dict())
        with Path(path).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(record.as_dict() for record in self)

    @classmethod
    def from_csv(cls, name: str, path: str | Path) -> DatasetManifest:
        with Path(path).open("r", newline="", encoding="utf-8") as stream:
            records = tuple(ManifestRecord.from_mapping(row) for row in csv.DictReader(stream))
        return cls(name, records)


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest_json(manifest: DatasetManifest, path: str | Path) -> None:
    payload = {
        "name": manifest.name,
        "digest": manifest.digest(),
        "records": [record.as_dict() for record in manifest],
    }
    with Path(path).open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)


def read_manifest_json(path: str | Path) -> DatasetManifest:
    with Path(path).open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict) or not isinstance(payload.get("records"), list):
        raise ValueError("invalid manifest document")
    manifest = DatasetManifest(
        str(payload["name"]),
        (ManifestRecord.from_mapping(record) for record in payload["records"]),
    )
    expected = str(payload.get("digest", ""))
    if expected and expected != manifest.digest():
        raise ValueError("manifest digest mismatch")
    return manifest
