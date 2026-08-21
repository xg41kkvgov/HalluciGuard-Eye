"""Deterministic clinical token and claim segmentation for evidence encoding."""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

TOKEN_PATTERN = re.compile(r"[A-Za-z]+(?:[-'][A-Za-z]+)*|\d+(?:\.\d+)?|[^\w\s]", re.UNICODE)
SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


@dataclass(frozen=True)
class Vocabulary:
    tokens: tuple[str, ...]
    unknown_token: str = "<unk>"
    padding_token: str = "<pad>"
    beginning_token: str = "<bos>"
    end_token: str = "<eos>"

    def __post_init__(self) -> None:
        if len(self.tokens) != len(set(self.tokens)):
            raise ValueError("vocabulary tokens must be unique")
        required = {self.unknown_token, self.padding_token, self.beginning_token, self.end_token}
        if not required <= set(self.tokens):
            raise ValueError("vocabulary is missing special tokens")

    @property
    def lookup(self) -> Mapping[str, int]:
        return {token: index for index, token in enumerate(self.tokens)}

    def identifier(self, token: str) -> int:
        mapping = self.lookup
        return mapping.get(token, mapping[self.unknown_token])

    def token(self, identifier: int) -> str:
        if not 0 <= identifier < len(self.tokens):
            raise IndexError("token identifier is out of range")
        return self.tokens[identifier]

    def save(self, path: str | Path) -> None:
        payload = {
            "tokens": self.tokens,
            "unknown_token": self.unknown_token,
            "padding_token": self.padding_token,
            "beginning_token": self.beginning_token,
            "end_token": self.end_token,
        }
        with Path(path).open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> Vocabulary:
        with Path(path).open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
        return cls(
            tokens=tuple(payload["tokens"]),
            unknown_token=str(payload["unknown_token"]),
            padding_token=str(payload["padding_token"]),
            beginning_token=str(payload["beginning_token"]),
            end_token=str(payload["end_token"]),
        )


@dataclass(frozen=True)
class EncodedSequence:
    identifiers: tuple[int, ...]
    attention_mask: tuple[bool, ...]
    offsets: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        if len(self.identifiers) != len(self.attention_mask) or len(self.identifiers) != len(
            self.offsets
        ):
            raise ValueError("encoded sequence fields must have equal lengths")


def normalize_clinical_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.replace("\u00a0", " ")
    return " ".join(normalized.split())


def tokenize_with_offsets(text: str) -> tuple[tuple[str, int, int], ...]:
    normalized = normalize_clinical_text(text)
    return tuple(
        (match.group(0), match.start(), match.end()) for match in TOKEN_PATTERN.finditer(normalized)
    )


def segment_claims(text: str) -> tuple[str, ...]:
    normalized = normalize_clinical_text(text)
    if not normalized:
        return ()
    sentences = SENTENCE_PATTERN.split(normalized)
    claims: list[str] = []
    for sentence in sentences:
        pieces = re.split(r"\s*;\s*|\s+(?:and|but)\s+(?=[A-Z])", sentence)
        claims.extend(piece.strip() for piece in pieces if piece.strip())
    return tuple(claims)


class ClinicalTokenizer:
    def __init__(self, vocabulary: Vocabulary, lowercase: bool = True) -> None:
        self.vocabulary = vocabulary
        self.lowercase = lowercase

    def _normalize_token(self, token: str) -> str:
        return token.casefold() if self.lowercase else token

    def encode(
        self, text: str, maximum_length: int, add_special_tokens: bool = True
    ) -> EncodedSequence:
        if maximum_length <= 0:
            raise ValueError("maximum length must be positive")
        pieces = tokenize_with_offsets(text)
        tokens = [self._normalize_token(token) for token, _, _ in pieces]
        offsets = [(start, end) for _, start, end in pieces]
        if add_special_tokens:
            tokens = [self.vocabulary.beginning_token, *tokens, self.vocabulary.end_token]
            offsets = [(0, 0), *offsets, (len(text), len(text))]
        tokens = tokens[:maximum_length]
        offsets = offsets[:maximum_length]
        identifiers = [self.vocabulary.identifier(token) for token in tokens]
        attention = [True] * len(identifiers)
        padding = maximum_length - len(identifiers)
        identifiers.extend([self.vocabulary.identifier(self.vocabulary.padding_token)] * padding)
        attention.extend([False] * padding)
        offsets.extend([(0, 0)] * padding)
        return EncodedSequence(tuple(identifiers), tuple(attention), tuple(offsets))

    def decode(self, identifiers: Sequence[int], skip_special_tokens: bool = True) -> str:
        special = {
            self.vocabulary.padding_token,
            self.vocabulary.beginning_token,
            self.vocabulary.end_token,
        }
        tokens = [self.vocabulary.token(identifier) for identifier in identifiers]
        if skip_special_tokens:
            tokens = [token for token in tokens if token not in special]
        output = ""
        for token in tokens:
            if (
                output
                and token not in {".", ",", ":", ";", "?", "!", ")"}
                and not output.endswith("(")
            ):
                output += " "
            output += token
        return output


def build_vocabulary(
    documents: Iterable[str],
    maximum_size: int,
    minimum_frequency: int = 1,
) -> Vocabulary:
    if maximum_size < 4 or minimum_frequency <= 0:
        raise ValueError("vocabulary controls are invalid")
    counts: dict[str, int] = {}
    for document in documents:
        for token, _, _ in tokenize_with_offsets(document):
            normalized = token.casefold()
            counts[normalized] = counts.get(normalized, 0) + 1
    specials = ("<pad>", "<unk>", "<bos>", "<eos>")
    candidates = [
        token
        for token, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        if count >= minimum_frequency and token not in specials
    ]
    return Vocabulary((*specials, *candidates[: maximum_size - len(specials)]))


def batch_sequences(
    sequences: Iterable[EncodedSequence],
) -> tuple[list[list[int]], list[list[bool]]]:
    items = tuple(sequences)
    if not items:
        raise ValueError("encoded sequence batch cannot be empty")
    lengths = {len(item.identifiers) for item in items}
    if len(lengths) != 1:
        raise ValueError("encoded sequences must have equal lengths")
    return [list(item.identifiers) for item in items], [list(item.attention_mask) for item in items]
