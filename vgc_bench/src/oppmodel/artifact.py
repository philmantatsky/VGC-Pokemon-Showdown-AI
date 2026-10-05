"""The opponent predictor's artifact file: one ``torch.save``'d dict.

Every model of the experiment (count tables, the neural predictor) is written
and read ONLY through this module, so a runtime needs one loader:

    save_artifact(path, kind="table", name="flags_table", featurizer=featurizer,
                  predictor_payload=table.to_payload(), extra={...})
    loaded = load_predictor(path)                # LoadedPredictor
    pred = loaded.predictor.predict(sheet_unknown_as_closed(batch))

Keys of the stored dict:

    format         'oppmodel-artifact'
    version        1
    kind           'table' | 'oppnet'
    name           the model's name (for example 'flags_table')
    featurizer     Featurizer.to_payload(): vocabulary, numeric tables, repertoire
    dex_signature  events.dex_signature() when the artifact was written
    predictor      the kind's own plain payload
    extra          free-form plain data: dataset tag, manifest sha256, config,
                   calibration, metrics

``load_predictor`` rebuilds the featurizer and hands the payload to the kind's
module, imported only then: ``tables.from_payload(payload, featurizer)`` for
``'table'``, ``model.from_payload(payload, featurizer)`` for ``'oppnet'``. Each
returns an object with ``predict(batch)``, ``name`` and ``kind``.

Storage. The file holds only what ``torch.load(weights_only=True)`` accepts:
dicts with string keys, lists, tuples, str / int / float / bool / None and
tensors. ``save_artifact`` converts on the way in: a numpy array becomes a
tagged dict with its dtype, shape and raw bytes (so every dtype, float16 and
uint16 included, comes back bit for bit), a numpy scalar a Python scalar, a
tuple / set / Counter a list or dict, a Path a str, and an instance of a
subclass of str / float / int (``np.float64``, ``np.str_``, a version string
object) the exact built-in type. Tensors stay tensors, on the CPU. Anything
else is refused with ``TypeError`` at save time, so no artifact needs
unpickling of arbitrary objects to be read. A tuple comes back as a list.

A different dex. Reading never fails because the installed dex differs from
the one the artifact was written with: the names of the differing signature
entries are returned in ``meta['dex_signature_diff']`` (and on
``featurizer.signature_diff``). ``strict=True`` turns that into a
``ValueError``.

Errors. ``save_artifact`` / ``read_artifact`` / ``load_predictor`` are loaders,
not decision-path code: they raise ``OSError`` for a file problem,
``TypeError`` for an unstorable payload and ``ValueError`` for everything else
that is wrong with an artifact. ``try_load_predictor`` is the same load for a
runtime that must not raise: it returns None and counts the reason in
``COUNTERS``.
"""

from __future__ import annotations

import importlib
import os
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, NamedTuple

import numpy as np

from vgc_bench.src.oppmodel.events import dex_signature
from vgc_bench.src.oppmodel.features import (
    ELO_BLANK,
    ELO_KEEP,
    Featurizer,
    signature_diff,
)

FORMAT = "oppmodel-artifact"
VERSION = 1
KIND_TABLE = "table"
KIND_OPPNET = "oppnet"
KINDS: tuple[str, ...] = (KIND_TABLE, KIND_OPPNET)
# kind -> module with ``from_payload(payload, featurizer)``; imported on demand.
_MODULES = {
    KIND_TABLE: "vgc_bench.src.oppmodel.tables",
    KIND_OPPNET: "vgc_bench.src.oppmodel.model",
}
_ARRAY_TAG = "__ndarray__"
_PLAIN_TYPES = (bool, int, float, str)

# Failures of ``try_load_predictor``, by exception type.
COUNTERS: Counter[str] = Counter()


class LoadedPredictor(NamedTuple):
    """What ``load_predictor`` returns.

    ``predictor`` implements ``features.Predictor`` and has ``name`` and
    ``kind``; ``featurizer`` encodes snapshots for it. ``meta`` holds
    ``format``, ``version``, ``path``, ``dex_signature`` (as stored),
    ``dex_signature_diff`` (names of entries that differ from the installed
    dex; empty when they agree), ``elo_mode`` and ``extra``.
    """

    predictor: Any
    featurizer: Featurizer
    kind: str
    name: str
    meta: dict[str, Any]


# --- plain-data conversion ----------------------------------------------------


def _torch() -> Any:
    import torch

    return torch


def _encode(value: Any, where: str) -> Any:
    """``value`` as weights_only-safe plain data. Raises ``TypeError``."""
    torch = _torch()
    # Exact types first: np.float64 is a float and np.str_ / a version object
    # are strs, and torch.load(weights_only=True) refuses every one of them.
    if value is None or type(value) in _PLAIN_TYPES:
        return value
    if isinstance(value, np.generic):
        return _encode(value.item(), where)
    if isinstance(value, str):
        return str.__str__(value)
    if isinstance(value, float):
        return float.__float__(value)
    if isinstance(value, int):  # bool cannot be subclassed
        return int.__int__(value)
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject or value.dtype.fields is not None:
            raise TypeError(
                f"{where}: an array of dtype {value.dtype} cannot be stored"
            )
        # tobytes() is C-ordered whatever the memory layout; the shape is the
        # array's own (ascontiguousarray would make a 0-d array 1-d).
        raw = np.frombuffer(value.tobytes(), dtype=np.uint8).copy()
        return {
            _ARRAY_TAG: True,
            "dtype": value.dtype.str,
            "shape": [int(size) for size in value.shape],
            "data": torch.from_numpy(raw),
        }
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{where}: dict key {key!r} is not a string")
            if key == _ARRAY_TAG:
                raise TypeError(f"{where}: the key {_ARRAY_TAG!r} is reserved")
            out[key] = _encode(item, f"{where}.{key}")
        return out
    if isinstance(value, (list, tuple)):
        return [_encode(item, f"{where}[{i}]") for i, item in enumerate(value)]
    if isinstance(value, (set, frozenset)):
        try:
            ordered = sorted(value)
        except TypeError as exc:
            raise TypeError(f"{where}: a set of unorderable items") from exc
        return [_encode(item, f"{where}[{i}]") for i, item in enumerate(ordered)]
    raise TypeError(f"{where}: cannot store a {type(value).__name__}")


def _decode(value: Any) -> Any:
    """Undo ``_encode``: tagged dicts become numpy arrays again."""
    if isinstance(value, dict):
        if value.get(_ARRAY_TAG) is True:
            dtype = np.dtype(str(value["dtype"]))
            if dtype.hasobject:
                raise ValueError("an object array cannot be read")
            shape = tuple(int(size) for size in value["shape"])
            raw = value["data"].detach().cpu().numpy().astype(np.uint8, copy=False)
            if raw.size != int(np.prod(shape, dtype=np.int64)) * dtype.itemsize:
                raise ValueError("a stored array does not fit its shape")
            return np.frombuffer(raw.tobytes(), dtype=dtype).reshape(shape).copy()
        return {key: _decode(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_decode(item) for item in value]
    return value


# --- writing ------------------------------------------------------------------


def save_artifact(
    path: Path | str,
    *,
    kind: str,
    name: str,
    featurizer: Featurizer,
    predictor_payload: Mapping[str, Any],
    extra: Mapping[str, Any] | None = None,
) -> None:
    """Write one artifact, through a temporary file next to it.

    The temporary file is read back the way ``read_artifact`` reads it before
    it takes the target's place, so a file that cannot be read never replaces
    one that can. Raises ``ValueError`` for an unknown kind, ``TypeError`` for
    a payload that is not plain data (see the module docstring) or that does
    not read back, ``OSError`` for the file.
    """
    if kind not in KINDS:
        raise ValueError(f"unknown artifact kind {kind!r}; expected one of {KINDS}")
    if not isinstance(name, str) or not name:
        raise ValueError("an artifact needs a name")
    document = {
        "format": FORMAT,
        "version": VERSION,
        "kind": kind,
        "name": name,
        "featurizer": _encode(featurizer.to_payload(), "featurizer"),
        "dex_signature": _encode(dex_signature(), "dex_signature"),
        "predictor": _encode(dict(predictor_payload), "predictor"),
        "extra": _encode(dict(extra or {}), "extra"),
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    scratch = target.with_name(f"{target.name}.tmp{os.getpid()}")
    try:
        _torch().save(document, scratch)
        try:
            _torch().load(scratch, map_location="cpu", weights_only=True)
        except OSError:
            raise
        except Exception as exc:
            raise TypeError(
                f"{target}: the artifact does not read back as plain data: {exc!r}"
            ) from exc
        scratch.replace(target)
    finally:
        if scratch.exists():
            scratch.unlink()


# --- reading ------------------------------------------------------------------


def read_artifact(path: Path | str, strict: bool = False) -> dict[str, Any]:
    """The stored dict, arrays as numpy again, plus ``dex_signature_diff``.

    ``dex_signature_diff`` lists the signature entries that differ between the
    artifact and the installed dex; it is added here and is not in the file.
    Raises ``OSError`` when the file cannot be opened and ``ValueError`` when it
    is not an artifact of this format and version, or, with ``strict``, when
    the dex signature differs.
    """
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"no artifact at {source}")
    try:
        stored = _torch().load(source, map_location="cpu", weights_only=True)
    except OSError:
        raise
    except Exception as exc:
        raise ValueError(f"{source} is not a readable artifact: {exc!r}") from exc
    if not isinstance(stored, dict) or stored.get("format") != FORMAT:
        raise ValueError(f"{source} is not an {FORMAT} file")
    if stored.get("version") != VERSION:
        raise ValueError(
            f"{source} has artifact version {stored.get('version')!r}, "
            f"this code reads {VERSION}"
        )
    missing = [
        key
        for key in ("kind", "name", "featurizer", "dex_signature", "predictor", "extra")
        if key not in stored
    ]
    if missing:
        raise ValueError(f"{source} lacks {missing}")
    try:
        document = {key: _decode(value) for key, value in stored.items()}
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError(f"{source} holds a damaged array: {exc!r}") from exc
    if document["kind"] not in KINDS:
        raise ValueError(f"{source} has unknown kind {document['kind']!r}")
    saved = document["dex_signature"]
    if not isinstance(saved, dict):
        raise ValueError(f"{source} has no dex signature")
    document["dex_signature_diff"] = signature_diff(saved, dex_signature())
    if strict and document["dex_signature_diff"]:
        raise ValueError(
            f"{source}: dex signature differs: {document['dex_signature_diff']}"
        )
    return document


def load_predictor(
    path: Path | str, elo_mode: str = ELO_KEEP, strict: bool = False
) -> LoadedPredictor:
    """Read an artifact and build its predictor and featurizer.

    ``elo_mode`` is ``'keep'`` or ``'blank'`` (the featurizer then encodes
    every rating as unknown: an Elo-blind model's runtime). A dex signature
    that differs from the installed dex is reported in
    ``meta['dex_signature_diff']`` and raised only with ``strict``. Raises
    ``OSError`` / ``ValueError`` (see ``read_artifact``); a failure inside the
    kind's own module also arrives as ``ValueError``.
    """
    if elo_mode not in (ELO_KEEP, ELO_BLANK):
        raise ValueError(f"elo_mode must be {ELO_KEEP!r} or {ELO_BLANK!r}")
    document = read_artifact(path, strict=strict)
    kind = str(document["kind"])
    try:
        featurizer = Featurizer.from_payload(document["featurizer"], elo_mode, strict)
        module = importlib.import_module(_MODULES[kind])
        predictor = module.from_payload(document["predictor"], featurizer)
    except ImportError as exc:
        raise ValueError(f"kind {kind!r} needs {_MODULES[kind]}: {exc!r}") from exc
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"{path}: the {kind} payload did not load: {exc!r}") from exc
    name = str(document["name"])
    meta = {
        "format": document["format"],
        "version": document["version"],
        "path": str(path),
        "dex_signature": document["dex_signature"],
        "dex_signature_diff": list(document["dex_signature_diff"]),
        "elo_mode": elo_mode,
        "extra": document["extra"],
    }
    return LoadedPredictor(predictor, featurizer, kind, name, meta)


def try_load_predictor(
    path: Path | str, elo_mode: str = ELO_KEEP, strict: bool = False
) -> LoadedPredictor | None:
    """``load_predictor`` for a caller that must not raise: None on any failure."""
    try:
        return load_predictor(path, elo_mode, strict)
    except Exception as exc:
        try:
            COUNTERS[f"load_error:{type(exc).__name__}"] += 1
        except Exception:
            pass
        return None
