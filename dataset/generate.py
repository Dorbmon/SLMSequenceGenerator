#!/usr/bin/env python3
"""Collect Dawn WebGPU WGS samples into sharded HDF5 files.

This process is deliberately *not* a solver.  It serves a localhost HTTP API
to the local Node/Dawn generator, validates every returned WebGPU result, and
writes only converged, numerically-valid samples.

``POST /api/sample`` uses a compact, versioned, little-endian protocol.  The
64-byte header is ``struct.Struct("<4sHHQ12I")``::

    magic[4] = b"SLMD"
    version u16 = 1
    header_bytes u16 = 64
    sample_id u64
    trap_count u32
    sampling_seed u32
    frame_crc32 u32       # IEEE CRC-32 of the raw frame bytes only
    width u32             # active SLM width, not FFT width
    height u32            # active SLM height, not FFT height
    positions_bytes u32   # exactly K * 2 * sizeof(float32)
    phases_bytes u32      # exactly K * sizeof(float32)
    trap_ids_bytes u32    # exactly K * sizeof(uint32)
    frame_bytes u32       # exactly width * height
    metrics_bytes u32     # UTF-8 JSON object
    flags u32 = 0
    reserved u32 = 0

The header is followed without alignment padding by positions ``<f4[K,2]``,
measured phases ``<f4[K]``, trap IDs ``<u4[K]``, frame ``u1[H,W]``, and the
metrics JSON.  HTTP Content-Length is the outer record length, so there is no
additional length prefix.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlsplit
import zlib

try:  # A clear CLI error is nicer than an import traceback before --help.
    import h5py  # type: ignore[import-not-found]
    import numpy as np  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - exercised only without requirements.
    h5py = None
    np = None


SCHEMA_VERSION = "slm-wgs-hdf5-v1"
MANIFEST_VERSION = 1
PROTOCOL_MAGIC = b"SLMD"
PROTOCOL_VERSION = 1
SAMPLE_HEADER = struct.Struct("<4sHHQ12I")
SAMPLE_HEADER_BYTES = SAMPLE_HEADER.size
STORAGE_MAX_TRAPS = 2000
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "data"
MAX_METRICS_BYTES = 1 << 20
MAX_JSON_BODY_BYTES = 1 << 20
UINT32_MAX = (1 << 32) - 1
UINT64_MAX = (1 << 64) - 1


class DatasetError(RuntimeError):
    """Base class for configuration, protocol, and persistence failures."""


class ProtocolError(DatasetError):
    """The Dawn runner supplied a malformed binary sample."""


class SampleRejectedError(DatasetError):
    """The supplied sample did not pass the collector's quality gates."""


class ConfigurationMismatchError(DatasetError):
    """An existing output directory belongs to a different run."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SampleRejectedError(f"{label} must be a finite number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as error:
        raise SampleRejectedError(f"{label} must be a finite number") from error
    if not math.isfinite(result):
        raise SampleRejectedError(f"{label} must be a finite number")
    return result


def _positive_int(value: str) -> int:
    try:
        result = int(value, 10)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if result <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return result


def _uint32(value: str) -> int:
    try:
        result = int(value, 0)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if not 0 <= result <= UINT32_MAX:
        raise argparse.ArgumentTypeError("must be between 0 and 4294967295")
    return result


def _positive_float(value: str) -> float:
    try:
        result = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a number") from error
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("must be a finite value greater than zero")
    return result


def _nonnegative_float(value: str) -> float:
    try:
        result = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a number") from error
    if not math.isfinite(result) or result < 0:
        raise argparse.ArgumentTypeError("must be a finite non-negative value")
    return result


def _unit_interval(value: str) -> float:
    result = _positive_float(value)
    if result > 1:
        raise argparse.ArgumentTypeError("must be no greater than one")
    return result


def next_power_of_two(value: int) -> int:
    return 1 << (value - 1).bit_length()


def is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def _parse_numeric_rows(text: str) -> list[float]:
    """Parse numeric CSV/whitespace text, including common ``code,phase`` CSV."""
    meaningful = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not meaningful:
        raise DatasetError("The phase-response LUT file is empty")

    delimiter = "," if any("," in line for line in meaningful) else (
        ";" if any(";" in line for line in meaningful) else None
    )
    if delimiter is not None:
        rows = [
            [cell.strip() for cell in next(csv.reader([line], delimiter=delimiter))]
            for line in meaningful
        ]
        header = rows and any(not _is_float_token(cell) for cell in rows[0])
        if header:
            normalized = [cell.lower().replace(" ", "").replace("_", "") for cell in rows[0]]
            phase_columns = [
                index for index, name in enumerate(normalized)
                if name in {"phase", "phaserad", "measuredphase", "phaseresponse"}
            ]
            if len(phase_columns) != 1:
                raise DatasetError("CSV LUT header must identify exactly one phase column")
            phase_index = phase_columns[0]
            rows = rows[1:]
            if not rows:
                raise DatasetError("The phase-response LUT contains a header but no values")
            if any(len(row) != len(normalized) or any(not cell for cell in row) for row in rows):
                raise DatasetError("Every CSV LUT row must match the header and contain no empty cells")
            try:
                phases = [float(row[phase_index]) for row in rows]
            except ValueError as error:
                raise DatasetError("The phase-response LUT contains a non-numeric phase") from error
            code_columns = [
                index for index, name in enumerate(normalized)
                if name in {"code", "displaycode", "gray", "grey", "grayscale", "index"}
            ]
            if len(code_columns) > 1:
                raise DatasetError("CSV LUT header identifies more than one display-code column")
            if code_columns:
                try:
                    codes = [float(row[code_columns[0]]) for row in rows]
                except ValueError as error:
                    raise DatasetError("The phase-response LUT contains a non-numeric display code") from error
                return _phases_from_display_code_table(codes, phases)
            return phases

        # A multi-line two-column table is interpreted as display-code, phase.
        if len(rows) >= 2 and all(len(row) == 2 for row in rows):
            try:
                codes = [float(row[0]) for row in rows]
                phases = [float(row[1]) for row in rows]
            except ValueError as error:
                raise DatasetError("The phase-response LUT contains a non-numeric value") from error
            return _phases_from_display_code_table(codes, phases)

        if any(any(not cell for cell in row) for row in rows):
            raise DatasetError("CSV LUT rows may not contain empty cells")
        tokens = [cell for row in rows for cell in row if cell]
    else:
        tokens = [token for line in meaningful for token in line.split()]
    try:
        return [float(token) for token in tokens]
    except ValueError as error:
        raise DatasetError("The phase-response LUT contains a non-numeric value") from error


def _is_float_token(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True


def _phases_from_display_code_table(codes: Sequence[float], phases: Sequence[float]) -> list[float]:
    """Normalize an explicit U8 display-code table without losing its code axis.

    An explicit code column always means literal U8 display codes. The table
    must cover both endpoints and is linearly expanded to one value per code.
    Uniformly sampled LUTs without literal codes should use JSON/one column.
    """
    if len(codes) != len(phases) or len(codes) < 2:
        raise DatasetError("A display-code LUT table needs at least two code/phase rows")
    if not all(math.isfinite(code) and code.is_integer() and 0 <= code <= 255 for code in codes):
        raise DatasetError("CSV LUT display codes must be integer values between 0 and 255")
    integer_codes = [int(code) for code in codes]
    if any(current <= previous for previous, current in zip(integer_codes, integer_codes[1:])):
        raise DatasetError("CSV LUT display codes must be strictly increasing")
    if integer_codes[0] != 0 or integer_codes[-1] != 255:
        raise DatasetError(
            "CSV LUTs with an explicit display-code column must include code 0 and code 255"
        )

    expanded: list[float] = []
    segment = 0
    for code in range(256):
        while segment + 1 < len(integer_codes) - 1 and code > integer_codes[segment + 1]:
            segment += 1
        low_code = integer_codes[segment]
        high_code = integer_codes[segment + 1]
        fraction = (code - low_code) / (high_code - low_code)
        expanded.append(float(phases[segment]) * (1 - fraction) + float(phases[segment + 1]) * fraction)
    return expanded


def parse_phase_response_lut(path: str | os.PathLike[str]) -> tuple[list[float], dict[str, Any]]:
    """Load and strictly validate a measured display-code-to-phase LUT.

    Accepted inputs are a JSON number array, a JSON object with one recognized
    LUT member, a numeric CSV (one row/column or ``code,phase``), or whitespace
    separated text.  Plateaus are allowed because measured responses commonly
    quantize, but the direction may never reverse and the total span must be
    non-zero.
    """
    source_path = Path(path).expanduser().resolve()
    if not source_path.is_file():
        raise DatasetError(f"Phase-response LUT does not exist: {source_path}")
    raw = source_path.read_bytes()
    try:
        text = raw.decode("utf-8-sig").strip()
    except UnicodeDecodeError as error:
        raise DatasetError("The phase-response LUT must be UTF-8 text") from error
    if not text:
        raise DatasetError("The phase-response LUT file is empty")

    values: list[float]
    try:
        candidate = json.loads(text, parse_constant=lambda token: (_raise_json_constant(token)))
    except json.JSONDecodeError:
        values = _parse_numeric_rows(text)
    else:
        if isinstance(candidate, dict):
            recognized = [
                key for key in ("phaseResponseLut", "phase_response_lut", "values", "phases")
                if key in candidate
            ]
            if len(recognized) != 1:
                raise DatasetError(
                    "JSON LUT object must contain exactly one of phaseResponseLut, "
                    "phase_response_lut, values, or phases"
                )
            candidate = candidate[recognized[0]]
        if not isinstance(candidate, list):
            raise DatasetError("JSON phase-response LUT must be a number array or recognized object")
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in candidate):
            raise DatasetError("Every phase-response LUT value must be numeric")
        values = [float(value) for value in candidate]

    if not 2 <= len(values) <= 65536:
        raise DatasetError("The measured phase-response LUT must contain 2 to 65536 values")
    if not all(math.isfinite(value) for value in values):
        raise DatasetError("Every phase-response LUT value must be finite")
    delta = values[-1] - values[0]
    if delta == 0:
        raise DatasetError("The measured phase-response LUT must have a non-zero phase span")
    increasing = delta > 0
    for index in range(1, len(values)):
        if (increasing and values[index] < values[index - 1]) or (
            not increasing and values[index] > values[index - 1]
        ):
            raise DatasetError(
                f"The measured phase-response LUT reverses direction at index {index}"
            )

    normalized = struct.pack(f"<{len(values)}d", *values)
    minimum_phase = min(values)
    maximum_phase = max(values)
    phase_convention = (
        "ZERO_TO_TWO_PI"
        if minimum_phase >= -1e-9 and maximum_phase > math.pi
        else "NEGATIVE_PI_TO_PI"
    )
    metadata = {
        "filename": source_path.name,
        "sourcePath": str(source_path),
        "sourceSha256": sha256_bytes(raw),
        "valuesSha256": sha256_bytes(normalized),
        "length": len(values),
        "direction": "INCREASING" if increasing else "DECREASING",
        "phaseConvention": phase_convention,
        "minimumPhaseRad": min(values),
        "maximumPhaseRad": max(values),
        "phaseSpanRad": abs(delta),
    }
    return values, metadata


def _raise_json_constant(token: str) -> Any:
    raise DatasetError(f"JSON value {token} is not finite")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a converged Dawn WebGPU WGS HDF5 dataset for SLMControl3",
    )
    parser.add_argument("--samples", type=_positive_int, required=True, help="accepted samples to generate")
    parser.add_argument(
        "--lut",
        type=Path,
        default=None,
        help=(
            "optional measured display-code phase-response LUT to bake into raw device codes; "
            "omit for logical 0-255 frames displayed through SLMControl3 with its LUT enabled"
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--shard-size", type=_positive_int, default=256)
    parser.add_argument("--compression", choices=("none", "lzf", "gzip"), default="none")

    optics = parser.add_argument_group("optical calibration")
    optics.add_argument("--width", type=_positive_int, default=1272, help="active SLM width")
    optics.add_argument("--height", type=_positive_int, default=1024, help="active SLM height")
    optics.add_argument("--fft-width", type=_positive_int, default=None)
    optics.add_argument("--fft-height", type=_positive_int, default=None)
    optics.add_argument("--wavelength-nm", type=_positive_float, default=407.0)
    optics.add_argument("--focal-length-mm", type=_positive_float, default=100.0)
    optics.add_argument("--pixel-pitch-um", type=_positive_float, default=12.5)
    optics.add_argument("--beam-diameter-x-mm", type=_positive_float, default=8.0)
    optics.add_argument("--beam-diameter-y-mm", type=_positive_float, default=8.0)
    optics.add_argument("--beam-center-x-mm", type=float, default=0.0)
    optics.add_argument("--beam-center-y-mm", type=float, default=0.0)
    optics.add_argument(
        "--phase-convention",
        choices=("auto", "negative-pi-to-pi", "zero-to-two-pi"),
        default="auto",
        help="override measured LUT phase convention when its absolute offset is ambiguous",
    )

    solver = parser.add_argument_group("Dawn WebGPU WGS solver")
    solver.add_argument("--iterations", type=_positive_int, default=12)
    solver.add_argument(
        "--max-iterations", type=_positive_int, default=64,
        help="hard cap on --iterations; the current WebGPU solver does not adaptively extend to this value",
    )
    solver.add_argument("--gamma", type=_positive_float, default=0.85)
    solver.add_argument("--epsilon", type=_positive_float, default=1e-8)
    solver.add_argument("--min-weight", type=_positive_float, default=0.1)
    solver.add_argument("--max-weight", type=_positive_float, default=10.0)
    solver.add_argument("--convergence-tolerance", type=_positive_float, default=1e-3)
    solver.add_argument("--solver-seed", type=_uint32, default=1)
    solver.add_argument("--background-policy", choices=("ZERO", "PRESERVE"), default="ZERO")

    sampling = parser.add_argument_group("random trap sampling")
    sampling.add_argument("--min-traps", type=_positive_int, default=1)
    sampling.add_argument("--max-traps", type=_positive_int, default=STORAGE_MAX_TRAPS)
    sampling.add_argument("--count-distribution", choices=("log-uniform", "uniform"), default="log-uniform")
    sampling.add_argument("--dataset-seed", type=_uint32, default=1)
    sampling.add_argument("--min-separation-um", type=_positive_float, default=None)
    sampling.add_argument("--field-fill-fraction", type=_unit_interval, default=0.9)
    sampling.add_argument("--x-min-um", type=float, default=None)
    sampling.add_argument("--x-max-um", type=float, default=None)
    sampling.add_argument("--y-min-um", type=float, default=None)
    sampling.add_argument("--y-max-um", type=float, default=None)
    sampling.add_argument("--zero-order-guard-um", type=_nonnegative_float, default=0.0)
    sampling.add_argument(
        "--max-attempts-per-point", "--max-sampling-attempts",
        dest="max_attempts_per_point", type=_positive_int, default=256,
    )
    sampling.add_argument(
        "--max-retries-per-sample", type=_positive_int, default=8,
        help="additional rejected position sets allowed per runner invocation; restart resumes at the next set",
    )

    dawn = parser.add_argument_group("Dawn runtime selection")
    dawn.add_argument("--dawn-backend", default=None, help="optional Dawn backend selector, for example d3d12 or vulkan")
    dawn.add_argument("--dawn-adapter", default=None, help="optional Dawn adapter name/filter")
    dawn.add_argument(
        "--dawn-option", action="append", default=[], metavar="OPTION",
        help="repeatable raw option passed to the webgpu/dawn.node create() configuration",
    )

    serving = parser.add_argument_group("localhost collector")
    serving.add_argument("--host", choices=("127.0.0.1", "localhost"), default="127.0.0.1")
    serving.add_argument("--port", type=_positive_int, default=8765)
    serving.add_argument(
        "--no-runner", action="store_true",
        help="serve the collector without starting the local Node/Dawn runner (tests/manual clients)",
    )
    return parser


def _finite_cli(value: float | None, label: str) -> float | None:
    if value is not None and not math.isfinite(value):
        raise DatasetError(f"{label} must be finite")
    return value


def configuration_from_args(args: argparse.Namespace) -> dict[str, Any]:
    lut: dict[str, Any] | None = None
    if args.lut is not None:
        values, lut_metadata = parse_phase_response_lut(args.lut)
        if args.phase_convention != "auto":
            lut_metadata["phaseConvention"] = {
                "negative-pi-to-pi": "NEGATIVE_PI_TO_PI",
                "zero-to-two-pi": "ZERO_TO_TWO_PI",
            }[args.phase_convention]
        if max(values) <= 1e-9 and min(values) < -math.pi:
            raise DatasetError(
                "A non-positive measured LUT spanning toward -2pi cannot be inverted by the "
                "device-ready phase mapper. Add 2*pi to every LUT phase so it is represented "
                "as 2pi -> 0 (or 0 -> 2pi), then use --phase-convention zero-to-two-pi."
            )
        lut = {**lut_metadata, "values": values}
    elif args.phase_convention != "auto":
        raise DatasetError("--phase-convention is only valid together with --lut")
    if args.samples > UINT32_MAX:
        raise DatasetError("--samples may not exceed 4294967295 (the deterministic sampler limit)")
    if args.width > 16384 or args.height > 16384:
        raise DatasetError("Active SLM dimensions may not exceed 16384")
    fft_width = args.fft_width or next_power_of_two(args.width)
    fft_height = args.fft_height or next_power_of_two(args.height)
    if not is_power_of_two(fft_width) or fft_width < args.width:
        raise DatasetError("--fft-width must be a power of two and at least --width")
    if not is_power_of_two(fft_height) or fft_height < args.height:
        raise DatasetError("--fft-height must be a power of two and at least --height")
    if args.iterations > args.max_iterations:
        raise DatasetError("--iterations may not exceed --max-iterations")
    if not 1 <= args.min_traps <= args.max_traps <= STORAGE_MAX_TRAPS:
        raise DatasetError("Trap limits must satisfy 1 <= min <= max <= 2000")
    if args.min_weight > args.max_weight:
        raise DatasetError("--min-weight may not exceed --max-weight")
    if not 1 <= args.port <= 65535:
        raise DatasetError("--port must be between 1 and 65535")
    for value, label, maximum_length in (
        (args.dawn_backend, "--dawn-backend", 128),
        (args.dawn_adapter, "--dawn-adapter", 512),
    ):
        if value is not None and (not value.strip() or len(value) > maximum_length):
            raise DatasetError(f"{label} must be non-empty and at most {maximum_length} characters")
    if any(not option.strip() or len(option) > 1024 for option in args.dawn_option):
        raise DatasetError("Each --dawn-option must be non-empty and at most 1024 characters")
    for field, label in (
        (args.beam_center_x_mm, "beam center X"),
        (args.beam_center_y_mm, "beam center Y"),
        (args.x_min_um, "X minimum"),
        (args.x_max_um, "X maximum"),
        (args.y_min_um, "Y minimum"),
        (args.y_max_um, "Y maximum"),
    ):
        _finite_cli(field, label)

    wavelength_um = args.wavelength_nm / 1000.0
    focal_length_um = args.focal_length_mm * 1000.0
    full_field_um = wavelength_um * focal_length_um / args.pixel_pitch_um
    half_sampled_field = full_field_um * args.field_fill_fraction / 2.0
    explicit_bounds = (args.x_min_um, args.x_max_um, args.y_min_um, args.y_max_um)
    if any(value is not None for value in explicit_bounds) and not all(
        value is not None for value in explicit_bounds
    ):
        raise DatasetError("Specify all four of --x-min-um/--x-max-um/--y-min-um/--y-max-um")
    if all(value is not None for value in explicit_bounds):
        x_min, x_max, y_min, y_max = (float(value) for value in explicit_bounds)
    else:
        x_min = y_min = -half_sampled_field
        x_max = y_max = half_sampled_field
    nyquist_half = full_field_um / 2.0
    if not (x_min < x_max and y_min < y_max):
        raise DatasetError("Sampling bounds must have strictly increasing minima and maxima")
    if min(x_min, y_min) < -nyquist_half or max(x_max, y_max) > nyquist_half:
        raise DatasetError("Sampling bounds must remain inside the optical Nyquist field")

    effective_x_mm = min(args.width * args.pixel_pitch_um / 1000.0, args.beam_diameter_x_mm)
    effective_y_mm = min(args.height * args.pixel_pitch_um / 1000.0, args.beam_diameter_y_mm)
    default_separation = 1.25 * max(
        wavelength_um * focal_length_um / (effective_x_mm * 1000.0),
        wavelength_um * focal_length_um / (effective_y_mm * 1000.0),
    )
    min_separation = args.min_separation_um or default_separation

    slmcontrol3_mode = lut is None
    configuration: dict[str, Any] = {
        "schemaVersion": SCHEMA_VERSION,
        "backend": "DAWN_WEBGPU",
        "requestedSamples": args.samples,
        "activeWidth": args.width,
        "activeHeight": args.height,
        "fftWidth": fft_width,
        "fftHeight": fft_height,
        "storageMaxTraps": STORAGE_MAX_TRAPS,
        "shardSize": args.shard_size,
        "compression": args.compression,
        "output": {
            "pixelFormat": "UINT8",
            "frameMode": "SLMCONTROL3_LOGICAL" if slmcontrol3_mode else "DEVICE_READY_LUT_BAKED",
            "deviceReady": not slmcontrol3_mode,
            "displayReady": True,
            "lutApplication": "SLMCONTROL3" if slmcontrol3_mode else "DAWN_NODE",
            "slmControl3LutMustBeEnabled": slmcontrol3_mode,
            "slmControl3LutMustBeDisabled": not slmcontrol3_mode,
            "frameSemantics": (
                "logical 0-255 phase codes; SLMControl3 applies its wavelength-specific LUT"
                if slmcontrol3_mode
                else "raw device codes with the supplied measured phase-response LUT baked in"
            ),
        },
        "targets": {
            "intensity": 1.0,
            "inputTargetPhaseRad": 0.0,
            "phaseLabel": (
                "solver-computed focal-plane phases from the final quantized SLMControl3 "
                "logical frame under ideal controller compensation"
                if slmcontrol3_mode
                else "solver-computed focal-plane phases from the final quantized device-ready frame"
            ),
            "positionUnits": "um",
            "phaseUnits": "rad",
        },
        "dawn": {
            "backend": args.dawn_backend.strip() if args.dawn_backend is not None else None,
            "adapter": args.dawn_adapter.strip() if args.dawn_adapter is not None else None,
            "options": list(args.dawn_option),
        },
        "lut": lut,
        "optics": {
            "wavelengthNm": args.wavelength_nm,
            "focalLengthMm": args.focal_length_mm,
            "pixelPitchUm": args.pixel_pitch_um,
            "incidentBeam": {
                "profile": "GAUSSIAN",
                "diameterXMm": args.beam_diameter_x_mm,
                "diameterYMm": args.beam_diameter_y_mm,
                "centerXMm": args.beam_center_x_mm,
                "centerYMm": args.beam_center_y_mm,
            },
        },
        "solver": {
            "targetPhaseMode": "REFERENCE_WGS",
            "iterations": args.iterations,
            "firstFrameIterations": args.iterations,
            "subsequentFrameIterations": args.iterations,
            "maxIterations": args.max_iterations,
            "gamma": args.gamma,
            "epsilon": args.epsilon,
            "minWeight": args.min_weight,
            "maxWeight": args.max_weight,
            "convergenceTolerance": args.convergence_tolerance,
            "deterministicSeed": args.solver_seed,
            "backgroundPolicy": args.background_policy,
            "oversampling": 1,
            "qualityGates": {},
            "requireConvergence": True,
            "measureSolveTime": True,
            "format": "UINT8",
            "independentSamples": True,
        },
        "sampling": {
            "minTraps": args.min_traps,
            "maxTraps": args.max_traps,
            "distribution": args.count_distribution,
            "datasetSeed": args.dataset_seed,
            "minSeparationUm": min_separation,
            "fieldFillFraction": args.field_fill_fraction,
            "xMinUm": x_min,
            "xMaxUm": x_max,
            "yMinUm": y_min,
            "yMaxUm": y_max,
            "zeroOrderGuardUm": args.zero_order_guard_um,
            "maxAttemptsPerPoint": args.max_attempts_per_point,
            "maxRetriesPerSample": args.max_retries_per_sample,
        },
    }
    # Source location is provenance, not a numerical setting. Moving an
    # identical optional LUT must not make an interrupted run impossible to resume.
    hashable = json.loads(canonical_json(configuration))
    if hashable["lut"] is not None:
        hashable["lut"].pop("sourcePath", None)
        hashable["lut"].pop("filename", None)
    configuration["configHash"] = sha256_bytes(canonical_json(hashable).encode("utf-8"))
    return configuration


@dataclass(frozen=True)
class SampleRecord:
    sample_id: int
    trap_count: int
    sampling_seed: int
    frame_crc32: int
    width: int
    height: int
    positions: Any
    measured_phases: Any
    trap_ids: Any
    frame: Any
    metrics: dict[str, Any]
    metrics_json: str


def _require_numpy() -> None:
    if np is None:
        raise DatasetError(
            "Missing dataset dependencies; install them with "
            "`python -m pip install -r dataset/requirements.txt`"
        )


def _json_object_no_duplicates(raw: bytes) -> dict[str, Any]:
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ProtocolError(f"metrics JSON repeats key {key!r}")
            result[key] = value
        return result

    try:
        decoded = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ProtocolError("metrics JSON must be valid UTF-8") from error
    try:
        value = json.loads(
            decoded,
            object_pairs_hook=object_pairs,
            parse_constant=lambda token: (_raise_protocol_constant(token)),
        )
    except json.JSONDecodeError as error:
        raise ProtocolError(f"metrics JSON is invalid: {error.msg}") from error
    if not isinstance(value, dict):
        raise ProtocolError("metrics JSON must be an object")
    return value


def _raise_protocol_constant(token: str) -> Any:
    raise ProtocolError(f"metrics JSON value {token} is not finite")


def parse_sample_payload(
    payload: bytes,
    *,
    expected_width: int | None = None,
    expected_height: int | None = None,
    maximum_traps: int = STORAGE_MAX_TRAPS,
) -> SampleRecord:
    _require_numpy()
    if len(payload) < SAMPLE_HEADER_BYTES:
        raise ProtocolError(f"sample payload is shorter than the {SAMPLE_HEADER_BYTES}-byte header")
    unpacked = SAMPLE_HEADER.unpack_from(payload)
    (
        magic,
        version,
        header_bytes,
        sample_id,
        trap_count,
        sampling_seed,
        frame_crc32,
        width,
        height,
        positions_bytes,
        phases_bytes,
        trap_ids_bytes,
        frame_bytes,
        metrics_bytes,
        flags,
        reserved,
    ) = unpacked
    if magic != PROTOCOL_MAGIC:
        raise ProtocolError("sample payload magic is not SLMD")
    if version != PROTOCOL_VERSION:
        raise ProtocolError(f"unsupported sample protocol version {version}")
    if header_bytes != SAMPLE_HEADER_BYTES:
        raise ProtocolError(f"sample header size must be {SAMPLE_HEADER_BYTES}")
    if flags != 0 or reserved != 0:
        raise ProtocolError("sample header flags and reserved fields must be zero")
    if not 1 <= trap_count <= min(maximum_traps, STORAGE_MAX_TRAPS):
        raise ProtocolError(f"trap_count must be between 1 and {min(maximum_traps, STORAGE_MAX_TRAPS)}")
    if width == 0 or height == 0:
        raise ProtocolError("frame dimensions must be non-zero")
    if expected_width is not None and width != expected_width:
        raise ProtocolError(f"frame width {width} does not match configured width {expected_width}")
    if expected_height is not None and height != expected_height:
        raise ProtocolError(f"frame height {height} does not match configured height {expected_height}")
    expected_lengths = (
        trap_count * 2 * 4,
        trap_count * 4,
        trap_count * 4,
        width * height,
    )
    supplied_lengths = (positions_bytes, phases_bytes, trap_ids_bytes, frame_bytes)
    if supplied_lengths != expected_lengths:
        raise ProtocolError(
            "sample section lengths are inconsistent with trap count/frame dimensions: "
            f"got {supplied_lengths}, expected {expected_lengths}"
        )
    if not 2 <= metrics_bytes <= MAX_METRICS_BYTES:
        raise ProtocolError(f"metrics JSON must contain 2 to {MAX_METRICS_BYTES} bytes")
    expected_total = SAMPLE_HEADER_BYTES + sum(expected_lengths) + metrics_bytes
    if len(payload) != expected_total:
        raise ProtocolError(f"sample payload has {len(payload)} bytes; expected exactly {expected_total}")

    cursor = SAMPLE_HEADER_BYTES
    positions = np.frombuffer(payload, dtype="<f4", count=trap_count * 2, offset=cursor).reshape(trap_count, 2)
    cursor += positions_bytes
    measured_phases = np.frombuffer(payload, dtype="<f4", count=trap_count, offset=cursor)
    cursor += phases_bytes
    trap_ids = np.frombuffer(payload, dtype="<u4", count=trap_count, offset=cursor)
    cursor += trap_ids_bytes
    frame = np.frombuffer(payload, dtype="u1", count=frame_bytes, offset=cursor).reshape(height, width)
    cursor += frame_bytes
    metrics_raw = payload[cursor:]
    metrics = _json_object_no_duplicates(metrics_raw)
    actual_crc32 = zlib.crc32(memoryview(frame).cast("B")) & UINT32_MAX
    if actual_crc32 != frame_crc32:
        raise ProtocolError(
            f"frame CRC-32 mismatch: header {frame_crc32:08x}, calculated {actual_crc32:08x}"
        )
    return SampleRecord(
        sample_id=sample_id,
        trap_count=trap_count,
        sampling_seed=sampling_seed,
        frame_crc32=frame_crc32,
        width=width,
        height=height,
        positions=positions,
        measured_phases=measured_phases,
        trap_ids=trap_ids,
        frame=frame,
        metrics=metrics,
        metrics_json=metrics_raw.decode("utf-8"),
    )


def pack_sample_payload(
    *,
    sample_id: int,
    sampling_seed: int,
    positions: Any,
    measured_phases: Any,
    trap_ids: Any,
    frame: Any,
    metrics: Mapping[str, Any],
) -> bytes:
    """Reference encoder used by tests and non-Node protocol clients."""
    _require_numpy()
    if not 0 <= int(sample_id) <= UINT64_MAX:
        raise ValueError("sample_id is outside uint64")
    if not 0 <= int(sampling_seed) <= UINT32_MAX:
        raise ValueError("sampling_seed is outside uint32")
    positions_array = np.ascontiguousarray(positions, dtype="<f4")
    phases_array = np.ascontiguousarray(measured_phases, dtype="<f4")
    ids_array = np.ascontiguousarray(trap_ids, dtype="<u4")
    frame_array = np.ascontiguousarray(frame, dtype="u1")
    if positions_array.ndim != 2 or positions_array.shape[1] != 2:
        raise ValueError("positions must have shape [K,2]")
    trap_count = int(positions_array.shape[0])
    if phases_array.shape != (trap_count,) or ids_array.shape != (trap_count,):
        raise ValueError("phase and trap ID counts must match positions")
    if frame_array.ndim != 2:
        raise ValueError("frame must have shape [height,width]")
    metrics_raw = canonical_json(dict(metrics)).encode("utf-8")
    sections = (
        positions_array.tobytes(order="C"),
        phases_array.tobytes(order="C"),
        ids_array.tobytes(order="C"),
        frame_array.tobytes(order="C"),
    )
    frame_crc32 = zlib.crc32(sections[3]) & UINT32_MAX
    header = SAMPLE_HEADER.pack(
        PROTOCOL_MAGIC,
        PROTOCOL_VERSION,
        SAMPLE_HEADER_BYTES,
        int(sample_id),
        trap_count,
        int(sampling_seed),
        frame_crc32,
        int(frame_array.shape[1]),
        int(frame_array.shape[0]),
        len(sections[0]),
        len(sections[1]),
        len(sections[2]),
        len(sections[3]),
        len(metrics_raw),
        0,
        0,
    )
    return b"".join((header, *sections, metrics_raw))


METRIC_DATASETS: dict[str, tuple[str, str, Any]] = {
    "frameIndex": ("frame_index", "<u8", 0),
    "timeUs": ("time_us", "<f8", 0.0),
    "iterations": ("iterations", "<u4", 0),
    "converged": ("converged", "?", False),
    "maximumRelativeAmplitudeError": ("maximum_relative_amplitude_error", "<f4", math.nan),
    "amplitudeConvergenceTolerance": ("amplitude_convergence_tolerance", "<f4", math.nan),
    "phaseConvergenceToleranceRad": ("phase_convergence_tolerance_rad", "<f4", math.nan),
    "targetIntensityMean": ("target_intensity_mean", "<f4", math.nan),
    "targetIntensityStd": ("target_intensity_std", "<f4", math.nan),
    "targetIntensityCoefficientOfVariation": ("target_intensity_coefficient_of_variation", "<f4", math.nan),
    "minimumToMeanIntensityRatio": ("minimum_to_mean_intensity_ratio", "<f4", math.nan),
    "diffractionEfficiency": ("diffraction_efficiency", "<f4", math.nan),
    "maximumGhostIntensity": ("maximum_ghost_intensity", "<f4", math.nan),
    "maximumWgsWeight": ("maximum_wgs_weight", "<f4", math.nan),
    "maximumTargetPhaseErrorRad": ("maximum_target_phase_error_rad", "<f4", math.nan),
    "targetPhaseChangeRad": ("target_phase_change_rad", "<f4", math.nan),
    "displayCodeChange": ("display_code_change", "<f4", math.nan),
    "estimatedTransitionMinimumIntensity": ("estimated_transition_minimum_intensity", "<f4", math.nan),
    "solveTimeMs": ("solve_time_ms", "<f4", math.nan),
    "refinementCount": ("refinement_count", "<u4", 0),
    "numericalValid": ("numerical_valid", "?", False),
    "accepted": ("accepted", "?", False),
}

REQUIRED_FINITE_METRICS = {
    "maximumRelativeAmplitudeError",
    "amplitudeConvergenceTolerance",
    "phaseConvergenceToleranceRad",
    "maximumTargetPhaseErrorRad",
    "targetIntensityMean",
    "targetIntensityStd",
    "targetIntensityCoefficientOfVariation",
    "minimumToMeanIntensityRatio",
    "diffractionEfficiency",
    "maximumGhostIntensity",
    "maximumWgsWeight",
    "targetPhaseChangeRad",
    "displayCodeChange",
    "estimatedTransitionMinimumIntensity",
    "solveTimeMs",
}


def _validate_json_finite(value: Any, path: str = "metrics") -> None:
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, (int, float)):
        try:
            finite = math.isfinite(float(value))
        except (OverflowError, ValueError) as error:
            raise SampleRejectedError(f"{path} is not finite") from error
        if not finite:
            raise SampleRejectedError(f"{path} is not finite")
        return
    if isinstance(value, list):
        for index, member in enumerate(value):
            _validate_json_finite(member, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, member in value.items():
            _validate_json_finite(member, f"{path}.{key}")
        return
    raise SampleRejectedError(f"{path} has an unsupported JSON value")


def validate_accepted_sample(record: SampleRecord, configuration: Mapping[str, Any]) -> None:
    _require_numpy()
    if record.width != configuration["activeWidth"] or record.height != configuration["activeHeight"]:
        raise SampleRejectedError("Frame dimensions do not match the active SLM calibration")
    if record.trap_count > configuration["sampling"]["maxTraps"]:
        raise SampleRejectedError("Trap count exceeds the configured sampling maximum")
    if not np.isfinite(record.positions).all():
        raise SampleRejectedError("Trap positions contain NaN or infinity")
    if not np.isfinite(record.measured_phases).all():
        raise SampleRejectedError("Measured phases contain NaN or infinity")
    phase_limit = math.pi + 2e-5
    if np.any(record.measured_phases < -phase_limit) or np.any(record.measured_phases > phase_limit):
        raise SampleRejectedError("Measured phases must be wrapped to [-pi, pi]")
    if np.unique(record.trap_ids).size != record.trap_count:
        raise SampleRejectedError("trap_ids must be unique within a sample")
    bounds = configuration["sampling"]
    tolerance = 1e-4
    if np.any(record.positions[:, 0] < bounds["xMinUm"] - tolerance) or np.any(
        record.positions[:, 0] > bounds["xMaxUm"] + tolerance
    ):
        raise SampleRejectedError("Trap X position lies outside configured sampling bounds")
    if np.any(record.positions[:, 1] < bounds["yMinUm"] - tolerance) or np.any(
        record.positions[:, 1] > bounds["yMaxUm"] + tolerance
    ):
        raise SampleRejectedError("Trap Y position lies outside configured sampling bounds")
    _validate_position_geometry(record.positions, bounds)

    metrics = record.metrics
    _validate_json_finite(metrics)
    for boolean_name in ("converged", "numericalValid", "accepted"):
        if metrics.get(boolean_name) is not True:
            raise SampleRejectedError(f"metrics.{boolean_name} must be true")
    flags = metrics.get("flags")
    if not isinstance(flags, list) or any(not isinstance(flag, str) for flag in flags):
        raise SampleRejectedError("metrics.flags must be a string array")
    forbidden = {"NOT_CONVERGED", "NUMERIC_ERROR", "ZERO_TARGET_OUTPUT"}.intersection(flags)
    if forbidden:
        raise SampleRejectedError(f"metrics contains rejection flags: {sorted(forbidden)}")
    iterations = metrics.get("iterations")
    if isinstance(iterations, bool) or not isinstance(iterations, int) or not (
        1 <= iterations <= configuration["solver"]["maxIterations"]
    ):
        raise SampleRejectedError("metrics.iterations is outside the configured solver budget")
    if iterations > configuration["solver"]["iterations"]:
        raise SampleRejectedError("metrics.iterations exceeds the configured per-sample budget")
    frame_index = metrics.get("frameIndex")
    if isinstance(frame_index, bool) or not isinstance(frame_index, int) or frame_index != record.sample_id:
        raise SampleRejectedError("metrics.frameIndex must equal sample_id")
    refinement_count = metrics.get("refinementCount")
    if isinstance(refinement_count, bool) or not isinstance(refinement_count, int) or refinement_count < 0:
        raise SampleRejectedError("metrics.refinementCount must be a non-negative integer")
    if "timeUs" not in metrics or _finite_number(metrics["timeUs"], "metrics.timeUs") < 0:
        raise SampleRejectedError("metrics.timeUs must be a finite non-negative number")
    for key in REQUIRED_FINITE_METRICS:
        if key not in metrics:
            raise SampleRejectedError(f"metrics.{key} is required")
        _finite_number(metrics[key], f"metrics.{key}")
    amplitude_error = float(metrics["maximumRelativeAmplitudeError"])
    amplitude_limit = float(metrics["amplitudeConvergenceTolerance"])
    phase_error = float(metrics["maximumTargetPhaseErrorRad"])
    phase_tolerance = float(metrics["phaseConvergenceToleranceRad"])
    configured_limit = float(configuration["solver"]["convergenceTolerance"])
    if not math.isclose(amplitude_limit, configured_limit, rel_tol=1e-5, abs_tol=1e-8):
        raise SampleRejectedError("Reported amplitude tolerance differs from collector configuration")
    if amplitude_error > amplitude_limit + 1e-7:
        raise SampleRejectedError("Reported amplitude error exceeds its convergence tolerance")
    if phase_error > phase_tolerance + 1e-7:
        raise SampleRejectedError("Reported phase error exceeds its convergence tolerance")
    nonnegative = (
        "maximumRelativeAmplitudeError",
        "amplitudeConvergenceTolerance",
        "phaseConvergenceToleranceRad",
        "maximumTargetPhaseErrorRad",
        "targetIntensityMean",
        "targetIntensityStd",
        "targetIntensityCoefficientOfVariation",
        "minimumToMeanIntensityRatio",
        "diffractionEfficiency",
        "maximumGhostIntensity",
        "maximumWgsWeight",
        "targetPhaseChangeRad",
        "displayCodeChange",
        "estimatedTransitionMinimumIntensity",
        "solveTimeMs",
    )
    if any(float(metrics[key]) < 0 for key in nonnegative):
        raise SampleRejectedError("Metrics that describe errors, powers, or times must be non-negative")


def _validate_position_geometry(positions: Any, sampling: Mapping[str, Any]) -> None:
    """Recheck the sampler's spacing and zero-order guard in linear time."""
    guard = float(sampling["zeroOrderGuardUm"])
    if guard > 0 and np.any(np.sum(np.square(positions.astype("<f8")), axis=1) < guard * guard):
        raise SampleRejectedError("A trap lies inside the configured zero-order guard")
    spacing = float(sampling["minSeparationUm"])
    if positions.shape[0] < 2 or spacing <= 0:
        return
    inverse_cell = 1.0 / spacing
    cells: dict[tuple[int, int], list[tuple[float, float]]] = {}
    threshold_squared = spacing * spacing * (1.0 - 1e-6)
    for raw_x, raw_y in positions:
        x, y = float(raw_x), float(raw_y)
        cell = (math.floor(x * inverse_cell), math.floor(y * inverse_cell))
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for other_x, other_y in cells.get((cell[0] + dx, cell[1] + dy), ()):
                    if (x - other_x) ** 2 + (y - other_y) ** 2 < threshold_squared:
                        raise SampleRejectedError("Trap positions violate the configured minimum separation")
        cells.setdefault(cell, []).append((x, y))


class DatasetCollector:
    """Transactional HDF5 shard writer with resumable manifest state."""

    def __init__(self, output_dir: Path, configuration: dict[str, Any]):
        if h5py is None or np is None:
            _require_numpy()
            raise DatasetError("h5py is required to write the dataset")
        self.output_dir = output_dir.expanduser().resolve()
        self.configuration = configuration
        self.config_hash = str(configuration["configHash"])
        self.requested_samples = int(configuration["requestedSamples"])
        self.shard_size = int(configuration["shardSize"])
        self.compression = str(configuration["compression"])
        self.manifest_path = self.output_dir / "manifest.json"
        self.lock = threading.RLock()
        self.complete_event = threading.Event()
        self._partial_file: Any | None = None
        self._partial_path: Path | None = None
        self._partial_index: int | None = None
        self._partial_capacity = 0
        self._partial_rows = 0
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.manifest = self._load_or_create_manifest()
        self._resume_and_validate()
        if self.manifest["complete"]:
            self.complete_event.set()

    @property
    def accepted_samples(self) -> int:
        return int(self.manifest["acceptedSamples"])

    @property
    def next_sample_id(self) -> int:
        return self.accepted_samples

    def _load_or_create_manifest(self) -> dict[str, Any]:
        if self.manifest_path.exists():
            try:
                manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise DatasetError(f"Cannot read complete manifest {self.manifest_path}: {error}") from error
            if not isinstance(manifest, dict):
                raise DatasetError("Existing manifest must be a JSON object")
            if manifest.get("manifestVersion") != MANIFEST_VERSION or manifest.get("schemaVersion") != SCHEMA_VERSION:
                raise ConfigurationMismatchError("Existing manifest uses an incompatible schema version")
            if manifest.get("configurationHash") != self.config_hash:
                raise ConfigurationMismatchError(
                    "Existing dataset configuration does not match this invocation; use a new output directory"
                )
            if manifest.get("requestedSamples") != self.requested_samples:
                raise ConfigurationMismatchError("Existing manifest has a different requested sample count")
            return manifest

        leftovers = list(self.output_dir.glob("shard-*.h5")) + list(self.output_dir.glob("shard-*.h5.partial"))
        if leftovers:
            raise DatasetError("Shard files exist without manifest.json; refusing to guess their configuration")
        now = utc_now()
        manifest = {
            "manifestVersion": MANIFEST_VERSION,
            "schemaVersion": SCHEMA_VERSION,
            "protocol": {
                "magic": PROTOCOL_MAGIC.decode("ascii"),
                "version": PROTOCOL_VERSION,
                "headerBytes": SAMPLE_HEADER_BYTES,
                "littleEndian": True,
            },
            "configurationHash": self.config_hash,
            "configuration": self.configuration,
            "createdUtc": now,
            "updatedUtc": now,
            "requestedSamples": self.requested_samples,
            "acceptedSamples": 0,
            "rejectedSamples": 0,
            "rejectionsByReason": {},
            "currentSampleRejections": 0,
            "complete": False,
            "shards": [],
            "partial": None,
        }
        self.manifest = manifest
        self._write_manifest()
        return manifest

    def _write_manifest(self) -> None:
        self.manifest["updatedUtc"] = utc_now()
        payload = (json.dumps(self.manifest, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode("utf-8")
        temporary = self.output_dir / f".manifest.{os.getpid()}.{threading.get_ident()}.tmp"
        try:
            with temporary.open("wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.manifest_path)
        finally:
            if temporary.exists():
                temporary.unlink()

    def _resume_and_validate(self) -> None:
        with self.lock:
            shards = self.manifest.get("shards")
            if not isinstance(shards, list):
                raise DatasetError("manifest.shards must be an array")
            manifest_accepted = self.manifest.get("acceptedSamples")
            if isinstance(manifest_accepted, bool) or not isinstance(manifest_accepted, int) or manifest_accepted < 0:
                raise DatasetError("manifest.acceptedSamples must be a non-negative integer")
            total = 0
            for expected_index, entry in enumerate(shards):
                if not isinstance(entry, dict) or entry.get("index") != expected_index:
                    raise DatasetError("Manifest shard indexes must be contiguous from zero")
                path = self.output_dir / str(entry.get("filename", ""))
                info = self._inspect_closed_shard(path, expected_index, total)
                if info["count"] != entry.get("count"):
                    raise DatasetError(f"Manifest count does not match {path.name}")
                for field in ("firstSampleId", "lastSampleId", "bytes", "sha256"):
                    if entry.get(field) != info[field]:
                        raise DatasetError(f"Manifest {field} does not match {path.name}")
                total += info["count"]

            partial_info = self.manifest.get("partial")
            next_index = len(shards)
            expected_partial = self.output_dir / f"shard-{next_index:05d}.h5.partial"
            expected_final = self.output_dir / f"shard-{next_index:05d}.h5"
            if partial_info is not None:
                if not isinstance(partial_info, dict) or partial_info.get("index") != next_index:
                    raise DatasetError("Manifest partial shard metadata is invalid")
                declared_path = self.output_dir / str(partial_info.get("filename", ""))
                if declared_path != expected_partial:
                    raise DatasetError("Manifest partial shard filename is invalid")
                if expected_partial.exists():
                    rows = self._open_existing_partial(expected_partial, next_index, total)
                    total += rows
                    self.manifest["partial"]["count"] = rows
                elif expected_final.exists():
                    # Recovery for a crash after atomic rename but before the
                    # corresponding atomic manifest update.
                    info = self._inspect_closed_shard(expected_final, next_index, total)
                    self.manifest["shards"].append(info)
                    self.manifest["partial"] = None
                    total += info["count"]
                else:
                    raise DatasetError("Manifest references a missing partial shard")
            elif expected_partial.exists():
                rows = self._open_existing_partial(expected_partial, next_index, total)
                total += rows
                self.manifest["partial"] = {
                    "index": next_index,
                    "filename": expected_partial.name,
                    "count": rows,
                    "capacity": self._partial_capacity,
                    "firstSampleId": total - rows,
                }
            elif expected_final.exists():
                info = self._inspect_closed_shard(expected_final, next_index, total)
                self.manifest["shards"].append(info)
                total += info["count"]

            known_names = {entry["filename"] for entry in self.manifest["shards"]}
            if self.manifest.get("partial"):
                known_names.add(self.manifest["partial"]["filename"])
            extras = [
                path.name for pattern in ("shard-*.h5", "shard-*.h5.partial")
                for path in self.output_dir.glob(pattern) if path.name not in known_names
            ]
            if extras:
                raise DatasetError(f"Unexpected shard files are present: {sorted(set(extras))}")
            if total > self.requested_samples:
                raise DatasetError("Existing shards contain more samples than requested")
            if bool(self.manifest.get("complete")) != ("completedUtc" in self.manifest):
                raise DatasetError("Manifest completion metadata is inconsistent")
            if self.manifest.get("complete") and total != self.requested_samples:
                raise DatasetError("A complete manifest does not contain the requested sample count")
            if total > manifest_accepted:
                # A row becomes durable when committed_rows is flushed, before
                # the manifest advances and clears retry state.  Recovering
                # such a row also advances next_sample_id, so retry metadata
                # from the now-accepted sample must not leak into its successor.
                self.manifest["currentSampleRejections"] = 0
                self.manifest.pop("currentTrapCount", None)
            self.manifest["acceptedSamples"] = total
            self._write_manifest()
            if self._partial_file is not None and self._partial_rows == self._partial_capacity:
                self._finalize_partial()

    def _validate_hdf_root(self, file: Any, index: int) -> None:
        if file.attrs.get("schema_version") != SCHEMA_VERSION:
            raise DatasetError(f"Shard {index} has an incompatible schema")
        if file.attrs.get("configuration_sha256") != self.config_hash:
            raise ConfigurationMismatchError(f"Shard {index} belongs to another configuration")
        if int(file.attrs.get("shard_index", -1)) != index:
            raise DatasetError(f"Shard {index} stores the wrong shard index")

    def _sample_datasets(self, file: Any) -> Iterable[Any]:
        for name in (
            "frames",
            "positions",
            "measured_phases",
            "trap_ids",
            "trap_count",
            "sample_id",
            "sampling_seed",
            "frame_crc32",
        ):
            yield file[name]
        metrics = file["metrics"]
        for name in metrics:
            yield metrics[name]

    def _validate_shapes_and_ids(self, file: Any, count: int, first_sample_id: int) -> None:
        expected_tail = {
            "frames": (self.configuration["activeHeight"], self.configuration["activeWidth"]),
            "positions": (STORAGE_MAX_TRAPS, 2),
            "measured_phases": (STORAGE_MAX_TRAPS,),
            "trap_ids": (STORAGE_MAX_TRAPS,),
            "trap_count": (),
            "sample_id": (),
            "sampling_seed": (),
            "frame_crc32": (),
        }
        for name, tail in expected_tail.items():
            dataset = file.get(name)
            if dataset is None or dataset.shape != (count, *tail):
                raise DatasetError(f"Shard dataset {name} has an invalid shape")
        for dataset in file["metrics"].values():
            if dataset.shape != (count,):
                raise DatasetError(f"Metric dataset {dataset.name} has an invalid shape")
        if count:
            ids = file["sample_id"][:]
            expected = np.arange(first_sample_id, first_sample_id + count, dtype="<u8")
            if not np.array_equal(ids, expected):
                raise DatasetError("Shard sample IDs are not contiguous")

    def _inspect_closed_shard(self, path: Path, index: int, first_sample_id: int) -> dict[str, Any]:
        if not path.is_file():
            raise DatasetError(f"Missing shard {path}")
        with h5py.File(path, "r") as file:
            self._validate_hdf_root(file, index)
            if "finalized_utc" not in file.attrs:
                raise DatasetError(f"Finalized shard {path.name} lacks its close marker")
            count = int(file.attrs.get("committed_rows", -1))
            if count <= 0:
                raise DatasetError(f"Finalized shard {path.name} is empty")
            self._validate_shapes_and_ids(file, count, first_sample_id)
        return {
            "index": index,
            "filename": path.name,
            "count": count,
            "firstSampleId": first_sample_id,
            "lastSampleId": first_sample_id + count - 1,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }

    def _open_existing_partial(self, path: Path, index: int, first_sample_id: int) -> int:
        file = h5py.File(path, "r+")
        try:
            self._validate_hdf_root(file, index)
            capacity = int(file.attrs.get("capacity", 0))
            committed = int(file.attrs.get("committed_rows", -1))
            if not 0 <= committed <= capacity <= self.shard_size:
                raise DatasetError(f"Partial shard {path.name} has invalid row counters")
            for dataset in self._sample_datasets(file):
                if dataset.shape[0] < committed:
                    raise DatasetError(f"Partial dataset {dataset.name} is shorter than committed_rows")
                if dataset.shape[0] != committed:
                    dataset.resize(committed, axis=0)
            self._validate_shapes_and_ids(file, committed, first_sample_id)
            file.flush()
        except Exception:
            file.close()
            raise
        self._partial_file = file
        self._partial_path = path
        self._partial_index = index
        self._partial_capacity = capacity
        self._partial_rows = committed
        return committed

    def _compression_kwargs(self) -> dict[str, Any]:
        if self.compression == "none":
            return {}
        if self.compression == "gzip":
            return {"compression": "gzip", "compression_opts": 1}
        return {"compression": "lzf"}

    def _create_partial(self) -> None:
        if self._partial_file is not None:
            return
        index = len(self.manifest["shards"])
        remaining = self.requested_samples - self.accepted_samples
        if remaining <= 0:
            raise DatasetError("No remaining samples need a shard")
        capacity = min(self.shard_size, remaining)
        path = self.output_dir / f"shard-{index:05d}.h5.partial"
        if path.exists() or path.with_suffix("").exists():
            raise DatasetError(f"Refusing to overwrite existing shard path {path}")
        file = h5py.File(path, "w", libver="latest")
        try:
            hdf_configuration = json.loads(canonical_json(self.configuration))
            if hdf_configuration["lut"] is not None:
                hdf_configuration["lut"].pop("values", None)
            output = self.configuration["output"]
            file.attrs.update({
                "schema_version": SCHEMA_VERSION,
                "configuration_sha256": self.config_hash,
                "shard_index": index,
                "capacity": capacity,
                "committed_rows": 0,
                "created_utc": utc_now(),
                "backend": self.configuration["backend"],
                "pixel_format": "UINT8",
                "frame_mode": output["frameMode"],
                "frame_semantics": output["frameSemantics"],
                "display_ready": bool(output["displayReady"]),
                "device_ready": bool(output["deviceReady"]),
                "lut_application": output["lutApplication"],
                "slmcontrol3_lut_must_be_enabled": bool(output["slmControl3LutMustBeEnabled"]),
                "slmcontrol3_lut_must_be_disabled": bool(output["slmControl3LutMustBeDisabled"]),
                "active_width": self.configuration["activeWidth"],
                "active_height": self.configuration["activeHeight"],
                "fft_width": self.configuration["fftWidth"],
                "fft_height": self.configuration["fftHeight"],
                "storage_max_traps": STORAGE_MAX_TRAPS,
                # Optional LUT values live in /calibration/phase_response_lut.
                # Keeping a potentially large array out of an HDF5 attribute
                # avoids object-header size limits without losing metadata.
                "configuration_json": canonical_json(hdf_configuration),
                "optics_json": canonical_json(self.configuration["optics"]),
                "solver_json": canonical_json(self.configuration["solver"]),
                "sampling_json": canonical_json(self.configuration["sampling"]),
                "dawn_json": canonical_json(self.configuration["dawn"]),
                "coordinate_units": "micrometres",
                "phase_units": "radians",
                "coordinate_convention": "Fraunhofer focal plane: +x right, +y up",
            })
            compression = self._compression_kwargs()
            height = self.configuration["activeHeight"]
            width = self.configuration["activeWidth"]
            file.create_dataset(
                "frames", shape=(0, height, width), maxshape=(capacity, height, width),
                chunks=(1, height, width), dtype="u1", **compression,
            )
            file.create_dataset(
                "positions", shape=(0, STORAGE_MAX_TRAPS, 2), maxshape=(capacity, STORAGE_MAX_TRAPS, 2),
                chunks=(1, STORAGE_MAX_TRAPS, 2), dtype="<f4", fillvalue=0.0,
            )
            file["positions"].attrs.update({"units": "um", "padding_value": 0.0})
            file.create_dataset(
                "measured_phases", shape=(0, STORAGE_MAX_TRAPS), maxshape=(capacity, STORAGE_MAX_TRAPS),
                chunks=(1, STORAGE_MAX_TRAPS), dtype="<f4", fillvalue=0.0,
            )
            file["measured_phases"].attrs.update({
                "units": "rad",
                "wrapped_range": "[-pi, pi]",
                "padding_value": 0.0,
                "semantics": self.configuration["targets"]["phaseLabel"],
                "experimental_measurement": False,
            })
            file.create_dataset(
                "trap_ids", shape=(0, STORAGE_MAX_TRAPS), maxshape=(capacity, STORAGE_MAX_TRAPS),
                chunks=(1, STORAGE_MAX_TRAPS), dtype="<u4", fillvalue=0,
            )
            file["trap_ids"].attrs["padding_value"] = 0
            for name, dtype in (
                ("trap_count", "<u2"),
                ("sample_id", "<u8"),
                ("sampling_seed", "<u4"),
                ("frame_crc32", "<u4"),
            ):
                file.create_dataset(name, shape=(0,), maxshape=(capacity,), chunks=(min(capacity, 256),), dtype=dtype)

            calibration = file.create_group("calibration")
            lut = self.configuration["lut"]
            if lut is None:
                calibration.attrs.update({
                    "phase_response_source": "SLMCONTROL3_INTERNAL_LUT",
                    "phase_convention": "NEGATIVE_PI_TO_PI",
                    "lut_semantics": (
                        "not embedded; frames are logical 0-255 phase codes and SLMControl3 "
                        "must apply its wavelength-specific calibration LUT"
                    ),
                })
            else:
                calibration.attrs.update({
                    "phase_response_source": "MEASURED_LUT_EMBEDDED",
                    "lut_filename": lut["filename"],
                    "lut_source_path": lut["sourcePath"],
                    "lut_source_sha256": lut["sourceSha256"],
                    "lut_values_sha256": lut["valuesSha256"],
                    "lut_direction": lut["direction"],
                    "phase_convention": lut["phaseConvention"],
                    "lut_semantics": (
                        "uniform samples over display codes 0..255 to measured phase radians; "
                        "baked into device-ready frame"
                    ),
                })
                calibration.create_dataset(
                    "phase_response_lut", data=np.asarray(lut["values"], dtype="<f8")
                )

            metrics_group = file.create_group("metrics")
            for _, (name, dtype, _) in METRIC_DATASETS.items():
                metrics_group.create_dataset(
                    name, shape=(0,), maxshape=(capacity,), chunks=(min(capacity, 256),), dtype=dtype
                )
            text_dtype = h5py.string_dtype(encoding="utf-8")
            metrics_group.create_dataset(
                "json", shape=(0,), maxshape=(capacity,), chunks=(min(capacity, 256),), dtype=text_dtype
            )
            metrics_group.create_dataset(
                "flags_json", shape=(0,), maxshape=(capacity,), chunks=(min(capacity, 256),), dtype=text_dtype
            )
            file.flush()
        except Exception:
            file.close()
            raise
        self._partial_file = file
        self._partial_path = path
        self._partial_index = index
        self._partial_capacity = capacity
        self._partial_rows = 0
        self.manifest["partial"] = {
            "index": index,
            "filename": path.name,
            "count": 0,
            "capacity": capacity,
            "firstSampleId": self.accepted_samples,
        }
        self._write_manifest()

    def _append_record(self, record: SampleRecord) -> tuple[int, int]:
        self._create_partial()
        file = self._partial_file
        assert file is not None and self._partial_index is not None
        row = self._partial_rows
        shard_index = self._partial_index
        datasets = list(self._sample_datasets(file))
        for dataset in datasets:
            dataset.resize(row + 1, axis=0)
        try:
            file["frames"][row] = record.frame
            positions = np.zeros((STORAGE_MAX_TRAPS, 2), dtype="<f4")
            positions[:record.trap_count] = record.positions
            phases = np.zeros((STORAGE_MAX_TRAPS,), dtype="<f4")
            phases[:record.trap_count] = record.measured_phases
            trap_ids = np.zeros((STORAGE_MAX_TRAPS,), dtype="<u4")
            trap_ids[:record.trap_count] = record.trap_ids
            file["positions"][row] = positions
            file["measured_phases"][row] = phases
            file["trap_ids"][row] = trap_ids
            file["trap_count"][row] = record.trap_count
            file["sample_id"][row] = record.sample_id
            file["sampling_seed"][row] = record.sampling_seed
            file["frame_crc32"][row] = record.frame_crc32
            metrics_group = file["metrics"]
            for key, (name, _, default) in METRIC_DATASETS.items():
                metrics_group[name][row] = record.metrics.get(key, default)
            metrics_group["json"][row] = record.metrics_json
            metrics_group["flags_json"][row] = canonical_json(record.metrics["flags"])
            file.flush()
            file.attrs.modify("committed_rows", row + 1)
            file.flush()
        except Exception:
            for dataset in datasets:
                dataset.resize(row, axis=0)
            file.flush()
            raise
        self._partial_rows = row + 1
        self.manifest["acceptedSamples"] = self.accepted_samples + 1
        self.manifest["currentSampleRejections"] = 0
        self.manifest.pop("currentTrapCount", None)
        self.manifest["partial"]["count"] = self._partial_rows
        self._write_manifest()
        if self._partial_rows == self._partial_capacity:
            self._finalize_partial()
        return shard_index, row

    def _finalize_partial(self) -> None:
        if self._partial_file is None or self._partial_path is None or self._partial_index is None:
            return
        if self._partial_rows <= 0:
            return
        file = self._partial_file
        partial_path = self._partial_path
        index = self._partial_index
        rows = self._partial_rows
        first_sample_id = self.accepted_samples - rows
        file.attrs["finalized_utc"] = utc_now()
        file.flush()
        file.close()
        self._partial_file = None
        final_path = self.output_dir / f"shard-{index:05d}.h5"
        if final_path.exists():
            raise DatasetError(f"Refusing to overwrite finalized shard {final_path.name}")
        os.replace(partial_path, final_path)
        info = {
            "index": index,
            "filename": final_path.name,
            "count": rows,
            "firstSampleId": first_sample_id,
            "lastSampleId": first_sample_id + rows - 1,
            "bytes": final_path.stat().st_size,
            "sha256": sha256_file(final_path),
        }
        self.manifest["shards"].append(info)
        self.manifest["partial"] = None
        self._partial_path = None
        self._partial_index = None
        self._partial_capacity = 0
        self._partial_rows = 0
        self._write_manifest()

    def accept_payload(self, payload: bytes) -> dict[str, Any]:
        with self.lock:
            if self.manifest["complete"]:
                raise DatasetError("Dataset is already complete")
            if self.accepted_samples >= self.requested_samples:
                raise DatasetError("All requested samples are already present; call /api/complete")
            record = parse_sample_payload(
                payload,
                expected_width=self.configuration["activeWidth"],
                expected_height=self.configuration["activeHeight"],
                maximum_traps=self.configuration["sampling"]["maxTraps"],
            )
            if record.sample_id != self.next_sample_id:
                raise ProtocolError(
                    f"Expected sample_id {self.next_sample_id}, received {record.sample_id}"
                )
            retried_trap_count = self.manifest.get("currentTrapCount")
            if retried_trap_count is not None and record.trap_count != retried_trap_count:
                raise ProtocolError("Accepted retry changed trap_count for the current sample_id")
            validate_accepted_sample(record, self.configuration)
            shard_index, row_index = self._append_record(record)
            return {
                "ok": True,
                "acceptedSamples": self.accepted_samples,
                "nextSampleId": self.next_sample_id,
                "remainingSamples": self.requested_samples - self.accepted_samples,
                "shardIndex": shard_index,
                "rowIndex": row_index,
            }

    def record_rejection(self, value: Mapping[str, Any]) -> dict[str, Any]:
        with self.lock:
            if self.manifest["complete"]:
                raise DatasetError("Dataset is already complete")
            sample_id = _parse_json_uint(value.get("sample_id"), "sample_id", UINT64_MAX)
            if sample_id != self.next_sample_id:
                raise ProtocolError(f"Expected rejection for sample_id {self.next_sample_id}")
            sampling_seed = _parse_json_uint(value.get("sampling_seed"), "sampling_seed", UINT32_MAX)
            reason = value.get("reason")
            if not isinstance(reason, str) or not reason.strip() or len(reason) > 512:
                raise ProtocolError("rejection reason must be a non-empty string of at most 512 characters")
            for optional_name in ("trap_count", "attempt"):
                if optional_name in value:
                    _parse_json_uint(value[optional_name], optional_name, UINT32_MAX)
            current_rejections = int(self.manifest.get("currentSampleRejections", 0))
            if "attempt" in value:
                attempt = int(value["attempt"])
                if attempt != current_rejections + 1:
                    raise ProtocolError(
                        f"rejection attempt must be {current_rejections + 1} for the current sample"
                    )
            if "trap_count" in value:
                trap_count = int(value["trap_count"])
                if not 1 <= trap_count <= self.configuration["sampling"]["maxTraps"]:
                    raise ProtocolError("rejection trap_count is outside configured sampling limits")
                configured_count = self.manifest.get("currentTrapCount")
                if configured_count is not None and trap_count != configured_count:
                    raise ProtocolError("retries for one sample_id must keep the same trap_count")
                self.manifest["currentTrapCount"] = trap_count
            if "metrics" in value:
                if not isinstance(value["metrics"], dict):
                    raise ProtocolError("rejection metrics must be an object")
                _validate_json_finite(value["metrics"])
            reason_detail = reason.strip()
            reason_key = reason_detail.split(":", 1)[0].strip()
            if not reason_key:
                raise ProtocolError("rejection reason must start with a non-empty reason code")
            self.manifest["rejectedSamples"] = int(self.manifest.get("rejectedSamples", 0)) + 1
            self.manifest["currentSampleRejections"] = current_rejections + 1
            by_reason = self.manifest.setdefault("rejectionsByReason", {})
            by_reason[reason_key] = int(by_reason.get(reason_key, 0)) + 1
            self.manifest["lastRejection"] = {
                "sampleId": sample_id,
                "samplingSeed": sampling_seed,
                "reasonCode": reason_key,
                "reason": reason_detail,
                "trapCount": value.get("trap_count"),
                "attempt": value.get("attempt"),
                "utc": utc_now(),
            }
            self._write_manifest()
            return {
                "ok": True,
                "sampleId": sample_id,
                "rejectedSamples": self.manifest["rejectedSamples"],
                "currentSampleRejections": self.manifest["currentSampleRejections"],
                "maxRetriesPerSample": self.configuration["sampling"]["maxRetriesPerSample"],
                "nextSampleId": self.next_sample_id,
            }

    def complete(self, summary: Any = None) -> dict[str, Any]:
        with self.lock:
            if self.manifest["complete"]:
                return self.status()
            if self.accepted_samples != self.requested_samples:
                raise DatasetError(
                    f"Cannot complete: {self.accepted_samples}/{self.requested_samples} accepted samples"
                )
            if summary is not None:
                _validate_json_finite(summary, "summary")
                encoded = canonical_json(summary).encode("utf-8")
                if len(encoded) > MAX_JSON_BODY_BYTES:
                    raise ProtocolError("completion summary is too large")
                self.manifest["runnerSummary"] = summary
            self._finalize_partial()
            self.manifest["complete"] = True
            self.manifest["completedUtc"] = utc_now()
            self._write_manifest()
            self.complete_event.set()
            return self.status()

    def public_config(self) -> dict[str, Any]:
        with self.lock:
            public = json.loads(canonical_json(self.configuration))
            public.update({
                "protocol": {
                    "magic": PROTOCOL_MAGIC.decode("ascii"),
                    "version": PROTOCOL_VERSION,
                    "headerBytes": SAMPLE_HEADER_BYTES,
                    "littleEndian": True,
                },
                "acceptedSamples": self.accepted_samples,
                "nextSampleId": self.next_sample_id,
                "remainingSamples": self.requested_samples - self.accepted_samples,
                "complete": bool(self.manifest["complete"]),
                "endpoints": {
                    "config": "/api/config",
                    "sample": "/api/sample",
                    "rejection": "/api/rejection",
                    "complete": "/api/complete",
                    "status": "/api/status",
                },
            })
            return public

    def status(self) -> dict[str, Any]:
        with self.lock:
            return {
                "ok": True,
                "complete": bool(self.manifest["complete"]),
                "requestedSamples": self.requested_samples,
                "acceptedSamples": self.accepted_samples,
                "remainingSamples": self.requested_samples - self.accepted_samples,
                "nextSampleId": self.next_sample_id,
                "rejectedSamples": int(self.manifest.get("rejectedSamples", 0)),
                "currentSampleRejections": int(self.manifest.get("currentSampleRejections", 0)),
                "currentTrapCount": self.manifest.get("currentTrapCount"),
                "rejectionsByReason": dict(self.manifest.get("rejectionsByReason", {})),
                "completedShards": len(self.manifest["shards"]),
                "shards": [
                    {
                        "filename": shard["filename"],
                        "count": shard["count"],
                        "firstSampleId": shard["firstSampleId"],
                        "lastSampleId": shard["lastSampleId"],
                        "bytes": shard["bytes"],
                        "sha256": shard["sha256"],
                    }
                    for shard in self.manifest["shards"]
                ],
                "partialShard": self.manifest.get("partial"),
                "configurationHash": self.config_hash,
                "outputDirectory": str(self.output_dir),
                "manifestPath": str(self.manifest_path),
            }

    def close(self) -> None:
        with self.lock:
            if self._partial_file is not None:
                self._partial_file.flush()
                self._partial_file.close()
                self._partial_file = None


def _parse_json_uint(value: Any, label: str, maximum: int) -> int:
    if isinstance(value, str):
        if not value.isascii() or not value.isdecimal():
            raise ProtocolError(f"{label} string must contain only decimal digits")
        result = int(value, 10)
    elif isinstance(value, int) and not isinstance(value, bool):
        result = value
    else:
        raise ProtocolError(f"{label} must be an integer or decimal string")
    if not 0 <= result <= maximum:
        raise ProtocolError(f"{label} is outside its unsigned integer range")
    return result


class CollectorHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], collector: DatasetCollector):
        self.collector = collector
        self.stop_event = threading.Event()
        self.completion_response_sent = threading.Event()
        super().__init__(address, CollectorRequestHandler)

    def serve_until_complete(self) -> None:
        self.timeout = 0.25
        while not self.stop_event.is_set() and not self.collector.complete_event.is_set():
            self.handle_request()
        if self.collector.complete_event.is_set():
            # The collector marks completion while evaluating the argument to
            # _send_json. Keep the CLI alive until that final response has
            # actually reached the Node/Dawn runner.
            self.completion_response_sent.wait(timeout=5)


class CollectorRequestHandler(BaseHTTPRequestHandler):
    server: CollectorHTTPServer
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:
        sys.stderr.write(f"[collector] {self.address_string()} - {format % args}\n")

    def _headers(self, content_length: int, status: int = HTTPStatus.OK) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(content_length))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()

    def _send_json(self, value: Any, status: int = HTTPStatus.OK) -> None:
        encoded = canonical_json(value).encode("utf-8")
        self._headers(len(encoded), status)
        self.wfile.write(encoded)

    def _send_error_json(self, error: Exception) -> None:
        if isinstance(error, (ProtocolError, SampleRejectedError)):
            status = HTTPStatus.UNPROCESSABLE_ENTITY
        elif isinstance(error, ConfigurationMismatchError):
            status = HTTPStatus.CONFLICT
        elif isinstance(error, DatasetError):
            status = HTTPStatus.CONFLICT
        else:
            status = HTTPStatus.INTERNAL_SERVER_ERROR
        self._send_json({"ok": False, "error": str(error), "errorType": type(error).__name__}, status)

    def _content_length(self, maximum: int) -> int:
        raw = self.headers.get("Content-Length")
        if raw is None:
            raise ProtocolError("Content-Length is required")
        try:
            length = int(raw, 10)
        except ValueError as error:
            raise ProtocolError("Content-Length must be an integer") from error
        if not 0 <= length <= maximum:
            raise ProtocolError(f"Request body exceeds the {maximum}-byte limit")
        return length

    def _read_json(self, *, allow_empty: bool = False) -> dict[str, Any]:
        length = self._content_length(MAX_JSON_BODY_BYTES)
        raw = self.rfile.read(length)
        if allow_empty and not raw:
            return {}
        try:
            value = json.loads(raw.decode("utf-8"), parse_constant=lambda token: (_raise_protocol_constant(token)))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ProtocolError("Request body must be valid UTF-8 JSON") from error
        if not isinstance(value, dict):
            raise ProtocolError("Request JSON must be an object")
        return value

    def do_OPTIONS(self) -> None:  # noqa: N802
        self._headers(0, HTTPStatus.NO_CONTENT)

    def do_GET(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path
        try:
            if path == "/api/config":
                self._send_json(self.server.collector.public_config())
            elif path == "/api/status":
                self._send_json(self.server.collector.status())
            else:
                self._send_json({"ok": False, "error": "Not found"}, HTTPStatus.NOT_FOUND)
        except Exception as error:  # pragma: no cover - unexpected I/O failures.
            self._send_error_json(error)

    def do_POST(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path
        try:
            if path == "/api/sample":
                config = self.server.collector.configuration
                maximum = (
                    SAMPLE_HEADER_BYTES
                    + int(config["sampling"]["maxTraps"]) * 16
                    + int(config["activeWidth"]) * int(config["activeHeight"])
                    + MAX_METRICS_BYTES
                )
                length = self._content_length(maximum)
                self._send_json(self.server.collector.accept_payload(self.rfile.read(length)))
            elif path == "/api/rejection":
                self._send_json(self.server.collector.record_rejection(self._read_json()))
            elif path == "/api/complete":
                body = self._read_json(allow_empty=True)
                response = self.server.collector.complete(body.get("summary"))
                self._send_json(response)
                self.wfile.flush()
                self.server.completion_response_sent.set()
            else:
                self._send_json({"ok": False, "error": "Not found"}, HTTPStatus.NOT_FOUND)
        except Exception as error:
            self._send_error_json(error)


def start_dawn_runner(repo_root: Path, collector_url: str) -> subprocess.Popen[Any]:
    node = shutil.which("node")
    tsx_entry = repo_root / "dataset" / "node_modules" / "tsx" / "dist" / "cli.mjs"
    runner_entry = repo_root / "dataset" / "src" / "generate.ts"
    if node is None:
        raise DatasetError("Node.js is required to run the Dawn dataset generator")
    if not tsx_entry.is_file():
        raise DatasetError(
            "The local dataset/tsx installation is missing; run `npm install` in dataset/ first"
        )
    if not runner_entry.is_file():
        raise DatasetError(f"The Dawn runner source is missing: {runner_entry}")
    command = [
        node,
        str(tsx_entry),
        str(runner_entry),
        "--collector",
        collector_url,
    ]
    # Inherit the collector console so JSONL status/progress from the Dawn
    # runner remains visible during long high-trap-count solves. No new console
    # is requested on Windows.
    return subprocess.Popen(command, cwd=repo_root, creationflags=0)


def stop_process(process: subprocess.Popen[Any] | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    collector: DatasetCollector | None = None
    server: CollectorHTTPServer | None = None
    server_thread: threading.Thread | None = None
    runner: subprocess.Popen[Any] | None = None
    try:
        configuration = configuration_from_args(args)
        collector = DatasetCollector(args.output_dir, configuration)
        if collector.manifest["complete"]:
            print(f"Dataset is already complete: {collector.manifest_path}")
            return 0

        server = CollectorHTTPServer((args.host, args.port), collector)
        server_thread = threading.Thread(target=server.serve_until_complete, name="hdf5-collector", daemon=True)
        server_thread.start()
        collector_origin = f"http://{args.host}:{args.port}"

        print(f"Collector: {collector_origin}", flush=True)
        print("Runner: manual / disabled" if args.no_runner else "Runner: Node + Dawn (dawn.node)", flush=True)
        print(
            f"Progress: {collector.accepted_samples}/{collector.requested_samples} accepted",
            flush=True,
        )
        if not args.no_runner:
            repo_root = Path(__file__).resolve().parent.parent
            runner = start_dawn_runner(repo_root, collector_origin)

        while server_thread.is_alive():
            server_thread.join(timeout=0.5)
            if runner is not None and runner.poll() is not None and not collector.complete_event.is_set():
                raise DatasetError(
                    f"Dawn runner exited with code {runner.returncode} before completing the dataset"
                )
        if runner is not None:
            try:
                runner_returncode = runner.wait(timeout=10)
            except subprocess.TimeoutExpired as error:
                raise DatasetError("Dawn runner did not exit after collector completion") from error
            if runner_returncode != 0:
                raise DatasetError(f"Dawn runner exited with code {runner_returncode}")
        print(f"Complete: {collector.accepted_samples} samples in {len(collector.manifest['shards'])} shards")
        print(f"Manifest: {collector.manifest_path}")
        return 0
    except KeyboardInterrupt:
        print("Interrupted; flushed partial shard can be resumed with the same command.", file=sys.stderr)
        return 130
    except (DatasetError, OSError) as error:
        print(f"dataset generation error: {error}", file=sys.stderr)
        return 2
    finally:
        # Stop new GPU work/uploads before closing the collector and its open
        # partial HDF5 file. This ordering makes Ctrl+C and runner failures
        # leave only a flushed, resumable prefix.
        stop_process(runner)
        if server is not None:
            server.stop_event.set()
            if server_thread is not None:
                server_thread.join(timeout=2)
            server.server_close()
        if collector is not None:
            collector.close()


if __name__ == "__main__":
    raise SystemExit(main())
