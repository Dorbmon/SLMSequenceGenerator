from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
h5py = pytest.importorskip("h5py")

import dataset.generate as generate_module  # noqa: E402
from dataset.generate import (  # noqa: E402
    ConfigurationMismatchError,
    DatasetCollector,
    DatasetError,
    DEFAULT_OUTPUT_DIR,
    CollectorRequestHandler,
    PROTOCOL_MAGIC,
    PROTOCOL_VERSION,
    ProtocolError,
    SAMPLE_HEADER_BYTES,
    SampleRejectedError,
    build_argument_parser,
    configuration_from_args,
    pack_sample_payload,
    parse_phase_response_lut,
    parse_sample_payload,
    start_dawn_runner,
)


def test_collector_access_log_suppresses_success_and_keeps_http_errors(
    capsys: pytest.CaptureFixture[str],
) -> None:
    handler = object.__new__(CollectorRequestHandler)
    handler.address_string = lambda: "127.0.0.1"  # type: ignore[method-assign]

    handler.log_message('"%s" %s %s', "POST /api/sample", "200", "-")
    assert capsys.readouterr().err == ""

    handler.log_message('"%s" %s %s', "POST /api/sample", "422", "-")
    assert '[collector] 127.0.0.1 - "POST /api/sample" 422 -' in capsys.readouterr().err


def test_lut_parses_supported_formats_and_rejects_invalid_data(tmp_path: Path) -> None:
    json_array = tmp_path / "array.json"
    json_array.write_text("[0, 1.1, 3.2, 6.3]", encoding="utf-8")
    values, metadata = parse_phase_response_lut(json_array)
    assert values == [0.0, 1.1, 3.2, 6.3]
    assert metadata["direction"] == "INCREASING"
    assert metadata["phaseConvention"] == "ZERO_TO_TWO_PI"
    assert metadata["length"] == 4
    assert len(metadata["sourceSha256"]) == 64

    json_object = tmp_path / "object.json"
    json_object.write_text('{"phaseResponseLut":[3.1,2.0,2.0,-3.2]}', encoding="utf-8")
    values, metadata = parse_phase_response_lut(json_object)
    assert values == [3.1, 2.0, 2.0, -3.2]
    assert metadata["direction"] == "DECREASING"
    assert metadata["phaseConvention"] == "NEGATIVE_PI_TO_PI"

    decreasing_zero_to_two_pi = tmp_path / "decreasing-zero-to-two-pi.json"
    decreasing_zero_to_two_pi.write_text("[6.4, 4.2, 2.1, 0]", encoding="utf-8")
    _, metadata = parse_phase_response_lut(decreasing_zero_to_two_pi)
    assert metadata["direction"] == "DECREASING"
    assert metadata["phaseConvention"] == "ZERO_TO_TWO_PI"

    code_phase_csv = tmp_path / "measured.csv"
    code_phase_csv.write_text(
        "code,phase_rad\n0,0\n85,0.8\n170,2.1\n255,6.4\n", encoding="utf-8"
    )
    code_values, _ = parse_phase_response_lut(code_phase_csv)
    assert len(code_values) == 256
    assert [code_values[index] for index in (0, 85, 170, 255)] == [0.0, 0.8, 2.1, 6.4]

    sparse_code_phase_csv = tmp_path / "sparse-measured.csv"
    sparse_code_phase_csv.write_text(
        "code,phase_rad\n0,0\n128,3.2\n255,6.4\n", encoding="utf-8"
    )
    sparse_values, _ = parse_phase_response_lut(sparse_code_phase_csv)
    assert len(sparse_values) == 256
    assert sparse_values[0] == 0.0
    assert sparse_values[128] == 3.2
    assert sparse_values[255] == 6.4

    whitespace = tmp_path / "measured.txt"
    whitespace.write_text("-3.14  -1.2\n0.4  3.14\n", encoding="utf-8")
    assert parse_phase_response_lut(whitespace)[0] == [-3.14, -1.2, 0.4, 3.14]

    for name, text in (
        ("reverse.txt", "0 2 1 6.2"),
        ("constant.txt", "1 1 1"),
        ("nan.json", "[0, NaN, 6.3]"),
        ("bad-object.json", '{"unknown":[0,1]}'),
        ("empty.csv", "code,phase_rad\n0,0\n1,\n"),
        ("bad-codes.csv", "code,phase_rad\n0,0\n2,1\n3,2\n"),
        ("short-explicit-codes.csv", "code,phase_rad\n0,0\n1,1\n2,2\n3,3\n"),
    ):
        path = tmp_path / name
        path.write_text(text, encoding="utf-8")
        with pytest.raises(DatasetError):
            parse_phase_response_lut(path)


def test_configuration_rejects_nonpositive_two_pi_lut(tmp_path: Path) -> None:
    lut_path = tmp_path / "negative-two-pi.json"
    lut_path.write_text(json.dumps([0.0, -2.1, -4.2, -6.3]), encoding="utf-8")
    parser = build_argument_parser()
    args = parser.parse_args(["--samples", "1", "--lut", str(lut_path)])

    with pytest.raises(DatasetError, match=r"Add 2\*pi"):
        configuration_from_args(args)


def test_configuration_defaults_to_slmcontrol3_logical_frames(tmp_path: Path) -> None:
    parser = build_argument_parser()
    logical = configuration_from_args(parser.parse_args(["--samples", "1"]))
    assert logical["solver"]["implementation"] == "webgpu-exact-nudft-export-certified-wgs-v2"
    assert logical["solver"]["convergenceTolerance"] == pytest.approx(0.02)
    assert logical["lut"] is None
    assert logical["output"] == {
        "pixelFormat": "UINT8",
        "frameMode": "SLMCONTROL3_LOGICAL",
        "deviceReady": False,
        "displayReady": True,
        "lutApplication": "SLMCONTROL3",
        "slmControl3LutMustBeEnabled": True,
        "slmControl3LutMustBeDisabled": False,
        "frameSemantics": (
            "logical 0-255 phase codes; SLMControl3 applies its wavelength-specific LUT"
        ),
    }

    with pytest.raises(DatasetError, match="only valid together with --lut"):
        configuration_from_args(parser.parse_args([
            "--samples", "1", "--phase-convention", "zero-to-two-pi",
        ]))

    lut_path = tmp_path / "lut.json"
    lut_path.write_text("[0, 3.2, 6.4]", encoding="utf-8")
    baked = configuration_from_args(parser.parse_args([
        "--samples", "1", "--lut", str(lut_path),
    ]))
    assert baked["lut"]["values"] == [0.0, 3.2, 6.4]
    assert baked["output"]["frameMode"] == "DEVICE_READY_LUT_BAKED"
    assert baked["output"]["lutApplication"] == "DAWN_NODE"
    assert baked["output"]["slmControl3LutMustBeDisabled"] is True
    assert baked["configHash"] != logical["configHash"]


def test_default_output_and_local_dawn_runner_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = build_argument_parser().parse_args(["--samples", "1"])
    assert args.output_dir == DEFAULT_OUTPUT_DIR

    repo_root = tmp_path / "repo"
    tsx_entry = repo_root / "dataset" / "node_modules" / "tsx" / "dist" / "cli.mjs"
    runner_entry = repo_root / "dataset" / "src" / "generate.ts"
    tsx_entry.parent.mkdir(parents=True)
    runner_entry.parent.mkdir(parents=True)
    tsx_entry.write_text("", encoding="utf-8")
    runner_entry.write_text("", encoding="utf-8")
    captured: dict[str, object] = {}

    class FakeProcess:
        pass

    fake_process = FakeProcess()
    monkeypatch.setattr(generate_module.shutil, "which", lambda executable: "C:/node/node.exe")

    def fake_popen(command: list[str], **kwargs: object) -> FakeProcess:
        captured["command"] = command
        captured["kwargs"] = kwargs
        return fake_process

    monkeypatch.setattr(generate_module.subprocess, "Popen", fake_popen)
    result = start_dawn_runner(repo_root, "http://127.0.0.1:8765")
    assert result is fake_process
    assert captured["command"] == [
        "C:/node/node.exe",
        str(tsx_entry),
        str(runner_entry),
        "--collector",
        "http://127.0.0.1:8765",
    ]
    assert captured["kwargs"] == {
        "cwd": repo_root,
        "creationflags": 0,
    }


def test_protocol_matches_typescript_cross_language_vector() -> None:
    positions = np.asarray([[1.5, -2.25], [3.75, 4.5]], dtype="<f4")
    phases = np.asarray([-math.pi, math.pi / 2], dtype="<f4")
    ids = np.asarray([1, 2], dtype="<u4")
    frame = np.asarray([[0x00, 0x7F, 0x80], [0xFF, 0x01, 0x02]], dtype="u1")
    payload = pack_sample_payload(
        sample_id=0x0123456789ABCDEF,
        sampling_seed=0xFEDCBA98,
        positions=positions,
        measured_phases=phases,
        trap_ids=ids,
        frame=frame,
        metrics={"converged": True, "iterations": 4},
    )
    expected_header = (
        "534c4d4401004000efcdab89674523010200000098badcfe0604946e03000000"
        "0200000010000000080000000800000006000000210000000000000000000000"
    )
    assert SAMPLE_HEADER_BYTES == 64
    assert payload[:64].hex() == expected_header
    assert len(payload) == 135
    parsed = parse_sample_payload(payload, expected_width=3, expected_height=2)
    assert PROTOCOL_MAGIC == b"SLMD" and PROTOCOL_VERSION == 1
    assert parsed.sample_id == 0x0123456789ABCDEF
    assert parsed.frame_crc32 == 0x6E940406
    np.testing.assert_array_equal(parsed.frame, frame)
    np.testing.assert_allclose(parsed.positions, positions)
    np.testing.assert_allclose(parsed.measured_phases, phases)
    np.testing.assert_array_equal(parsed.trap_ids, ids)


def _configuration(
    tmp_path: Path,
    *,
    samples: int,
    shard_size: int,
    with_lut: bool = False,
) -> dict:
    parser = build_argument_parser()
    arguments = [
        "--samples", str(samples),
        "--output-dir", str(tmp_path / "data"),
        "--shard-size", str(shard_size),
        "--width", "4",
        "--height", "3",
        "--fft-width", "4",
        "--fft-height", "4",
        "--iterations", "2",
        "--convergence-tolerance", "0.1",
        "--min-traps", "1",
        "--max-traps", "4",
        "--min-separation-um", "0.1",
        "--x-min-um", "-100",
        "--x-max-um", "100",
        "--y-min-um", "-100",
        "--y-max-um", "100",
        "--dawn-backend", "d3d12",
        "--dawn-adapter", "test-adapter",
        "--dawn-option", "enable-features=timestamp-query",
        "--dawn-option", "validation=full",
        "--no-runner",
    ]
    if with_lut:
        lut = tmp_path / "lut.json"
        lut.write_text("[0, 2.1, 4.2, 6.4]", encoding="utf-8")
        arguments.extend(["--lut", str(lut)])
    args = parser.parse_args(arguments)
    configuration = configuration_from_args(args)
    assert configuration["backend"] == "DAWN_WEBGPU"
    assert configuration["output"]["lutApplication"] == (
        "DAWN_NODE" if with_lut else "SLMCONTROL3"
    )
    assert configuration["dawn"] == {
        "backend": "d3d12",
        "adapter": "test-adapter",
        "options": ["enable-features=timestamp-query", "validation=full"],
    }
    return configuration


def _metrics(*, converged: bool = True, frame_index: int = 0) -> dict:
    return {
        "frameIndex": frame_index,
        "timeUs": 0,
        "iterations": 2,
        "converged": converged,
        "maximumRelativeAmplitudeError": 0.01,
        "amplitudeConvergenceTolerance": 0.1,
        "phaseConvergenceToleranceRad": 0.2,
        "targetIntensityMean": 2.0,
        "targetIntensityStd": 0.01,
        "targetIntensityCoefficientOfVariation": 0.005,
        "minimumToMeanIntensityRatio": 0.99,
        "diffractionEfficiency": 0.5,
        "maximumGhostIntensity": 0.01,
        "maximumWgsWeight": 1.2,
        "maximumTargetPhaseErrorRad": 0.1,
        "targetPhaseChangeRad": 0.0,
        "displayCodeChange": 0.0,
        "estimatedTransitionMinimumIntensity": 1.0,
        "solveTimeMs": 12.5,
        "refinementCount": 0,
        "numericalValid": True,
        "accepted": converged,
        "flags": [] if converged else ["NOT_CONVERGED"],
    }


def _payload(sample_id: int, *, converged: bool = True) -> bytes:
    return pack_sample_payload(
        sample_id=sample_id,
        sampling_seed=100 + sample_id,
        positions=np.asarray([[-1.0, 2.0], [3.0, -4.0]], dtype="<f4"),
        measured_phases=np.asarray([-0.5, 1.25], dtype="<f4"),
        trap_ids=np.asarray([10, 20], dtype="<u4"),
        frame=(np.arange(12, dtype="u1").reshape(3, 4) + sample_id),
        metrics=_metrics(converged=converged, frame_index=sample_id),
    )


def test_hdf5_sharding_schema_padding_manifest_and_resume(tmp_path: Path) -> None:
    configuration = _configuration(tmp_path, samples=4, shard_size=3)
    output = tmp_path / "data"
    collector = DatasetCollector(output, configuration)
    collector.accept_payload(_payload(0))
    collector.close()

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["partial"]["count"] == 1
    assert (output / "shard-00000.h5.partial").exists()

    resumed = DatasetCollector(output, configuration)
    assert resumed.next_sample_id == 1
    for sample_id in range(1, 4):
        resumed.accept_payload(_payload(sample_id))
    result = resumed.complete({"gpu": "test-adapter"})
    resumed.close()
    assert result["complete"] is True

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["acceptedSamples"] == 4
    assert manifest["complete"] is True
    assert [entry["count"] for entry in manifest["shards"]] == [3, 1]
    assert manifest["runnerSummary"] == {"gpu": "test-adapter"}
    assert not list(output.glob("*.partial"))

    with h5py.File(output / "shard-00000.h5", "r") as shard:
        assert shard["frames"].shape == (3, 3, 4)
        assert shard["frames"].chunks == (1, 3, 4)
        assert shard["positions"].shape == (3, 2000, 2)
        assert shard["measured_phases"].shape == (3, 2000)
        assert shard["trap_ids"].shape == (3, 2000)
        assert shard["trap_count"][:].tolist() == [2, 2, 2]
        assert shard["sample_id"][:].tolist() == [0, 1, 2]
        assert np.all(shard["positions"][0, 2:] == 0)
        assert np.all(shard["measured_phases"][0, 2:] == 0)
        assert np.all(shard["trap_ids"][0, 2:] == 0)
        assert shard["metrics/converged"][:].tolist() == [True, True, True]
        assert "phase_response_lut" not in shard["calibration"]
        assert shard["calibration"].attrs["phase_response_source"] == "SLMCONTROL3_INTERNAL_LUT"
        assert shard.attrs["backend"] == "DAWN_WEBGPU"
        assert shard.attrs["frame_mode"] == "SLMCONTROL3_LOGICAL"
        assert shard.attrs["lut_application"] == "SLMCONTROL3"
        assert json.loads(shard.attrs["dawn_json"]) == configuration["dawn"]
        assert bool(shard.attrs["display_ready"]) is True
        assert bool(shard.attrs["device_ready"]) is False
        assert bool(shard.attrs["slmcontrol3_lut_must_be_enabled"]) is True
        assert bool(shard.attrs["slmcontrol3_lut_must_be_disabled"]) is False
        assert bool(shard["measured_phases"].attrs["experimental_measurement"]) is False

    completed = DatasetCollector(output, configuration)
    assert completed.status()["complete"] is True
    completed.close()

    changed = dict(configuration)
    changed["configHash"] = "0" * 64
    with pytest.raises(ConfigurationMismatchError):
        DatasetCollector(output, changed)


def test_optional_measured_lut_keeps_baked_device_code_mode(tmp_path: Path) -> None:
    configuration = _configuration(tmp_path, samples=1, shard_size=1, with_lut=True)
    output = tmp_path / "data"
    collector = DatasetCollector(output, configuration)
    collector.accept_payload(_payload(0))
    collector.complete()
    collector.close()

    with h5py.File(output / "shard-00000.h5", "r") as shard:
        assert shard["calibration/phase_response_lut"][:].tolist() == [0.0, 2.1, 4.2, 6.4]
        assert shard["calibration"].attrs["phase_response_source"] == "MEASURED_LUT_EMBEDDED"
        assert shard.attrs["frame_mode"] == "DEVICE_READY_LUT_BAKED"
        assert shard.attrs["lut_application"] == "DAWN_NODE"
        assert bool(shard.attrs["device_ready"]) is True
        assert bool(shard.attrs["slmcontrol3_lut_must_be_enabled"]) is False
        assert bool(shard.attrs["slmcontrol3_lut_must_be_disabled"]) is True


def test_resume_clears_retry_state_when_hdf_commit_leads_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configuration = _configuration(tmp_path, samples=2, shard_size=2)
    output = tmp_path / "data"
    collector = DatasetCollector(output, configuration)
    collector.record_rejection({
        "sample_id": 0,
        "sampling_seed": 101,
        "trap_count": 2,
        "attempt": 1,
        "reason": "NOT_CONVERGED",
    })
    collector._create_partial()

    def fail_manifest_write() -> None:
        raise OSError("simulated crash after HDF commit")

    monkeypatch.setattr(collector, "_write_manifest", fail_manifest_write)
    with pytest.raises(OSError, match="simulated crash"):
        collector.accept_payload(_payload(0))
    collector.close()

    stale = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert stale["acceptedSamples"] == 0
    assert stale["currentSampleRejections"] == 1
    assert stale["currentTrapCount"] == 2

    resumed = DatasetCollector(output, configuration)
    status = resumed.status()
    assert status["acceptedSamples"] == 1
    assert status["nextSampleId"] == 1
    assert status["currentSampleRejections"] == 0
    assert status["rejectedSamples"] == 1
    assert status["rejectionsByReason"] == {"NOT_CONVERGED": 1}
    assert "currentTrapCount" not in resumed.manifest
    resumed.close()

    repaired = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert repaired["acceptedSamples"] == 1
    assert repaired["currentSampleRejections"] == 0
    assert "currentTrapCount" not in repaired


def test_rejection_attempts_continue_across_per_run_retry_windows(tmp_path: Path) -> None:
    configuration = _configuration(tmp_path, samples=1, shard_size=1)
    output = tmp_path / "data"
    retry_window = configuration["sampling"]["maxRetriesPerSample"]
    collector = DatasetCollector(output, configuration)

    # One runner invocation tries the initial candidate plus maxRetriesPerSample
    # retries. Every rejection is durable so a later run must start after it.
    for attempt in range(1, retry_window + 2):
        collector.record_rejection({
            "sample_id": 0,
            "sampling_seed": 1000 + attempt,
            "trap_count": 2,
            "attempt": attempt,
            "reason": "NOT_CONVERGED",
        })
    collector.close()

    resumed = DatasetCollector(output, configuration)
    assert resumed.status()["currentSampleRejections"] == retry_window + 1
    continued_attempt = retry_window + 2
    resumed.record_rejection({
        "sample_id": 0,
        "sampling_seed": 1000 + continued_attempt,
        "trap_count": 2,
        "attempt": continued_attempt,
        "reason": "NOT_CONVERGED",
    })
    assert resumed.status()["currentSampleRejections"] == continued_attempt

    with pytest.raises(ProtocolError, match=f"must be {continued_attempt + 1}"):
        resumed.record_rejection({
            "sample_id": 0,
            "sampling_seed": 9999,
            "trap_count": 2,
            "attempt": continued_attempt + 2,
            "reason": "NOT_CONVERGED",
        })

    resumed.accept_payload(_payload(0))
    status = resumed.status()
    assert status["acceptedSamples"] == 1
    assert status["currentSampleRejections"] == 0
    assert "currentTrapCount" not in resumed.manifest
    resumed.close()


def test_nonconverged_and_internally_inconsistent_metrics_are_rejected(tmp_path: Path) -> None:
    configuration = _configuration(tmp_path, samples=1, shard_size=1)
    collector = DatasetCollector(tmp_path / "data", configuration)
    with pytest.raises(SampleRejectedError, match="converged"):
        collector.accept_payload(_payload(0, converged=False))
    assert collector.accepted_samples == 0

    metrics = _metrics()
    metrics["maximumRelativeAmplitudeError"] = 0.2
    inconsistent = pack_sample_payload(
        sample_id=0,
        sampling_seed=9,
        positions=np.asarray([[-1.0, 2.0], [3.0, -4.0]], dtype="<f4"),
        measured_phases=np.asarray([-0.5, 1.25], dtype="<f4"),
        trap_ids=np.asarray([10, 20], dtype="<u4"),
        frame=np.arange(12, dtype="u1").reshape(3, 4),
        metrics=metrics,
    )
    with pytest.raises(SampleRejectedError, match="exceeds"):
        collector.accept_payload(inconsistent)
    assert collector.accepted_samples == 0
    assert not list((tmp_path / "data").glob("shard-*.h5.partial"))
    collector.close()
