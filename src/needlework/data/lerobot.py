"""Explicit LeRobot v2.0/v2.1 -> Needlework adapter for right-arm targets.

Run ``python -m needlework.data.lerobot --help``. No robot SDK is imported.
The source dataset is read-only; an import is published only after validation.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import shutil
import tempfile
from pathlib import Path

import numpy as np
import zarr
from numcodecs import Blosc

from needlework.constants import IMAGE_SHAPE, OUTCOMES
from needlework.data.schema import StoreSpec, store_attrs, validate_store
from needlework.geometry.rotations import matrix_to_rot6d, rot6d_to_matrix

FORMAT = 1


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _inside(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"dataset path escapes its root: {relative}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _info(root: Path) -> dict:
    info = read_json(root / "meta/info.json")
    if info.get("codebase_version") not in ("v2.0", "v2.1"):
        raise ValueError(
            "only LeRobot v2.0 and v2.1 are supported; no implicit v3 migration"
        )
    if not np.isfinite(info["fps"]) or info["fps"] <= 0 or info["chunks_size"] <= 0:
        raise ValueError("fps and chunks_size must be positive")
    return info


def _path(root: Path, info: dict, episode: int, key: str, **kwargs) -> Path:
    return _inside(
        root,
        info[key].format(
            episode_index=episode,
            episode_chunk=episode // info["chunks_size"],
            **kwargs,
        ),
    )


def _table(path: Path):
    import pyarrow.parquet as pq

    return pq.read_table(path)


def _column(table, key: str) -> np.ndarray:
    if key not in table.column_names:
        raise ValueError(f"missing parquet column {key!r}")
    try:
        return np.asarray(table[key].to_pylist())
    except ValueError as error:
        raise ValueError(f"ragged or invalid column {key!r}") from error


def _scalar(table, key: str) -> np.ndarray:
    value = _column(table, key)
    if value.shape not in ((len(table),), (len(table), 1)):
        raise ValueError(f"{key}: expected one scalar per frame, got {value.shape}")
    return value.reshape(-1)


def field(table, selector: dict, width: int) -> np.ndarray:
    """Read a vector, or explicit indices of a packed vector; never guess slices."""
    if set(selector) - {"column", "indices"}:
        raise ValueError(f"unknown selector options: {selector}")
    value = _column(table, selector["column"])
    if value.ndim == 1:
        value = value[:, None]
    if value.ndim != 2:
        raise ValueError(f"{selector}: expected a scalar/vector column")
    if "indices" in selector:
        indices = selector["indices"]
        if (
            not isinstance(indices, list)
            or any(type(i) is not int or i < 0 or i >= value.shape[1] for i in indices)
            or len(set(indices)) != len(indices)
        ):
            raise ValueError(f"{selector}: invalid or duplicated indices")
        value = value[:, indices]
    if value.shape != (len(table), width):
        raise ValueError(
            f"{selector}: got {value.shape}, expected [{len(table)}, {width}]"
        )
    value = value.astype(np.float32)
    if not np.isfinite(value).all():
        raise ValueError(f"{selector}: null, NaN or infinite values")
    return value


def columns_to_rows(value: np.ndarray) -> np.ndarray:
    """Input [r00,r10,r20,r01,r11,r21]; output Needlework's row-based 6D.

    Orthonormalize columns first, matching the user's execution path. A mere
    reshape/permutation of the six raw numbers cannot perform this conversion.
    """
    rotation = rot6d_to_matrix(value).swapaxes(-1, -2)
    return matrix_to_rot6d(rotation)


def rows_to_columns(value: np.ndarray) -> np.ndarray:
    return matrix_to_rot6d(rot6d_to_matrix(value).swapaxes(-1, -2))


def inspect_dataset(root: Path, sample_episodes: int = 3) -> dict:
    """Small read-only inventory. Numeric similarity is evidence, not semantics."""
    info = _info(root)
    episodes_path = root / "meta/episodes.jsonl"
    if episodes_path.is_file():
        episodes = [
            json.loads(line)
            for line in episodes_path.read_text().splitlines()
            if line.strip()
        ]
    else:
        episodes = [{"episode_index": e} for e in range(info["total_episodes"])]
    samples = []
    for episode in episodes[:sample_episodes]:
        idx = int(episode["episode_index"])
        table = _table(_path(root, info, idx, "data_path"))
        numeric = {}
        for name in table.column_names:
            try:
                value = _column(table, name).astype(float)
                if value.size and value.ndim <= 2:
                    finite = value[np.isfinite(value)]
                    numeric[name] = {
                        "shape": list(value.shape),
                        "finite": bool(np.isfinite(value).all()),
                        "min": float(finite.min()) if finite.size else None,
                        "max": float(finite.max()) if finite.size else None,
                    }
            except (TypeError, ValueError):
                continue
        samples.append(
            {
                "episode": episode,
                "rows": len(table),
                "schema": str(table.schema),
                "numeric": numeric,
            }
        )
    return {
        "info": info,
        "samples": samples,
        "unresolved": [
            "Target callback vs measured-state fallback; names do not prove semantics",
            "Command/observation timing, camera offsets, command rate and takeovers",
            "Joint order, historical mode, units, FK tool/base and rotation order",
            "Outcomes and homogeneous task/calibration/controller settings",
        ],
    }


def _config(config: dict) -> StoreSpec:
    required = {
        "format",
        "action_mode",
        "action_semantics",
        "action_row_offset",
        "task_index",
        "units",
        "rotation_6d",
        "joint_names",
        "gripper_range",
        "fields",
        "cameras",
        "evidence",
        "timestamp_tolerance_s",
        "video_tolerance_s",
    }
    if set(config) != required:
        raise ValueError(
            f"config keys differ: missing={required - set(config)}, "
            f"extra={set(config) - required}"
        )
    if config["format"] != FORMAT or config["action_mode"] not in ("joint", "ee"):
        raise ValueError("format must be 1 and action_mode must be joint or ee")
    if config["action_semantics"] != "command_target":
        raise ValueError(
            "actions must be verified command_target; "
            "measured/unknown actions are not causal labels"
        )
    if (
        type(config["action_row_offset"]) is not int
        or type(config["task_index"]) is not int
    ):
        raise ValueError(
            "action_row_offset and task_index must be explicitly verified integers"
        )
    if config["units"] != {"position": "m", "joint": "rad", "gripper": "motor_rad"}:
        raise ValueError(
            "expected m, rad and motor_rad; convert units explicitly before import"
        )
    if config["rotation_6d"] != "columns":
        raise ValueError("this adapter expects column-based source rotation_6d")
    evidence_keys = {
        "action_source",
        "time_alignment",
        "control_and_sampling",
        "coordinates_and_units",
        "outcomes",
    }
    if (
        not isinstance(config["evidence"], dict)
        or set(config["evidence"]) != evidence_keys
    ):
        raise ValueError(f"evidence must document {sorted(evidence_keys)}")
    if any(
        not isinstance(v, str)
        or not v.strip()
        or v.strip().lower() in ("unknown", "todo")
        for v in config["evidence"].values()
    ):
        raise ValueError(
            "all evidence entries need real file/log references; "
            "see the AI handoff guide"
        )
    names = config["joint_names"]
    if (
        not isinstance(names, list)
        or any(not isinstance(n, str) or not n for n in names)
        or len(names) != len(set(names))
    ):
        raise ValueError("joint_names must be unique names in controller order")
    if config["action_mode"] == "joint" and not names:
        raise ValueError("joint mode requires verified joint_names")
    bounds = np.asarray(config["gripper_range"], dtype=float)
    if (
        bounds.shape != (2,)
        or not np.isfinite(bounds).all()
        or bounds[0] != 0
        or not 0 < bounds[1] <= 2.0
    ):
        raise ValueError(
            "gripper_range must be [0, verified upper limit <= 2.0] motor radians"
        )
    for name in ("timestamp_tolerance_s", "video_tolerance_s"):
        value = config[name]
        if not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a verified positive tolerance")
    fields = config["fields"]
    arm_key = "joint" if config["action_mode"] == "joint" else "ee"
    expected = {
        f"action_{arm_key}",
        f"observation_{arm_key}",
        "action_gripper",
        "observation_gripper",
    }
    if set(fields) != expected:
        raise ValueError(f"fields must be exactly {sorted(expected)}")
    if not isinstance(config["cameras"], dict) or not config["cameras"]:
        raise ValueError("at least one camera is required")
    for alias, camera in config["cameras"].items():
        if not re.fullmatch(r"camera_[A-Za-z0-9_]+", alias):
            raise ValueError(
                "camera aliases must use camera_<name>, without dots or slashes"
            )
        if set(camera) != {"key", "timestamp_offset_s"} or not np.isfinite(
            camera["timestamp_offset_s"]
        ):
            raise ValueError(f"{alias}: require key and finite timestamp_offset_s")
    proprio = {"right_gripper": 1}
    if arm_key == "joint":
        proprio = {"right_joint_position": len(names), **proprio}
    else:
        proprio = {"right_ee_pos": 3, "right_ee_rot6d": 6, **proprio}
    return StoreSpec(
        arms=("right",),
        frame=(
            "base_link -> tool_link; meters; absolute command targets; "
            "joint angles in radians"
        ),
        gripper=(
            "motor position, radians: 0 closed, application open 1.6; SDK limit 2.0"
        ),
        cameras=tuple(config["cameras"]),
        proprio=proprio,
        state_dim=None,
        joint_names=tuple(names) if arm_key == "joint" else (),
    )


def _episodes(manifest: list[dict], config: dict) -> list[dict]:
    if not isinstance(manifest, list) or not manifest:
        raise ValueError("episodes manifest must be a nonempty JSON list")
    seen = set()
    for ep in manifest:
        required = {
            "episode_index",
            "success",
            "action_semantics",
            "policy_mode",
            "evidence",
        }
        if set(ep) != required:
            raise ValueError(
                f"episode manifest entries require exactly {sorted(required)}"
            )
        idx = ep["episode_index"]
        if (
            type(idx) is not int
            or idx < 0
            or idx in seen
            or type(ep["success"]) is not bool
        ):
            raise ValueError(
                "episodes need unique nonnegative indices "
                "and actual boolean success labels"
            )
        seen.add(idx)
        if ep["action_semantics"] != config["action_semantics"] or ep[
            "policy_mode"
        ] not in ("joint", "ee", "mixed"):
            raise ValueError(
                "each selected episode needs verified command targets "
                "and historical policy_mode"
            )
        if (
            not isinstance(ep["evidence"], str)
            or not ep["evidence"].strip()
            or ep["evidence"].lower() in ("todo", "unknown")
        ):
            raise ValueError(
                "each episode needs evidence for its outcome and recording provenance"
            )
    if not any(ep["success"] for ep in manifest):
        raise ValueError("at least one successful episode is required for stitching")
    return sorted(manifest, key=lambda ep: (not ep["success"], ep["episode_index"]))


def _numeric(table, episode: int, info: dict, config: dict) -> tuple[dict, np.ndarray]:
    count = len(table)
    for key in ("episode_index", "frame_index", "task_index"):
        value = _scalar(table, key)
        if not np.issubdtype(value.dtype, np.integer):
            raise ValueError(f"{key} must contain integer indices")
    if not np.all(_scalar(table, "episode_index") == episode):
        raise ValueError(f"episode {episode}: parquet contains another episode")
    if not np.array_equal(_scalar(table, "frame_index"), np.arange(count)):
        raise ValueError(
            f"episode {episode}: frame_index must start at zero with no gaps/duplicates"
        )
    if not np.all(_scalar(table, "task_index") == config["task_index"]):
        raise ValueError(
            "mixed tasks are not stitchable; import one task_index at a time"
        )
    times = _scalar(table, "timestamp").astype(float)
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("timestamps must be finite and strictly increasing")
    if np.any(
        np.abs(np.diff(times) - 1 / info["fps"]) > config["timestamp_tolerance_s"]
    ):
        raise ValueError(
            "timestamps disagree with fps: "
            "repair drops/rate/alignment before conversion"
        )
    offset = config["action_row_offset"]
    obs_rows = np.arange(max(0, -offset), min(count, count - offset))
    if len(obs_rows) < 2:
        raise ValueError("alignment leaves fewer than two frames")
    action_rows = obs_rows + offset
    fields = config["fields"]
    action_grip = field(table, fields["action_gripper"], 1)[action_rows]
    obs_grip = field(table, fields["observation_gripper"], 1)[obs_rows]
    low, high = config["gripper_range"]
    for value in (action_grip, obs_grip):
        if np.any((value < low - 1e-6) | (value > high + 1e-6)):
            raise ValueError(
                "gripper exceeds the verified motor-radian bounds; no silent clipping"
            )
    if config["action_mode"] == "joint":
        width = len(config["joint_names"])
        arm = field(table, fields["action_joint"], width)[action_rows]
        proprio = {
            "right_joint_position": field(table, fields["observation_joint"], width)[
                obs_rows
            ]
        }
    else:
        arm = field(table, fields["action_ee"], 9)[action_rows].copy()
        measured = field(table, fields["observation_ee"], 9)[obs_rows]
        arm[:, 3:9] = columns_to_rows(arm[:, 3:9])
        proprio = {
            "right_ee_pos": measured[:, :3],
            "right_ee_rot6d": columns_to_rows(measured[:, 3:9]),
        }
    return {
        "action": np.concatenate([arm, action_grip], axis=1),
        **proprio,
        "right_gripper": obs_grip,
    }, obs_rows


def _rgb(image) -> np.ndarray:
    from PIL import Image, ImageOps

    return np.asarray(
        ImageOps.pad(
            image.convert("RGB"), IMAGE_SHAPE[:2], method=Image.Resampling.BILINEAR
        ),
        dtype=np.uint8,
    )


def _video(path: Path, targets: np.ndarray, tolerance: float):
    """Stream nearest frames by presentation timestamp; reject missing/stale frames."""
    import av
    from PIL import Image

    with av.open(str(path)) as container:
        frames = iter(container.decode(video=0))

        def next_frame():
            frame = next(frames, None)
            if frame is None:
                return None
            if frame.pts is None:
                raise ValueError(f"{path}: video frame has no presentation timestamp")
            return float(frame.pts * frame.time_base), frame

        current = next_frame()
        previous = None
        last_selected = -np.inf
        for target in targets:
            while current is not None and current[0] < target:
                previous, current = current, next_frame()
                if current is not None and current[0] <= previous[0]:
                    raise ValueError(f"{path}: non-increasing video timestamps")
            candidates = [v for v in (previous, current) if v is not None]
            if not candidates:
                raise ValueError(f"{path}: empty video")
            best = min(candidates, key=lambda v: abs(v[0] - target))
            if abs(best[0] - target) > tolerance:
                raise ValueError(
                    f"{path}: no video frame aligned with timestamp {target}"
                )
            if best[0] <= last_selected:
                raise ValueError(
                    f"{path}: repeated video frame; check camera rate/alignment"
                )
            last_selected = best[0]
            yield _rgb(Image.fromarray(best[1].to_ndarray(format="rgb24")))


def _images(root, info, table, episode, camera, rows, tolerance, hashes):
    from PIL import Image

    key = camera["key"]
    feature = info["features"].get(key)
    if feature is None or feature.get("dtype") not in ("video", "image"):
        raise ValueError(f"{key}: expected a declared LeRobot video/image feature")
    if feature["dtype"] == "video":
        path = _path(root, info, episode, "video_path", video_key=key)
        hashes[str(path.relative_to(root.resolve()))] = sha256(path)
        times = _scalar(table, "timestamp")[rows] + camera["timestamp_offset_s"]
        yield from _video(path, times, tolerance)
        return
    if camera["timestamp_offset_s"] != 0:
        raise ValueError(
            "embedded images are row-aligned; "
            "nonzero image offsets need explicit preprocessing"
        )
    for row in rows:
        value = table[key][int(row)].as_py()
        if isinstance(value, dict) and value.get("bytes") is not None:
            with Image.open(io.BytesIO(value["bytes"])) as image:
                yield _rgb(image)
        elif isinstance(value, dict) and value.get("path"):
            path = _inside(root, value["path"])
            hashes[str(path.relative_to(root.resolve()))] = sha256(path)
            with Image.open(path) as image:
                yield _rgb(image)
        else:
            raise ValueError(
                f"{key}: expected encoded image bytes/path; unsupported image encoding"
            )


def _digest_store(path: Path, provenance: str) -> str:
    digest = hashlib.sha256(provenance.encode())
    for file in sorted(p for p in path.rglob("*") if p.is_file()):
        digest.update(file.relative_to(path).as_posix().encode())
        digest.update(sha256(file).encode())
    return digest.hexdigest()


def convert(
    root: Path, output: Path, *, task: str, config: dict, episodes: list[dict]
) -> dict:
    root, output = root.resolve(), output.resolve()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", task):
        raise ValueError("task must contain only letters, digits, underscore or hyphen")
    if output.exists() or output.is_relative_to(root) or root.is_relative_to(output):
        raise ValueError("output must be new and separate from the source dataset")
    spec = _config(config)
    episodes = _episodes(episodes, config)
    info = _info(root)
    if (
        config["timestamp_tolerance_s"] >= 0.5 / info["fps"]
        or config["video_tolerance_s"] >= 1 / info["fps"]
    ):
        raise ValueError(
            "tolerances must be below half a frame (sampling) and one frame (video)"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=output.name + ".tmp-", dir=output.parent))
    hashes = {"meta/info.json": sha256(root / "meta/info.json")}
    mapping, stores = [], []
    global_start = 0
    try:
        for outcome in OUTCOMES:
            selected = [
                ep for ep in episodes if ep["success"] == (outcome == "success")
            ]
            if not selected:
                continue
            path = temporary / f"{outcome}.zarr"
            stores.append(path)
            group = zarr.open_group(str(path), mode="w")
            group.attrs.update(store_attrs("lerobot", task, spec))
            data = group.create_group("data")
            compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
            for key, width in spec.data_widths().items():
                data.zeros(
                    key,
                    shape=(0, width),
                    chunks=(1024, width),
                    dtype="f4",
                    compressor=compressor,
                )
            for key in spec.cameras:
                data.zeros(
                    key,
                    shape=(0, *IMAGE_SHAPE),
                    chunks=(1, *IMAGE_SHAPE),
                    dtype="u1",
                    compressor=compressor,
                )
            ends, start = [], 0
            for ep in selected:
                idx = ep["episode_index"]
                parquet = _path(root, info, idx, "data_path")
                hashes[str(parquet.relative_to(root))] = sha256(parquet)
                table = _table(parquet)
                numeric, rows = _numeric(table, idx, info, config)
                stop = start + len(rows)
                for key, value in numeric.items():
                    data[key].append(value.astype(np.float32), axis=0)
                for alias, camera in config["cameras"].items():
                    data[alias].resize((stop, *IMAGE_SHAPE))
                    images = _images(
                        root,
                        info,
                        table,
                        idx,
                        camera,
                        rows,
                        config["video_tolerance_s"],
                        hashes,
                    )
                    written = 0
                    for j, image in enumerate(images):
                        data[alias][start + j] = image
                        written += 1
                    if written != len(rows):
                        raise ValueError(
                            f"{alias}: image count differs from observations"
                        )
                mapping.append(
                    {
                        **ep,
                        "outcome": outcome,
                        "store_episode": len(ends),
                        "global_start": global_start,
                        "length": len(rows),
                        "original_observation_start": int(rows[0]),
                        "original_action_start": int(
                            rows[0] + config["action_row_offset"]
                        ),
                        "gripper_above_application_open": int(
                            np.count_nonzero(numeric["action"][:, -1] > 1.6)
                        ),
                    }
                )
                global_start += len(rows)
                ends.append(stop)
                start = stop
            meta = group.create_group("meta")
            meta.array("episode_ends", np.asarray(ends, dtype=np.int64))
            meta.array(
                "episode_success", np.full(len(ends), outcome == "success", dtype=bool)
            )
            validate_store(path, domain="lerobot", task=task)
        report = {
            "format": FORMAT,
            "domain": "lerobot",
            "task": task,
            "fps": info["fps"],
            "codebase_version": info["codebase_version"],
            "config": config,
            "episodes": mapping,
            "source_sha256": hashes,
            "image_preprocessing": (
                "RGB, aspect-preserving bilinear resize "
                "and centered black padding to 224x224"
            ),
            "observation_keys": list(spec.proprio),
            "cameras": list(spec.cameras),
            "action_dim": spec.action_dim,
            "warnings": [
                "No real-data or robot validation is implied by conversion.",
                "Success/failure labels apply to whole original episodes, "
                "including takeovers; no automatic fragment relabeling.",
                "Command targets must reflect the verified recording/downsampling/"
                "controller contract; no smoothing is reapplied.",
            ],
        }
        provenance = json.dumps(report, sort_keys=True)
        for path in stores:
            path.with_name(path.name + ".sha256").write_text(
                _digest_store(path, provenance) + "\n"
            )
        write_json(temporary / "import_manifest.json", report)
        temporary.rename(output)
        return report
    except BaseException:
        shutil.rmtree(temporary)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    inspect = sub.add_parser(
        "inspect", help="read metadata/schema; never infer action semantics"
    )
    inspect.add_argument("--root", type=Path, required=True)
    inspect.add_argument("--sample-episodes", type=int, default=3)
    inspect.add_argument("--output", type=Path, required=True)
    convert_parser = sub.add_parser(
        "convert", help="import verified targets into new Zarr stores"
    )
    convert_parser.add_argument("--root", type=Path, required=True)
    convert_parser.add_argument("--output", type=Path, required=True)
    convert_parser.add_argument("--task", required=True)
    convert_parser.add_argument("--config", type=Path, required=True)
    convert_parser.add_argument("--episodes", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "inspect":
        if args.sample_episodes < 1:
            parser.error("--sample-episodes must be positive")
        if args.output.exists() or args.output.resolve().is_relative_to(
            args.root.resolve()
        ):
            parser.error("inspection output must be new and outside the source dataset")
        write_json(args.output, inspect_dataset(args.root, args.sample_episodes))
        print(
            f"Inventory written to {args.output}; action semantics remain unverified."
        )
    else:
        report = convert(
            args.root,
            args.output,
            task=args.task,
            config=read_json(args.config),
            episodes=read_json(args.episodes),
        )
        print(
            f"Imported {len(report['episodes'])} episodes / "
            f"{report['action_dim']} action dimensions to {args.output}"
        )


if __name__ == "__main__":
    main()
