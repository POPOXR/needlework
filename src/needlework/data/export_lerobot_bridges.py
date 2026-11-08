"""Export verified bridge supervision in source LeRobot action conventions.

This writes an NPZ sidecar, NOT fabricated LeRobot episodes. The target continuation
is not appended and no observation is assigned to generated intermediate actions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import zarr

from needlework import paths
from needlework.data import bridges
from needlework.data.lerobot import read_json, rows_to_columns
from needlework.data.store import EpisodeStore


def export(task: str, bridge_path: Path, output: Path) -> None:
    if output.suffix != ".npz":
        raise ValueError("output must have a .npz suffix")
    store = EpisodeStore.open("lerobot", task)
    task_dir = paths.data_dir() / "lerobot" / task
    manifest = read_json(task_dir / "import_manifest.json")
    metadata = dict(zarr.open_group(str(bridge_path), mode="r").attrs)
    stride, horizon = metadata["stride"], metadata["horizon"]
    if stride != 1:
        raise ValueError("LeRobot bridge export requires stride=1")
    b = bridges.load(bridge_path, store, horizon=horizon, stride=stride)
    mapping = manifest["episodes"]
    if (
        len(mapping) != store.n_episodes
        or [e["global_start"] for e in mapping] != store.episode_starts.tolist()
        or [e["length"] for e in mapping]
        != np.diff(np.r_[0, store.episode_ends]).tolist()
    ):
        raise ValueError("import manifest does not match the store's episode mapping")

    def original(frames):
        episodes = store.episode_of_frame(frames)
        ids = np.asarray(
            [mapping[e]["episode_index"] for e in episodes], dtype=np.int64
        )
        starts = np.asarray(
            [mapping[e]["original_observation_start"] for e in episodes], dtype=np.int64
        )
        return ids, frames - store.episode_starts[episodes] + starts

    source_ep, source_frame = original(b.source_frame)
    target_ep, target_frame = original(b.target_frame)
    actions = b.actions.copy()
    if not store.spec.joint_names:
        actions[:, 3:9] = rows_to_columns(actions[:, 3:9])
    meta = {
        "format": 1,
        "kind": "bridge_supervision_sidecar",
        "task": task,
        "action_mode": manifest["config"]["action_mode"],
        "rotation_6d": "columns" if not store.spec.joint_names else None,
        "joint_names": list(store.spec.joint_names),
        "gripper": "motor radians, unchanged; 0 closed, application open 1.6",
        "fps": manifest["fps"],
        "stride": stride,
        "horizon": horizon,
        "store_identity": store.identity,
        "source_sha256": manifest["source_sha256"],
        "action_row_offset": manifest["config"]["action_row_offset"],
        "training_contract": (
            "source observations + logged prefix + bridge; mask loss after bridge; "
            "retain original examples; never append target actions"
        ),
        "not_a_robot_command": True,
        "gripper_outside_sdk_range": int(
            np.count_nonzero((actions[:, -1] < 0) | (actions[:, -1] > 2.0))
        ),
    }
    # Exclusive create: do not replace a previous export. File handle avoids NumPy's
    # automatic filename extension and stores strings without object/pickle arrays.
    with output.open("xb") as stream:
        np.savez_compressed(
            stream,
            actions=actions,
            action_start=b.action_start,
            action_len=b.action_len,
            source_episode_index=source_ep,
            source_frame_index=source_frame,
            target_episode_index=target_ep,
            target_frame_index=target_frame,
            stage=b.stage,
            verifier_margin=b.verifier_margin,
            metadata_json=np.asarray(json.dumps(meta, ensure_ascii=False)),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", required=True)
    parser.add_argument("--bridges", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export(args.task, args.bridges, args.output)
    print(f"Bridge supervision written to {args.output}")


if __name__ == "__main__":
    main()
