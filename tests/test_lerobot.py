"""Synthetic v2 fixtures exercise the real importer and CPU training data paths."""

from __future__ import annotations

import copy
import io
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import zarr
from PIL import Image
from scipy.spatial.transform import Rotation

from needlework.config import compose, compose_stitch
from needlework.data import bridges
from needlework.data.export_lerobot_bridges import export
from needlework.data.lerobot import (
    _numeric,
    _video,
    columns_to_rows,
    convert,
    field,
    inspect_dataset,
    rows_to_columns,
    sha256,
    write_json,
)
from needlework.data.schema import spec_for, store_attrs
from needlework.data.store import EpisodeStore
from needlework.geometry.actions import ActionLayout, action_poses, to_env_command
from needlework.geometry.rotations import matrix_to_rot6d, rot6d_to_matrix
from needlework.sampling.augmented_dataset import AugmentedPolicyDataset, SamplerOptions
from needlework.sampling.observations import make_observations
from needlework.sampling.pairs import IdmPairs
from needlework.sampling.policy_dataset import PolicyDataset, WindowShape
from needlework.sampling.roles import RoleWeights
from needlework.training.common import identity_dims

pytestmark = pytest.mark.cpu


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "source"
    (root / "meta").mkdir(parents=True)
    info = {
        "codebase_version": "v2.1",
        "fps": 20,
        "chunks_size": 2,
        "total_episodes": 4,
        "data_path": (
            "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
        ),
        "video_path": (
            "videos/chunk-{episode_chunk:03d}/{video_key}/"
            "episode_{episode_index:06d}.mp4"
        ),
        "features": {
            "observation.images.front": {"dtype": "image", "shape": [24, 32, 3]}
        },
    }
    write_json(root / "meta/info.json", info)
    buffer = io.BytesIO()
    Image.new("RGB", (32, 24), (220, 30, 10)).save(buffer, format="PNG")
    count = 40
    frames = np.arange(count)
    angles = np.column_stack((frames * 0.01, frames * -0.02, frames * 0.03))
    rotation = Rotation.from_euler("xyz", [25, -40, 73], degrees=True).as_matrix()
    columns = rotation[:, :2].T.reshape(6)
    ee = np.tile(np.r_[[0.1, 0.2, 0.3], columns], (count, 1))
    tables = {}
    for episode in (0, 2, 3):
        table = pa.table(
            {
                "episode_index": [episode] * count,
                "frame_index": frames,
                "task_index": [7] * count,
                "timestamp": frames / 20.0,
                "actions.arm_joints": angles.tolist(),
                "observations.arm_joints": (angles - 0.2).tolist(),
                "actions.arm_ee": ee.tolist(),
                "observations.arm_ee": (ee + np.r_[[0.02, 0, 0], np.zeros(6)]).tolist(),
                "actions.gripper": np.linspace(0, 1.8, count),
                "observations.gripper": [0.4] * count,
                "observation.images.front": [{"bytes": buffer.getvalue(), "path": None}]
                * count,
            }
        )
        path = root / info["data_path"].format(
            episode_chunk=episode // 2, episode_index=episode
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, path)
        tables[episode] = table
    config = json.loads(
        (
            Path(__file__).parents[1] / "examples/lerobot/right_arm.template.json"
        ).read_text()
    )
    config.update(
        action_semantics="command_target",
        action_row_offset=1,
        task_index=7,
        joint_names=["shoulder", "elbow", "wrist"],
        timestamp_tolerance_s=0.002,
        video_tolerance_s=0.02,
        cameras={
            "camera_front": {
                "key": "observation.images.front",
                "timestamp_offset_s": 0.0,
            }
        },
    )
    config["evidence"] = dict.fromkeys(
        config["evidence"], "synthetic fixture; known by construction"
    )
    episodes = [
        {
            "episode_index": e,
            "success": e != 0,
            "action_semantics": "command_target",
            "policy_mode": "joint",
            "evidence": "synthetic fixture",
        }
        for e in (0, 3, 2)
    ]
    return root, info, config, episodes, tables, rotation


def imported(source, tmp_path, monkeypatch):
    root, _, config, episodes, _, _ = source
    monkeypatch.setenv("NEEDLEWORK_ROOT", str(tmp_path))
    output = tmp_path / "data/lerobot/fixture"
    report = convert(root, output, task="fixture", config=config, episodes=episodes)
    return EpisodeStore.open("lerobot", "fixture"), report, output


def test_joint_import_keeps_targets_alignment_and_provenance(
    source, tmp_path, monkeypatch
):
    root, info, _, _, tables, _ = source
    before = {
        str(p.relative_to(root)): sha256(p) for p in root.rglob("*") if p.is_file()
    }
    store, report, output = imported(source, tmp_path, monkeypatch)
    assert store.spec.action_dim == 4
    assert store.layout.joint_names == ("shoulder", "elbow", "wrist")
    assert store.episode_ends.tolist() == [39, 78, 117]
    assert store.episode_success.tolist() == [True, True, False]
    assert [e["episode_index"] for e in report["episodes"]] == [2, 3, 0]
    np.testing.assert_allclose(
        store.arrays["action"][:39, :3], tables[2]["actions.arm_joints"].to_pylist()[1:]
    )
    np.testing.assert_allclose(
        store.arrays["right_joint_position"][:39],
        tables[2]["observations.arm_joints"].to_pylist()[:-1],
    )
    assert store.arrays["action"][:39, -1].max() == pytest.approx(1.8)
    assert report["episodes"][0]["gripper_above_application_open"] > 0
    assert report["episodes"][0]["original_action_start"] == 1
    assert report["source_sha256"] == before
    assert before == {
        str(p.relative_to(root)): sha256(p) for p in root.rglob("*") if p.is_file()
    }
    images = zarr.open(str(output / "success.zarr"), mode="r")["data/camera_front"]
    assert images.shape == (78, 224, 224, 3)
    assert images.chunks == (1, 224, 224, 3)
    assert images[0, 112, 112].tolist() == [220, 30, 10]
    assert images[0, 0, 0].tolist() == [0, 0, 0]
    with pytest.raises(ValueError, match="new and separate"):
        convert(root, output, task="fixture", config=source[2], episodes=source[3])
    audit = inspect_dataset(root, sample_episodes=1)
    assert audit["info"] == info
    assert audit["samples"][0]["numeric"]["actions.arm_joints"]["shape"] == [40, 3]


def test_column_rotation_conversion_is_not_identity_or_transpose():
    rotation = Rotation.from_euler("xyz", [31, -27, 69], degrees=True).as_matrix()
    columns = rotation[:, :2].T.reshape(6)
    # Non-unit policy output is interpreted with column Gram-Schmidt as at execution.
    raw = columns.copy()
    raw[:3] *= 1.8
    raw[3:] = raw[3:] * 0.7 + raw[:3] * 0.3
    actual = rot6d_to_matrix(columns_to_rows(raw))
    np.testing.assert_allclose(actual, rotation, atol=2e-7)
    assert not np.allclose(actual, rotation.T)
    np.testing.assert_allclose(
        rows_to_columns(matrix_to_rot6d(rotation)), columns, atol=2e-7
    )
    with pytest.raises(ValueError, match="parallel"):
        columns_to_rows(np.array([1, 0, 0, 2, 0, 0]))


@pytest.mark.parametrize("mode", ["joint", "ee"])
def test_training_windows_idm_and_masked_bridge_export(
    source, tmp_path, monkeypatch, mode
):
    config = source[2]
    if mode == "ee":
        config["action_mode"] = mode
        for prefix in ("action", "observation"):
            config["fields"].pop(prefix + "_joint")
            config["fields"][prefix + "_ee"] = {
                "column": ("actions" if prefix == "action" else "observations")
                + ".arm_ee"
            }
    store, _, _ = imported(source, tmp_path, monkeypatch)
    cfg = compose(
        [
            "component=idm",
            f"task=lerobot/{mode}",
            "task.name=fixture",
            "task.obs.cameras=[camera_front]",
            "run.name=test",
        ]
    )
    features = {
        "camera_front": np.arange(store.n_frames * 4, dtype=np.float32).reshape(
            store.n_frames, 4
        )
    }
    obs = make_observations(
        store, features, (*cfg.task.obs.cameras, *cfg.task.obs.proprio), cfg.task
    )
    data = PolicyDataset(
        store=store,
        observations=obs,
        episodes=np.array([0, 1]),
        shape=WindowShape(8, 2, 3, 1),
        relative_actions=False,
        train=False,
    )
    batch = data.batch(np.array([0, 1]))
    assert batch["action"].shape == (2, 8, 4 if mode == "joint" else 10)
    assert identity_dims(1, list(cfg.task.action_identity), store.layout).shape == (
        store.spec.action_dim,
    )
    pairs = IdmPairs(data, features, seed=123, resample=False)
    idm = pairs.batch(np.array([0, 1]))
    for i, offset in enumerate(pairs.offset[:2]):
        last = offset  # n_obs - 2 + offset
        np.testing.assert_array_equal(
            idm["action"][i, last:],
            np.repeat(idm["action"][i, last : last + 1], 8 - last, axis=0),
        )
    b = bridges.Bridges(
        source_frame=np.array([5], np.int64),
        target_frame=np.array([20], np.int64),
        stage=np.array([1], np.int64),
        verifier_margin=np.array([0.3], np.float32),
        action_start=np.array([0], np.int64),
        action_len=np.array([2], np.int64),
        actions=store.arrays["action"][[6, 7]].copy(),
    )
    augmented = AugmentedPolicyDataset(
        data,
        b,
        weight=0.5,
        role_weights=RoleWeights(**dict(cfg.task.role_weights)),
        epoch_length="windows",
        options=SamplerOptions(**dict(cfg.task.sampler_options)),
        seed=1,
    )
    departure = np.flatnonzero(augmented.crossings[:, 1] == 5)
    _, action, valid = augmented.bridge_arrays(departure)
    np.testing.assert_array_equal(action[0, 1:3], b.actions)
    assert valid[0, :3].all() and not valid[0, 3:].any()
    assert len(augmented.twins) == 1
    _, original, original_valid = augmented.twin_arrays(augmented.twins)
    assert original_valid.all()
    np.testing.assert_array_equal(original[0, 1], store.arrays["action"][5])
    bridge_path = tmp_path / "bridges.zarr"
    bridges.save(bridge_path, b, store, horizon=7, stride=1, recipe={})
    output = tmp_path / "bridges.npz"
    export("fixture", bridge_path, output)
    with np.load(output, allow_pickle=False) as saved:
        assert saved["source_episode_index"].tolist() == [2]
        assert saved["source_frame_index"].tolist() == [5]
        assert saved["target_frame_index"].tolist() == [20]
        assert saved["action_len"].tolist() == [2]
        if mode == "ee":
            np.testing.assert_allclose(
                saved["actions"][:, 3:9],
                np.tile(source[-1][:, :2].T.reshape(6), (2, 1)),
                atol=2e-7,
            )
        else:
            np.testing.assert_array_equal(saved["actions"], b.actions)
        assert json.loads(str(saved["metadata_json"]))["not_a_robot_command"]
    with pytest.raises(FileExistsError):
        export("fixture", bridge_path, output)


@pytest.mark.parametrize("mode", ["joint", "ee"])
@pytest.mark.parametrize("component", ["policy", "idm", "verifier"])
def test_task_configs_compose(mode, component):
    cfg = compose(
        [
            f"component={component}",
            f"task=lerobot/{mode}",
            "task.name=fixture",
            "run.name=smoke",
        ]
    )
    assert cfg.task.domain == "lerobot" and cfg.task.action_mode == mode
    assert cfg.task.stride == 1 and not cfg.task.relative_actions
    assert cfg.task.verifier_serving.mode == "independent"
    stitched = compose_stitch(
        [
            f"task=lerobot/{mode}",
            "task.name=fixture",
            "run.name=smoke",
            "idm_checkpoint=/tmp/idm",
            "verifier_checkpoint=/tmp/verifier",
        ]
    )
    assert stitched.task.obs == cfg.task.obs


@pytest.mark.parametrize(
    "key,value",
    [
        ("action_semantics", "measured_state"),
        ("action_semantics", "unknown"),
        ("action_row_offset", None),
        ("joint_names", []),
        ("rotation_6d", "rows"),
        ("units", {"position": "mm", "joint": "deg", "gripper": "probability"}),
    ],
)
def test_reject_unverified_or_incompatible_semantics(source, tmp_path, key, value):
    config = copy.deepcopy(source[2])
    config[key] = value
    output = tmp_path / "bad"
    with pytest.raises(ValueError):
        convert(source[0], output, task="bad", config=config, episodes=source[3])
    assert not output.exists()


def test_bad_episode_label_is_not_guessed(source, tmp_path):
    episodes = copy.deepcopy(source[3])
    episodes[0]["success"] = "false"
    with pytest.raises(ValueError, match="boolean"):
        convert(
            source[0], tmp_path / "bad", task="bad", config=source[2], episodes=episodes
        )


@pytest.mark.parametrize(
    "key,value,message",
    [
        ("timestamp", [0.1] * 40, "strictly increasing"),
        ("timestamp", (np.arange(40) / 10).tolist(), "disagree with fps"),
        ("frame_index", [0] * 40, "gaps/duplicates"),
        ("task_index", [8] * 40, "mixed tasks"),
        ("episode_index", [8] * 40, "another episode"),
        ("actions.gripper", [2.1] * 40, "bounds"),
        ("actions.gripper", [None] * 40, "NaN"),
    ],
)
def test_numeric_corruption_is_rejected(source, key, value, message):
    table = source[4][2]
    table = table.set_column(table.schema.get_field_index(key), key, pa.array(value))
    with pytest.raises(ValueError, match=message):
        _numeric(table, 2, source[1], source[2])


def test_negative_offset_and_explicit_packed_slices(source):
    config = copy.deepcopy(source[2])
    config["action_row_offset"] = -1
    numeric, rows = _numeric(source[4][2], 2, source[1], config)
    assert rows.tolist() == list(range(1, 40))
    np.testing.assert_array_equal(numeric["action"][0, :3], [0, 0, 0])
    table = pa.table({"action": [[99, 1, 2, 3, 0.8]]})
    np.testing.assert_array_equal(
        field(table, {"column": "action", "indices": [1, 2, 3]}, 3), [[1, 2, 3]]
    )
    with pytest.raises(ValueError, match="duplicated"):
        field(table, {"column": "action", "indices": [1, 1, 2]}, 3)


def test_success_only_and_incomplete_write_cleanup(source, tmp_path):
    root, _, config, episodes, _, _ = source
    selected = [e for e in episodes if e["success"]]
    output = tmp_path / "success_only"
    convert(root, output, task="fixture", config=config, episodes=selected)
    store = EpisodeStore.open_paths(
        domain="lerobot",
        task="fixture",
        store_paths={"success": output / "success.zarr"},
    )
    assert store.n_episodes == 2
    config["cameras"]["camera_front"]["key"] = "missing_camera"
    failed = tmp_path / "failed"
    with pytest.raises(ValueError, match="declared"):
        convert(root, failed, task="fixture", config=config, episodes=selected)
    assert not failed.exists() and not list(tmp_path.glob("failed.tmp-*"))


def test_legacy_action_schema_and_joint_pose_guard():
    for domain, task in (
        ("umi", "sweater"),
        ("robomimic", "square"),
        ("robomimic", "transport"),
    ):
        spec = spec_for(domain, task)
        attrs = store_attrs(domain, task)
        assert set(attrs) == {"domain", "task", "action"}
        layout = ActionLayout.from_spec(attrs["action"])
        assert layout.dim == 10 * len(spec.arms)
    layout = ActionLayout(("right",), ("q1", "q2", "q3"))
    for func in (action_poses, to_env_command):
        with pytest.raises(ValueError, match="FK/IK"):
            func(layout, np.zeros((2, 4), np.float32))


def test_video_presentation_timestamps_and_rgb(tmp_path):
    import av

    path = tmp_path / "tiny.mp4"
    with av.open(str(path), "w") as output:
        stream = output.add_stream("mpeg4", rate=20)
        stream.width, stream.height, stream.pix_fmt = 32, 24, "yuv420p"
        for i in range(5):
            pixels = np.full((24, 32, 3), (20 + i * 30, 5, 5), dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    decoded = list(_video(path, np.array([0.05, 0.15]), 0.005))
    assert len(decoded) == 2
    assert abs(int(decoded[0][112, 112, 0]) - 50) < 8
    assert abs(int(decoded[1][112, 112, 0]) - 110) < 8
    with pytest.raises(ValueError, match="no video frame"):
        list(_video(path, np.array([1.0]), 0.005))


@pytest.mark.parametrize("action_dim", [4, 10])
def test_policy_idm_verifier_accept_variable_action_dim(action_dim):
    import torch

    from needlework.models.idm import IdmPolicy
    from needlework.models.policy import DiffusionPolicy, PolicyShape
    from needlework.models.verifier import Verifier

    torch.manual_seed(5)
    shape = PolicyShape(
        ("camera_front",), 4, {"right_state": action_dim}, action_dim, 8, 2, 3
    )
    kwargs = {
        "shape": shape,
        "unet": {
            "down_dims": [8, 16],
            "step_embed_dim": 8,
            "kernel_size": 3,
            "n_groups": 4,
        },
        "scheduler": {
            "train_steps": 8,
            "beta_start": 0.0001,
            "beta_end": 0.02,
            "beta_schedule": "linear",
        },
        "inference_steps": 2,
    }
    obs = {
        "camera_front": torch.randn(2, 2, 4),
        "right_state": torch.randn(2, 2, action_dim),
    }
    action = torch.randn(2, 8, action_dim)
    valid = torch.tensor([[True] * 4 + [False] * 4] * 2)
    for model, inputs in (
        (DiffusionPolicy(**kwargs), obs),
        (
            IdmPolicy(**kwargs, hidden=(8,)),
            {"obs": obs, "goal": {"camera_front": torch.randn(2, 4)}},
        ),
    ):
        loss = model.loss(inputs, action, valid)
        assert torch.isfinite(loss)
        loss.backward()
        assert any(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in model.parameters()
        )
        predicted = model.predict(
            inputs, [torch.Generator().manual_seed(i) for i in range(2)]
        )
        assert predicted.shape == action.shape and torch.isfinite(predicted).all()
    verifier = Verifier(
        n_cameras=1,
        patch_grid=2,
        feature_dim=4,
        proprio=shape.proprio,
        action_dim=action_dim,
        horizon=7,
        n_obs=2,
        hidden=16,
        layers=1,
        heads=4,
        ff_mult=2,
    )
    batch = {
        "obs": obs,
        "action": action,
        "source_patches": torch.randn(2, 1, 4, 4),
        "goal_patches": torch.randn(2, 1, 4, 4),
    }
    logits = verifier(batch)
    assert logits.shape == (2, 7) and torch.isfinite(logits).all()
    logits.square().mean().backward()
    # A future command must not change earlier-prefix reachability scores.
    changed = {**batch, "action": action.clone()}
    changed["action"][:, -1] += 100
    torch.testing.assert_close(verifier(changed)[:, :-1], logits[:, :-1])
