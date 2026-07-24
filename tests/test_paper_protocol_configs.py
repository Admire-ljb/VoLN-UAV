from pathlib import Path

from voln_uav.common.config import load_config


ROOT = Path(__file__).resolve().parents[1]
DINO_V3 = "hf:facebook/dinov3-vitb16-pretrain-lvd1689m"
CLIP_B16 = "open_clip:ViT-B-16:openai"


def _load(name: str) -> dict:
    return load_config(ROOT / "configs" / name)


def test_voln_training_config_matches_paper_architecture():
    adapter = _load("train_adapter_dataset_release.yaml")
    planner = _load("train_planner_dataset_release.yaml")

    assert adapter["model"]["dino_backbone"] == DINO_V3
    assert adapter["model"]["dino_dim"] == 768
    assert adapter["model"]["clip_image_encoder"] == CLIP_B16
    assert planner["model"]["dino_backbone"] == DINO_V3
    assert planner["model"]["text_encoder"] == CLIP_B16
    assert planner["model"]["planner_backbone"] == "lmsys/vicuna-7b-v1.5"
    assert planner["model"]["lora_rank"] == 16
    assert planner["model"]["horizon"] == 8
    assert planner["model"]["proprio_schema"] == "body_linear_angular_relative_v1"
    assert "anchor_weight" not in planner["loss"]
    assert planner["success_radius"] == 4.0


def test_benchmark_config_matches_beacon_count_protocol():
    benchmark = _load("benchmark_dataset_release.yaml")

    assert benchmark["beacons"]["task_beacons_min_per_route"] == 3
    assert benchmark["beacons"]["task_beacons_max_per_route"] == 5
    assert benchmark["beacons"]["background_per_scene"] == 150


def test_release_evaluation_configs_share_paper_protocol():
    names = [
        "eval_offline_dataset_release.yaml",
        "eval_offline_seq2seq_dataset_release.yaml",
        "eval_offline_cma_dataset_release.yaml",
        "eval_offline_lag_dataset_release.yaml",
        "eval_airsim_dataset_release.yaml",
        "eval_airsim_seq2seq_dataset_release.yaml",
        "eval_airsim_cma_dataset_release.yaml",
        "eval_airsim_lag_dataset_release.yaml",
    ]
    for name in names:
        config = _load(name)
        assert config["max_steps"] == 128, name
        assert config["success_radius"] == 4.0, name
        assert config["stop_probability"] is None, name
        assert config["paper_protocol"] == "paper_protocol.yaml", name
        assert config["strict_paper_protocol"] is True, name
        assert config["strict_scenes"] is False, name
        assert config["planner_ckpt"].endswith("planner_best.pt"), name
        assert config["model"]["dino_backbone"] == DINO_V3, name
        assert config["model"]["clip_image_encoder"] == CLIP_B16, name
        assert config["model"]["text_encoder"] == CLIP_B16, name
        assert config["model"]["horizon"] == 8, name
        assert config["model"]["proprio_schema"] == "body_linear_angular_relative_v1", name

        if name.startswith("eval_airsim"):
            assert config["termination_mode"] == "paper", name
            assert config["reference_bootstrap_steps"] == 0, name
            assert config["min_steps_before_stop"] == 0, name


def test_ablation_configs_match_manuscript_definitions():
    no_align_adapter = _load("train_adapter_no_align_dataset_release.yaml")
    no_align = _load("train_planner_no_align_dataset_release.yaml")
    no_lora = _load("train_planner_no_lora_dataset_release.yaml")
    clip_input = _load("train_planner_clip_input_dataset_release.yaml")

    assert no_align_adapter["alignment_mode"] == "no_align"
    assert no_align["adapter_ckpt"].endswith("adapter_no_align/adapter_best.pt")
    assert no_lora["model"]["lora_enabled"] is False
    assert clip_input["model"]["vision_input"] == "clip"
    assert clip_input["adapter_ckpt"] is None

    for variant in (no_align, no_lora, clip_input):
        assert variant["train_records"] == "records/train.jsonl"
        assert variant["val_records"] == "records/val.jsonl"
        assert variant["model"]["planner_backbone"] == "lmsys/vicuna-7b-v1.5"
        assert variant["model"]["horizon"] == 8


def test_online_controller_defaults_are_explicit():
    assert _load("eval_airsim_dataset_release.yaml")["controller"] == "random"
    for name in (
        "eval_airsim_seq2seq_dataset_release.yaml",
        "eval_airsim_cma_dataset_release.yaml",
        "eval_airsim_lag_dataset_release.yaml",
        "eval_airsim_no_align_dataset_release.yaml",
        "eval_airsim_no_lora_dataset_release.yaml",
        "eval_airsim_clip_input_dataset_release.yaml",
    ):
        assert _load(name)["controller"] == "policy", name
