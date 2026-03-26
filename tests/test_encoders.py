from voln_uav.models.encoders import parse_open_clip_spec


def test_parse_open_clip_spec_defaults():
    model, pretrained = parse_open_clip_spec("open_clip:")
    assert model == "ViT-B-32"
    assert pretrained == "laion2b_s34b_b79k"


def test_parse_open_clip_spec_with_model_only():
    model, pretrained = parse_open_clip_spec("open_clip:ViT-L-14")
    assert model == "ViT-L-14"
    assert pretrained == "laion2b_s34b_b79k"


def test_parse_open_clip_spec_with_model_and_pretrained():
    model, pretrained = parse_open_clip_spec("open_clip:ViT-B-32:openai")
    assert model == "ViT-B-32"
    assert pretrained == "openai"
