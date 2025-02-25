import pytest
import torch
from bubbleformer.models import get_model

@pytest.mark.parametrize("fields", [1, 2])
@pytest.mark.parametrize("patch_size", [4, 8, 16])
@pytest.mark.parametrize("embed_dim", [192, 384])
def test_avit(fields, patch_size, embed_dim):
    """
    Test AViT model with random configs
    """
    model_params = {
        "fields": fields,
        "patch_size": patch_size,
        "embed_dim": embed_dim,
        "num_heads": 4,
        "processor_blocks": 4,
        "drop_path": 0.1
    }
    spatial_dims = 64, 64
    model = get_model("avit", **model_params)
    x = torch.randn(2, 3, fields, *spatial_dims) # (B, T, C, H, W)
    y = model(x) # (B, T, C, H, W)

    assert y.shape == (2, 3, fields, *spatial_dims)

@pytest.mark.parametrize("time_window", [1, 3])
@pytest.mark.parametrize("fields", [1, 3])
@pytest.mark.parametrize("hidden_channels", [16, 32])
def test_unet_classic(time_window, fields, hidden_channels):
    """
    Test UNet model with random configs
    """
    model_params = {
        "time_window": time_window,
        "fields": fields,
        "hidden_channels": hidden_channels
    }
    spatial_dims = 64, 64
    model = get_model("unet_classic", **model_params)
    x = torch.randn(2, time_window, fields, *spatial_dims)
    y = model(x)

    assert y.shape == (2, time_window, fields, *spatial_dims)

@pytest.mark.parametrize("time_window", [1, 3])
@pytest.mark.parametrize("fields", [1, 3])
@pytest.mark.parametrize("hidden_channels", [16, 32])
def test_unet_modern(time_window, fields, hidden_channels):
    """
    Test UNet model with random configs
    """
    model_params = {
        "time_window": time_window,
        "fields": fields,
        "hidden_channels": hidden_channels,
        "ch_mults": [1, 2, 2],
        "norm": True,
    }
    spatial_dims = 64, 64
    model = get_model("unet_modern", **model_params)
    x = torch.randn(2, time_window, fields, *spatial_dims)
    y = model(x)

    assert y.shape == (2, time_window, fields, *spatial_dims)

@pytest.mark.parametrize("fields", [1, 2])
@pytest.mark.parametrize("patch_size", [4, 8, 16])
@pytest.mark.parametrize("embed_dim", [192, 384])
def test_avit_moe(fields, patch_size, embed_dim):
    """
    Test AViT model with random configs
    """
    model_params = {
        "fields": fields,
        "patch_size": patch_size,
        "embed_dim": embed_dim,
        "num_heads": 4,
        "processor_blocks": 4,
        "drop_path": 0.1,
        "n_experts": 6,
        "n_shared_experts":1,
        "top_k":2,
    }
    spatial_dims = 64, 64
    model = get_model("avit_moe", **model_params)
    x = torch.randn(2, 3, fields, *spatial_dims) # (B, T, C, H, W)
    y = model(x) # (B, T, C, H, W)

    assert y.shape == (2, 3, fields, *spatial_dims)