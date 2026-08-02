import torch

from cardiac_aug.models import AttentionUNet, ConditionalRefiner, MaskConditionedVAE, PatchDiscriminator


def test_model_shapes_and_backward():
    image = torch.rand(2, 1, 64, 64)
    mask = torch.randint(0, 4, (2, 64, 64))
    vae = MaskConditionedVAE(64, 4, 8, 8)
    reconstruction, mu, logvar = vae(image, mask)
    assert reconstruction.shape == image.shape and mu.shape == logvar.shape == (2, 8)
    refiner = ConditionalRefiner(4, 8)
    refined = refiner(reconstruction, mask)
    assert refined.shape == image.shape
    patch = PatchDiscriminator(4, 8)(refined, mask)
    assert patch.ndim == 4 and patch.shape[1] == 1
    logits = AttentionUNet(1, 4, 8)(image)
    assert logits.shape == (2, 4, 64, 64)
    (logits.mean() + patch.mean() + reconstruction.mean()).backward()
