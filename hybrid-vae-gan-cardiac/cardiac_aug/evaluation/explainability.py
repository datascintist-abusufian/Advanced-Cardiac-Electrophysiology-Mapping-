from __future__ import annotations

import torch
import torch.nn.functional as F


def segmentation_input_attribution(model, image: torch.Tensor, class_index: int) -> torch.Tensor:
    """Gradient × input attribution for the selected class's predicted segmentation region."""
    model.eval()
    x = image.detach().clone().requires_grad_(True)
    logits = model(x)
    predicted_region = (logits.argmax(1) == class_index).float()
    if predicted_region.sum() == 0:
        score = logits[:, class_index].mean()
    else:
        score = (logits[:, class_index] * predicted_region).sum() / predicted_region.sum()
    model.zero_grad(set_to_none=True)
    score.backward()
    attribution = (x.grad * x).abs().sum(1, keepdim=True)
    minimum, maximum = attribution.amin((-2, -1), keepdim=True), attribution.amax((-2, -1), keepdim=True)
    return (attribution - minimum) / (maximum - minimum + 1e-8)


@torch.no_grad()
def attention_gate_map(model, image: torch.Tensor) -> torch.Tensor:
    """Expose the finest Attention U-Net gate as a segmentation-specific explanation."""
    model.eval()
    model(image)
    return F.interpolate(model.last_attention_maps[-1], image.shape[-2:], mode="bilinear", align_corners=False)
