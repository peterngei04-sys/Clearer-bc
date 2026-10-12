import sys
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter

sys.path.insert(0, str(Path(__file__).parent / "models"))

from u2net import U2NETP


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "u2netp.pth"

_model = None


def get_model():
    global _model

    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model weights not found: {MODEL_PATH}"
            )

        print("Loading Clearer.bc AI model...")

        _model = U2NETP(3, 1)

        state_dict = torch.load(
            MODEL_PATH,
            map_location="cpu",
            weights_only=True
        )

        _model.load_state_dict(state_dict)
        _model.eval()

        print("Clearer.bc AI model loaded.")

    return _model


def prepare_image(image, size):
    """Prepare an RGB image using the working Termux tensor method."""

    image = image.convert("RGB")
    original_size = image.size

    resized = image.resize(
        (size, size),
        Image.Resampling.BILINEAR
    )

    pixels = list(resized.getdata())

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    channels = [[], [], []]

    for r, g, b in pixels:
        channels[0].append(
            (r / 255.0 - mean[0]) / std[0]
        )
        channels[1].append(
            (g / 255.0 - mean[1]) / std[1]
        )
        channels[2].append(
            (b / 255.0 - mean[2]) / std[2]
        )

    tensor = torch.tensor(
        channels,
        dtype=torch.float32
    ).reshape(1, 3, size, size)

    return tensor, original_size


def create_mask(prediction, original_size):
    """Normalize, refine and resize the predicted foreground mask."""

    mask = prediction.squeeze().detach().cpu()

    if mask.ndim != 2:
        raise RuntimeError(
            f"Unexpected mask dimensions: {tuple(mask.shape)}"
        )

    minimum = mask.min().item()
    maximum = mask.max().item()

    print(
        f"Raw mask range: {minimum:.6f} - {maximum:.6f}"
    )

    if maximum - minimum > 1e-8:
        mask = (mask - minimum) / (maximum - minimum)
    else:
        raise RuntimeError(
            "The model returned a constant mask. "
            "Check the model weights and input image."
        )

    mask_np = mask.numpy().astype(np.float32)

    # Keep the same refinement strategy as v3.
    low = 0.30
    high = 0.70

    refined = np.clip(
        (mask_np - low) / (high - low),
        0.0,
        1.0
    )

    alpha = (refined * 255.0).round().astype(np.uint8)

    mask_image = Image.fromarray(alpha)

    mask_image = mask_image.resize(
        original_size,
        Image.Resampling.LANCZOS
    )

    mask_image = mask_image.filter(
        ImageFilter.GaussianBlur(radius=0.35)
    )

    return mask_image


def remove_background(input_path, output_path, inference_size=384):
    """Remove a background using the selected inference resolution."""

    if inference_size not in (320, 384, 512):
        raise ValueError(
            "Supported sizes are 320, 384 and 512."
        )

    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input image not found: {input_path}"
        )

    print("=" * 48)
    print("Clearer.bc Background Removal")
    print(f"Inference resolution: {inference_size}x{inference_size}")
    print("=" * 48)

    with Image.open(input_path) as opened:
        image = opened.convert("RGB")

    original_size = image.size

    print(f"Original image: {original_size[0]}x{original_size[1]}")

    tensor, original_size = prepare_image(
        image,
        inference_size
    )

    print(f"Input tensor: {tuple(tensor.shape)}")

    model = get_model()

    print("Running CPU inference...")

    with torch.inference_mode():
        outputs = model(tensor)
        prediction = outputs[0]

    print("Inference complete.")

    # Release intermediate model outputs before saving.
    del outputs
    del tensor

    mask_image = create_mask(
        prediction,
        original_size
    )

    del prediction

    result = image.convert("RGBA")
    result.putalpha(mask_image)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    result.save(output_path, "PNG")

    print(f"Saved: {output_path}")
    print("Clearer.bc processing complete.")

    return str(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Clearer.bc local background remover"
    )

    parser.add_argument("input", help="Input image path")
    parser.add_argument("output", help="Output PNG path")

    parser.add_argument(
        "--size",
        type=int,
        choices=[320, 384, 512],
        default=384,
        help="Model input resolution (default: 384)"
    )

    args = parser.parse_args()

    remove_background(
        args.input,
        args.output,
        inference_size=args.size
    )
