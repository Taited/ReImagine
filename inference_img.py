import argparse
import glob
import os.path as osp

import numpy as np
import torch
from diffsynth.pipelines.flux_image_new import ControlNetInput, FluxImagePipeline, ModelConfig
from PIL import Image


_DEFAULT_POSITIVE_PROMPT = (
    "person, clean white background, highly detailed, high resolution, 4k"
)
_DEFAULT_NEGATIVE_PROMPT = (
    "low quality, blurry, overexposed, underexposed, bad anatomy, deformed body, "
    "extra limbs, extra fingers, missing fingers, distorted face, watermark, subtitle, text, logo, "
    "jpeg artifacts, noisy background, cluttered background"
)


def resize_with_ratio_and_padding(img, target_height=1328, target_width=800):
    """Resize to target height (fixed aspect for width), pad or center-crop to target width."""
    img_np = np.array(img)
    h, w = img_np.shape[:2]

    scale = target_height / h
    new_width = int(w * scale)

    img_np = np.array(
        Image.fromarray(img_np).resize((new_width, target_height), Image.Resampling.LANCZOS)
    )

    if new_width < target_width:
        pad = target_width - new_width
        left_pad, right_pad = pad // 2, pad - pad // 2
        left_edge = img_np[:, :1]
        right_edge = img_np[:, -1:]
        img_np = np.concatenate(
            [
                np.tile(left_edge, (1, left_pad, 1)),
                img_np,
                np.tile(right_edge, (1, right_pad, 1)),
            ],
            axis=1,
        )

    elif new_width > target_width:
        start = (new_width - target_width) // 2
        img_np = img_np[:, start : start + target_width]

    return Image.fromarray(img_np)


def _transformer_shard_paths(kontext_dir: str) -> list:
    pat = osp.join(kontext_dir, "transformer", "*.safetensors")
    shards = sorted(glob.glob(pat))
    if not shards:
        raise SystemExit(f"No .safetensors under {pat!r}")
    return shards


def parse_args():
    p = argparse.ArgumentParser(description="ReImagine Demo")
    p.add_argument("--kontext_dir", type=str, default="models/FLUX.1-Kontext-dev", help="FLUX.1-Kontext-dev root")
    p.add_argument(
        "--controlnet_dir",
        type=str,
        default="models/Flux.1-dev-Controlnet-Surface-Normals",
        help="Flux.1-dev-Controlnet-Surface-Normals root",
    )
    p.add_argument("--lora_dir", type=str, default="models/ReImagine-Pretrained", help="ReImagine LoRA folder")
    p.add_argument("-o", "--save_path", type=str, default="reimagine_demo.png", help="Output image path")
    p.add_argument(
        "--reference_path",
        type=str,
        default="assets/reference/101936_kontext.png",
        help="Wide reference image (left=front, right=back)",
    )
    p.add_argument(
        "--normal_path",
        type=str,
        default="assets/normal/103444_cam08.jpg",
        help="Normal map image",
    )
    p.add_argument("--lora_alpha", type=float, default=1.0, help="LoRA strength")
    p.add_argument("--controlnet_scale", type=float, default=1.0, help="ControlNet scale")
    p.add_argument("--num_inference_steps", type=int, default=20, help="Denoising steps")
    p.add_argument("--seed", type=int, default=43, help="RNG seed")
    p.add_argument("--height", type=int, default=1328, help="Output height")
    p.add_argument("--width", type=int, default=800, help="Output width")
    p.add_argument(
        "--prompt",
        type=str,
        default=_DEFAULT_POSITIVE_PROMPT,
        help="Positive text prompt (empty string falls back to built-in default)",
    )
    p.add_argument(
        "--negative_prompt",
        type=str,
        default=_DEFAULT_NEGATIVE_PROMPT,
        help="Negative text prompt (empty string falls back to built-in default)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    kontext_image = Image.open(args.reference_path)
    pose_image = Image.open(args.normal_path)
    pose_image = resize_with_ratio_and_padding(
        pose_image, target_height=args.height, target_width=args.width
    )

    shard_paths = _transformer_shard_paths(args.kontext_dir)
    pipe = FluxImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(path=shard_paths),
            ModelConfig(path=osp.join(args.kontext_dir, "text_encoder/model.safetensors")),
            ModelConfig(path=osp.join(args.kontext_dir, "text_encoder_2/")),
            ModelConfig(path=osp.join(args.kontext_dir, "ae.safetensors")),
            ModelConfig(path=osp.join(args.controlnet_dir, "diffusion_pytorch_model.safetensors")),
        ],
    )
    lora_file = osp.join(args.lora_dir, "kontext-wo_smplx-lora.safetensors")
    pipe.load_lora(pipe.dit, lora_file, alpha=args.lora_alpha)

    image = pipe(
        prompt=(args.prompt or "").strip() or _DEFAULT_POSITIVE_PROMPT,
        negative_prompt=(args.negative_prompt or "").strip() or _DEFAULT_NEGATIVE_PROMPT,
        kontext_images=[kontext_image],
        controlnet_inputs=[ControlNetInput(image=pose_image, scale=args.controlnet_scale)],
        num_inference_steps=args.num_inference_steps,
        height=args.height,
        width=args.width,
        seed=args.seed,
    )
    image.save(args.save_path)
    print(f"Saved: {args.save_path}")


if __name__ == "__main__":
    main()
