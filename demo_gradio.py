from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path
from typing import Any

import gradio as gr
import jax.numpy as jnp
import numpy as np
from flax import nnx

from visual_util import predictions_to_glb
from vggt.models import ModelConfig, VGGT
from vggt.models.vggt import create_vggt_from_nnx_npz, create_vggt_from_orbax_checkpoint
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.io import load_and_preprocess_images_np
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


class ModelRunner:
    def __init__(
        self,
        *,
        weights_orbax: str | None,
        weights_npz: str | None,
        from_pretrained: bool,
    ):
        self.weights_orbax = weights_orbax
        self.weights_npz = weights_npz
        self.from_pretrained = from_pretrained
        self.model: VGGT | None = None

    def get(self) -> VGGT:
        if self.model is not None:
            return self.model

        config = ModelConfig.vggt_base()
        rngs = nnx.Rngs(0)

        if self.from_pretrained:
            model = VGGT.from_pretrained(config=config, rngs=rngs, prefer_hf_weights=True)
        elif self.weights_orbax is not None and Path(self.weights_orbax).exists():
            model = create_vggt_from_orbax_checkpoint(self.weights_orbax, config=config, rngs=rngs, strict=True)
        elif self.weights_npz is not None and Path(self.weights_npz).exists():
            model = create_vggt_from_nnx_npz(self.weights_npz, config=config, rngs=rngs, strict=True)
        else:
            model = VGGT(config, rngs=rngs)

        model.eval()
        self.model = model
        return model


def _to_local_paths(files: list[Any] | None) -> list[str]:
    if not files:
        return []
    out: list[str] = []
    for file_obj in files:
        if isinstance(file_obj, str):
            out.append(file_obj)
            continue
        name = getattr(file_obj, "name", None)
        if isinstance(name, str):
            out.append(name)
    return sorted(out)


def _prepare_run_dir(image_paths: list[str]) -> tuple[Path, list[str]]:
    run_dir = Path(tempfile.mkdtemp(prefix="vggt_gradio_"))
    image_dir = run_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    copied_paths: list[str] = []
    for index, src in enumerate(image_paths):
        src_path = Path(src)
        dst = image_dir / f"{index:04d}{src_path.suffix.lower()}"
        shutil.copy2(src_path, dst)
        copied_paths.append(str(dst))
    return run_dir, copied_paths


def _predict(
    model: VGGT,
    image_paths: list[str],
    *,
    target_size: int,
) -> dict[str, np.ndarray]:
    images = load_and_preprocess_images_np(image_paths, target_size=target_size)
    batch = jnp.asarray(images)[None, ...]
    preds = model(batch)

    outputs: dict[str, np.ndarray] = {}
    for key, value in preds.items():
        if isinstance(value, list):
            continue
        array = np.asarray(value)
        if array.ndim >= 1 and array.shape[0] == 1:
            array = array[0]
        outputs[key] = array

    outputs["images"] = images
    image_h, image_w = images.shape[-3], images.shape[-2]
    extrinsic, intrinsic = pose_encoding_to_extri_intri(outputs["pose_enc"], image_size_hw=(image_h, image_w))
    outputs["extrinsic"] = extrinsic[0]
    outputs["intrinsic"] = intrinsic[0]
    outputs["world_points_from_depth"] = unproject_depth_map_to_point_map(
        outputs["depth"], outputs["extrinsic"], outputs["intrinsic"]
    )
    return outputs


def _update_frame_choices(files: list[Any] | None):
    paths = _to_local_paths(files)
    choices = ["All"] + [f"{idx}: {Path(path).name}" for idx, path in enumerate(paths)]
    return gr.Dropdown(choices=choices, value="All")


def build_app(runner: ModelRunner, *, default_target_size: int):
    def reconstruct(
        files,
        conf_thres,
        frame_filter,
        prediction_mode,
        mask_black_bg,
        mask_white_bg,
        show_cam,
        target_size,
    ):
        image_paths = _to_local_paths(files)
        if len(image_paths) == 0:
            return None, "请先上传图片。"

        run_dir, copied_paths = _prepare_run_dir(image_paths)
        try:
            predictions = _predict(runner.get(), copied_paths, target_size=int(target_size))
            scene = predictions_to_glb(
                predictions,
                conf_thres=float(conf_thres),
                filter_by_frames=str(frame_filter),
                mask_black_bg=bool(mask_black_bg),
                mask_white_bg=bool(mask_white_bg),
                show_cam=bool(show_cam),
                prediction_mode=str(prediction_mode),
                target_dir=str(run_dir),
            )
            output_glb = run_dir / "scene.glb"
            scene.export(output_glb)
            return str(output_glb), f"完成：{len(copied_paths)} 张图，结果保存在 {output_glb}"
        except Exception as exc:
            return None, f"重建失败：{exc}"

    with gr.Blocks(title="VGGT JAX Gradio Demo") as app:
        gr.Markdown("# VGGT JAX/Flax Gradio Demo")
        with gr.Row():
            files = gr.File(label="上传图片", file_count="multiple", type="filepath")
            with gr.Column():
                target_size = gr.Slider(112, 518, value=default_target_size, step=14, label="推理尺寸")
                conf_thres = gr.Slider(0.0, 100.0, value=50.0, step=0.1, label="置信度百分位")
                frame_filter = gr.Dropdown(["All"], value="All", label="显示帧")
                prediction_mode = gr.Radio(
                    choices=["Predicted Pointmap", "Predicted Depthmap"],
                    value="Predicted Pointmap",
                    label="重建来源",
                )
                mask_black_bg = gr.Checkbox(value=False, label="过滤黑背景")
                mask_white_bg = gr.Checkbox(value=False, label="过滤白背景")
                show_cam = gr.Checkbox(value=True, label="显示相机")
                run_btn = gr.Button("开始重建")

        viewer = gr.Model3D(label="重建结果 (GLB)")
        status = gr.Textbox(label="状态", interactive=False)

        files.change(_update_frame_choices, inputs=[files], outputs=[frame_filter])
        run_btn.click(
            reconstruct,
            inputs=[files, conf_thres, frame_filter, prediction_mode, mask_black_bg, mask_white_bg, show_cam, target_size],
            outputs=[viewer, status],
        )

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VGGT JAX/Flax Gradio demo")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--target-size", type=int, default=224)
    parser.add_argument("--weights-orbax", type=str, default=None)
    parser.add_argument("--weights-npz", type=str, default=None)
    parser.add_argument("--from-pretrained", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runner = ModelRunner(
        weights_orbax=args.weights_orbax,
        weights_npz=args.weights_npz,
        from_pretrained=args.from_pretrained,
    )
    app = build_app(runner, default_target_size=args.target_size)
    app.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
