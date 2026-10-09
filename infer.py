import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from tqdm import tqdm

from datasets.breast_nifti import DATA_ROOT, PROTOCOL, BreastIndex, PhaseCache
from common import (MODEL_KIND, DEFAULT_INFERENCE_STEPS, build_model, device_for, load_breast_checkpoint, predict_volume,
                           save_json, save_patient_pngs, save_volume, write_csv)


def resolve_checkpoint(model_path=None):
    if model_path:
        return Path(model_path).resolve()
    run = Path('runs/tavl_syn').resolve()
    path = run / "checkpoints/model_final.pth"
    if not path.is_file():
        raise FileNotFoundError(f"No final-epoch Rectified Flow checkpoint yet: {path}. "
                                "Finish training first, or explicitly specify --model_path for an intermediate checkpoint.")
    return path


def create_result_directory(root, name):
    if not name or name in ('.', '..') or '/' in name or '\\' in name or ':' in name:
        raise ValueError('run_name must be a single directory name, not a path')
    desired = Path(root).resolve() / name
    directory = desired
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    suffix = 0
    while True:
        try:
            directory.mkdir(parents=True, exist_ok=False)
            return directory
        except FileExistsError:
            suffix += 1
            directory = desired.with_name(f"{desired.name}_{stamp}_{suffix}")


def infer(args):
    if not 0 <= args.overlap < 1 or args.batch_size < 1:
        raise ValueError("Invalid overlap or inference batch size")
    index = BreastIndex(args.data_root)
    cases = index.select(args.split, args.limit_patients)
    device = device_for(args.device)
    checkpoint = resolve_checkpoint(args.model_path)
    state = load_breast_checkpoint(checkpoint, index)
    config = state["args"]
    model = build_model(config["img_size"], config["model_chn"], config["mr_base_ch"],
                        config["tavs_num_dirs"], config["k"]).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    ode_steps = args.ode_steps if args.ode_steps is not None else DEFAULT_INFERENCE_STEPS
    seed = args.seed if args.seed is not None else config["inference_seed"]
    if ode_steps < 1:
        raise ValueError("ODE steps must be positive")
    output = create_result_directory(args.output_dir, args.run_name or f"epoch_{state['epoch']}_{args.split}")
    print(f"Results: {output}; Euler steps={ode_steps}, NFE/patch={ode_steps}; seed={seed}", flush=True)
    (output / "nifti").mkdir()
    (output / "npy_volumes").mkdir()
    (output / "png").mkdir()
    if args.save_npy:
        (output / "npy").mkdir()
    cache = PhaseCache(args.cache_dir)
    manifest = []
    run_config = {
        "protocol": PROTOCOL, "model_kind": MODEL_KIND, "checkpoint": str(checkpoint), "epoch": state["epoch"],
        "index_fingerprint": index.fingerprint, "split": args.split,
        "patient_ids": [c["patient_id"] for c in cases], "args": vars(args),
        "model_args": config, "status": "running", "intensity": "raw MRI via pre-only inverse transform",
        "geometry": "original canonical XYZ shape and affine; 2D axial sliding windows",
        "npy_volume_axes": "ZXY", "npy_volume_intensity": "raw_mri",
        "target_read_during_inference": False,
        "ode_method": "euler", "ode_steps": ode_steps, "euler_k": config["k"], "ode_nfe_per_patch": ode_steps,
        "inference_seed": seed,
        "noise_protocol": "patient/slice-seeded Gaussian field shared by overlapping patches",
        "png_export": "every slice: pre and generated; uint8 0-255; shared patient pre window"}
    save_json(output / "run_config.json", run_config)
    for case in tqdm(cases, desc=f"Generate {args.split} patients"):
        pre = cache.get(case, 0)
        pred = predict_volume(model, pre, case, device, config["img_size"],
                              args.overlap, args.batch_size, ode_steps=ode_steps, seed=seed)
        if not np.isfinite(pred).all():
            raise FloatingPointError(case["patient_id"])
        pid = case["patient_id"]
        destination = output / "nifti" / f"{pid}_post1_generated.nii.gz"
        save_volume(pred, case["path"], destination)
        npy_destination = output / "npy_volumes" / f"{pid}_post1_generated.npy"
        np.save(npy_destination, np.ascontiguousarray(np.moveaxis(pred, 2, 0)), allow_pickle=False)
        if args.save_npy:
            npy_dir = output / "npy" / pid
            npy_dir.mkdir()
            for z in range(pred.shape[2]):
                np.save(npy_dir / f"{pid}_z{z:04d}.npy", pred[:, :, z], allow_pickle=False)
        png_dir, display = save_patient_pngs(pre, pred, case, output / "png")
        manifest.append({"patient_id": pid, "default_split": args.split,
                         "generated_post1_path": str(destination),
                         "generated_post1_npy_path": str(npy_destination),
                         "npy_axes": "ZXY",
                         "source_dce_path": case.get("source_dce_path", str(case["path"])),
                         "source_metadata_path": str(case["path"]) if "npy_paths" in case else "",
                         "source_phase_index": 0, "pre_robust_center": case["center"],
                         "pre_robust_scale": case["scale"], "shape_xyz": "x".join(map(str, pred.shape)),
                         "intensity_space": "raw_mri", "checkpoint": str(checkpoint),
                         "model_kind": MODEL_KIND, "ode_method": "euler", "ode_steps": ode_steps, "euler_k": config["k"],
                         "inference_seed": seed,
                         "generated_png_dir": str(png_dir / "generated"),
                         "pre_png_dir": str(png_dir / "pre"),
                         "png_window_low": display["window_low"],
                         "png_window_high": display["window_high"]})
        write_csv(output / "generated_manifest.csv", manifest)
    run_config["status"] = "complete"
    save_json(output / "run_config.json", run_config)
    save_json(output / "completion.json", {"status": "complete", "patients": len(manifest)})
    save_json(Path(args.output_dir).resolve() / "latest_result.json",
              {"directory": output.name, "split": args.split, "index_fingerprint": index.fingerprint})
    print(f"Saved {len(manifest)} generated 3D volumes: {output}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Generate breast Post1 volumes with TAVL-Syn and Euler sampling.")
    parser.add_argument("--data_root", default=str(DATA_ROOT))
    parser.add_argument("--model_path", help="Default: runs/tavl_syn/checkpoints/model_final.pth")
    parser.add_argument("--output_dir", default='outputs')
    parser.add_argument("--run_name")
    parser.add_argument("--cache_dir", default="", help="Optional phase cache; sequential inference needs no disk cache")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch_size", type=int, default=4, help="Number of patches per model call")
    parser.add_argument("--overlap", type=float, default=.5)
    parser.add_argument("--ode_steps", type=int, default=DEFAULT_INFERENCE_STEPS,
                        help="Euler inference steps (default: 5; independent of training previews)")
    parser.add_argument("--seed", type=int, help="Default: inference seed stored in checkpoint")
    parser.add_argument("--limit_patients", type=int, help="Debug only")
    parser.add_argument("--save_npy", action="store_true", help="Also save each original-size axial slice")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


if __name__ == "__main__":
    infer(parse_args())
