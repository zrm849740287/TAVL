import base64
import csv
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from PIL import Image

from datasets.breast_nifti import PROTOCOL, normalize, denormalize
from models.tavs_unet import build_velocity_model
from models.rectified_flow import RectifiedFlow, exponential_curve


MODEL_KIND = "breast-rectified-tavs-dsc-v1"
DEFAULT_INFERENCE_STEPS = 5


def build_model(img_size=256, model_chn=128, mr_base_ch=32, tavs_num_dirs=8, k=0.):
    model = build_velocity_model(model_chn=model_chn, image_size=img_size,
                                 mr_base_ch=mr_base_ch, tavs_num_dirs=tavs_num_dirs)
    model.euler_k = float(k)
    return model


def device_for(name):
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Use --device cpu for CPU execution.")
    return torch.device(name)


def starts(length, size, stride):
    positions = list(range(0, max(length - size, 0) + 1, stride))
    if positions[-1] != length - size:
        positions.append(length - size)
    return positions


@torch.inference_mode()
def integrate_flow(model, pre, noise, ode_steps=DEFAULT_INFERENCE_STEPS):

    if not isinstance(ode_steps, int) or isinstance(ode_steps, bool) or ode_steps < 1:
        raise ValueError("ode_steps must be a positive integer")
    if pre.shape != noise.shape or pre.ndim != 4 or pre.shape[1] != 1:
        raise ValueError("pre and noise must have matched [B,1,H,W] shapes")
    if pre.device != noise.device:
        raise ValueError("pre and noise must be on the same device")

    k = getattr(model, 'euler_k', 0.)
    times = exponential_curve(torch.linspace(0., 1., ode_steps+1, device=noise.device), k)
    if not torch.isfinite(times).all() or not torch.all(times[1:] > times[:-1]):
        raise ValueError('Invalid Euler time grid')
    cached = model.prepare_condition(pre) if hasattr(model, 'prepare_condition') else None
    x = noise.clone()
    for i in range(ode_steps):
        t = times[i].expand(x.shape[0])
        velocity = (model(t, x, pre, condition=cached) if cached is not None
                    else model(t, x, pre))
        x = RectifiedFlow.euler(x, velocity, times[i+1]-times[i])
    return x


def _slice_noise(shape, patient_id, slice_index, seed):

    key = f"{MODEL_KIND}|{int(seed)}|{patient_id}|{int(slice_index)}".encode("utf-8")
    local_seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "little")
    generator = np.random.default_rng(local_seed)
    return generator.standard_normal(shape, dtype=np.float32)


@torch.inference_mode()
def predict_slice(model, raw_pre, case, device, size=256, overlap=0.5, batch_size=4,
                  ode_steps=DEFAULT_INFERENCE_STEPS, seed=20260727, slice_index=0):
    if not 0 <= overlap < 1 or batch_size < 1:
        raise ValueError("Require 0 <= overlap < 1 and batch_size >= 1")
    height, width = raw_pre.shape
    px, py = max(0, size - height), max(0, size - width)
    pads = ((px // 2, px - px // 2), (py // 2, py - py // 2))
    image = normalize(np.pad(raw_pre, pads), case)
    noise = _slice_noise(image.shape, case["patient_id"], slice_index, seed)
    stride = max(1, int(size * (1 - overlap)))
    points = [(x, y) for x in starts(image.shape[0], size, stride)
              for y in starts(image.shape[1], size, stride)]
    coordinate = np.linspace(-1, 1, size, dtype=np.float32)
    weight = np.maximum(np.exp(-0.5 * ((coordinate[:, None] / .4) ** 2 +
                                      (coordinate[None, :] / .4) ** 2)), 1e-4)
    output, weights = np.zeros_like(image), np.zeros_like(image)
    for first in range(0, len(points), batch_size):
        batch_points = points[first:first + batch_size]
        batch = np.stack([image[x:x + size, y:y + size] for x, y in batch_points])[:, None]
        noise_batch = np.stack([noise[x:x + size, y:y + size]
                                for x, y in batch_points])[:, None]
        pred = integrate_flow(model, torch.from_numpy(batch).to(device),
                              torch.from_numpy(noise_batch).to(device),
                              ode_steps=ode_steps).float().cpu().numpy()
        if pred.shape != batch.shape or not np.isfinite(pred).all():
            raise ValueError(f"Invalid prediction: {pred.shape}, expected {batch.shape}")
        for value, (x, y) in zip(pred[:, 0], batch_points):
            output[x:x + size, y:y + size] += value * weight
            weights[x:x + size, y:y + size] += weight
    output /= weights
    return output[pads[0][0]:pads[0][0] + height, pads[1][0]:pads[1][0] + width]


def predict_volume(model, pre, case, device, size=256, overlap=.5, batch_size=4,
                   ode_steps=DEFAULT_INFERENCE_STEPS, seed=20260727):
    output = np.zeros(pre.shape, dtype=np.float32)

    for z in range(pre.shape[2]):
        image = np.asarray(pre[:, :, z])
        if np.any(image != 0):
            value = predict_slice(model, image, case, device, size, overlap, batch_size,
                                  ode_steps=ode_steps, seed=seed, slice_index=z)
            output[:, :, z] = denormalize(value, case)


    return output


def reference_geometry(reference):
    if Path(reference).suffix.lower() == ".json":
        meta = json.loads(Path(reference).read_text(encoding="utf-8"))
        return (tuple(meta["shape_xyz"]), np.asarray(meta["affine"]),
                nib.Nifti1Header(binaryblock=base64.b64decode(meta["nifti_header_base64"])),
                meta["qform"], meta["qform_code"], meta["sform"], meta["sform_code"])
    source = nib.load(str(reference))
    return (source.shape[:3], source.affine, source.header.copy(), source.get_qform(),
            int(source.header["qform_code"]), source.get_sform(), int(source.header["sform_code"]))


def save_volume(array, reference, destination):
    shape, affine, header, qform, qcode, sform, scode = reference_geometry(reference)
    if tuple(array.shape) != tuple(shape):
        raise ValueError("Generated volume must retain the patient's canonical XYZ shape")
    spacing = header.get_zooms()[:3]
    header.set_data_shape(array.shape)
    header.set_data_dtype(np.float32)
    header.set_zooms(spacing)
    image = nib.Nifti1Image(np.asarray(array, dtype=np.float32), affine, header)
    image.set_qform(qform, qcode)
    image.set_sform(sform, scode)
    nib.save(image, str(destination))


def save_preview(pre, pred, case, destination, target=None):

    foreground = pre[pre != 0]
    if not foreground.size:
        foreground = pre.reshape(-1)
    low, high = np.percentile(foreground, [.5, 99.5])
    high = max(high, low + 1e-6)
    panels = [pre, pred] if target is None else [pre, target, pred]
    pixels = [np.rot90(np.uint8(np.clip((a - low) / (high - low), 0, 1) * 255))
              for a in panels]
    Image.fromarray(np.concatenate(pixels, axis=1)).save(destination)


def png_window_from_pre(pre):

    values = pre[np.isfinite(pre) & (pre != 0)]
    if not values.size:
        return 0., 1.
    low, high = np.percentile(values, [.5, 99.5])
    return float(low), float(max(high, low + 1e-6))


def to_display_uint8(array, low, high):
    if not np.isfinite([low, high]).all() or high <= low:
        raise ValueError("Invalid PNG display window")
    if not np.isfinite(array).all():
        raise ValueError("Nonfinite PNG input")
    values = np.clip((np.asarray(array, dtype=np.float32) - low) / (high - low), 0, 1)
    return np.rot90(np.rint(values * 255).astype(np.uint8))


def save_patient_pngs(pre, pred, case, png_root):

    if pre.shape != pred.shape or pre.ndim != 3:
        raise ValueError("PNG export expects matched XYZ volumes")
    root = Path(png_root) / case["patient_id"]
    for folder in ("pre", "generated", "previews"):
        (root / folder).mkdir(parents=True, exist_ok=True)
    low, high = png_window_from_pre(pre)
    preview_z = {case["slices"][int((len(case["slices"]) - 1) * f)] for f in (.25, .5, .75)}
    for z in range(pre.shape[2]):
        name = f"{case['patient_id']}_z{z:04d}.png"
        source_pixels = to_display_uint8(pre[:, :, z], low, high)
        generated_pixels = to_display_uint8(pred[:, :, z], low, high)
        Image.fromarray(source_pixels).save(root / "pre" / name)
        Image.fromarray(generated_pixels).save(root / "generated" / name)
        if z in preview_z:
            Image.fromarray(np.concatenate((source_pixels, generated_pixels), axis=1)).save(
                root / "previews" / name)
    info = {"patient_id": case["patient_id"], "slice_count": pre.shape[2],
            "window_source": "whole_patient_pre_finite_nonzero_percentiles_0.5_99.5",
            "window_low": low, "window_high": high, "dtype": "uint8", "range": [0, 255],
            "mapping": "round(clip((raw-low)/(high-low),0,1)*255)",
            "display_transform": "numpy.rot90(XY_slice,k=1); NIfTI and NPY are not rotated",
            "preview_panels": ["PRE", "GENERATED_POST1"],
            "display_only": True, "target_intensity_used": False}
    save_json(root / "display.json", info)
    return root, info


def write_csv(path, rows, fields=None):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def load_breast_checkpoint(path, index, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(ckpt, dict) or ckpt.get("model_kind") != MODEL_KIND:
        raise ValueError("Checkpoint model kind does not match this breast synthesis pipeline")
    if ckpt.get("protocol") != PROTOCOL:
        raise ValueError("Checkpoint preprocessing protocol does not match this pipeline")
    if ckpt.get("index_fingerprint") != index.fingerprint:
        raise ValueError("Dataset indexes differ from checkpoint; check patient splits")
    return ckpt
