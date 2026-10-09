import csv
import hashlib
import json
import os
import shutil
import uuid
from collections import OrderedDict
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

DATA_ROOT = Path(os.environ.get('TAVL_DATA_ROOT', 'data/breast'))
PROJECT_DIR = Path(__file__).resolve().parents[1]
IMAGE_SIZE = 256
PROTOCOL = "breast-pre-post1-2d-pre-robust-v1"


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


class BreastIndex:
    def __init__(self, data_root=DATA_ROOT):
        self.root = Path(data_root).resolve()
        descriptor_path = self.root / "dataset.json"
        self.shared_npy = descriptor_path.is_file()
        descriptor = None
        if self.shared_npy:
            descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
            if descriptor.get("schema") != "breast-shared-npy-zxy-float32-v1" or descriptor.get("status") != "complete":
                raise ValueError("Shared NPY dataset is not fully converted and verified yet")
        split_path = self.root / "metadata/splits.csv"
        volume_path = self.root / "tasks/generation_3d.csv"
        slice_path = self.root / "tasks/generation_2d.csv"
        self.fingerprint = hashlib.sha256(b"".join(
            p.read_bytes() for p in (split_path, volume_path, slice_path)
        )).hexdigest()
        if descriptor is not None and descriptor.get("source_index_fingerprint") != self.fingerprint:
            raise ValueError("Shared NPY index fingerprint differs from its conversion record")
        split_rows = read_csv(split_path)
        splits = {row["patient_id"]: row for row in split_rows}
        if len(splits) != len(split_rows):
            raise ValueError("Duplicate patient IDs in splits.csv")
        self.cases = {}
        for row in read_csv(volume_path):
            pid = row["patient_id"]
            if pid in self.cases or pid not in splits:
                raise ValueError(f"Duplicate or unknown patient: {pid}")
            reference = splits[pid]
            for key in ("default_split", "official_split", "cv_fold"):
                if row[key] != reference[key]:
                    raise ValueError(f"Split mismatch: {pid}, {key}")
            split = row["default_split"]
            if split not in ("train", "val", "test"):
                raise ValueError(f"Invalid split: {pid}")
            if (row["official_split"] == "test") != (split == "test"):
                raise ValueError(f"Official test leakage: {pid}")
            if row["phase_semantics"] != "pre|post1" or row["phase_count"] != "2":
                raise ValueError(f"Expected [pre, post1]: {pid}")
            if row["input_phase_index"] != "0" or row["target_phase_index"] != "1":
                raise ValueError(f"Unexpected generation direction: {pid}")
            center, scale = float(row["pre_robust_center"]), float(row["pre_robust_scale"])
            if not np.isfinite([center, scale]).all() or scale <= 0:
                raise ValueError(f"Invalid pre normalization: {pid}")

            path = self.root / "cases" / pid / "dce_raw_ras_1mm.nii.gz"
            npy_info = {}
            if self.shared_npy:
                folder = self.root / split / pid
                path = folder / "metadata.json"
                meta = json.loads(path.read_text(encoding="utf-8"))
                if (meta.get("status") != "complete" or meta.get("patient_id") != pid or
                        meta.get("default_split") != split or meta.get("axes") != "ZXY" or
                        meta.get("dtype") != "float32" or meta.get("intensity_transform") != "none"):
                    raise ValueError(f"Invalid shared NPY metadata: {pid}")
                if not np.allclose([meta["pre_robust_center"], meta["pre_robust_scale"]], [center, scale], rtol=1e-7):
                    raise ValueError(f"Shared NPY normalization differs: {pid}")
                paths = [folder / "pre.npy", folder / "post1.npy"]
                if not all(p.is_file() for p in paths):
                    raise FileNotFoundError(f"Missing shared NPY phase: {pid}")
                npy_info = {"npy_paths": paths, "npy_metadata": meta,
                            "source_dce_path": meta["source_signature"]["path"]}
            if not path.is_file():
                raise FileNotFoundError(path)
            self.cases[pid] = dict(row, path=path, center=center, scale=scale, slices=[], **npy_info)
        if set(self.cases) != set(splits):
            raise ValueError("Generation index and patient split populations differ")
        for row in read_csv(slice_path):
            pid = row["patient_id"]
            case = self.cases[pid]
            if row["default_split"] != case["default_split"]:
                raise ValueError(f"2D/3D split mismatch: {pid}")
            if row["slice_axis_xyz"] != "2" or row["slice_index_type"] != "standard_axial_ras":
                raise ValueError("Only standard axial slices (XYZ axis 2) are supported")
            if (row["input_phase_index"], row["target_phase_index"]) != ("0", "1"):
                raise ValueError(f"2D generation direction mismatch: {pid}")
            for column, value in (("pre_robust_center", case["center"]),
                                  ("pre_robust_scale", case["scale"])):
                if not np.isclose(float(row[column]), value, rtol=1e-7, atol=1e-7):
                    raise ValueError(f"2D normalization mismatch: {pid}")
            case["slices"].append(int(row["slice_index"]))
        for pid, case in self.cases.items():
            if not case["slices"] or len(set(case["slices"])) != len(case["slices"]):
                raise ValueError(f"Empty/duplicate slice index: {pid}")
            case["slices"].sort()

    def select(self, split, limit=None):
        rows = [c for c in self.cases.values() if c["default_split"] == split]
        if not rows:
            raise ValueError(f"Empty split: {split}")
        if limit is not None and limit <= 0:
            raise ValueError("Patient limit must be positive")
        return rows if limit is None else rows[:limit]

    def summary(self):
        return {split: {"patients": len(self.select(split)),
                        "slices": sum(len(c["slices"]) for c in self.select(split))}
                for split in ("train", "val", "test")}


class PhaseCache:

    def __init__(self, directory=None, max_open=4):
        self.directory = Path(directory) if directory else None
        self.max_open = max_open if directory else 2
        self.open_arrays = OrderedDict()

    def get(self, case, phase):
        if "npy_paths" in case:
            path = case["npy_paths"][phase]
            stat = path.stat()
            key = f"shared:{path}:{stat.st_size}:{stat.st_mtime_ns}"
            if key in self.open_arrays:
                self.open_arrays.move_to_end(key)
                return self.open_arrays[key]
            while len(self.open_arrays) >= self.max_open:
                self.open_arrays.popitem(last=False)
            stored = np.load(path, mmap_mode="r", allow_pickle=False)
            meta = case["npy_metadata"]
            if (list(stored.shape) != meta["shape_zxy"] or stored.dtype != np.float32 or
                    not stored.flags.c_contiguous):
                raise ValueError(f"Shared NPY shape/dtype/order mismatch: {path}")


            array = np.moveaxis(stored, 0, 2)
            if min(case["slices"]) < 0 or max(case["slices"]) >= array.shape[2]:
                raise ValueError(f"Out-of-bounds shared slice index: {path}")
            self.open_arrays[key] = array
            return array
        path = case["path"]
        stat = path.stat()
        identity = f"{path}|{stat.st_size}|{stat.st_mtime_ns}|{phase}|raw-f32-v1"
        key = hashlib.sha256(identity.encode()).hexdigest()[:24]
        if key in self.open_arrays:
            self.open_arrays.move_to_end(key)
            return self.open_arrays[key]


        while len(self.open_arrays) >= self.max_open:
            self.open_arrays.popitem(last=False)
        cached = None
        if self.directory:
            self.directory.mkdir(parents=True, exist_ok=True)
            cached = self.directory / f"{case['patient_id']}_phase{phase}_{key}.npy"
        if cached is None or not cached.exists():
            image = nib.load(str(path))
            if len(image.shape) != 4 or image.shape[3] != 2:
                raise ValueError(f"Expected [X,Y,Z,2]: {path}, got {image.shape}")
            if nib.aff2axcodes(image.affine) != ("R", "A", "S"):
                raise ValueError(f"Expected RAS: {path}")
            if not np.allclose(image.header.get_zooms()[:3], (1, 1, 1), atol=1e-4):
                raise ValueError(f"Expected 1mm spacing: {path}")

            array = np.array(image.dataobj[..., phase], dtype=np.float32)
            if not np.isfinite(array).all():
                raise ValueError(f"Nonfinite phase {phase}: {path}")
            if min(case["slices"]) < 0 or max(case["slices"]) >= array.shape[2]:
                raise ValueError(f"Out-of-bounds slice index: {path}")
            if cached:
                if shutil.disk_usage(self.directory).free < array.nbytes + 1024 ** 3:
                    raise OSError(f"Insufficient cache disk space: {self.directory}. "
                                  "Choose --cache_dir on a larger drive or --cache_dir '' to disable.")
                temp = cached.with_name(cached.name + f".{os.getpid()}.{uuid.uuid4().hex}.tmp")
                try:
                    with temp.open("wb") as stream:
                        np.save(stream, array, allow_pickle=False)

                    if not cached.exists():
                        try:
                            os.replace(temp, cached)
                        except PermissionError:
                            if not cached.exists():
                                raise
                finally:
                    temp.unlink(missing_ok=True)
        if cached:
            array = np.load(cached, mmap_mode="r", allow_pickle=False)
        self.open_arrays[key] = array
        while len(self.open_arrays) > self.max_open:
            self.open_arrays.popitem(last=False)
        return array


def normalize(array, case):
    return (np.asarray(array, dtype=np.float32) - case["center"]) / case["scale"]


def denormalize(array, case):

    return np.asarray(array, dtype=np.float32) * case["scale"] + case["center"]


class BreastPatchDataset(Dataset):

    def __init__(self, index, image_size=256, cache_dir=None, patches_per_patient=4,
                 patient_limit=None):
        self.cases = index.select("train", patient_limit)
        self.size = image_size
        self.repeats = patches_per_patient
        if image_size < 32 or image_size % 16 or patches_per_patient < 1:
            raise ValueError("Patch must be >=32 and a multiple of 16; repeats >=1")
        self.cache = PhaseCache(cache_dir)

    def __len__(self):
        return len(self.cases) * self.repeats

    def __getitem__(self, item):
        case = self.cases[item % len(self.cases)]
        z = case["slices"][int(torch.randint(len(case["slices"]), ()).item())]
        pre = np.array(self.cache.get(case, 0)[:, :, z], copy=True)
        post = np.array(self.cache.get(case, 1)[:, :, z], copy=True)
        if pre.shape != post.shape:
            raise ValueError("Paired image shapes differ")

        pads = tuple((max(0, self.size - n) // 2,
                      max(0, self.size - n) - max(0, self.size - n) // 2) for n in pre.shape)
        pre, post = np.pad(pre, pads), np.pad(post, pads)

        if torch.rand(()).item() < 0.8 and np.any(pre != 0):
            positions = np.flatnonzero(pre != 0)
            point = np.unravel_index(positions[torch.randint(len(positions), ()).item()], pre.shape)
            start = [int(np.clip(p - self.size // 2, 0, n - self.size))
                     for p, n in zip(point, pre.shape)]
        else:
            start = [torch.randint(n - self.size + 1, ()).item() for n in pre.shape]
        region = tuple(slice(p, p + self.size) for p in start)
        pre, post = normalize(pre[region], case), normalize(post[region], case)
        return {"image": torch.from_numpy(pre.copy()[None]),
                "label": torch.from_numpy(post.copy()[None]),
                "patient_id": case["patient_id"], "slice_index": z}


class PatientBlockSampler(Sampler):

    def __init__(self, dataset, generator=None):
        self.patient_count = len(dataset.cases)
        self.repeats = dataset.repeats
        self.generator = generator

    def __len__(self):
        return self.patient_count * self.repeats

    def __iter__(self):
        order = torch.randperm(self.patient_count, generator=self.generator).tolist()
        for patient in order:
            for repeat in range(self.repeats):
                yield patient + repeat * self.patient_count


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Audit breast generation indexes (read-only)")
    parser.add_argument("--data_root", default=str(DATA_ROOT))
    args = parser.parse_args()
    index = BreastIndex(args.data_root)
    print(json.dumps({"protocol": PROTOCOL, "splits": index.summary(),
                      "index_fingerprint": index.fingerprint}, indent=2))
