import argparse
from concurrent.futures import ThreadPoolExecutor
import math
from pathlib import Path

import nibabel as nib
import numpy as np
from skimage.metrics import structural_similarity
from tqdm import tqdm

from datasets.breast_nifti import BreastIndex, PhaseCache, DATA_ROOT
from common import reference_geometry, write_csv


def slice_metrics(truth, prediction):
    truth, prediction = np.asarray(truth, dtype=np.float64), np.asarray(prediction, dtype=np.float64)
    if truth.ndim != 2 or truth.shape != prediction.shape or min(truth.shape) < 11:
        raise ValueError('SSIM requires matched 2D slices at least 11 pixels wide')
    if not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise ValueError('Nonfinite slice')
    delta = prediction-truth
    mse = float(np.mean(delta**2))
    return dict(psnr=math.inf if mse == 0 else -10*math.log10(mse),
                ssim=float(structural_similarity(truth, prediction, data_range=1.,
                    win_size=11, gaussian_weights=True, sigma=1.5, use_sample_covariance=False,
                    channel_axis=None, K1=.01, K2=.03)),
                mae=float(np.abs(delta).mean()))


def evaluate(args):
    if args.workers < 1:
        raise ValueError('workers must be positive')
    index = BreastIndex(args.data_root)
    cases = index.select(args.split)
    paths = [Path(args.predictions)/(c['patient_id']+'_post1_generated.nii.gz') for c in cases]
    missing = [p for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f'Missing predictions; evaluation never drops patients: {missing[:3]}')
    rows = []
    cache = PhaseCache()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for case, path in tqdm(list(zip(cases, paths)), desc='Evaluate patients'):
            image = nib.as_closest_canonical(nib.load(str(path)))
            shape, affine, *_ = reference_geometry(case['path'])
            if tuple(image.shape) != tuple(shape) or not np.allclose(image.affine, affine, rtol=0, atol=1e-4):
                raise ValueError(f'Prediction geometry differs: {path}')
            pred = image.get_fdata(dtype=np.float32)
            pre, truth = cache.get(case, 0), cache.get(case, 1)
            if not np.isfinite(truth).all() or not np.isfinite(pred).all() or not np.isfinite(pre).all():
                raise ValueError(f'Nonfinite volume: {case["patient_id"]}')
            low, high = float(truth.min()), float(truth.max())
            if high <= low:
                raise ValueError(f'Constant target cannot define min-max range: {case["patient_id"]}')
            zs = np.flatnonzero(np.any(pre != 0, axis=(0,1)))
            if not zs.size:
                raise ValueError(f'No valid source slices: {case["patient_id"]}')
            def score(z):

                gt = (np.asarray(truth[:,:,z], dtype=np.float64)-low)/(high-low)
                generated = (np.asarray(pred[:,:,z], dtype=np.float64)-low)/(high-low)
                return slice_metrics(gt, generated)
            scores = list(pool.map(score, zs))
            rows.append(dict(patient_id=case['patient_id'], slices=len(scores),
                **{m:float(np.mean([s[m] for s in scores])) for m in ('psnr','ssim','mae')}))
    print(f'Patients: {len(rows)}; valid slices: {sum(r["slices"] for r in rows)}')
    for metric in ('psnr','ssim','mae'):
        values = np.asarray([r[metric] for r in rows])
        std = f'{values.std(ddof=1):.4f}' if len(values)>1 and np.isfinite(values).all() else 'N/A'
        print(f'{metric.upper()}: {values.mean():.4f} +/- {std}')
    if args.patient_csv:
        output = Path(args.patient_csv)
        if output.exists():
            raise FileExistsError(f'Existing metric file is protected: {output}')
        output.parent.mkdir(parents=True, exist_ok=True)
        write_csv(output, rows)
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Patient-balanced 2D whole-slice breast synthesis metrics.")
    parser.add_argument('--data_root', default=str(DATA_ROOT))
    parser.add_argument('--predictions', required=True, help='The nifti subfolder written by infer.py')
    parser.add_argument('--split', choices=['val','test'], default='test')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--patient_csv', help='Optional per-patient output; otherwise print only')
    evaluate(parser.parse_args())
