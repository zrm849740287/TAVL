import argparse
import base64
import csv
import hashlib
import json
from pathlib import Path
import re

import nibabel as nib
import numpy as np
from tqdm import tqdm

SCHEMA = 'breast-shared-npy-zxy-float32-v1'


def write_csv(path, rows):
    with Path(path).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prepare(manifest, output):
    manifest, output = Path(manifest).resolve(), Path(output).resolve()
    with manifest.open(encoding='utf-8-sig', newline='') as stream:
        records = list(csv.DictReader(stream))
    if not records or not {'patient_id', 'split', 'pre', 'post1'} <= set(records[0]):
        raise ValueError('CSV requires patient_id,split,pre,post1')
    ids, phase_paths = set(), set()
    for row in records:
        pid = row['patient_id']
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', pid) or pid in ids:
            raise ValueError(f'Unsafe or duplicate patient ID: {pid}')
        ids.add(pid)
        if row['split'] not in ('train', 'val', 'test'):
            raise ValueError(f'Invalid split: {pid}')
        for key in ('pre', 'post1'):
            path = Path(row[key])
            path = (manifest.parent/path).resolve() if not path.is_absolute() else path.resolve()
            if not path.is_file() or not path.name.lower().endswith(('.nii', '.nii.gz')):
                raise FileNotFoundError(f'Expected a 3D NIfTI file: {path}')
            if path in phase_paths:
                raise ValueError(f'Reused image across patients/phases: {path}')
            phase_paths.add(path)
            row[key] = path
    if {r['split'] for r in records} != {'train', 'val', 'test'}:
        raise ValueError('Provide explicit patient-level train, val and test splits')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Output must be empty; existing datasets are never overwritten')
    output.mkdir(parents=True, exist_ok=True)
    (output/'metadata').mkdir()
    (output/'tasks').mkdir()
    descriptor = output/'dataset.json'
    descriptor.write_text(json.dumps(dict(schema=SCHEMA, status='incomplete')), encoding='utf-8')
    splits, volumes, slices = [], [], []
    for row in tqdm(records, desc='Prepare paired volumes'):
        pid, split = row['patient_id'], row['split']
        images = [nib.as_closest_canonical(nib.load(str(row[key]))) for key in ('pre', 'post1')]
        pre_img, post_img = images
        if any(len(im.shape) != 3 for im in images):
            raise ValueError(f'Expected separately selected 3D pre and post1: {pid}')
        if pre_img.shape != post_img.shape or not np.allclose(pre_img.affine, post_img.affine, rtol=0, atol=1e-4):
            raise ValueError(f'Unaligned pair: {pid}; register/resample before preparing data')
        if nib.aff2axcodes(pre_img.affine) != ('R','A','S') or not np.allclose(
                nib.affines.voxel_sizes(pre_img.affine), [1,1,1], rtol=0, atol=1e-4):
            raise ValueError(f'Expected a 1 mm RAS grid: {pid}; this script does not resample')
        pre, post = [im.get_fdata(dtype=np.float32) for im in images]
        if not np.isfinite(pre).all() or not np.isfinite(post).all():
            raise ValueError(f'Nonfinite image: {pid}')
        support = pre[pre != 0]
        if not support.size:
            raise ValueError(f'Empty source: {pid}')
        center = float(np.median(support))
        scale = 1.4826 * float(np.median(np.abs(support-center)))
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(f'Zero/invalid source MAD: {pid}; inspect the raw source')
        valid_z = np.flatnonzero(np.any(pre != 0, axis=(0,1)))
        split_row = dict(patient_id=pid, default_split=split,
                         official_split='test' if split == 'test' else 'train', cv_fold='')
        splits.append(split_row)
        base = dict(**split_row, pre_robust_center=center, pre_robust_scale=scale,
                    input_phase_index=0, target_phase_index=1)
        volumes.append(dict(**base, phase_semantics='pre|post1', phase_count=2))
        slices.extend(dict(**base, slice_axis_xyz=2, slice_index_type='standard_axial_ras',
                           slice_index=int(z)) for z in valid_z)
        folder = output/split/pid
        folder.mkdir(parents=True)
        for key, array in (('pre', pre), ('post1', post)):
            np.save(folder/(key+'.npy'), np.ascontiguousarray(np.moveaxis(array, 2, 0)), allow_pickle=False)
        qform, qcode = pre_img.get_qform(coded=True)
        sform, scode = pre_img.get_sform(coded=True)
        meta = dict(schema=SCHEMA, status='complete', **split_row, axes='ZXY', dtype='float32',
                    intensity_transform='none', shape_xyz=list(pre.shape),
                    shape_zxy=[pre.shape[2], pre.shape[0], pre.shape[1]],
                    affine=pre_img.affine.tolist(), spacing_xyz_mm=list(map(float, pre_img.header.get_zooms()[:3])),
                    pre_robust_center=center, pre_robust_scale=scale,
                    source_signature={'path':str(row['pre'])},
                    nifti_header_base64=base64.b64encode(pre_img.header.binaryblock).decode('ascii'),
                    qform=None if qform is None else qform.tolist(), qform_code=int(qcode),
                    sform=None if sform is None else sform.tolist(), sform_code=int(scode))
        (folder/'metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    paths = [output/'metadata/splits.csv', output/'tasks/generation_3d.csv', output/'tasks/generation_2d.csv']
    for path, rows in zip(paths, (splits, volumes, slices)):
        write_csv(path, rows)
    fingerprint = hashlib.sha256(b''.join(p.read_bytes() for p in paths)).hexdigest()
    descriptor.write_text(json.dumps(dict(schema=SCHEMA, status='complete',
        source_index_fingerprint=fingerprint, normalization='source nonzero median/MAD'), indent=2), encoding='utf-8')
    print(f'Prepared {len(records)} patients at {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Package aligned raw-intensity 1 mm NIfTI pairs for TAVL-Syn.")
    parser.add_argument('--manifest', required=True, help='Patient-level CSV of aligned raw NIfTI pairs')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    prepare(args.manifest, args.output)
