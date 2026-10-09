import hashlib
import numpy as np
import torch
from .breast_nifti import BreastPatchDataset, normalize

SAMPLING_PROTOCOL = 'pre-slice-cycles-grid-crops-v1'


def permutation(length, seed, *key):
    digest = hashlib.sha256(repr((seed, key)).encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], 'little')).permutation(length)


def starts(length, size):
    end = length - size
    return sorted(set(list(range(0, end + 1, max(1, size // 2))) + [end]))


class BreastCoverageDataset(BreastPatchDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epoch = 1
        self.seed = 20260727

    def set_epoch(self, epoch, seed=20260727):
        self.epoch, self.seed = epoch, seed

    def selection(self, item):
        patient = item % len(self.cases)
        slot = item // len(self.cases)
        case = self.cases[patient]
        serial = (self.epoch - 1) * self.repeats + slot
        cycle, offset = divmod(serial, len(case['slices']))
        order = permutation(len(case['slices']), self.seed, case['patient_id'], 'slice', cycle)
        return case, case['slices'][int(order[offset])], cycle

    def __getitem__(self, item):
        case, z, visit = self.selection(item)
        pre = np.array(self.cache.get(case, 0)[:, :, z], copy=True)
        post = np.array(self.cache.get(case, 1)[:, :, z], copy=True)
        if pre.shape != post.shape:
            raise ValueError('Paired image shapes differ')
        pads = tuple((max(0, self.size - n) // 2,
                      max(0, self.size - n) - max(0, self.size - n) // 2) for n in pre.shape)
        pre, post = np.pad(pre, pads), np.pad(post, pads)


        points = [(x, y) for x in starts(pre.shape[0], self.size)
                  for y in starts(pre.shape[1], self.size)
                  if np.any(pre[x:x+self.size, y:y+self.size] != 0)]
        if not points:
            raise ValueError(f"Indexed slice is empty: {case['patient_id']} z={z}")
        crop_cycle, crop_offset = divmod(visit, len(points))
        order = permutation(len(points), self.seed, case['patient_id'], z, 'crop', crop_cycle)
        x, y = points[int(order[crop_offset])]
        region = (slice(x, x+self.size), slice(y, y+self.size))
        return {'image': torch.from_numpy(normalize(pre[region], case).copy()[None]),
                'label': torch.from_numpy(normalize(post[region], case).copy()[None]),
                'patient_id': case['patient_id'], 'slice_index': z,
                'crop_start': torch.tensor([x, y]), 'slice_cycle': visit}
