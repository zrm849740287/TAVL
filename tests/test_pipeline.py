import csv
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import nibabel as nib
import numpy as np
import torch

import common
import infer
import train
from datasets.breast_nifti import BreastIndex, PhaseCache, normalize, denormalize
from evaluate import evaluate, slice_metrics
from prepare_data import prepare


class ZeroVelocity(torch.nn.Module):
    def forward(self, t, x, pre, condition=None):
        return torch.zeros_like(x)


class InferenceDefaultsTests(unittest.TestCase):
    def test_inference_defaults_are_five_and_previews_remain_twenty(self):
        import inspect
        self.assertEqual(infer.parse_args([]).ode_steps, 5)
        self.assertEqual(infer.parse_args(['--ode_steps', '10']).ode_steps, 10)
        self.assertEqual(train.parse_args([]).sample_step, 20)
        for function in (common.integrate_flow, common.predict_slice, common.predict_volume):
            self.assertEqual(inspect.signature(function).parameters['ode_steps'].default, 5)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        rng = np.random.default_rng(91)
        rows = []
        self.affine = np.eye(4)
        self.affine[:3,3] = [-9,12,4]
        for split in ('train','val','test'):
            pid = 'SYNTHETIC_'+split.upper()
            pre = rng.uniform(2,20,(32,32,3)).astype(np.float32)
            pre[:,:,0] = 0
            post = pre*1.2
            paths = [self.root/f'{pid}_{p}.nii.gz' for p in ('pre','post1')]
            for path, volume in zip(paths, (pre,post)):
                nib.save(nib.Nifti1Image(volume,self.affine),path)
            rows.append(dict(patient_id=pid,split=split,pre=paths[0].name,post1=paths[1].name))
        self.manifest = self.root/'pairs.csv'
        with self.manifest.open('w',newline='') as f:
            w = csv.DictWriter(f,fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        self.data = self.root/'dataset'
        prepare(self.manifest,self.data)

    def tearDown(self):
        self.temp.cleanup()

    def test_index_axes_normalization_and_no_overwrite(self):
        index = BreastIndex(self.data)
        self.assertEqual(index.summary()['train']['patients'],1)
        case = index.select('train')[0]
        pre = PhaseCache().get(case,0)
        self.assertEqual(pre.shape,(32,32,3))
        np.testing.assert_allclose(denormalize(normalize(pre,case),case),pre,atol=2e-6)
        with self.assertRaises(FileExistsError):
            prepare(self.manifest,self.data)

    def test_slice_metric_reference(self):
        from skimage.metrics import structural_similarity
        a = np.random.default_rng(1).random((32,32))
        b = a+.1
        metrics = slice_metrics(a,b)
        self.assertAlmostEqual(metrics['psnr'],20.,places=10)
        self.assertAlmostEqual(metrics['mae'],.1,places=10)
        expected = structural_similarity(a,b,data_range=1,win_size=11,gaussian_weights=True,
                                         sigma=1.5,use_sample_covariance=False)
        self.assertAlmostEqual(metrics['ssim'],expected,places=12)

    def test_overlap_noise_is_independent_of_patch_batching(self):
        case = dict(patient_id='SYNTHETIC',center=5.,scale=2.)
        pre = np.full((48,56),10.,dtype=np.float32)
        a = common.predict_slice(ZeroVelocity(),pre,case,'cpu',32,.5,1,2,7,0)
        b = common.predict_slice(ZeroVelocity(),pre,case,'cpu',32,.5,3,2,7,0)
        np.testing.assert_array_equal(a,b)

    def test_train_resume_pre_only_inference_and_evaluate(self):
        args = train.parse_args(['--data_root',str(self.data),'--output_dir',str(self.root/'run'),
            '--device','cpu','--img_size','32','--model_chn','32','--mr_base_ch','8',
            '--batch_size','2','--patches_per_patient','2','--max_epoch','1',
            '--sc_warmup_epochs','0','--sc_ramp_epochs','1','--sc_update_interval','1',
            '--sc_max_start_time','0.999999','--sample_step','2'])
        train.main(args)
        args.max_epoch = 2
        train.main(args)
        index = BreastIndex(self.data)
        checkpoint = self.root/'run/checkpoints/model_final.pth'
        state = common.load_breast_checkpoint(checkpoint,index)
        self.assertEqual(state['epoch'],2)
        self.assertGreater(state['history'][0]['sc_active_samples'],0)
        original_get = PhaseCache.get
        def pre_only(cache,case,phase):
            self.assertEqual(phase,0,'Inference must not read target pixels')
            return original_get(cache,case,phase)
        options = SimpleNamespace(data_root=str(self.data),model_path=str(checkpoint),split='test',
            limit_patients=None,device='cpu',ode_steps=None,seed=7,output_dir=str(self.root/'outputs'),
            run_name='fixture',cache_dir='',save_npy=False,overlap=.5,batch_size=2)
        with patch.object(PhaseCache,'get',pre_only):
            infer.infer(options)
        out = self.root/'outputs/fixture'
        self.assertEqual(json.loads((out/'run_config.json').read_text())['ode_steps'], 5)
        pid = index.select('test')[0]['patient_id']
        image = nib.load(out/f'nifti/{pid}_post1_generated.nii.gz')
        stored = np.load(out/f'npy_volumes/{pid}_post1_generated.npy')
        np.testing.assert_array_equal(np.asarray(image.dataobj),np.moveaxis(stored,0,2))
        np.testing.assert_allclose(image.affine,self.affine)
        self.assertFalse(np.any(stored[0]))
        results = evaluate(SimpleNamespace(data_root=str(self.data),split='test',
            predictions=str(out/'nifti'),workers=1,patient_csv=None))
        self.assertEqual(len(results),1)
        self.assertEqual(results[0]['slices'],2)
        self.assertTrue(np.isfinite(results[0]['psnr']))
        self.assertEqual(json.loads((out/'completion.json').read_text())['status'],'complete')
