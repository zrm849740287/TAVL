import argparse
import math
from pathlib import Path
import random

import numpy as np
import torch
from torch.amp import GradScaler
from torch.utils.data import DataLoader

from datasets.breast_nifti import DATA_ROOT, PROTOCOL, BreastIndex, PhaseCache, PatientBlockSampler
from datasets.breast_coverage import BreastCoverageDataset, SAMPLING_PROTOCOL
from common import (MODEL_KIND, build_model, device_for, load_breast_checkpoint,
                           predict_volume, save_json, save_preview, write_csv)
from models.rectified_flow import RectifiedFlow
import engine


class PairDataset(BreastCoverageDataset):
    def __getitem__(self,item):
        batch = super().__getitem__(item)
        return batch['image'],batch['label']


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def save_checkpoint(state,path):
    path = Path(path)
    temp = path.with_suffix(path.suffix+'.tmp')
    torch.save(state,temp)
    temp.replace(path)


def validate_volume(model,index,args,device,output,epoch):
    model.eval()
    cache = PhaseCache()
    scores = []
    preview_dir = output/'validation_previews'/f'epoch_{epoch}'
    preview_dir.mkdir(parents=True,exist_ok=True)
    for case in index.select('val',args.val_limit):
        pre,post = cache.get(case,0),cache.get(case,1)
        pred = predict_volume(model,pre,case,device,args.img_size,args.overlap,
                              args.infer_batch_size,args.sample_step,args.inference_seed)
        if pred.shape != post.shape or not np.isfinite(pred).all():
            raise ValueError('Invalid validation volume: '+case['patient_id'])
        scores.append(float(np.abs(pred-post).mean()))
        z = case['slices'][len(case['slices'])//2]
        save_preview(pre[:,:,z],pred[:,:,z],case,preview_dir/(case['patient_id']+'.png'),post[:,:,z])
    return float(np.mean(scores))


def main(args):
    engine.validate_arguments(args)
    if (min(args.max_epoch,args.batch_size,args.sample_step,args.patches_per_patient) < 1 or
            args.num_workers < 0 or args.val_every < 0 or args.save_interval < 0 or
            not 0 <= args.overlap < 1 or args.infer_batch_size < 1 or
            not math.isfinite(args.k) or args.lr <= 0 or args.grad_clip <= 0):
        raise ValueError('Invalid breast training options.')
    output = Path(args.output_dir).resolve()
    latest = output/'checkpoints/model_latest.pth'
    resume = Path(args.resume).resolve() if args.resume else (latest if latest.exists() else None)
    if args.fresh_start:
        if args.resume or args.continue_train:
            raise ValueError('fresh_start cannot be combined with resume.')
        if output.exists() and any(output.iterdir()):
            raise FileExistsError('Use a new --output_dir for a fresh run; existing files are preserved.')
        resume = None
    if args.continue_train and resume is None:
        raise FileNotFoundError('No breast checkpoint to resume.')
    if not resume and output.exists() and any(output.iterdir()) and not args.dry_run:
        raise FileExistsError('Nonempty output directory without a latest checkpoint; use a new --output_dir.')
    engine.set_seed(args.seed)
    device = device_for(args.device)
    index = BreastIndex(args.data_root)
    print('Breast pre -> post1:',index.summary(),flush=True)
    dataset = PairDataset(index,args.img_size,args.cache_dir,args.patches_per_patient,args.train_limit)
    model = build_model(args.img_size,args.model_chn,args.mr_base_ch,args.tavs_num_dirs,args.k).to(device)
    if args.dry_run:
        dataset.set_epoch(1,args.seed)
        pre,post = dataset[0]
        model.eval()
        with torch.no_grad():
            pre,post = pre[None].to(device),post[None].to(device)
            t = torch.full((1,),.5,device=device)
            flow = RectifiedFlow()
            x,noise = flow.create_flow(post,t)
            velocity = model(t,x,pre)
            loss = flow.calc_loss(velocity,post,noise,t)
        if velocity.shape != post.shape or not torch.isfinite(loss):
            raise ValueError('Invalid dry-run prediction.')
        print(dict(shape=list(post.shape),loss=float(loss),pre_range=[float(pre.min()),float(pre.max())],
                   target_range=[float(post.min()),float(post.max())],
                   normalization='pre-median-MAD; no clipping/minmax/HU',optimization=False),flush=True)
        return
    state = load_breast_checkpoint(resume,index) if resume else None
    if state:
        if state.get('sampling_protocol') != SAMPLING_PROTOCOL:
            raise ValueError('Checkpoint sampling protocol mismatch.')
        for field in ('img_size','model_chn','mr_base_ch','tavs_num_dirs','batch_size',
                      'patches_per_patient','seed','train_limit','num_workers','lr','grad_clip',
                      'fp16','k','sc_loss_weight','sc_warmup_epochs','sc_ramp_epochs',
                      'sc_update_interval','sc_min_step_ratio','sc_max_step_ratio',
                      'sc_max_start_time','sc_correction_fraction','val_every','val_limit',
                      'sample_step','inference_seed','overlap','infer_batch_size'):
            if state['args'][field] != getattr(args,field):
                raise ValueError('Resume configuration changed: '+field)
        model.load_state_dict(state['model'])
    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr)
    scaler = GradScaler('cuda',enabled=args.fp16 and device.type=='cuda')
    sc_generator = torch.Generator(device='cpu').manual_seed(args.seed+engine.SC_RANDOM_SEED_OFFSET)
    loader_generator = torch.Generator().manual_seed(args.seed)
    start,best,history = 0,float('inf'),[]
    if state:
        optimizer.load_state_dict(state['optimizer'])
        scaler.load_state_dict(state['scaler'])
        engine.restore_rng_state(state['rng_state'],sc_generator)
        loader_generator.set_state(state['loader_rng'])
        start,best,history = state['epoch'],state['best_val_mae'],state['history']
        del state
        print(f'Resumed completed epoch {start}; next epoch {start+1}.',flush=True)
    if start >= args.max_epoch:
        print('Requested epochs already completed.',flush=True)
        return
    output.mkdir(parents=True,exist_ok=True)
    checkpoints = output/'checkpoints'
    checkpoints.mkdir(exist_ok=True)
    manifest = dict(protocol=PROTOCOL,model_kind=MODEL_KIND,sampling_protocol=SAMPLING_PROTOCOL,
                    index_fingerprint=index.fingerprint,args=vars(args),
                    normalization='(raw-pre_median)/(1.4826*pre_MAD); no clipping',
                    training_patients=[c['patient_id'] for c in dataset.cases],
                    objective='RectifiedFlow MSE + damped self-correction',
                    selection='final_epoch',ode_method='euler',limited_run=args.train_limit is not None,
                    sampling='source-indexed slice cycles and 50% overlap nonempty crops')
    save_json(output/'run_config.json',manifest)
    loader = DataLoader(dataset,batch_size=args.batch_size,
                        sampler=PatientBlockSampler(dataset,loader_generator),
                        num_workers=args.num_workers,worker_init_fn=seed_worker,
                        generator=loader_generator,pin_memory=device.type=='cuda')
    print(f'{len(dataset)} patches/epoch; TAVS + DSC; validation every {args.val_every} (0=off).',flush=True)
    for epoch in range(start+1,args.max_epoch+1):
        dataset.set_epoch(epoch,args.seed)
        metrics = engine.train_one_epoch(model,loader,optimizer,scaler,RectifiedFlow(),
                                          device,args,epoch,sc_generator)
        if not all(math.isfinite(float(x)) for x in metrics.values()):
            raise FloatingPointError('Nonfinite epoch metrics.')
        score = None
        improved = False
        if args.val_every and (epoch % args.val_every == 0 or epoch == args.max_epoch):

            rng = engine.capture_rng_state(sc_generator)
            score = validate_volume(model,index,args,device,output,epoch)
            engine.restore_rng_state(rng,sc_generator)
            if not math.isfinite(score):
                raise FloatingPointError('Nonfinite validation metric.')
            improved = score < best
            best = min(best,score)
        history.append(dict(epoch=epoch,**metrics,val_patient_raw_mae=score))
        state = dict(protocol=PROTOCOL,model_kind=MODEL_KIND,sampling_protocol=SAMPLING_PROTOCOL,
                     index_fingerprint=index.fingerprint,args=vars(args),epoch=epoch,
                     model=model.state_dict(),optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),
                     rng_state=engine.capture_rng_state(sc_generator),loader_rng=loader_generator.get_state(),
                     history=history,best_val_mae=best)
        save_checkpoint(state,latest)
        if improved:
            save_checkpoint(state,checkpoints/'model_best.pth')
        if args.save_interval and epoch % args.save_interval == 0:
            save_checkpoint(state,checkpoints/f'model_epoch_{epoch}.pth')
        if epoch == args.max_epoch:
            save_checkpoint(state,checkpoints/'model_final.pth')
        write_csv(output/'train_log.csv',history)
        print(f'Epoch {epoch}: objective={metrics["objective"]:.6g}, validation={score}',flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train TAVL-Syn for paired Pre-to-Post1 breast MRI synthesis.")
    parser.add_argument('--data_root', default=str(DATA_ROOT))
    parser.add_argument('--output_dir', default='runs/tavl_syn')
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--model_chn', type=int, default=128)
    parser.add_argument('--mr_base_ch', type=int, default=32, help='Source MRI encoder base channels')
    parser.add_argument('--tavs_num_dirs', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--grad_clip', type=float, default=1.)
    parser.add_argument('--seed', type=int, default=20260727)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--sample_step', type=int, default=20, help='Euler steps for optional validation previews')
    parser.add_argument('--k', type=float, default=0., help='Euler time-grid exponent; 0 gives uniform steps')
    parser.add_argument('--save_interval', type=int, default=0)
    parser.add_argument('--continue_train', action='store_true')
    for name, dtype, value in (
            ('sc_loss_weight', float, .1), ('sc_warmup_epochs', int, 20),
            ('sc_ramp_epochs', int, 20), ('sc_update_interval', int, 4),
            ('sc_min_step_ratio', float, .1), ('sc_max_step_ratio', float, .3),
            ('sc_max_start_time', float, .8), ('sc_correction_fraction', float, .25)):
        parser.add_argument('--'+name, '--'+name.replace('_','-'), type=dtype, default=value)
    parser.add_argument('--cache_dir',default='')
    parser.add_argument('--patches_per_patient',type=int,default=16)
    parser.add_argument('--train_limit',type=int)
    parser.add_argument('--val_limit',type=int,default=4)
    parser.add_argument('--val_every',type=int,default=0)
    parser.add_argument('--overlap',type=float,default=.5)
    parser.add_argument('--infer_batch_size',type=int,default=4)
    parser.add_argument('--inference_seed',type=int,default=20260727)
    parser.add_argument('--resume')
    parser.add_argument('--fresh_start',action='store_true')
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--dry_run',action='store_true')
    args = parser.parse_args(argv)
    return args


if __name__ == '__main__':
    main(parse_args())
