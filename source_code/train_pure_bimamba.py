"""
Main training script - causal forward-only Mamba.

Usage:
  python train_pure_bimamba.py --model causal_mamba
  python train_pure_bimamba.py --model split_causal_mamba
"""

import argparse
import hashlib
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.navigation_models import (
    CausalMambaNavigator, SplitCausalMambaNavigator
)
from datasets.nclt_fusion_dataset import NCLTFusionDataset
from training.trainer import CausalMambaTrainer, make_three_phase_schedule
from evaluation.evaluator import evaluate_table2

MODEL_MAP = {
    'causal_mamba':       (CausalMambaNavigator,      'KalmanNet'),
    'split_causal_mamba': (SplitCausalMambaNavigator, 'Split_KalmanNet'),
}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_worker_init_fn(seed):
    def _seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        random.seed(worker_seed)
        np.random.seed(worker_seed)

    return _seed_worker


def checkpoint_metadata(path):
    stat = os.stat(path)
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return {
        'best_checkpoint': path,
        'checkpoint_size_bytes': stat.st_size,
        'checkpoint_mtime': stat.st_mtime,
        'checkpoint_sha256': h.hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=list(MODEL_MAP.keys()),
                        default='causal_mamba')
    parser.add_argument('--data_path',   type=str,   default='data/nclt_1hz.pt')
    parser.add_argument('--d_model',     type=int,   default=96)
    parser.add_argument('--d_state',     type=int,   default=16)
    parser.add_argument('--n_layers',    type=int,   default=2)
    parser.add_argument('--batch_size',  type=int,   default=22)
    parser.add_argument('--seed',        type=int,   default=42)
    parser.add_argument('--num_workers', type=int,   default=4)
    parser.add_argument('--out_json',    type=str,   default=None)
    parser.add_argument('--resume',      type=str,   default=None)
    parser.add_argument('--output_dir',  type=str,   default='results')
    # Single-phase overrides (for grid sweep / ablation)
    parser.add_argument('--D',           type=int,   default=None)
    parser.add_argument('--k',           type=int,   default=None)
    parser.add_argument('--w',           type=int,   default=None)
    parser.add_argument('--gps_dropout', type=float, default=None)
    parser.add_argument('--lr',          type=float, default=None)
    parser.add_argument('--epochs',      type=int,   default=None)
    parser.add_argument('--mamba_context', type=int, default=32)
    parser.add_argument('--gps_anchor_alpha', type=float, default=0.03)
    parser.add_argument('--eval_gps_anchor_alpha', type=float, default=0.03)
    parser.add_argument('--gps_fusion_mode',
                        choices=['fixed', 'learned'], default='fixed')
    parser.add_argument('--gps_filter',
                        choices=['none', 'moving_avg', 'median', 'kalman'],
                        default='none')
    parser.add_argument('--gps_filter_window', type=int, default=5)
    parser.add_argument('--gps_confidence_prior', type=float, default=0.1)
    parser.add_argument('--gps_confidence_reg', type=float, default=0.0)
    parser.add_argument('--train_samples_per_traj', type=int, default=16)
    args = parser.parse_args()

    seed_everything(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device} | Model: {args.model}")

    if args.D is not None:
        train_chunk_D = args.D
    else:
        train_chunk_D = max(phase['D'] for phase in make_three_phase_schedule())

    train_ds = NCLTFusionDataset(
        args.data_path,
        split='train',
        D=train_chunk_D,
        samples_per_traj=args.train_samples_per_traj,
        gps_filter=args.gps_filter,
        gps_filter_window=args.gps_filter_window,
    )
    val_ds   = NCLTFusionDataset(
        args.data_path,
        split='val',
        gps_filter=args.gps_filter,
        gps_filter_window=args.gps_filter_window,
    )
    test_ds  = NCLTFusionDataset(
        args.data_path,
        split='test',
        gps_filter=args.gps_filter,
        gps_filter_window=args.gps_filter_window,
    )

    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed)
    worker_init_fn = make_worker_init_fn(args.seed)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                               shuffle=True, num_workers=args.num_workers,
                               pin_memory=True, generator=loader_generator,
                               worker_init_fn=worker_init_fn)
    val_loader   = DataLoader(val_ds, batch_size=1,
                               shuffle=False, num_workers=2,
                               worker_init_fn=worker_init_fn)
    test_loader  = DataLoader(test_ds, batch_size=1,
                               shuffle=False, num_workers=2,
                               worker_init_fn=worker_init_fn)
    print(
        f"Train samples/epoch: {len(train_ds)} "
        f"({len(train_ds.trajs)} trajectories x {args.train_samples_per_traj} crops)"
    )

    model_cls, paper_ref = MODEL_MAP[args.model]
    model = model_cls(d_model=args.d_model, d_state=args.d_state,
                       n_layers=args.n_layers)
    n_params = sum(p.numel() for p in model.parameters()
                   if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    if args.resume and os.path.exists(args.resume):
        missing, unexpected = model.load_state_dict(
            torch.load(args.resume, map_location=device, weights_only=True),
            strict=False,
        )
        print(f"Resumed from {args.resume}")
        if missing or unexpected:
            print(f"Resume state mismatch: missing={missing}, unexpected={unexpected}")

    if args.D is not None:
        phases = [dict(
            epochs      = range(args.epochs or 50),
            D           = args.D,
            k           = args.k or 2,
            w           = args.w or 4,
            gps_dropout = args.gps_dropout if args.gps_dropout is not None
                          else 0.35,
            lr          = args.lr or 1e-3,
            label       = (f'single_D{args.D}_k{args.k or 2}'
                           f'_w{args.w or 4}'),
        )]
    else:
        phases = make_three_phase_schedule()

    save_prefix = os.path.join(
        args.output_dir, f'causal_{args.model}_d{args.d_model}'
    )

    trainer = CausalMambaTrainer(
        model, train_loader, val_loader, device,
        config={
            'loss_scales': (1, 5, 20, 50),
            'warmup_epochs': 15,
            'mamba_context': args.mamba_context,
            'gps_anchor_alpha': args.gps_anchor_alpha,
            'eval_gps_anchor_alpha': args.eval_gps_anchor_alpha,
            'gps_fusion_mode': args.gps_fusion_mode,
            'gps_confidence_prior': args.gps_confidence_prior,
            'gps_confidence_reg': args.gps_confidence_reg,
        },
        save_prefix=save_prefix,
    )
    best_ckpt = trainer.train(phases)

    print("\nLoading best checkpoint for test evaluation...")
    model.load_state_dict(
        torch.load(best_ckpt, map_location=device, weights_only=True),
        strict=False,
    )

    out_json = args.out_json or os.path.join(
        args.output_dir, f'causal_{args.model}_table2.json'
    )

    from training.trainer import CausalMambaTrainer as _T
    evaluate_table2(
        model, test_loader, device, _T,
        model_name=args.model.replace('_', '-').title(),
        paper_ref=paper_ref,
        out_json=out_json,
        mamba_context=args.mamba_context,
        eval_gps_anchor_alpha=args.eval_gps_anchor_alpha,
        gps_fusion_mode=args.gps_fusion_mode,
        metadata={
            **checkpoint_metadata(best_ckpt),
            'config': {
                'model': args.model,
                'data_path': args.data_path,
                'd_model': args.d_model,
                'd_state': args.d_state,
                'n_layers': args.n_layers,
                'batch_size': args.batch_size,
                'seed': args.seed,
                'num_workers': args.num_workers,
                'D': args.D,
                'k': args.k,
                'w': args.w,
                'gps_dropout': args.gps_dropout,
                'lr': args.lr,
                'epochs': args.epochs,
                'mamba_context': args.mamba_context,
                'gps_anchor_alpha': args.gps_anchor_alpha,
                'eval_gps_anchor_alpha': args.eval_gps_anchor_alpha,
                'gps_fusion_mode': args.gps_fusion_mode,
                'gps_filter': args.gps_filter,
                'gps_filter_window': args.gps_filter_window,
                'gps_confidence_prior': args.gps_confidence_prior,
                'gps_confidence_reg': args.gps_confidence_reg,
                'train_samples_per_traj': args.train_samples_per_traj,
            },
        },
    )


if __name__ == '__main__':
    main()
