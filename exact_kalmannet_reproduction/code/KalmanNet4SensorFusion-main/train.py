import argparse
import os
import os.path as osp
from typing import Dict

import numpy as np
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from mmengine.config import Config, DictAction
from torch import Tensor

from Net.dataset.track_dataset import TrackDataModule
from Net.utils import (MODELS, generate_save_dir,
                           training_info)


TBPTT_SETTINGS = {
    "2_4_50": (2, 4, 50),
    "2_8_50": (2, 8, 50),
    "50_50_50": (50, 50, 50),
    "2_4_100": (2, 4, 100),
    "2_8_100": (2, 8, 100),
    "100_100_100": (100, 100, 100),
    "2_4_200": (2, 4, 200),
    "2_8_200": (2, 8, 200),
    "200_200_200": (200, 200, 200),
}


def make_output_dir(path: str) -> Dict:
    save_dirs: Dict[str, str] = {
        "experiments_dir": path,
        "new_name": osp.basename(osp.normpath(path)),
        "weight_dir": osp.join(path, "checkpoints"),
        "config_dir": osp.join(path, "configs"),
        "eval_dir": osp.join(path, "eval_results"),
        "log_images_dir": osp.join(path, "log_images"),
        "log_metrics_dir": osp.join(path, "log_metrics"),
    }
    for key, directory in save_dirs.items():
        if key.endswith("dir") or key.endswith("_dir"):
            os.makedirs(directory, exist_ok=True)
    return save_dirs


def apply_tbptt_setting(cfg: Config, tbptt: str | None) -> None:
    if not tbptt:
        return
    if tbptt not in TBPTT_SETTINGS:
        raise ValueError(f"Unknown --tbptt {tbptt!r}; expected one of {sorted(TBPTT_SETTINGS)}")
    k, w, d = TBPTT_SETTINGS[tbptt]
    cfg.trainer.detach_step = k
    cfg.trainer.slide_win_size = w
    cfg.trainer.gradient_clip_val = 5.0
    cfg.data.train_dataset.seq_len = d
    if hasattr(cfg, "train_seq_len"):
        cfg.train_seq_len = d
    cfg.logger.name = f"{cfg.logger.name}_tbptt_{tbptt}"


def main(args: argparse.ArgumentParser, cfg: Config) -> None:
    training_info()
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    save_dir: Dict = make_output_dir(args.output_dir) if args.output_dir else generate_save_dir(root='./runs',
                                                                                                project=cfg.logger.project,
                                                                                                name=cfg.logger.name)
    cfg.logger.name = save_dir['new_name']
    cfg.dump(osp.join(save_dir['config_dir'], 'config.py'))

    data_module = TrackDataModule(
        cfg, use_transform=cfg.data.transforms.use_transform)
    data_module.setup()
    # model.

    model = MODELS.build(
        dict(type=cfg.trainer.type, cfg=cfg, save_dir=save_dir))
    # trainer
    lr_monitor = LearningRateMonitor(logging_interval='step')
    model_monitor = ModelCheckpoint(
        dirpath=save_dir['weight_dir'],
        filename='{epoch}-{val_loss:.2f}-{val_MSE_dB:.2f}',
        mode='min',
        save_top_k=10,
        monitor='val_MSE_dB')
    callbacks = [lr_monitor, model_monitor]
    wandb_logger = WandbLogger(project=cfg.logger.project,
                               name=cfg.logger.name,
                               offline=cfg.logger.offline)
    trainer = Trainer(
        accelerator='gpu',
        max_epochs=cfg.trainer.epochs,
        logger=wandb_logger,
        log_every_n_steps=1,
        detect_anomaly=cfg.trainer.detect_anomaly,
        callbacks=callbacks,
        devices=cfg.trainer.device,
        num_sanity_val_steps=0,
        check_val_every_n_epoch = cfg.trainer.check_val_every_n_epoch if cfg.trainer.check_val_every_n_epoch is not None else 1

    )
    trainer.fit(model, datamodule=data_module, ckpt_path=args.ckpt_path)


def parse_args():
    parser = argparse.ArgumentParser(
        prog='KalmanNet',
        description='Dataset, training and network parameters')
    parser.add_argument('--config',
                        '--cfg',
                        type=str,
                        metavar='config',
                        help='model and seq ')

    parser.add_argument(
        '--cfg_options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument('--ckpt-path',
                        type=str,
                        default=None,
                        help='Optional Lightning checkpoint path to resume from.')
    parser.add_argument('--tbptt',
                        type=str,
                        default=None,
                        choices=sorted(TBPTT_SETTINGS),
                        help='TBPTT setting encoded as k_w_D, e.g. 2_4_50.')
    parser.add_argument('--output-dir',
                        type=str,
                        default=None,
                        help='Explicit experiment directory. Overrides runs/project/name_vN.')
    parser.add_argument('--epochs',
                        type=int,
                        default=None,
                        help='Override cfg.trainer.epochs for smoke tests.')
    args = parser.parse_known_args()[0]
    return args


if __name__ == '__main__':
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    apply_tbptt_setting(cfg, args.tbptt)
    if args.epochs is not None:
        cfg.trainer.epochs = args.epochs
    print(cfg)
    main(args, cfg)
