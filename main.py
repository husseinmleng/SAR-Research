import os
import sys
import random
import numpy as np
import torch

from argument import parser, print_args
from data import self_DataLoader
from trainer import Trainer, TrainerBaseline
from utils import create_logger, mkdir


def main(args):
    # -----------------------------------------------------------------------
    # Folder setup
    # -----------------------------------------------------------------------
    folder_name = '%dway_%dshot_%s_%s' % (
        args.nway, args.shots, args.model_type, args.affix
    )
    model_folder = os.path.join(args.model_root, folder_name)
    log_folder   = os.path.join(args.log_root,   folder_name)

    mkdir(args.model_root)
    mkdir(args.log_root)
    mkdir(model_folder)
    mkdir(log_folder)
    mkdir(args.eval_output)

    setattr(args, 'model_folder', model_folder)
    setattr(args, 'log_folder',   log_folder)
    logger = create_logger(log_folder, args.todo)
    print_args(args, logger)

    # -----------------------------------------------------------------------
    # Data loader
    # -----------------------------------------------------------------------
    tr_dataloader = self_DataLoader(
        root=args.data_root,
        dataset=args.dataset,
        seed=args.seed,
        nway=args.nway,
        unseen_class=args.unseen_class,
        unseen_ratio=args.unseen_ratio,
        gan_augment=args.gan_augment,
        gan_output_dir=args.gan_output_dir,
        augment_rotation=args.augment_rotation,
        augment_speckle=args.augment_speckle,
        speckle_sigma=args.speckle_sigma,
    )

    # -----------------------------------------------------------------------
    # --eval_only mode
    # -----------------------------------------------------------------------
    if args.eval_only:
        if not args.load:
            logger.error('--eval_only requires --load to be set')
            sys.exit(1)

        trainer_dict = {
            'args': args,
            'logger': logger,
            'tr_dataloader': tr_dataloader,
        }
        trainer = Trainer(trainer_dict)
        model_path = os.path.join(args.load_dir, 'model.pth')
        trainer.load_model(model_path)
        trainer.model_cuda()

        _, overall_acc, seen_acc, unseen_acc, y_true, y_pred = \
            trainer.eval(tr_dataloader)

        # Build class names for the seen slots + "unseen"
        seen_keys = sorted(tr_dataloader.full_test_dict.keys())
        class_names = [str(k) for k in seen_keys] + ['unseen']

        from evaluate import compute_metrics, save_report
        metrics = compute_metrics(y_true, y_pred, class_names,
                                  unseen_label=args.nway)
        config = folder_name
        path = save_report(metrics, config, args.eval_output, 'eval_only')
        logger.info('Evaluation report saved to: %s' % path)
        return

    # -----------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------
    trainer_dict = {
        'args': args,
        'logger': logger,
        'tr_dataloader': tr_dataloader,
    }

    if args.model_type == 'gnn':
        trainer = Trainer(trainer_dict)
    elif args.model_type == 'cnn':
        trainer = TrainerBaseline(trainer_dict)
    else:
        raise ValueError('Unknown model_type: %s' % args.model_type)

    if args.load:
        model_path = os.path.join(args.load_dir, 'model.pth')
        if args.model_type == 'gnn':
            trainer.load_model(model_path)

    trainer.train()


if __name__ == '__main__':
    args = parser()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.use_gpu
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True   # Ada Lovelace TF32
        torch.backends.cudnn.allow_tf32 = True          # Ada Lovelace TF32
        device = torch.device('cuda')
        props = torch.cuda.get_device_properties(device)
        print(f'GPU: {props.name}, VRAM: {props.total_memory / 1e9:.1f}GB')
        if props.total_memory < 12e9:
            print('Optimizing for low-VRAM GPU (< 12GB)')
            print(f'  - batch_size={args.batch_size}, eval_batch_size={args.eval_batch_size}')
            print(f'  - eval_sample={args.eval_sample_8gb}, gradient_checkpointing={args.gradient_checkpointing}')

    main(args)
