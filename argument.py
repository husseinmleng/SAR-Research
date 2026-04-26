import argparse

def parser():
    parser = argparse.ArgumentParser(description='DLCV_final')
    parser.add_argument('--todo', choices=['train', 'valid'], default='train',
        help='what behavior want to do: train | valid | test')
    parser.add_argument('--dataset', default='MSTAR', help='the dataset to train')
    parser.add_argument('--model_type', default='gnn', help='model to use')
    parser.add_argument('--use_gpu', default='0', help='which use want to use')
    parser.add_argument('--seed', default=1, type=int, help='the seed for noise')
    parser.add_argument('--batch_size', default=4, type=int, help='size of data per training iteration (reduced to 4 for 8GB VRAM)')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--max_iteration', default=100000, type=int, help='max iteration to train')
    parser.add_argument('--log_interval', default=100, type=int)
    parser.add_argument('--eval_interval', default=500, type=int) # comment
    parser.add_argument('--early_stop', default=10, type=int,
        help='the number of epochs to stop training if the loss is not decrease')
    parser.add_argument('--early_stop_pretrain', default=10, type=int, help='early stop for pretrain')
    parser.add_argument('--test_dir', default='')
    parser.add_argument('--data_root', default='data', help='root for train data')
    parser.add_argument('--log_root', default='log', help='the root to save log')
    parser.add_argument('--model_root', default='model', help='the root to save model')
    parser.add_argument('--affix', default='', help='affix for the name of save folder')
    parser.add_argument('--save', action='store_true', help='whether to save model and logs')
    parser.add_argument('--load', action='store_true', help='whether to load model')
    parser.add_argument('--load_dir', default='model/3way_20shot_gnn_', help='the model to load')
    parser.add_argument('--output_dir', default='output', help='the folder to save output')
    parser.add_argument('--output_name', default='output.txt', help='filename of output')
    parser.add_argument('--nway', default=3, type=int)
    parser.add_argument('--shots', default=20, type=int)
    parser.add_argument('--freeze_cnn', action='store_true', help='whether to freeze cnn-embedding layer')

    # OSR / data arguments
    parser.add_argument('--unseen_class', default='T72', type=str,
        help='directory name of the unseen class (default: T72)')
    parser.add_argument('--unseen_ratio', default=1.0, type=float,
        help='ratio of unseen to seen queries during training (default: 1.0 for 1:1)')
    parser.add_argument('--warmup_iters', default=2000, type=int,
        help='iterations to train with seen-only queries before introducing unseen (0 = disabled)')

    # GAN augmentation
    parser.add_argument('--gan_augment', action='store_true',
        help='augment support set with GAN-generated images')
    parser.add_argument('--gan_output_dir', default='gan_output', type=str,
        help='directory containing GAN-generated images per class')

    # Physics-informed regularization
    parser.add_argument('--physics_lambda', default=0.0, type=float,
        help='weight for intra-class embedding variance penalty (0 = disabled)')

    # Augmentation transforms
    parser.add_argument('--augment_rotation', action='store_true',
        help='apply random 0-360 degree rotation to support images')
    parser.add_argument('--augment_speckle', action='store_true',
        help='apply multiplicative speckle noise to support images')
    parser.add_argument('--speckle_sigma', default=0.1, type=float,
        help='sigma for speckle noise (default: 0.1)')

    # Evaluation
    parser.add_argument('--eval_only', action='store_true',
        help='run evaluation once on loaded checkpoint then exit')
    parser.add_argument('--eval_output', default='results', type=str,
        help='directory to save evaluation JSON reports')
    parser.add_argument('--baseline_kshot', action='store_true',
        help='use K-shot subsampling for CNN baseline training')
    parser.add_argument('--amp', action='store_true', default=True,
        help='enable automatic mixed precision (FP16) training for faster GPU utilization')
    parser.add_argument('--eval_batch_size', default=2, type=int,
        help='batch size for evaluation (separate from training for memory efficiency)')
    parser.add_argument('--gradient_checkpointing', action='store_true', default=True,
        help='enable gradient checkpointing to reduce activation memory during backprop')
    parser.add_argument('--eval_sample_8gb', default=2000, type=int,
        help='number of evaluation samples for 8GB VRAM (default 5000 for larger GPUs)')

    args = parser.parse_args()

    # Validation
    if args.unseen_ratio <= 0:
        raise ValueError('--unseen_ratio must be > 0, got %f' % args.unseen_ratio)
    if args.physics_lambda < 0:
        raise ValueError('--physics_lambda must be >= 0, got %f' % args.physics_lambda)

    return args

def print_args(args, logger=None):
    for k, v in vars(args).items():
        if logger is not None:
            logger.info('{:<16} : {}'.format(k, v))
        else:
            print('{:<16} : {}'.format(k, v))