import argparse

def args_setting():
    parser = argparse.ArgumentParser(description='EXP Training')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')  # 1e-3#
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')  # 256#
    parser.add_argument('--epochs', default=20, type=int, help='number epochs')  # 12#
    parser.add_argument('--num_classes', default=8, type=int, help='number classes')
    parser.add_argument('--weight_decay', default=5e-4, type=float)  # 5e-4#
    parser.add_argument('--seq_len', default=3, type=int)
    parser.add_argument('--num_workers', default=8, type=int)

    parser.add_argument('--temp', type=float, default=0.07)
    parser.add_argument('--alpha', type=float, default=0.7,
                        help='mixup interpolation coefficient (default: 1)')
    parser.add_argument('--remix_tau', type=float, default=0.9,
                        help='remixup interpolation coefficient tau(default: 1)')
    parser.add_argument('--remix_kappa', type=float, default=15.,
                        help='remixup interpolation coefficient kappa(default: 1)')

    parser.add_argument('--dataset', type=str, default="v2")  # ["v1","v2","v3"]
    parser.add_argument('--net', type=str, default="RNN")  # ["RES","INC","HRN",'C','M','T']
    parser.add_argument('--mode', type=str, default="train")  # ["train","trainMixup","trainRemix"]
    parser.add_argument('--file', type=str, default=f"{parser.parse_args().net}_ex_best")

    parser.add_argument('--sample', type=str, default="ori")
    parser.add_argument('--resume', type=bool, default=False)

    parser.add_argument('--loss', type=str, default='CrossEntropyLoss')  # ['CrossEntropyLabelAwareSmooth','SuperContrastive']
    parser.add_argument('--warmup', type=bool, default=False)
    parser.add_argument('--optim', type=str, default='SGD')

    parser.add_argument('--arch', default='resnet101', type=str, help='baseline of the training network.')
    parser.add_argument('--rep', default='pretrain', type=str,
                        help='Choose methods for representation learning.')  # ['SSL','pretrain']
    parser.add_argument('--gpu-index', default=0, type=int, help='Gpu index.')
    parser.add_argument('--fp16-precision', action='store_true',
                        help='Whether or not to use 16-bit precision GPU training.')
    parser.add_argument('--log-every-n-steps', default=100, type=int,
                        help='Log every n steps')

    args = parser.parse_args()

    return args
