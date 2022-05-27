import argparse
import warnings

import torch

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader

from ABAW_img import ABAW_trainer
from data_loader import train_transform, test_transform, Aff2_Dataset_series_shuffle, Aff2_Dataset_static_shuffle
from models.model_R3D_DFEW import DFEW_SSL
from models.model_RES import RES_SSL
from utils import seed_everything, create_original_data

warnings.filterwarnings("ignore")
device = "cuda:0" if torch.cuda.is_available() else 'cpu'

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

parser.add_argument('--gpu-index', default=0, type=int, help='Gpu index.')
parser.add_argument('--fp16-precision', action='store_true',
                    help='Whether or not to use 16-bit precision GPU training.')
parser.add_argument('--log-every-n-steps', default=100, type=int,
                    help='Log every n steps')

args = parser.parse_args()

def setup(df_train, df_valid, args):
    train_dataset = Aff2_Dataset_static_shuffle(df=df_train, root=False,
                                                transform=train_transform)
    valid_dataset = Aff2_Dataset_static_shuffle(df=df_valid, root=False,
                                                transform=test_transform)

    # train_dataset = Aff2_Dataset_series_shuffle(df=df_train, root=False,
    #                                             transform=train_transform,
    #                                             type_partition='ex', length_seq=3)
    # valid_dataset = Aff2_Dataset_series_shuffle(df=df_valid, root=False,
    #                                             transform=test_transform,
    #                                             type_partition='ex', length_seq=3)

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size,
                              num_workers=args.num_workers,
                              shuffle=True,
                              drop_last=False)
    valid_loader = DataLoader(dataset=valid_dataset,
                              batch_size=args.batch_size,
                              num_workers=args.num_workers,
                              shuffle=False,
                              drop_last=False)

    best_acc = 0

    return train_loader, valid_loader, best_acc

def main():
    # args = parser.parse_args()
    seed_everything()
    if torch.cuda.is_available():
        args.device = torch.device('cuda')
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        args.device = torch.device('cpu')

    # train set
    df_train = create_original_data(
        f'/data/users/ys221/data/ABAW/labels_save/expression/Train_Set_{args.dataset}/*')
    # valid set
    df_valid = create_original_data(
        f'/data/users/ys221/data/ABAW/labels_save/expression/Validation_Set_{args.dataset}/*')

    train_loader, valid_loader, best_acc = setup(df_train, df_valid, args)

    # model = DFEW_SSL(mode='SCHW')
    model = RES_SSL()

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)

    num_steps = len(train_loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=0,
                                                           last_epoch=-1)

    # criterion = nn.CrossEntropyLoss()
    optimizer.zero_grad()
    optimizer.step()

    #  It’s a no-op if the 'gpu_index' argument is a negative integer or None.
    with torch.cuda.device(args.gpu_index):
        exp_train = ABAW_trainer(model=model, optimizer=optimizer, scheduler=scheduler, args=args)
        exp_train.run(train_loader, valid_loader)


if __name__ == "__main__":
    main()