import argparse
import warnings
import logging
import torch.optim
import torchvision.models
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR, MultiStepLR, CosineAnnealingWarmRestarts, StepLR
from pytorch_metric_learning import distances, losses, miners, reducers, testers
import pytorch_warmup as warmup
from sklearn.metrics import confusion_matrix, classification_report
from data_loader import *
from config import *
from loss import *
from models.model_RES import RES_SSL
from models.model_RES_imagenet import RES_feature
from utils import *
from config.default_loss import _C as cfg_loss
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

warnings.filterwarnings("ignore")

device = "cuda:0" if torch.cuda.is_available() else 'cpu'

def args_config():
    parser = argparse.ArgumentParser(description='EXP Training')
    parser.add_argument('--lr', default=1e-2, type=float, help='learning rate')  # 1e-3#
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')  # 256#
    parser.add_argument('--num_epochs', default=50, type=int, help='number epochs')  # 12#
    parser.add_argument('--num_classes', default=8, type=int, help='number classes')
    parser.add_argument('--weight_decay', default=5e-4, type=float)  # 5e-4#
    parser.add_argument('--seq_len', default=3, type=int)

    parser.add_argument('--temp', type=float, default=0.07)
    parser.add_argument('--alpha', type=float, default=0.7,
                        help='mixup interpolation coefficient (default: 1)')
    parser.add_argument('--remix_tau', type=float, default=0.9,
                        help='remixup interpolation coefficient tau(default: 1)')
    parser.add_argument('--remix_kappa', type=float, default=15.,
                        help='remixup interpolation coefficient kappa(default: 1)')

    parser.add_argument('--dataset', type=str, default="v2")  # ["v1","v2","v3"]
    parser.add_argument('--net', type=str, default="RES")  # ["RES","INC","HRN",'C','M','T']
    parser.add_argument('--mode', type=str, default="train")  # ["train","trainMixup","trainRemix"]
    parser.add_argument('--file', type=str, default=f"{parser.parse_args().net}_ex_best")

    parser.add_argument('--sample', type=str, default="ori")
    parser.add_argument('--resume', type=bool, default=False)

    parser.add_argument('--loss', type=str, default='CrossEntropyLabelAwareSmooth')  #['CrossEntropyLabelAwareSmooth','SuperContrastive']
    parser.add_argument('--warmup', type=bool, default=False)
    parser.add_argument('--optim', type=str, default='SGD')

    args = parser.parse_args()

    return args


def train(train_loader, model, itr):
    cost_list = 0

    cat_preds = []
    cat_labels = []

    model.train()

    for batch_idx, samples in tqdm(enumerate(train_loader), total=len(train_loader), desc=f'{args.net} Train_mode with warmup {args.warmup}'):
        images = samples['images'].to(device).float()
        labels_cat = samples['labels'].to(device).long()
        # import pdb; pdb.set_trace()

        pred_cat = model(images)
        loss = criterion(pred_cat, labels_cat)
        cost_list += loss.item()
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()
        scheduler.step(itr)
        if args.warmup:
            warmup_scheduler.dampen()
        itr += 1

        pred_cat = F.softmax(pred_cat)
        pred_cat = torch.argmax(pred_cat, dim=1)

        cat_preds.append(pred_cat.detach().cpu().numpy())
        cat_labels.append(labels_cat.detach().cpu().numpy())
        t.set_postfix(Lr=optimizer.param_groups[0]['lr'],
                      Loss=f'{cost_list / (batch_idx + 1):04f}',
                      itr=itr)

    cat_preds = np.concatenate(cat_preds, axis=0)
    cat_labels = np.concatenate(cat_labels, axis=0)
    cm = confusion_matrix(cat_labels, cat_preds)
    cr = classification_report(cat_labels, cat_preds)
    f1, acc, total = EXPR_metric(cat_preds, cat_labels)
    print(f'f1 = {f1} \n'
          f'acc = {acc} \n'
          f'total = {total} \n',
          'confusion metrix: \n', cm, '\n',
          'classification report: \n', cr, '\n')

def valid(valid_loader, model):
    cost_test = 0
    model.eval()
    with torch.no_grad():
        cat_preds = []
        cat_labels = []
        for batch_idx, samples in tqdm(enumerate(valid_loader), total=len(valid_loader), desc='Valid_mode'):
            images = samples['images'].to(device).float()
            labels_cat = samples['labels'].to(device).long()

            pred_cat = model(images)
            test_loss = criterion(pred_cat, labels_cat)
            cost_test += test_loss.item()

            pred_cat = F.softmax(pred_cat)
            pred_cat = torch.argmax(pred_cat, dim=1)

            cat_preds.append(pred_cat.detach().cpu().numpy())
            cat_labels.append(labels_cat.detach().cpu().numpy())
            t.set_postfix(Loss=f'{cost_test / (batch_idx + 1):04f}')

        cat_preds = np.concatenate(cat_preds, axis=0)
        cat_labels = np.concatenate(cat_labels, axis=0)
        cm = confusion_matrix(cat_labels, cat_preds)
        cr = classification_report(cat_labels, cat_preds)
        f1, acc, total = EXPR_metric(cat_preds, cat_labels)
        print(f'f1 = {f1} \n'
              f'acc = {acc} \n'
              f'total = {total} \n',
              'confusion metrix: \n', cm, '\n',
              'classification report: \n', cr, '\n')

        lr = optimizer.param_groups[0]['lr']

    return f1, acc, total, cm, cr, lr

def setup(df_train, df_valid, args):
    train_dataset = Aff2_Dataset_static_shuffle(df=df_train, root=False,
                                                transform=train_transform)

    valid_dataset = Aff2_Dataset_static_shuffle(df=df_valid, root=False,
                                                transform=test_transform)

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size,
                              num_workers=12,
                              shuffle=True,
                              drop_last=False)

    valid_loader = DataLoader(dataset=valid_dataset,
                              batch_size=args.batch_size,
                              num_workers=12,
                              shuffle=False,
                              drop_last=False)

    best_acc = 0

    return train_loader, valid_loader, best_acc


if __name__ == '__main__':
    args = args_config()
    seed_everything()
    # train set
    df_train = create_original_data(
        f'/data/users/ys221/data/ABAW/labels_save/expression/Train_Set_{args.dataset}/*')
    # valid set
    df_valid = create_original_data(
        f'/data/users/ys221/data/ABAW/labels_save/expression/Validation_Set_{args.dataset}/*')

    list_labels_ex = np.array(df_train['labels_ex'].values)
    weight, num_class_list = ex_count_weight(list_labels_ex)
    weight = weight.to(device)
    print("Exp weight: ", weight)

    train_loader, valid_loader, best_acc = setup(df_train, df_valid, args)

    model = RES_feature()
    # model = torchvision.models.resnet50(pretrained=True, num_classes=8)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model.to(device)

    para_dict = {
        "num_classes": args.num_classes,
        "num_class_list": num_class_list,
        "device": device,
        "cfg": cfg_loss,
    }

    # loss
    if args.loss == 'CrossEntropyLabelAwareSmooth':
        criterion = CrossEntropyLabelAwareSmooth(para_dict=para_dict)
    elif args.loss == 'BalancedSoftmaxCE':
        criterion = BalancedSoftmaxCE(para_dict=para_dict)
    else:
        criterion = nn.CrossEntropyLoss()

    distance = distances.CosineSimilarity()
    reducer = reducers.ThresholdReducer(low=0)
    criterion_tri = losses.TripletMarginLoss(margin=0.1, distance=distance, reducer=reducer)
    mining_func = miners.TripletMarginMiner(margin=0.1, distance=distance, type_of_triplets="semihard")

    # optim & scheduler
    num_steps = len(train_loader) * args.num_epochs
    if args.warmup:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
        scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=num_steps, eta_min=1e-6, last_epoch=-1)
        warmup_scheduler = warmup.ExponentialWarmup(optimizer, warmup_period=len(train_loader))
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
        scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=num_steps, eta_min=1e-6, last_epoch=-1)
        # scheduler = StepLR(optimizer=optimizer, step_size=len(train_loader), gamma=0.5, last_epoch=-1)
        # scheduler = CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=len(train_loader), T_mult=2, eta_min=0, last_epoch=-1)

    optimizer.zero_grad()
    optimizer.step()

    itr = 0

    with trange(args.num_epochs, total=args.num_epochs, desc='Epoch') as t:
        for epoch in t:
            t.set_description('Epoch %i' % epoch)
            # train here
            train(train_loader, model, itr)

            itr += len(train_loader)
            f1, acc, total, cm, cr, lr = valid(valid_loader, model)

            state = {
                'net': model,
                'acc': total,
                'epoch': epoch,
                'optim': optimizer.state_dict(),
                'rng_state': torch.get_rng_state()
            }
            os.makedirs(f'./checkpoint/expression/{args.net}_{args.dataset}_{args.mode}', exist_ok=True)
            torch.save(state, f'./checkpoint/expression/{args.net}_{args.dataset}_{args.mode}/{args.file}_{epoch}.pth')

            os.makedirs(f'./log/expression/{args.net}_{args.dataset}_{args.mode}', exist_ok=True)
            logging.basicConfig(level=logging.INFO,
                                filename=f'./log/expression/{args.net}_{args.dataset}_{args.mode}/{args.file}.log',
                                filemode='a+')

            logging.info(f'start epoch {epoch} ----------------------------')
            logging.info(f'Currently {args.mode} using {args.net} model')
            logging.info(f'Sampling the data in {args.sample} way')
            logging.info(f'Using {args.dataset} dataset')
            logging.info(f'Training loss: {args.loss}')
            logging.info(f'optimizer: {args.optim}')
            logging.info(f'hyperparameter setting: ')
            logging.info(f'learning rate: {str(lr)} ')
            logging.info(f'batch size: {str(args.batch_size)} ')
            logging.info(f'trianing epochs: {str(args.num_epochs)}')
            logging.info(f'epoch {epoch}: f1 = {f1}, acc = {acc}, total = {total}')
            logging.info('end   -------------------------------------------')