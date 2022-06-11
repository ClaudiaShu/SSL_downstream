from torch.utils.data import DataLoader
from tqdm import tqdm

from data_aug import get_image_transform
from data_loader import AFFECTNET_hdf5, AFFECTNET_Dataset
from utils import create_original_data

seq_len = 5
train_transform, test_transform = get_image_transform(seq_len)
# train set
train_data = "/mnt/c/Data/Yuxuan/AffectNet/train_set_data.csv"
#
dataset = AFFECTNET_Dataset(df=train_data,
                            root=False,
                            transform=train_transform)
#
# hdf5_file_name = "/mnt/c/Data/Yuxuan/AffectNet/AffectNet_valid.hdf5"
#
# dataset = AFFECTNET_hdf5(hdf5_file_name, transform=train_transform)

data_loader = DataLoader(dataset=dataset,
                          batch_size=64,
                          num_workers=8,
                          shuffle=True,
                          drop_last=False)

for samples in tqdm(data_loader):
    images = samples['images']
    labels = samples['labels']
    labels_va = samples['labels_conti']


