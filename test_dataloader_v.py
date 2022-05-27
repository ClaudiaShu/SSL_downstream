from torch.utils.data import DataLoader

from data_aug import get_video_transform
from data_loader import Aff2_Dataset_series_shuffle
from utils import create_original_data

seq_len = 5
train_transform, test_transform = get_video_transform(seq_len)
# train set
df_train = create_original_data(
    f'/data/users/ys221/data/ABAW/labels_save/expression/Train_Set_v2/*')
# valid set
df_valid = create_original_data(
    f'/data/users/ys221/data/ABAW/labels_save/expression/Validation_Set_v2/*')

dataset = Aff2_Dataset_series_shuffle(df=df_valid, root=False,
                                            transform=test_transform,
                                            length_seq=seq_len)

data_loader = DataLoader(dataset=dataset,
                          batch_size=64,
                          num_workers=6,
                          shuffle=True,
                          drop_last=False)

for samples in data_loader:
    imgs = samples['images']
    print()