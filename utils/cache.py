from utils import *
file = '/data/users/ys221/data/ABAW/labels_save/expression/Train_Set_v1/*'

csv_train, csv_split = split_csv_data(file1=file, file2=None, number_fold=5)
# create_original_data(file_name=file)