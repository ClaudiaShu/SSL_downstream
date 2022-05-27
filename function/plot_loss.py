import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

file_path = '/data/users/ys221/software/SSL_downstream/runs/May16_15-48-52_fuhe.doc.ic.ac.uk/training.log'
file = open(file_path, 'r')
loss = []
line1 = file.readline()
line2 = file.readline()
for lines in tqdm(file.readlines()):
    # print(lines)
    loss.append(lines.split('\t')[-2].split(':')[-1])

loss = np.stack(loss).astype(float)

plt.plot(loss)
plt.show()