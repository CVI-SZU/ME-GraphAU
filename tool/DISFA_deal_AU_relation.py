import os

import numpy as np

list_path = "../data/DISFA/list"
class_num = 8

for i in range(1, 4):
    read_list_name = "DISFA_train_label_fold" + str(i) + ".txt"
    save_list_name = "DISFA_train_AU_relation_fold" + str(i) + ".txt"
    aus = np.loadtxt(os.path.join(list_path, read_list_name))
    le = aus.shape[0]
    new_aus = np.zeros((le, class_num * class_num))
    for j in range(class_num):
        for k in range(class_num):
            new_aus[:, j * class_num + k] = 2 * aus[:, j] + aus[:, k]
    np.savetxt(os.path.join(list_path, save_list_name), new_aus, fmt="%d")
