import numpy as np

list_path_prefix = '../data/BP4D/list/'

'''
example of content in 'BP4D_train_label_fold1.txt':
0 0 0 0 0 1 1 0 0 0 0 0
0 0 0 0 0 1 1 0 0 0 0 0
0 0 0 0 0 1 1 0 0 0 0 0
'''

for fold in range(1,4):
    imgs_AUoccur = np.loadtxt(list_path_prefix + 'BP4D_train_label_fold'+str(fold)+'.txt')
    AUoccur_rate = np.zeros((1, imgs_AUoccur.shape[1]))
    for i in range(imgs_AUoccur.shape[1]):
        AUoccur_rate[0, i] = sum(imgs_AUoccur[:,i]>0) / float(imgs_AUoccur.shape[0])

    AU_weight = 1.0 / AUoccur_rate
    AU_weight = AU_weight / AU_weight.sum() * AU_weight.shape[1]
    np.savetxt(list_path_prefix+'BP4D_train_weight_fold'+str(fold)+'.txt', AU_weight, fmt='%f', delimiter='\t')
