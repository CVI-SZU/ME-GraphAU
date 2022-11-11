import os
import pandas as pd
import numpy as np
# #
AUs = ['1', '2' ,'4', '5', '6', '7', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '22' ,'23', '24', '25', '26', '27', '32', '38', '39']
mcro_AUs = ['L1', 'R1', 'L2', 'R2', 'L4', 'R4', 'L6', 'R6', 'L10', 'R10', 'L12', 'R12', 'L14', 'R14']
total_AUs = AUs+mcro_AUs

new_dataset_train_img_list = []
new_dataset_val_img_list = []
new_dataset_test_img_list = []

new_dataset_train_label_list = []
new_dataset_val_label_list = []
new_dataset_test_label_list = []


#BP4d
print("processing BP4D------------------------------------------------------------")
TRAIN_BP4D_Sequence_split = ['F001','M007','F018','F008','F002','M004','F009','M012','M001','F020','M014','F014',
                             'F023','M008','M010','M002','F005','F022','M018','M017','F013','M013']
VAL_BP4D_Sequence_split =  ['F003','M016','F011','M005', 'F016','M011']
TEST_BP4D_Sequence_split = ['F007','F015','F006','F019','M006','M009','F012','M003','F004','F021','F017','M015','F010']

tasks = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8']
label_folder = 'BP4D/AUCoding/AU_OCC'
list_path_prefix = 'Datasets/hybrid_dataset/BP4D/list'


def get_AUlabels(seq, task, path):
	path_label = os.path.join(path, '{sequence}_{task}.csv'.format(sequence=seq, task=task))
	usecols = ['0', '1', '2' ,'4', '5', '6', '7', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '22' ,'23', '24', '25', '26', '27', '32', '38', '39']
	df = pd.read_csv(path_label, header=0, index_col=0, usecols=usecols)
	frames = [str(item) for item in list(df.index.values)]
	frames_path = ['{}/{}/{}'.format(seq, task, item) for item in frames]
	labels = df.values
	# 返回的frames是list，值是排好序的int变量，指示对应的帧。labels是N*12的np.ndarray，对应AU标签
	return frames_path, labels


with open(os.path.join(list_path_prefix,  'BP4D_train_img_path.txt'),'w') as f:
    u = 0
with open(os.path.join(list_path_prefix,  'BP4D_val_img_path.txt'),'w') as f:
    u = 0
with open(os.path.join(list_path_prefix,  'BP4D_test_img_path.txt'),'w') as f:
    u = 0

frames = None
labels = None
for seq in TRAIN_BP4D_Sequence_split:
    for t in tasks:
        temp_frames, temp_labels = get_AUlabels(seq, t, label_folder)
        temp_labels = temp_labels.astype(int)
        temp_labels[temp_labels == 9] = -1
        padding = np.zeros((temp_labels.shape[0], len(mcro_AUs))) -1
        temp_labels = np.concatenate((temp_labels, padding), axis=-1)
        if frames is None:
            labels = temp_labels
            frames = temp_frames  # str list
        else:
            labels = np.concatenate((labels, temp_labels), axis=0)  # np.ndarray
            frames = frames + temp_frames  # str list

BP4D_train_image_path_list = frames
BP4D_train_image_label = labels

for frame in BP4D_train_image_path_list:
    frame_img_name = frame + '.jpg'
    with open(os.path.join(list_path_prefix,  'BP4D_train_img_path.txt'), 'a+') as f:
        f.write(os.path.join('BP4D', frame_img_name + '\n'))
        new_dataset_train_img_list.append(os.path.join('BP4D', frame_img_name + '\n'))
np.savetxt( os.path.join(list_path_prefix,  'BP4D_train_label.txt') , BP4D_train_image_label ,fmt='%d', delimiter=' ')
new_dataset_train_label_list.append(BP4D_train_image_label)
# print("Train label shape:", BP4D_train_image_label.shape)
# print("Train label fre:", BP4D_train_image_label.sum(0)[[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,23,24,25,26]])

frames = None
labels = None
for seq in VAL_BP4D_Sequence_split:
    for t in tasks:
        temp_frames, temp_labels = get_AUlabels(seq, t, label_folder)
        temp_labels = temp_labels.astype(int)
        temp_labels[temp_labels == 9] = -1
        padding = np.zeros((temp_labels.shape[0], len(mcro_AUs))) -1
        temp_labels = np.concatenate((temp_labels, padding), axis=-1)
        if frames is None:
            labels = temp_labels
            frames = temp_frames  # str list
        else:
            labels = np.concatenate((labels, temp_labels), axis=0)  # np.ndarray
            frames = frames + temp_frames  # str list

BP4D_val_image_path_list = frames
BP4D_val_image_label = labels

for frame in BP4D_val_image_path_list:
    frame_img_name = frame + '.jpg'
    with open(os.path.join(list_path_prefix,  'BP4D_val_img_path.txt'), 'a+') as f:
        f.write(os.path.join('BP4D', frame_img_name + '\n'))
        new_dataset_val_img_list.append(os.path.join('BP4D', frame_img_name + '\n'))

np.savetxt( os.path.join(list_path_prefix,  'BP4D_val_label.txt') , BP4D_val_image_label ,fmt='%d', delimiter=' ')
new_dataset_val_label_list.append(BP4D_val_image_label)

# print("Val label shape:", BP4D_val_image_label.shape)
# print("Val label fre:", BP4D_val_image_label.sum(0)[[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,23,24,25,26]])

frames = None
labels = None
for seq in TEST_BP4D_Sequence_split:
    for t in tasks:
        temp_frames, temp_labels = get_AUlabels(seq, t, label_folder)
        temp_labels = temp_labels.astype(int)
        temp_labels[temp_labels == 9] = -1
        padding = np.zeros((temp_labels.shape[0], len(mcro_AUs))) -1
        temp_labels = np.concatenate((temp_labels, padding), axis=-1)
        if frames is None:
            labels = temp_labels
            frames = temp_frames  # str list
        else:
            labels = np.concatenate((labels, temp_labels), axis=0)  # np.ndarray
            frames = frames + temp_frames  # str list

BP4D_test_image_path_list = frames
BP4D_test_image_label = labels

for frame in BP4D_test_image_path_list:
    frame_img_name = frame + '.jpg'
    with open(os.path.join(list_path_prefix,  'BP4D_test_img_path.txt'), 'a+') as f:
        f.write(os.path.join('BP4D', frame_img_name + '\n'))
        new_dataset_test_img_list.append(os.path.join('BP4D', frame_img_name + '\n'))

np.savetxt( os.path.join(list_path_prefix,  'BP4D_test_label.txt') , BP4D_test_image_label ,fmt='%d', delimiter=' ')
new_dataset_test_label_list.append(BP4D_test_image_label)

# print("Test label shape:", BP4D_test_image_label.shape)
# print("Test label fre:", BP4D_test_image_label.sum(0)[[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,23,24,25,26]])


#-------------------------------------------------------------------------------------------------------------------------------
#DISFA
print("processing DISFA------------------------------------------------------------")
#You nead downloading DISFA including 'ActionUnit_Labels'
label_path = 'DISFA/ActionUnit_Labels'
list_path_prefix = 'Datasets/hybrid_dataset/DISFA/list'

TRAIN_DISFA_Sequence_split = ['SN002','SN010','SN027','SN032','SN030','SN009','SN016','SN013','SN018','SN011','SN028','SN024','SN003','SN029']
VAL_DISFA_Sequence_split = ['SN012','SN006','SN031','SN021']
TEST_DISFA_Sequence_split = ['SN023','SN025','SN008','SN005','SN007','SN017','SN004','SN001','SN026']

# fold1:  train : part1+part2 test: part3
# fold2:  train : part1+part3 test: part2
# fold3:  train : part2+part3 test: part1

au_idx = [1, 2, 4, 5, 6, 9, 12, 15, 17, 20, 25, 26]


with open(os.path.join(list_path_prefix,'DISFA_train_img_path.txt'),'w') as f:
    u = 0

TRAIN_frame_list = []
TRAIN_numpy_list = []
for fr in TRAIN_DISFA_Sequence_split:
    fr_path = os.path.join(label_path,fr)
    au1_path = os.path.join(fr_path,fr+'_au1.txt')
    with open(au1_path, 'r') as label:
        total_frame = len(label.readlines())
    au_label_array = np.zeros((total_frame,12),dtype=np.int32)
    for ai, au in enumerate(au_idx):
        AULabel_path = os.path.join(fr_path,fr+'_au'+str(au) +'.txt')
        if not os.path.isfile(AULabel_path):
            continue
        # print("--Checking AU:" + str(au) + " ...")
        with open(AULabel_path, 'r') as label:
            for t, lines in enumerate(label.readlines()):
                frameIdx, AUIntensity = lines.split(',')
                frameIdx, AUIntensity = int(frameIdx), int(AUIntensity)
                if AUIntensity >= 1:
                    AUIntensity = 1
                else:
                    AUIntensity = 0
                au_label_array[t,ai] = AUIntensity
    for i in range(total_frame):
        frame_img_name = fr + '/' + str(i) + '.png'
        TRAIN_frame_list.append(frame_img_name)
        with open(os.path.join(list_path_prefix,'DISFA_train_img_path.txt'), 'a+') as f:
            f.write(os.path.join('DISFA',frame_img_name+'\n'))
            new_dataset_train_img_list.append(os.path.join('DISFA',frame_img_name+'\n'))

    TRAIN_numpy_list.append(au_label_array)

TRAIN_numpy_list = np.concatenate(TRAIN_numpy_list,axis=0)
# TRAIN test for fold3
DISFA_train_image_label = np.zeros((TRAIN_numpy_list.shape[0], len(total_AUs))) -1
for i, au in enumerate(au_idx):
    au = str(au)
    index = total_AUs.index(au)
    DISFA_train_image_label[:, index] = TRAIN_numpy_list[:, i]

np.savetxt(os.path.join(list_path_prefix,'DISFA_train_label.txt'), DISFA_train_image_label,fmt='%d', delimiter=' ')
new_dataset_train_label_list.append(DISFA_train_image_label)
# print(TRAIN_numpy_list.sum(0))


with open(os.path.join(list_path_prefix,'DISFA_val_img_path.txt'),'w') as f:
    u = 0

VAL_frame_list = []
VAL_numpy_list = []
for fr in VAL_DISFA_Sequence_split:
    fr_path = os.path.join(label_path,fr)
    au1_path = os.path.join(fr_path,fr+'_au1.txt')
    with open(au1_path, 'r') as label:
        total_frame = len(label.readlines())
    au_label_array = np.zeros((total_frame,12),dtype=np.int32)
    for ai, au in enumerate(au_idx):
        AULabel_path = os.path.join(fr_path,fr+'_au'+str(au) +'.txt')
        if not os.path.isfile(AULabel_path):
            continue
        # print("--Checking AU:" + str(au) + " ...")
        with open(AULabel_path, 'r') as label:
            for t, lines in enumerate(label.readlines()):
                frameIdx, AUIntensity = lines.split(',')
                frameIdx, AUIntensity = int(frameIdx), int(AUIntensity)
                if AUIntensity >= 1:
                    AUIntensity = 1
                else:
                    AUIntensity = 0
                au_label_array[t,ai] = AUIntensity
    for i in range(total_frame):
        frame_img_name = fr + '/' + str(i) + '.png'
        VAL_frame_list.append(frame_img_name)
        with open(os.path.join(list_path_prefix,'DISFA_val_img_path.txt'), 'a+') as f:
            f.write(os.path.join('DISFA',frame_img_name+'\n'))
            new_dataset_val_img_list.append(os.path.join('DISFA',frame_img_name+'\n'))

    VAL_numpy_list.append(au_label_array)

VAL_numpy_list = np.concatenate(VAL_numpy_list,axis=0)
DISFA_val_image_label = np.zeros((VAL_numpy_list.shape[0], len(total_AUs))) -1
for i, au in enumerate(au_idx):
    au = str(au)
    index = total_AUs.index(au)
    DISFA_val_image_label[:, index] = VAL_numpy_list[:, i]

np.savetxt(os.path.join(list_path_prefix,'DISFA_val_label.txt'), DISFA_val_image_label,fmt='%d', delimiter=' ')
new_dataset_val_label_list.append(DISFA_val_image_label)

# print(VAL_numpy_list.sum(0))


with open(os.path.join(list_path_prefix,'DISFA_test_img_path.txt'),'w') as f:
    u = 0

TEST_frame_list = []
TEST_numpy_list = []
for fr in TEST_DISFA_Sequence_split:
    fr_path = os.path.join(label_path,fr)
    au1_path = os.path.join(fr_path,fr+'_au1.txt')
    with open(au1_path, 'r') as label:
        total_frame = len(label.readlines())
    au_label_array = np.zeros((total_frame,12),dtype=np.int32)
    for ai, au in enumerate(au_idx):
        AULabel_path = os.path.join(fr_path,fr+'_au'+str(au) +'.txt')
        if not os.path.isfile(AULabel_path):
            continue
        # print("--Checking AU:" + str(au) + " ...")
        with open(AULabel_path, 'r') as label:
            for t, lines in enumerate(label.readlines()):
                frameIdx, AUIntensity = lines.split(',')
                frameIdx, AUIntensity = int(frameIdx), int(AUIntensity)
                if AUIntensity >= 1:
                    AUIntensity = 1
                else:
                    AUIntensity = 0
                au_label_array[t,ai] = AUIntensity
    for i in range(total_frame):
        frame_img_name = fr + '/' + str(i) + '.png'
        TEST_frame_list.append(frame_img_name)
        with open(os.path.join(list_path_prefix,'DISFA_test_img_path.txt'), 'a+') as f:
            f.write(os.path.join('DISFA',frame_img_name+'\n'))
            new_dataset_test_img_list.append(os.path.join('DISFA',frame_img_name+'\n'))

    TEST_numpy_list.append(au_label_array)

TEST_numpy_list = np.concatenate(TEST_numpy_list,axis=0)
DISFA_test_image_label = np.zeros((TEST_numpy_list.shape[0], len(total_AUs))) -1
for i, au in enumerate(au_idx):
    au = str(au)
    index = total_AUs.index(au)
    DISFA_test_image_label[:, index] = TEST_numpy_list[:, i]

np.savetxt(os.path.join(list_path_prefix,'DISFA_test_label.txt'), DISFA_test_image_label,fmt='%d', delimiter=' ')
new_dataset_test_label_list.append(DISFA_test_image_label)

# print(TEST_numpy_list.sum(0))


#-------------------------------------------------------------------------------------------------------------------------------
#CK+
print("processing CK+------------------------------------------------------------")

import os
import random
from collections import Counter
all_list = []
path = 'CK+/FACS'
list_path_prefix = 'Datasets/hybrid_dataset/CK+/list'
img_dir_prefix = 'Datasets/hybrid_dataset/img/CK+'

train_subjects = ['S034', 'S147', 'S149', 'S502', 'S046', 'S061', 'S037', 'S115', 'S076', 'S506', 'S080', 'S067', 'S096', 'S134', 'S092', 'S137', 'S073', 'S086', 'S099', 'S104', 'S042', 'S129', 'S059', 'S139', 'S106', 'S110', 'S085', 'S111', 'S053', 'S113', 'S503', 'S103', 'S093', 'S077', 'S069', 'S094', 'S108', 'S074', 'S505', 'S131', 'S062', 'S116', 'S097', 'S151', 'S126', 'S101', 'S088', 'S075', 'S082', 'S127', 'S136', 'S130', 'S098', 'S044', 'S029', 'S045', 'S054', 'S057', 'S133', 'S999', 'S032', 'S072', 'S064', 'S135', 'S084', 'S117', 'S079', 'S078', 'S122', 'S091', 'S119', 'S022', 'S089', 'S014', 'S050', 'S125', 'S107']
val_subjects = ['S102', 'S895', 'S011', 'S056', 'S109', 'S026', 'S028', 'S051', 'S148', 'S055', 'S154', 'S128', 'S118', 'S504']
test_subjects = ['S068', 'S060', 'S105', 'S070', 'S090', 'S157', 'S087', 'S156', 'S155', 'S095', 'S071', 'S066', 'S058', 'S160', 'S121', 'S501', 'S112', 'S035', 'S083', 'S005', 'S063', 'S120', 'S100', 'S124', 'S065', 'S158', 'S052', 'S138', 'S081', 'S132', 'S010', 'S114']

au_idx = [1,2,4,5,6,7,9,10,11,12,13,14,15,16,17,18,20,22,23,24,25,26,27,38,39]

img_path_list = []
au_label_list = []

for subject in train_subjects:
    sub_path = os.path.join(path, subject)
    sequences = os.listdir(sub_path)
    for sq in sequences:
        sq_pth = os.path.join(sub_path, sq)
        txts = os.listdir(sq_pth)
        for tx in txts:
            tx_path = os.path.join(sq_pth, tx)
            label = np.zeros((1,len(au_idx)))
            flag =0
            with open(tx_path, 'r') as f:
                lines = f.readlines()
                img_path = os.path.join(subject, sq, tx.split('_facs')[0] + '.png')
                if os.path.exists(os.path.join(img_dir_prefix, img_path)):
                    for line in lines:
                        AU = line.strip().split(' ')[0]
                        if AU != '' and int(float(AU)) in au_idx:
                            flag = 1
                            AU = int(float(AU))
                            AUIntensity = int(float(line.strip().split(' ')[3]))
                            if AUIntensity > 1:
                                AUIntensity = 1
                            else:
                                AUIntensity = 0
                            label[0, au_idx.index(AU)] = AUIntensity
                    if flag > 0:
                        img_path_list.append(img_path)
                        au_label_list.append(label)

TRAIN_numpy_list = np.concatenate(au_label_list,axis=0)
CKPlus_train_image_label = np.zeros((TRAIN_numpy_list.shape[0], len(total_AUs))) -1
for i, au in enumerate(au_idx):
    au = str(au)
    index = total_AUs.index(au)
    CKPlus_train_image_label[:, index] = TRAIN_numpy_list[:, i]

np.savetxt(os.path.join(list_path_prefix,'CK+_train_label.txt'), CKPlus_train_image_label,fmt='%d', delimiter=' ')
new_dataset_train_label_list.append(CKPlus_train_image_label)

with open(os.path.join(list_path_prefix, 'CK+_train_img_path.txt'), 'w+') as f:
    l = 0
for img_path in img_path_list:
    with open(os.path.join(list_path_prefix,'CK+_train_img_path.txt'), 'a+') as f:
        f.write(os.path.join('CK+', img_path+'\n'))
        new_dataset_train_img_list.append(os.path.join('CK+', img_path+'\n'))

img_path_list = []
au_label_list = []

for subject in val_subjects:
    sub_path = os.path.join(path, subject)
    sequences = os.listdir(sub_path)
    for sq in sequences:
        sq_pth = os.path.join(sub_path, sq)
        txts = os.listdir(sq_pth)
        for tx in txts:
            tx_path = os.path.join(sq_pth, tx)
            label = np.zeros((1,len(au_idx)))
            flag =0
            with open(tx_path, 'r') as f:
                lines = f.readlines()
                img_path = os.path.join(subject, sq, tx.split('_facs')[0] + '.png')
                if os.path.exists(os.path.join(img_dir_prefix, img_path)):
                    for line in lines:
                        AU = line.strip().split(' ')[0]
                        if AU != '' and int(float(AU)) in au_idx:
                            flag = 1
                            AU = int(float(AU))
                            AUIntensity = int(float(line.strip().split(' ')[3]))
                            if AUIntensity > 1:
                                AUIntensity = 1
                            else:
                                AUIntensity = 0
                            label[0, au_idx.index(AU)] = AUIntensity
                    if flag > 0:
                        img_path_list.append(img_path)
                        au_label_list.append(label)

VAL_numpy_list = np.concatenate(au_label_list,axis=0)
CKPlus_val_image_label = np.zeros((VAL_numpy_list.shape[0], len(total_AUs))) -1
for i, au in enumerate(au_idx):
    au = str(au)
    index = total_AUs.index(au)
    CKPlus_val_image_label[:, index] = VAL_numpy_list[:, i]

np.savetxt(os.path.join(list_path_prefix,'CK+_val_label.txt'), CKPlus_val_image_label,fmt='%d', delimiter=' ')
new_dataset_val_label_list.append(CKPlus_val_image_label)

with open(os.path.join(list_path_prefix, 'CK+_val_img_path.txt'), 'w+') as f:
    l = 0
for img_path in img_path_list:
    with open(os.path.join(list_path_prefix,'CK+_val_img_path.txt'), 'a+') as f:
        f.write(os.path.join('CK+', img_path+'\n'))
        new_dataset_val_img_list.append(os.path.join('CK+', img_path+'\n'))


img_path_list = []
au_label_list = []

for subject in test_subjects:
    sub_path = os.path.join(path, subject)
    sequences = os.listdir(sub_path)
    for sq in sequences:
        sq_pth = os.path.join(sub_path, sq)
        txts = os.listdir(sq_pth)
        for tx in txts:
            tx_path = os.path.join(sq_pth, tx)
            label = np.zeros((1,len(au_idx)))
            flag =0
            with open(tx_path, 'r') as f:
                lines = f.readlines()
                img_path = os.path.join(subject, sq, tx.split('_facs')[0] + '.png')
                if os.path.exists(os.path.join(img_dir_prefix, img_path)):
                    for line in lines:
                        AU = line.strip().split(' ')[0]
                        if AU != '' and int(float(AU)) in au_idx:
                            flag = 1
                            AU = int(float(AU))
                            AUIntensity = int(float(line.strip().split(' ')[3]))
                            if AUIntensity > 1:
                                AUIntensity = 1
                            else:
                                AUIntensity = 0
                            label[0, au_idx.index(AU)] = AUIntensity
                    if flag > 0:
                        img_path_list.append(img_path)
                        au_label_list.append(label)

TEST_numpy_list = np.concatenate(au_label_list,axis=0)
CKPlus_test_image_label = np.zeros((TEST_numpy_list.shape[0], len(total_AUs))) -1
for i, au in enumerate(au_idx):
    au = str(au)
    index = total_AUs.index(au)
    CKPlus_test_image_label[:, index] = TEST_numpy_list[:, i]

np.savetxt(os.path.join(list_path_prefix,'CK+_test_label.txt'), CKPlus_test_image_label,fmt='%d', delimiter=' ')
new_dataset_test_label_list.append(CKPlus_test_image_label)

with open(os.path.join(list_path_prefix, 'CK+_test_img_path.txt'), 'w+') as f:
    l = 0
for img_path in img_path_list:
    with open(os.path.join(list_path_prefix,'CK+_test_img_path.txt'), 'a+') as f:
        f.write(os.path.join('CK+', img_path+'\n'))
        new_dataset_test_img_list.append(os.path.join('CK+', img_path+'\n'))




#-------------------------------------------------------------------------------------------------------------------------------
#RAF-AU
print("processing RAF-AU------------------------------------------------------------")

import os
import random
au_idx = ['1', '2' ,'4', '5', '6', '7', '9', '10', '12', '14', '15', '16', '17', '18', '19' ,'20', '22', '23', '24', '25', '26', '27', '32', '39', 'L1', 'R1', 'L2', 'R2', 'L4', 'R4', 'L6', 'R6','L10','R10', 'L12', 'R12', 'L14', 'R14']
list_path_prefix = 'Datasets/hybrid_dataset/RAF-AU/list'
train_idx = []

with open('RAF-AU-train_ids.txt', 'r') as f:
    lines = f.readlines()
    for idx in lines:
        idx = int(idx.strip())
        train_idx.append(idx)

val_idx = []
with open('RAF-AU-val_ids.txt', 'r') as f:
    lines = f.readlines()
    for idx in lines:
        idx = int(idx.strip())
        val_idx.append(idx)

test_idx = []
with open('RAF-AU-test_ids.txt', 'r') as f:
    lines = f.readlines()
    for idx in lines:
        idx = int(idx.strip())
        test_idx.append(idx)

with open('RAF-AU/RAFAU_label.txt', 'r') as f:
    label_lines = f.readlines()

img_path_list = []
au_label_list = []
for idx in train_idx:
    train_au_item = label_lines[idx]
    items = train_au_item.split(' ')
    img_path = items[0]
    labels = items[1].strip()
    flag = 0
    if labels!= 'null':
        label_items = labels.split('+')
        au_label = np.zeros((1, len(au_idx)))
        for item in label_items:
            if item in au_idx:
                flag = 1
                au_label[0, au_idx.index(item)] = 1
    if flag >0:
        img_path_list.append(img_path.split('.')[0].zfill(4) + '_aligned.jpg')
        au_label_list.append(au_label)

TRAIN_numpy_list = np.concatenate(au_label_list,axis=0)
RAF_AU_train_image_label = np.zeros((TRAIN_numpy_list.shape[0], len(total_AUs))) -1
# print(TRAIN_numpy_list.sum(0))

for i, au in enumerate(au_idx):
    au = str(au)
    index = total_AUs.index(au)
    RAF_AU_train_image_label[:, index] = TRAIN_numpy_list[:, i]

np.savetxt(os.path.join(list_path_prefix,'RAF_AU_train_label.txt'), RAF_AU_train_image_label,fmt='%d', delimiter=' ')
new_dataset_train_label_list.append(RAF_AU_train_image_label)


with open(os.path.join(list_path_prefix, 'RAF_AU_train_img_path.txt'), 'a+') as f:
    l=0
for img_path in img_path_list:
    with open(os.path.join(list_path_prefix,'RAF_AU_train_img_path.txt'), 'a+') as f:
        f.write(os.path.join('RAF-AU',img_path+'\n'))
        new_dataset_train_img_list.append(os.path.join('RAF-AU',img_path+'\n'))


img_path_list = []
au_label_list = []
for idx in val_idx:
    val_au_item = label_lines[idx]
    items = val_au_item.split(' ')
    img_path = items[0]
    labels = items[1].strip()
    flag = 0
    if labels!= 'null':
        label_items = labels.split('+')
        au_label = np.zeros((1, len(au_idx)))
        for item in label_items:
            if item in au_idx:
                flag = 1
                au_label[0, au_idx.index(item)] = 1
    if flag >0:
        img_path_list.append(img_path.split('.')[0].zfill(4) + '_aligned.jpg')
        au_label_list.append(au_label)

VAL_numpy_list = np.concatenate(au_label_list,axis=0)
# print(VAL_numpy_list.sum(0))

RAF_AU_val_image_label = np.zeros((VAL_numpy_list.shape[0], len(total_AUs))) -1

for i, au in enumerate(au_idx):
    au = str(au)
    index = total_AUs.index(au)
    RAF_AU_val_image_label[:, index] = VAL_numpy_list[:, i]

np.savetxt(os.path.join(list_path_prefix,'RAF_AU_val_label.txt'), RAF_AU_val_image_label,fmt='%d', delimiter=' ')
new_dataset_val_label_list.append(RAF_AU_val_image_label)


with open(os.path.join(list_path_prefix, 'RAF_AU_val_img_path.txt'), 'a+') as f:
    l=0
for img_path in img_path_list:
    with open(os.path.join(list_path_prefix,'RAF_AU_val_img_path.txt'), 'a+') as f:
        f.write(os.path.join('RAF-AU',img_path+'\n'))
        new_dataset_val_img_list.append(os.path.join('RAF-AU',img_path+'\n'))


img_path_list = []
au_label_list = []
for idx in test_idx:
    test_au_item = label_lines[idx]
    items = test_au_item.split(' ')
    img_path = items[0]
    labels = items[1].strip()
    flag = 0
    if labels!= 'null':
        label_items = labels.split('+')
        au_label = np.zeros((1, len(au_idx)))
        for item in label_items:
            if item in au_idx:
                flag = 1
                au_label[0, au_idx.index(item)] = 1
    if flag >0:
        img_path_list.append(img_path.split('.')[0].zfill(4) + '_aligned.jpg')
        au_label_list.append(au_label)

TEST_numpy_list = np.concatenate(au_label_list,axis=0)
RAF_AU_test_image_label = np.zeros((TEST_numpy_list.shape[0], len(total_AUs))) -1
# print(TEST_numpy_list.sum(0))

for i, au in enumerate(au_idx):
    au = str(au)
    index = total_AUs.index(au)
    RAF_AU_test_image_label[:, index] = TEST_numpy_list[:, i]

np.savetxt(os.path.join(list_path_prefix,'RAF_AU_test_label.txt'),  RAF_AU_test_image_label,fmt='%d', delimiter=' ')
new_dataset_test_label_list.append(RAF_AU_test_image_label)

with open(os.path.join(list_path_prefix, 'RAF_AU_test_img_path.txt'), 'a+') as f:
    l=0

for img_path in img_path_list:
    with open(os.path.join(list_path_prefix,'RAF_AU_test_img_path.txt'), 'a+') as f:
        f.write(os.path.join('RAF-AU',img_path+'\n'))
        new_dataset_test_img_list.append(os.path.join('RAF-AU',img_path+'\n'))

#-------------------------------------------------------------------------------------------------------------------------------
#CASME2
print("processing CASME2------------------------------------------------------------")

df = pd.read_excel('CASME2/CASME2-coding-20140508.xlsx')
list_path_prefix = 'Datasets/hybrid_dataset/CASME2/list'
all_list = []

CASME2_train_subjects_split = ['sub01', 'sub02', 'sub04', 'sub06', 'sub7', 'sub11', 'sub12', 'sub17', 'sub19', 'sub20', 'sub21', 'sub24','sub25']
CASME2_val_subjects_split = ['sub03', 'sub05', 'sub16', 'sub22']
CASME2_test_subjects_split = ['sub08', 'sub09', 'sub10', 'sub15', 'sub23','sub26']

au_ids  = ['1', '2' ,'4', '5', '6', '7', '9', '10','12', '14', '15','17', '18', '20', '24', '25', '26', '38' ,'L1', 'R1', 'L2', 'R2', 'L4', 'R4', 'L6', 'R6', 'L10', 'R10', 'L12', 'R12', 'L14', 'R14']

df.iloc[:, 0] =  df.iloc[:, 0].astype(str)
df = df.iloc[:,[0,1,3,5,7]]
values = df.values

train_img_path_list = []
train_au_label_list = []

val_img_path_list = []
val_au_label_list = []

test_img_path_list = []
test_au_label_list = []

for line in values:
    subject = 'sub'+ line[0].zfill(2)
    sequence = line[1]
    OnsetFrame = line[2]
    OffsetFrame = line[3]
    au = str(line[4])
    flag = 0
    au_label = np.zeros((1,len(au_ids)))
    if au !='?':
        au_items = au.split('+')
        for item in au_items:
            # print(item)
            if item in au_ids:
                flag=1
                au_label[0, au_ids.index(item)] = 1

    if flag>0:
        for i in range(OnsetFrame, OffsetFrame+1):
            img_path = os.path.join(subject,str(sequence),'reg_img'+ str(i) +'.jpg')
            if subject in CASME2_train_subjects_split:
                train_img_path_list.append(img_path)
                train_au_label_list.append(au_label)
            elif subject in CASME2_val_subjects_split:
                val_img_path_list.append(img_path)
                val_au_label_list.append(au_label)
            else:
                test_img_path_list.append(img_path)
                test_au_label_list.append(au_label)

TRAIN_numpy_list = np.concatenate(train_au_label_list,axis=0)
VAL_numpy_list = np.concatenate(val_au_label_list,axis=0)
TEST_numpy_list = np.concatenate(test_au_label_list,axis=0)

# print(TRAIN_numpy_list.sum(0))
# print(VAL_numpy_list.sum(0))
# print(TEST_numpy_list.sum(0))
CASME2_train_image_label = np.zeros((TRAIN_numpy_list.shape[0], len(total_AUs))) - 1
for i, au in enumerate(au_ids):
    au = str(au)
    index = total_AUs.index(au)
    CASME2_train_image_label[:, index] = TRAIN_numpy_list[:, i]

np.savetxt(os.path.join(list_path_prefix,'CASME2_train_label.txt'),  CASME2_train_image_label, fmt='%d', delimiter=' ')
new_dataset_train_label_list.append(CASME2_train_image_label)


with open(os.path.join(list_path_prefix, 'CASME2_train_img_path.txt'), 'w+') as f:
    i=0
for img_path in train_img_path_list:
    with open(os.path.join(list_path_prefix,'CASME2_train_img_path.txt'), 'a+') as f:
        f.write(os.path.join('CASME2', img_path+'\n'))
        new_dataset_train_img_list.append(os.path.join('CASME2', img_path+'\n'))



CASME2_val_image_label = np.zeros((VAL_numpy_list.shape[0], len(total_AUs))) - 1
for i, au in enumerate(au_ids):
    au = str(au)
    index = total_AUs.index(au)
    CASME2_val_image_label[:, index] = VAL_numpy_list[:, i]

np.savetxt(os.path.join(list_path_prefix,'CASME2_val_label.txt'),  CASME2_val_image_label, fmt='%d', delimiter=' ')
new_dataset_val_label_list.append(CASME2_val_image_label)

with open(os.path.join(list_path_prefix, 'CASME2_val_img_path.txt'), 'w+') as f:
    i=0
for img_path in val_img_path_list:
    with open(os.path.join(list_path_prefix,'CASME2_val_img_path.txt'), 'a+') as f:
        f.write(os.path.join('CASME2', img_path+'\n'))
        new_dataset_val_img_list.append(os.path.join('CASME2', img_path+'\n'))


CASME2_test_image_label = np.zeros((TEST_numpy_list.shape[0], len(total_AUs))) - 1
for i, au in enumerate(au_ids):
    au = str(au)
    index = total_AUs.index(au)
    CASME2_test_image_label[:, index] = TEST_numpy_list[:, i]




np.savetxt(os.path.join(list_path_prefix,'CASME2_test_label.txt'),  CASME2_test_image_label, fmt='%d', delimiter=' ')

with open(os.path.join(list_path_prefix, 'CASME2_test_img_path.txt'), 'w+') as f:
    i=0
for img_path in test_img_path_list:
    with open(os.path.join(list_path_prefix,'CASME2_test_img_path.txt'), 'a+') as f:
        f.write(os.path.join('CASME2', img_path+'\n'))
        new_dataset_test_img_list.append(os.path.join('CASME2', img_path+'\n'))
new_dataset_test_label_list.append(CASME2_test_image_label)


#-------------------------------------------------------------------------------------------------------------------------------
# AFFW-2
print("processing AFFW-2------------------------------------------------------------")

au_ids  = ['1', '2' ,'4', '6', '7', '10', '12', '15', '23', '24', '25', '26']


list_path_prefix = 'Datasets/hybrid_dataset/AFFW-2/list'

train_path = 'TrainFromTrain_Set'
val_path = 'ValFromTrain_Set'
test_path = 'Validation_Set'

label_root = 'AFFW-2/list/AU_Set'

train_list = os.listdir(os.path.join(label_root, train_path))

train_labels = os.path.join(list_path_prefix, 'AFFW-2_train_label.txt')
with open(train_labels, 'w') as  f:
    i = 0

val_list = os.listdir(os.path.join(label_root, val_path))

val_labels = os.path.join(list_path_prefix, 'AFFW-2_val_label.txt')
with open(val_labels, 'w') as  f:
    i = 0

test_list = os.listdir(os.path.join(label_root, test_path))

test_labels = os.path.join(list_path_prefix, 'AFFW-2_test_label.txt')
with open(test_labels, 'w') as  f:
    i = 0


train_img_path = os.path.join(list_path_prefix, 'AFFW-2_train_img_path.txt')
with open(train_img_path, 'w') as f:
    i = 0
val_img_path = os.path.join(list_path_prefix, 'AFFW-2_val_img_path.txt')
with open(val_img_path, 'w') as f:
    i = 0
test_img_path = os.path.join(list_path_prefix, 'AFFW-2_test_img_path.txt')
with open(test_img_path, 'w') as f:
    i = 0




au_labels = []
au_img_path = []
for train_txt in train_list:
    with open(os.path.join(os.path.join(label_root, train_path), train_txt), 'r') as f:
        lines = f.readlines()
    lines = lines[1:]
    for j, line in enumerate(lines):
        line = line.rstrip('\n').split(',')
        line = np.array(line).astype(np.int32)
        if -1 in line:
            continue
        au_labels.append(line.reshape(1, -1))
        au_img_path.append(os.path.join(train_txt.split('.')[0], str(j+1).zfill(5)+'.jpg'))


au_labels = np.concatenate(au_labels, axis=0)
AFFW_train_image_label = np.zeros((au_labels.shape[0], len(total_AUs))) - 1
for i, au in enumerate(au_ids):
    au = str(au)
    index = total_AUs.index(au)
    AFFW_train_image_label[:, index] = au_labels[:, i]

with open(train_img_path, 'a+') as f:
    for line in au_img_path:
        f.write(os.path.join('AFFW',line+'\n'))
        new_dataset_train_img_list.append(os.path.join('AFFW',line+'\n'))

np.savetxt(train_labels, AFFW_train_image_label ,fmt='%d', delimiter=' ')
new_dataset_train_label_list.append(AFFW_train_image_label)




au_labels = []
au_img_path = []
for val_txt in val_list:
    with open(os.path.join(os.path.join(label_root, val_path), val_txt), 'r') as f:
        lines = f.readlines()
    lines = lines[1:]
    for j, line in enumerate(lines):
        line = line.rstrip('\n').split(',')
        line = np.array(line).astype(np.int32)
        if -1 in line:
            continue
        au_labels.append(line.reshape(1, -1))
        au_img_path.append(os.path.join(val_txt.split('.')[0], str(j+1).zfill(5)+'.jpg'))


au_labels = np.concatenate(au_labels, axis=0)
AFFW_val_image_label = np.zeros((au_labels.shape[0], len(total_AUs))) - 1
for i, au in enumerate(au_ids):
    au = str(au)
    index = total_AUs.index(au)
    AFFW_val_image_label[:, index] = au_labels[:, i]

with open(val_img_path, 'a+') as f:
    for line in au_img_path:
        f.write(os.path.join('AFFW',line+'\n'))
        new_dataset_val_img_list.append(os.path.join('AFFW',line+'\n'))

np.savetxt(val_labels, AFFW_val_image_label ,fmt='%d', delimiter=' ')
new_dataset_val_label_list.append(AFFW_val_image_label)





au_labels = []
au_img_path = []
for test_txt in test_list:
    with open(os.path.join(os.path.join(label_root, test_path), test_txt), 'r') as f:
        lines = f.readlines()
    lines = lines[1:]
    for j, line in enumerate(lines):
        line = line.rstrip('\n').split(',')
        line = np.array(line).astype(np.int32)
        if -1 in line:
            continue
        au_labels.append(line.reshape(1, -1))
        au_img_path.append(os.path.join(test_txt.split('.')[0], str(j+1).zfill(5)+'.jpg'))


au_labels = np.concatenate(au_labels, axis=0)
AFFW_test_image_label = np.zeros((au_labels.shape[0], len(total_AUs))) - 1
for i, au in enumerate(au_ids):
    au = str(au)
    index = total_AUs.index(au)
    AFFW_test_image_label[:, index] = au_labels[:, i]

with open(test_img_path, 'a+') as f:
    for line in au_img_path:
        f.write(os.path.join('AFFW',line+'\n'))
        new_dataset_test_img_list.append(os.path.join('AFFW',line+'\n'))

np.savetxt(test_labels, AFFW_test_image_label ,fmt='%d', delimiter=' ')
new_dataset_test_label_list.append(AFFW_test_image_label)


#
# print(len(new_dataset_train_img_list))
# print(len(new_dataset_val_img_list))
# print(len(new_dataset_test_img_list))

new_dataset_train_label_list = np.concatenate(new_dataset_train_label_list, axis=0)
new_dataset_val_label_list = np.concatenate(new_dataset_val_label_list, axis=0)
new_dataset_test_label_list = np.concatenate(new_dataset_test_label_list, axis=0)


sub_list = [0,1,2,4,7,8,11]

for i in range(new_dataset_train_label_list.shape[0]):
    for j in range(27, 41):
        sub_au_label = new_dataset_train_label_list[i, j]
        if sub_au_label >0:
            main_au_index = sub_list[ (j - 27) // 2]
            new_dataset_train_label_list[i, main_au_index] = 1


for i in range(new_dataset_val_label_list.shape[0]):
    for j in range(27, 41):
        sub_au_label = new_dataset_val_label_list[i, j]
        if sub_au_label >0:
            main_au_index = sub_list[ (j - 27) // 2]
            new_dataset_val_label_list[i, main_au_index] = 1

for i in range(new_dataset_test_label_list.shape[0]):
    for j in range(27, 41):
        sub_au_label = new_dataset_test_label_list[i, j]
        if sub_au_label >0:
            main_au_index = sub_list[ (j - 27) // 2]
            new_dataset_test_label_list[i, main_au_index] = 1

np.savetxt('Datasets/hybrid_dataset/list/hybrid_train_label.txt', new_dataset_train_label_list ,fmt='%d', delimiter=' ')
np.savetxt('Datasets/hybrid_dataset/list/hybrid_val_label.txt', new_dataset_val_label_list ,fmt='%d', delimiter=' ')
np.savetxt('Datasets/hybrid_dataset/list/hybrid_test_label.txt', new_dataset_test_label_list ,fmt='%d', delimiter=' ')

with open('Datasets/hybrid_dataset/list/hybrid_train_img_path.txt', 'w+') as f:
    for line in new_dataset_train_img_list:
        f.write(line)

with open('Datasets/hybrid_dataset/list/hybrid_val_img_path.txt', 'w+') as f:
    for line in new_dataset_val_img_list:
        f.write(line)

with open('Datasets/hybrid_dataset/list/hybrid_test_img_path.txt', 'w+') as f:
    for line in new_dataset_test_img_list:
        f.write(line)

# print(new_dataset_train_label_list.shape)
# print(new_dataset_val_label_list.shape)
# print(new_dataset_test_label_list.shape)

# new_dataset_train_label_list[new_dataset_train_label_list==-1] = 0
# new_dataset_val_label_list[new_dataset_val_label_list==-1] = 0
# new_dataset_test_label_list[new_dataset_test_label_list==-1] = 0
#
# print(new_dataset_train_label_list.sum(0))
# print(new_dataset_val_label_list.sum(0))
# print(new_dataset_test_label_list.sum(0))
