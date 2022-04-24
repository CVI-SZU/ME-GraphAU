import os
import numpy as np

#You nead downloading DISFA including 'ActionUnit_Labels'
label_path = '../data/DISFA/ActionUnit_Labels'
list_path_prefix = '../data/DISFA/list/'

part1 = ['SN002','SN010','SN001','SN026','SN027','SN032','SN030','SN009','SN016']
part2 = ['SN013','SN018','SN011','SN028','SN012','SN006','SN031','SN021','SN024']
part3 = ['SN003','SN029','SN023','SN025','SN008','SN005','SN007','SN017','SN004']

# fold1:  train : part1+part2 test: part3
# fold2:  train : part1+part3 test: part2
# fold3:  train : part2+part3 test: part1

au_idx = [1, 2, 4, 6, 9, 12, 25, 26]


with open(list_path_prefix + 'DISFA_test_img_path_fold3.txt','w') as f:
    u = 0

part1_frame_list = []
part1_numpy_list = []
for fr in part1:
    fr_path = os.path.join(label_path,fr)
    au1_path = os.path.join(fr_path,fr+'_au1.txt')
    with open(au1_path, 'r') as label:
        total_frame = len(label.readlines())
    au_label_array = np.zeros((total_frame,8),dtype=np.int)
    for ai, au in enumerate(au_idx):
        AULabel_path = os.path.join(fr_path,fr+'_au'+str(au) +'.txt')
        if not os.path.isfile(AULabel_path):
            continue
        print("--Checking AU:" + str(au) + " ...")
        with open(AULabel_path, 'r') as label:
            for t, lines in enumerate(label.readlines()):
                frameIdx, AUIntensity = lines.split(',')
                frameIdx, AUIntensity = int(frameIdx), int(AUIntensity)
                if AUIntensity >= 2:
                    AUIntensity = 1
                else:
                    AUIntensity = 0
                au_label_array[t,ai] = AUIntensity
    for i in range(total_frame):
        frame_img_name = fr + '/' + str(i) + '.png'
        part1_frame_list.append(frame_img_name)
        with open(list_path_prefix + 'DISFA_test_img_path_fold3.txt', 'a+') as f:
            f.write(frame_img_name+'\n')
    part1_numpy_list.append(au_label_array)

part1_numpy_list = np.concatenate(part1_numpy_list,axis=0)
# part1 test for fold3
np.savetxt(list_path_prefix + 'DISFA_test_label_fold3.txt', part1_numpy_list,fmt='%d', delimiter=' ')

#################################################################################
with open(list_path_prefix + 'DISFA_test_img_path_fold2.txt','w') as f:
    u =0

part2_frame_list = []
part2_numpy_list = []
for fr in part2:
    fr_path = os.path.join(label_path,fr)
    au1_path = os.path.join(fr_path,fr+'_au1.txt')
    with open(au1_path, 'r') as label:
        total_frame = len(label.readlines())
    au_label_array = np.zeros((total_frame,8),dtype=np.int)
    for ai, au in enumerate(au_idx):
        AULabel_path = os.path.join(fr_path,fr+'_au'+str(au) +'.txt')
        if not os.path.isfile(AULabel_path):
            continue
        print("--Checking AU:" + str(au) + " ...")
        with open(AULabel_path, 'r') as label:
            for t, lines in enumerate(label.readlines()):
                frameIdx, AUIntensity = lines.split(',')
                frameIdx, AUIntensity = int(frameIdx), int(AUIntensity)
                if AUIntensity >= 2:
                    AUIntensity = 1
                else:
                    AUIntensity = 0
                au_label_array[t,ai] = AUIntensity
    for i in range(total_frame):
        frame_img_name = fr + '/' + str(i) + '.png'
        part2_frame_list.append(frame_img_name)
        with open(list_path_prefix + 'DISFA_test_img_path_fold2.txt', 'a+') as f:
            f.write(frame_img_name + '\n')
    part2_numpy_list.append(au_label_array)

part2_numpy_list = np.concatenate(part2_numpy_list,axis=0)
# part2 test for fold2
np.savetxt(list_path_prefix + 'DISFA_test_label_fold2.txt', part2_numpy_list,fmt='%d')

#################################################################################
with open(list_path_prefix + 'DISFA_test_img_path_fold1.txt','w') as f:
    u =0

part3_frame_list = []
part3_numpy_list = []
for fr in part3:
    fr_path = os.path.join(label_path,fr)
    au1_path = os.path.join(fr_path,fr+'_au1.txt')
    with open(au1_path, 'r') as label:
        total_frame = len(label.readlines())
    au_label_array = np.zeros((total_frame,8),dtype=np.int)
    for ai, au in enumerate(au_idx):
        AULabel_path = os.path.join(fr_path,fr+'_au'+str(au) +'.txt')
        if not os.path.isfile(AULabel_path):
            continue
        print("--Checking AU:" + str(au) + " ...")
        with open(AULabel_path, 'r') as label:
            for t, lines in enumerate(label.readlines()):
                frameIdx, AUIntensity = lines.split(',')
                frameIdx, AUIntensity = int(frameIdx), int(AUIntensity)
                if AUIntensity >= 2:
                    AUIntensity = 1
                else:
                    AUIntensity = 0
                au_label_array[t,ai] = AUIntensity
    for i in range(total_frame):
        frame_img_name = fr + '/' + str(i) + '.png'
        part3_frame_list.append(frame_img_name)
        with open(list_path_prefix + 'DISFA_test_img_path_fold1.txt', 'a+') as f:
            f.write(frame_img_name + '\n')
    part3_numpy_list.append(au_label_array)

part3_numpy_list = np.concatenate(part3_numpy_list,axis=0)
# part3 test for fold1
np.savetxt(list_path_prefix + 'DISFA_test_label_fold1.txt', part3_numpy_list,fmt='%d')

#################################################################################
with open(list_path_prefix + 'DISFA_train_img_path_fold1.txt','w') as f:
    u = 0
train_img_label_fold1_list = part1_frame_list + part2_frame_list
for frame_img_name in train_img_label_fold1_list:
	with open(list_path_prefix + 'DISFA_train_img_path_fold1.txt', 'a+') as f:
		f.write(frame_img_name + '\n')
train_img_label_fold1_numpy_list = np.concatenate((part1_numpy_list, part2_numpy_list), axis=0)
np.savetxt(list_path_prefix + 'DISFA_train_label_fold1.txt', train_img_label_fold1_numpy_list, fmt='%d')

#################################################################################
with open(list_path_prefix + 'DISFA_train_img_path_fold2.txt','w') as f:
    u = 0
train_img_label_fold2_list = part1_frame_list + part3_frame_list
for frame_img_name in train_img_label_fold2_list:
	with open(list_path_prefix + 'DISFA_train_img_path_fold2.txt', 'a+') as f:
		f.write(frame_img_name + '\n')
train_img_label_fold2_numpy_list = np.concatenate((part1_numpy_list, part3_numpy_list), axis=0)
np.savetxt(list_path_prefix + 'DISFA_train_label_fold2.txt', train_img_label_fold2_numpy_list, fmt='%d')

#################################################################################
with open(list_path_prefix + 'DISFA_train_img_path_fold3.txt','w') as f:
    u = 0
train_img_label_fold3_list = part2_frame_list + part3_frame_list
for frame_img_name in train_img_label_fold3_list:
	with open(list_path_prefix + 'DISFA_train_img_path_fold3.txt', 'a+') as f:
		f.write(frame_img_name + '\n')
train_img_label_fold3_numpy_list = np.concatenate((part2_numpy_list, part3_numpy_list), axis=0)
np.savetxt(list_path_prefix + 'DISFA_train_label_fold3.txt', train_img_label_fold3_numpy_list, fmt='%d')



