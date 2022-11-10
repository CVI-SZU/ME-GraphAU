# OpenGrpahAU



This repo is the OpenGprahAU tool.

| AU1 | AU2 | AU4 | AU5 | AU6 | AU7 | AU9 |   AU10 | AU11  | AU12 | AU13 | AU14 | AU15 | AU16 | 
| :---:  | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Inner brow raiser| Outer brow raiser
![image](https://user-images.githubusercontent.com/35754447/201079730-c175a5ee-3d8b-440d-80b1-491c1abdf74a.png)
 | AU19 | AU20  | AU22 | AU23 | AU24 | AU25 | AU26 | AU27 | AU32 | AU38 | AU39 | - |
| AU17 | AU18 | AU19 | AU20  | AU22 | AU23 | AU24 | AU25 | AU26 | AU27 | AU32 | AU38 | AU39 | - |
| :---:  | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |



41 action unit categories, ('Outer brow raiser',
        'Brow lowerer',
        'Upper lid raiser',
        'Cheek raiser',
        'Lid tightener',
        'Nose wrinkler',
        'Upper lip raiser',
        'Nasolabial deepener',
        'Lip corner puller',
        'Sharp lip puller',
        'Dimpler',
        'Lip corner depressor',
        'Lower lip depressor',
        'Chin raiser',
        'Lip pucker',
        'Tongue show',
        'Lip stretcher',
        'Lip funneler',
        'Lip tightener',
        'Lip pressor',
        'Lips part',
        'Jaw drop',
        'Mouth stretch',
        'Lip bite',
        'Nostril dilator',
        'Nostril compressor',
        'Left Inner brow raiser',
        'Right Inner brow raiser',
        'Left Outer brow raiser',
        'Right Outer brow raiser',
        'Left Brow lowerer',
        'Right Brow lowerer',
        'Left Cheek raiser',
        'Right Cheek raiser',
        'Left Upper lip raiser',
        'Right Upper lip raiser',
        'Left Nasolabial deepener',
        'Right Nasolabial deepener',
        'Left Dimpler',
        'Right Dimpler'])

pretraiend on hybrid dataset of 2,000k images.

demo:
<p align="center">
<img src="demo_imgs/1014_pred.jpg" width="70%" />
</p>


## Pretrained models

Hybrid Dataset

### Stage1:

|arch_type|GoogleDrive link| Average F1-score|
| :--- | :---: |  :---: |
|`Ours (ResNet-18)`| -| - |
|`Ours (ResNet-50)`| [link](https://drive.google.com/file/d/11xh9r2e4qCpWEtQ-ptJGWut_TQ0_AmSp/view?usp=share_link) | - |
|`Ours (ResNet-101)`| - | -  |
|`Ours (Swin-Tiny)`| - | - |
|`Ours (Swin-Small)`| - | - |
|`Ours (Swin-Base)`| - | - |


### Stage2:

|arch_type|GoogleDrive link| Average F1-score|
| :--- | :---: |  :---: |
|`Ours (ResNet-18)`| -| - |
|`Ours (ResNet-50)`| - | - |
|`Ours (ResNet-101)`| - | -  |
|`Ours (Swin-Tiny)`| - | - |
|`Ours (Swin-Small)`| - | - |
|`Ours (Swin-Base)`| - | - |



## Demo
- to detect facial action units in an image, run:
```
python demo.py --arc resnet50 --exp-name demo --resume checkpoints/OpenGprahAU-resent50.pth --input demo_imgs/1014.jpg
```
