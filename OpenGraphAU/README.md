# OpenGrpahAU



This repo is the OpenGprahAU tool.

| AU1 | AU2 | AU4 | AU5 | AU6 | AU7 | AU9 |   AU10 | AU11  | AU12 | AU13 | AU14 | AU15 | AU16 | 
| AU17 | AU18 | AU19 | AU20  | AU22 | AU23 | AU24 | AU25 | AU26 | AU27 | AU32 | AU38 | AU39 | - |

| :-------: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|   EAC-Net  | 39.0 | 35.2 | 48.6 | 76.1 | 72.9 | 81.9 | 86.2 | 58.8 | 37.5 | 59.1 |  35.9 | 35.8 | 55.9 |
|   JAA-Net  |  47.2 | 44.0 |54.9 |77.5 |74.6 |84.0 |86.9 |61.9 |43.6 |60.3 |42.7 |41.9 |60.0|
|   LP-Net |  43.4  | 38.0  | 54.2  | 77.1  | 76.7  | 83.8  | 87.2  |63.3  |45.3  |60.5  |48.1  |54.2  |61.0|
|   ARL | 45.8 |39.8 |55.1 |75.7 |77.2 |82.3 |86.6 |58.8 |47.6 |62.1 |47.4 |55.4 |61.1|
|   SEV-Net | 58.2 |50.4 |58.3 |81.9 |73.9 |87.8 |87.5 |61.6 |52.6 |62.2 |44.6 |47.6 |63.9|
|   FAUDT | 51.7 |49.3 |61.0 |77.8 |79.5 |82.9 |86.3 |67.6 |51.9 |63.0 |43.7 |56.3 |64.2 |
|   SRERL | 46.9 |45.3 |55.6 |77.1 |78.4 |83.5 |87.6 |63.9 |52.2 |63.9  |47.1 |53.3 |62.9 |
|   UGN-B | 54.2  |46.4  |56.8  |76.2  |76.7  |82.4  |86.1  |64.7  |51.2  |63.1  |48.5  |53.6  |63.3 |
|   HMP-PS | 53.1 |46.1 |56.0 |76.5 |76.9 |82.1 |86.4 |64.8 |51.5 |63.0 |49.9 | 54.5  |63.4 |
|   Ours (ResNet-50) | 53.7 |46.9 |59.0 |78.5 |80.0 |84.4 |87.8 |67.3 |52.5 |63.2 |50.6 |52.4 |64.7 |
|   Ours (Swin-B) | 52.7 |44.3 |60.9 |79.9 |80.1| 85.3 |89.2| 69.4| 55.4| 64.4| 49.8 |55.1 |65.5|


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
