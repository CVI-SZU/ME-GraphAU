# OpenGrpahAU



This repo is the OpenGprahAU tool.

41 action unit categories,

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
