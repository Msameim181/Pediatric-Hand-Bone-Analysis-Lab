# Pediatric-Hand-Bone-Analysis-Lab
This is a revision of the RSNA Hand dataset which includes Hand Detection, Pose Estimation, and Instance Segmentation.



## Installations

### Install PyTorch and MMCV 
```shell
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```

```shell
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
```

### Install MMDetection
#### Install from source

```shell
rm -rf mmdetection
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -e .
```
Clean workspace
```shell
cd ..
rm -rf mmdetection
```
#### Install from pip
```shell
pip install mmdet
```
