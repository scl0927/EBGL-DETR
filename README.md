# EBGL-DETR, The Visual Computer

The corresponding paper title for this project is “Enhanced UAV Small Object Detection via Entangled Transformer with Spatial Aggregation”.

# Installation

```
$ git clone https://github.com/scl0927/EBGL-DETR.git
$ cd EBGL-DETR-master
$ conda create -n EBGL-DETR python=3.10
$ conda activate EBGL-DETR
$ pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
$ pip install -r requirements.txt
```
<summary>Install</summary>

[**Python>=3.8.0**](https://www.python.org/) is required with all
[requirements.txt](https://github.com/lyuwenyu/RT-DETR/blob/main/rtdetr_pytorch/requirements.txt) installed including
[**PyTorch>=1.8**](https://pytorch.org/get-started/locally/):

# Train
Single GPU training
```python
import warnings, os
warnings.filterwarnings('ignore')
from ultralytics import RTDETR
if __name__ == '__main__':
    model = RTDETR('ultralytics/cfg/models/rt-detr/EBGL-DETR.yaml')
    model.train(data='dataset/VisDrone2019.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=8,
                workers=4,
                # device='0,1', 
                # resume='', 
                project='runs/train',
                name='exp',
                )
```
```bash
$ python train.py
```

# Val
```python
import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR
#用于打印训练结果的指标
if __name__ == '__main__':
    model = RTDETR('runs/train/exp/weights/best.pt')
    model.val(data='dataset/data.yaml',
              split='val',
              imgsz=640,
              batch=8,
            #   save_json=True, # if you need to cal coco metrice
              project='runs/val',
              name='exp',
              )
```
```bash
$ python val.py
```

## 📊 Statistical Validation

To ensure the robustness of our results, we conducted multiple independent training runs
and report the **mean ± standard deviation** of all evaluation metrics.

| Dataset       | mAP⁵⁰ₜₑₛₜ (Mean ± Std) |
|:-------------:|:-----------------------:|
| VisDrone2019  | 42.0 ± 0.3              |
| AI-TOD        | 53.7 ± 0.4              |
| TinyPerson    | 26.4 ± 0.4              |
| NWPU VHR-10   | 89.1 ± 0.2              |

The small standard deviations confirm that **EBGL-DETR** achieves consistent and stable
performance across multiple independent runs.

> **Note:** Due to space limitations, these detailed statistical results are not included
> in the main manuscript. All results will be made publicly available in this repository
> to ensure full reproducibility and transparency.

# Datasets
*[VisDrone](https://github.com/VisDrone/VisDrone-Dataset)
*[AI-TOD](https://github.com/jwwangchn/AI-TOD)
*[TinyPerson](https://github.com/xixu-me/YOLO-TinyPerson)
*[NWPU-VHR 10](https://github.com/Gaoshuaikun/NWPU-VHR-10)

dataset/
├── images/
│   ├── train/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── val/
│   │   ├── image1001.jpg
│   │   ├── image1002.jpg
│   │   └── ...
│   └── test/       
├── labels/
│   ├── train/
│   │   ├── image1.txt
│   │   ├── image2.txt
│   │   └── ...
│   ├── val/
│   │   ├── image1001.txt
│   │   ├── image1002.txt
│   │   └── ...
│   └── test/       
└── dataset.yaml    
