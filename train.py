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