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