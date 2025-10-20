import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

# 预测框粗细和颜色修改问题可看<使用说明.md>下方的<YOLOV8源码常见疑问解答小课堂>第六点

if __name__ == '__main__':
    model = RTDETR('') # select your model.pt path
    model.predict(source='',
                  conf=0.30,
                  project='runs/detect',
                  name='', # set the name of the output folder
                  save=True,
                  # visualize=True # visualize model features maps
                  line_width=2, # line width of the bounding boxes
                  show_conf=True, # do not show prediction confidence
                  show_labels=True, # do not show prediction labels
                  # save_txt=True, # save results as .txt file
                  # save_crop=True, # save cropped images with results
                  )