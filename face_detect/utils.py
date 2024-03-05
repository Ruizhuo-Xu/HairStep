import os

import cv2
import numpy as np
from insightface.app import FaceAnalysis
 
def detect_and_crop_faces(image_path, output_path, scale=1.5):
    print("Crop Cur GPU ID: " + os.environ.get('CUDA_VISIBLE_DEVICES'))
    app = FaceAnalysis(name='buffalo_l')   # 使用的检测模型名为buffalo_sc
    app.prepare(ctx_id=0, det_size=(640, 640))  # ctx_id小于0表示用cpu预测，det_size表示resize后的图片分辨率  
    
    img = cv2.imread(image_path)  # 读取图片
    img_h, img_w = img.shape[:-1]
    faces = app.get(img)   # 得到人脸信息
    bbox = faces[0]['bbox']
    x1, y1 = bbox[0], bbox[1]
    x2, y2 = bbox[2], bbox[3]
    new_x1 = max(0, int(x1 - 0.5 * (scale - 1) * (x2 - x1)))
    new_y1 = max(0, int(y1 - 0.5 * (scale - 1) * (y2 - y1)))
    new_x2 = min(img_w, int(x2 + 0.5 * (scale - 1) * (x2 - x1)))
    new_y2 = min(img_h, int(y2 + 0.5 * (scale - 1) * (y2 - y1)))
    face_roi = img[new_y1:new_y2, new_x1:new_x2]

    return cv2.imwrite(output_path, face_roi)


if __name__ == '__main__':
    image_path = '/workspace/face_detect/temp_images/uploaded_image.png'
    output_path = '/workspace/results/real_imgs/img/output.png'
    detect_and_crop_faces(image_path, output_path, scale=2.0)