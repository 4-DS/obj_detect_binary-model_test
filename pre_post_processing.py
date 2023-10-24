import numpy as np
import cv2
from PIL import Image


class PrePostProcessing:
    def __init__(self, mean=None, std=None):
        self.input_size = (640, 640)
        self.mean = mean if mean else np.array([123.675, 116.28, 103.53])
        self.std = std if std else np.array([58.395, 57.12, 57.375]) 
    
    def prep_processing(self, file_stream):        
        pil_img=Image.open(file_stream)
        image_array = np.asarray(pil_img)
        resized = cv2.resize(image_array, self.input_size).astype(np.float32)
        scale_y, scale_x = self.input_size[0]/image_array.shape[0], self.input_size[1]/image_array.shape[1]
        scale_factors = [scale_x, scale_y]*2
        resized -= self.mean
        resized /= self.std
        input_data = resized.transpose(2,0,1)[np.newaxis, ...].astype(np.float32)
        return input_data, scale_factors
    
    def post_processing(self, output_data, scale_factors):        
        scale_x, scale_y = scale_factors[:2]
        dets, labels = output_data
        dets = dets*np.array(scale_factors + [1.0])
        return dets, labels 