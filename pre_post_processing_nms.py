import numpy as np
import cv2
from PIL import Image
from yolox_nms_postprocessing import YOLOXPostProcessing


class PrePostProcessing:
    def __init__(self, mean=None, std=None, score_thr:float=0.01, iou_threshold:float=0.5, num_classes:int=1):
        self.score_thr = score_thr
        self.iou_threshold = iou_threshold
        self.input_size = (640, 640)
        self.mean = mean if mean else np.array([123.675, 116.28, 103.53])
        self.std = std if std else np.array([58.395, 57.12, 57.375]) 
        self.num_classes = num_classes
        self.yolox_postprocessing = YOLOXPostProcessing(strides = [(8, 8), (16, 16), (32, 32)],
                                                        offset = 0, 
                                                        num_levels = 3, 
                                                        cls_out_channels = num_classes,
                                                        num_classes = num_classes) 
        
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
        cls_scores = [output_data[i] for i in range(3)]
        bbox_preds = [output_data[i] for i in range(3, 6)]
        objectnesses =[output_data[i] for i in range(6, 9)]
        
        scale_factors = np.array([scale_factors])
        
        bbox_results = self.yolox_postprocessing.get_bboxes(cls_scores, 
                                                            bbox_preds, 
                                                            objectnesses, 
                                                            scale_factor=scale_factors, 
                                                            score_thr=self.score_thr, 
                                                            iou_threshold=self.iou_threshold)        
        return bbox_results  