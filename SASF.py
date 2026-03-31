import cv2
import numpy as np
import os
from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name
modelSASF = AntiSpoofPredict(0)
image_cropper = CropImage()
WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")
FINETUNED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "finetuned_weights")

class aSASF:
    def __init__(self,threshold=0.085):
        self.threshold=threshold
    def __call__(self,image,bbox,landmarks): # image RGB 
        model_probs = []
        img =[0,0]
        base_models = ['2.7_80x80_MiniFASNetV2.pth','4_0_0_80x80_MiniFASNetV1SE.pth']
        bbox = np.asarray(bbox, dtype=float).copy()
        # Detector provides xyxy; keep a fallback for xywh-style boxes.
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            bbox[2] = bbox[0] + max(0.0, bbox[2])
            bbox[3] = bbox[1] + max(0.0, bbox[3])
        bbox = bbox.tolist()
        imageBGR = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
        for index,model_name in enumerate(base_models):
            h_input, w_input, model_type, scale = parse_model_name(model_name)
            param = {"org_img": imageBGR,"bbox": bbox,"scale": scale,"out_w": w_input,"out_h": h_input,"crop": True}
            img[index] = image_cropper.crop(**param)
            finetuned_name = model_name.replace(".pth", "_finetuned.pth")
            finetuned_path = os.path.join(FINETUNED_DIR, finetuned_name)
            if os.path.exists(finetuned_path):
                model_path = finetuned_path
            else:
                model_path = os.path.join(WEIGHTS_DIR, model_name)
            pred=modelSASF.predict(img[index], model_path)
            probs = np.asarray(pred).reshape(-1)
            if probs.size < 3:
                raise ValueError(f"Unexpected SASF output shape from {model_name}: {np.asarray(pred).shape}")
            model_probs.append(probs)

        # model_probs[i] is [spoof_low, real, spoof_high]
        # We combine the spoof classes (index 0 and 2) and average across models
        prob_spoof_m1 = float(model_probs[0][0] + model_probs[0][2])
        prob_spoof_m2 = float(model_probs[1][0] + model_probs[1][2])
        prob_spoof = (prob_spoof_m1 + prob_spoof_m2) / 2.0
        
        spoof      = int(prob_spoof>self.threshold)
        return spoof,prob_spoof,img[0]
