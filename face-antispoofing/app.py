import logging
import os
import time
import sys
import cv2
import numpy as np
import gradio as gr
import IADG
import SASF

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    ModelD = IADG.aFaceDetect()
    Model1 = SASF.aSASF(threshold=0.0094)
    Model2 = IADG.aSpoofONNX('modelrgb', threshold=0.0553)
    Model3 = IADG.aSpoof('ICM2O', threshold=0.9980)
    Model4 = IADG.aSpoof('IOM2C', threshold=0.9944)
except Exception as e:
    logger.error(f"Error loading models: {e}")
    ModelD = None
    Model1 = None
    Model2 = None
    Model3 = None
    Model4 = None


def prob(p, thre):
    if p < thre:
        return (thre - p) / thre
    return (p - thre) / (1 - thre)


def run_image(input_image, text):  # input_image - RGB
    if input_image is None or not hasattr(input_image, "shape"):
        return None, None, "Please upload an image first."
    
    if ModelD is None:
        return None, None, "Models failed to load. Ensure the 'weights' folder is present in the Space."

    try:
        thre = [float(v.strip()) for v in text.split('\n') if v.strip()]
        if len(thre) != 4:
            return None, None, "Enter exactly 4 threshold values (one per line)."
    except (ValueError, AttributeError):
        return None, None, "Threshold values must be numeric (one per line)."

    Model1.threshold = thre[0]
    Model2.threshold = thre[1]
    Model3.threshold = thre[2]
    Model4.threshold = thre[3]
    bboxes, landmarks = ModelD(input_image)
    if len(landmarks) < 1:
        return input_image, input_image, 'No face detected or multiple faces detected'

    spoof1, spoof_prob1, img1 = Model1(input_image, bboxes[0], landmarks[0])
    spoof2, spoof_prob2, img2 = Model2(input_image, bboxes[0], landmarks[0])
    spoof3, spoof_prob3, img3 = Model3(input_image, bboxes[0], landmarks[0])
    spoof4, spoof_prob4, img4 = Model4(input_image, bboxes[0], landmarks[0])

    names = ['Real photo', 'Spoof']
    text = f'SASF :\t P={spoof_prob1:.4f} ({names[spoof1]}). Confidence: {prob(spoof_prob1, Model1.threshold)}\n'
    text += f'FLRGB:\t P={spoof_prob2:.4f} ({names[spoof2]}). Confidence: {prob(spoof_prob2, Model2.threshold)}\n'
    text += f'ICM2O:\t P={spoof_prob3:.4f} ({names[spoof3]}). Confidence: {prob(spoof_prob3, Model3.threshold)}\n'
    text += f'IOM2C:\t P={spoof_prob4:.4f} ({names[spoof4]}). Confidence: {prob(spoof_prob4, Model4.threshold)}\n'
    return img2, img3, text


def demo():
    with gr.Blocks(title='Face anti-spoofing detector test. Version 2.') as demo:
        with gr.Row():
            with gr.Column():
                gr.Markdown('<h1><center>Face anti-spoofing detector testing</center></h1>')
                gr.Markdown('''
* Model 1. Silent-Face-Anti-Spoofing (SASF). <https://github.com/minivision-ai/Silent-Face-Anti-Spoofing>
* Model 2. Face Liveness Detection Model-RGB (**FLRGB**). <https://modelscope.cn/models/iic/cv_manual_face-liveness_flrgb>
* Model 3. Instance-Aware Domain Generalization for Face Anti-Spoofing (**ICM2O**). <https://openaccess.thecvf.com/content/CVPR2023/papers/Zhou_Instance-Aware_Domain_Generalization_for_Face_Anti-Spoofing_CVPR_2023_paper.pdf>
* Model 4. Instance-Aware Domain Generalization for Face Anti-Spoofing (**IOM2C**). <https://openaccess.thecvf.com/content/CVPR2023/papers/Zhou_Instance-Aware_Domain_Generalization_for_Face_Anti-Spoofing_CVPR_2023_paper.pdf>
* Result interpretation: the further P is above the threshold, the more confident the model is that the image is a spoof.
* Threshold values should be selected during testing.
''')

        with gr.Row():
            with gr.Column():
                with gr.Row():
                    image = gr.Image(type='numpy', sources=['upload', 'webcam'], label='Spoof or real human face photo')
                with gr.Row():
                    input_text = gr.Textbox(label='Threshold values', value='0.0094\n0.2808\n0.9980\n0.9944', lines=4)
                with gr.Row():
                    submit = gr.Button('Run prediction')
                    clear = gr.Button('Clear')
            with gr.Column():
                with gr.Row():
                    with gr.Column():
                        output_image1 = gr.Image(type='numpy', label='Input image 1')
                    with gr.Column():
                        output_image2 = gr.Image(type='numpy', label='Input image 2')
                with gr.Row():
                    output_text = gr.TextArea(label='Prediction results', lines=6, value='')

        submit.click(run_image, [image, input_text], [output_image1, output_image2, output_text])
        clear.click(lambda: [None] * 4, None, [image, output_image1, output_image2, output_text])
        demo.launch()


if __name__ == '__main__':
    demo()
