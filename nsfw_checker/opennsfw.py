import cv2
import torch
import onnx
import onnxruntime
import numpy as np
from tqdm import tqdm

# https://github.com/yahoo/open_nsfw

class NSFWChecker:
    def __init__(self, model_path=None, providers=["CPUExecutionProvider"]):
        model = onnx.load(model_path)
        self.input_name = model.graph.input[0].name
        session_options = onnxruntime.SessionOptions()
        self.session = onnxruntime.InferenceSession(model_path, sess_options=session_options, providers=providers)

    def is_nsfw(self, img_paths, threshold = 0.85):
        return False
