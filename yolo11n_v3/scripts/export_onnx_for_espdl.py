import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.nn.modules.head import Detect
import os
from pathlib import Path

# 路径配置（通过环境变量覆盖，默认使用相对路径）
CCPD_DIR = os.environ.get("CCPD_DIR", "./data/CCPD2020")
CCPD_AUG_DIR = os.environ.get("CCPD_AUG_DIR", "./data/CCPD2020_augmented")
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "./outputs"))
PRETRAINED_WEIGHTS = os.environ.get("PRETRAINED_WEIGHTS", "./pretrained/yolo11n.pt")

class DetectESPDL(Detect):
    def forward(self, x):
        if self.training:
            for i in range(self.nl):
                x[i] = self.cv2[i](x[i])
                x[i] = torch.cat([self.cv3[i](x[i]), x[i]], 1)
            return x
        
        outputs = []
        for i in range(self.nl):
            bbox = self.cv2[i](x[i])
            score = self.cv3[i](x[i])
            outputs.append(bbox)
            outputs.append(score)
        
        return tuple(outputs)

def export_yolo11_for_espdl(model_path, output_path, imgsz=320):
    print(f"Loading model from: {model_path}")
    
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    model = checkpoint['model']
    
    print(f"Model info:")
    print(f"  - Number of classes: {model.nc}")
    print(f"  - Class names: {model.names}")
    
    detect_layer = None
    detect_idx = None
    for i, m in enumerate(model.model):
        if isinstance(m, Detect):
            detect_layer = m
            detect_idx = i
            print(f"\nFound Detect layer at index {i}")
            print(f"  nc: {m.nc}")
            print(f"  nl: {m.nl}")
            print(f"  reg_max: {m.reg_max}")
            print(f"  training: {m.training}")
            break
    
    if detect_layer is None:
        raise ValueError("Detect layer not found")
    
    model.eval()
    model.float()
    
    print(f"\nAfter model.eval():")
    print(f"  Model training: {model.training}")
    print(f"  Detect layer training: {detect_layer.training}")
    
    print("\nTesting original model...")
    dummy_input = torch.randn(1, 3, imgsz, imgsz)
    with torch.no_grad():
        orig_output = model(dummy_input)
        print(f"Original output type: {type(orig_output)}")
        if isinstance(orig_output, (list, tuple)):
            for i, o in enumerate(orig_output):
                if isinstance(o, torch.Tensor):
                    print(f"  Output {i}: Tensor shape = {o.shape}")
    
    print("\nCreating ESPDL Detect layer...")
    input_channels = []
    for i in range(detect_layer.nl):
        cv2_first_conv = detect_layer.cv2[i][0]
        if hasattr(cv2_first_conv, 'conv'):
            in_ch = cv2_first_conv.conv.in_channels
        else:
            in_ch = cv2_first_conv.in_channels
        input_channels.append(in_ch)
        print(f"  Layer {i} input channels: {in_ch}")
    
    detect_espdl = DetectESPDL(
        nc=detect_layer.nc,
        ch=input_channels
    )
    
    detect_espdl.cv2 = detect_layer.cv2
    detect_espdl.cv3 = detect_layer.cv3
    detect_espdl.dfl = detect_layer.dfl
    detect_espdl.nl = detect_layer.nl
    detect_espdl.no = detect_layer.no
    detect_espdl.reg_max = detect_layer.reg_max
    detect_espdl.stride = detect_layer.stride
    
    if hasattr(detect_layer, 'na'):
        detect_espdl.na = detect_layer.na
    
    if hasattr(detect_layer, 'f'):
        detect_espdl.f = detect_layer.f
    else:
        detect_espdl.f = -1
    
    if hasattr(detect_layer, 'i'):
        detect_espdl.i = detect_layer.i
    else:
        detect_espdl.i = detect_idx
    
    detect_espdl.eval()
    
    print(f"\nESPDL Detect layer training: {detect_espdl.training}")
    
    model.model[detect_idx] = detect_espdl
    
    model.eval()
    
    print(f"\nAfter replacement, model training: {model.training}")
    print(f"Detect layer training: {model.model[detect_idx].training}")
    
    print("\nTesting modified model...")
    with torch.no_grad():
        new_output = model(dummy_input)
        print(f"New output type: {type(new_output)}")
        if isinstance(new_output, (list, tuple)):
            for i, o in enumerate(new_output):
                if isinstance(o, torch.Tensor):
                    print(f"  Output {i}: Tensor shape = {o.shape}")
    
    output_names = []
    for i in range(3):
        output_names.append(f'box{i}')
        output_names.append(f'score{i}')
    
    print("\nExporting to ONNX...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=['images'],
        output_names=output_names,
        dynamic_axes=None,
        opset_version=11,
        do_constant_folding=True,
    )
    
    print(f"\nONNX model exported to: {output_path}")
    print(f"Output names: {output_names}")
    return output_path

if __name__ == "__main__":
    model_path = str(OUTPUT_DIR / "320x320_qat.pt")
    output_path = str(OUTPUT_DIR / "yolo11n_320x320_espdl.onnx")
    
    export_yolo11_for_espdl(model_path, output_path, imgsz=320)
