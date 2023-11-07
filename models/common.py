# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Common modules
"""

import json
import math
import platform
import warnings
from collections import OrderedDict, namedtuple
from copy import copy
from pathlib import Path
from torchvision import transforms as T

import os
import cv2
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
import yaml
from PIL import Image
from torch.cuda import amp
import torch.nn.functional as F
from utils.datasets import exif_transpose, letterbox
from utils.general import (LOGGER, check_requirements, check_suffix, check_version, colorstr, increment_path,
                           make_divisible, non_max_suppression,skip_nms, scale_coords, xywh2xyxy, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import copy_attr, time_sync
from utils.downloads import attempt_download
from utils.loss import get_dissim, get_cossim, get_ssim
from einops import rearrange
from sklearn.cluster import KMeans
from utils.cnn import *
from tqdm import tqdm
from torchvision.ops import roi_pool,roi_align,RoIAlign,MultiScaleRoIAlign,RoIAlign
def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else (x // 2 for x in k)  # auto-pad
    return p

def init_weight(m):

    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
    elif isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight)


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.LeakyReLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        # self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

class DWConv(Conv):
    # Depth-wise convolution class
    def __init__(self, c1, c2, k=1, s=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), act=act)

class TransformerLayer(nn.Module):
    # Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)
    def __init__(self, c, num_heads):
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x):
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        x = self.fc2(self.fc1(x)) + x
        return x

class TransformerBlock(nn.Module):
    # Vision Transformer https://arxiv.org/abs/2010.11929
    def __init__(self, c1, c2, num_heads, num_layers):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.linear = nn.Linear(c2, c2)  # learnable position embedding
        self.tr = nn.Sequential(*(TransformerLayer(c2, num_heads) for _ in range(num_layers)))
        self.c2 = c2

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2).permute(2, 0, 1)
        return self.tr(p + self.linear(p)).permute(1, 2, 0).reshape(b, self.c2, w, h)

class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))

class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))
        # self.m = nn.Sequential(*(CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))

class C3TR(C3):
    # C3 module with TransformerBlock()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)

class C3SPP(C3):
    # C3 module with SPP()
    def __init__(self, c1, c2, k=(5, 9, 13), n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = SPP(c_, c_, k)

class C3Ghost(C3):
    # C3 module with GhostBottleneck()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))

class SPP(nn.Module):
    # Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))

class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))

class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)
        # self.contract = Contract(gain=2)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))
        # return self.conv(self.contract(x))

class GhostConv(nn.Module):
    # Ghost Convolution https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):  # ch_in, ch_out, kernel, stride, groups
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act)

    def forward(self, x):
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)

class GhostBottleneck(nn.Module):
    # Ghost Bottleneck https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=3, s=1):  # ch_in, ch_out, kernel, stride
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(GhostConv(c1, c_, 1, 1),  # pw
                                  DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
                                  GhostConv(c_, c2, 1, 1, act=False))  # pw-linear
        self.shortcut = nn.Sequential(DWConv(c1, c1, k, s, act=False),
                                      Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)

class Contract(nn.Module):
    # Contract width-height into channels, i.e. x(1,64,80,80) to x(1,256,40,40)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        b, c, h, w = x.size()  # assert (h / s == 0) and (W / s == 0), 'Indivisible gain'
        s = self.gain
        x = x.view(b, c, h // s, s, w // s, s)  # x(1,64,40,2,40,2)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # x(1,2,2,64,40,40)
        return x.view(b, c * s * s, h // s, w // s)  # x(1,256,40,40)

class Expand(nn.Module):
    # Expand channels into width-height, i.e. x(1,64,80,80) to x(1,16,160,160)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        b, c, h, w = x.size()  # assert C / s ** 2 == 0, 'Indivisible gain'
        s = self.gain
        x = x.view(b, s, s, c // s ** 2, h, w)  # x(1,2,2,16,80,80)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # x(1,16,80,2,80,2)
        return x.view(b, c // s ** 2, h * s, w * s)  # x(1,16,160,160)

class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)

class DetectMultiBackend(nn.Module):
    # YOLOv5 MultiBackend class for python inference on various backends
    def __init__(self, weights='yolov5s.pt', device=torch.device('cpu'), dnn=False, data=None, fp16=False):
        # Usage:
        #   PyTorch:              weights = *.pt
        #   TorchScript:                    *.torchscript
        #   ONNX Runtime:                   *.onnx
        #   ONNX OpenCV DNN:                *.onnx with --dnn
        #   OpenVINO:                       *.xml
        #   CoreML:                         *.mlmodel
        #   TensorRT:                       *.engine
        #   TensorFlow SavedModel:          *_saved_model
        #   TensorFlow GraphDef:            *.pb
        #   TensorFlow Lite:                *.tflite
        #   TensorFlow Edge TPU:            *_edgetpu.tflite
        from models.experimental import attempt_download, attempt_load  # scoped to avoid circular import

        super().__init__()
        w = str(weights[0] if isinstance(weights, list) else weights)
        pt, jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs = self.model_type(w)  # get backend
        stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults
        w = attempt_download(w)  # download if not local
        fp16 &= (pt or jit or onnx or engine) and device.type != 'cpu'  # FP16
        if data:  # data.yaml path (optional)
            with open(data, errors='ignore') as f:
                names = yaml.safe_load(f)['names']  # class names

        if pt:  # PyTorch
            model = attempt_load(weights if isinstance(weights, list) else w, map_location=device)
            stride = max(int(model.stride.max()), 32)  # model stride
            names = model.module.names if hasattr(model, 'module') else model.names  # get class names
            model.half() if fp16 else model.float()
            self.model = model  # explicitly assign for to(), cpu(), cuda(), half()
        elif jit:  # TorchScript
            LOGGER.info(f'Loading {w} for TorchScript inference...')
            extra_files = {'config.txt': ''}  # model metadata
            model = torch.jit.load(w, _extra_files=extra_files)
            model.half() if fp16 else model.float()
            if extra_files['config.txt']:
                d = json.loads(extra_files['config.txt'])  # extra_files dict
                stride, names = int(d['stride']), d['names']
        elif dnn:  # ONNX OpenCV DNN
            LOGGER.info(f'Loading {w} for ONNX OpenCV DNN inference...')
            check_requirements(('opencv-python>=4.5.4',))
            net = cv2.dnn.readNetFromONNX(w)
        elif onnx:  # ONNX Runtime
            LOGGER.info(f'Loading {w} for ONNX Runtime inference...')
            cuda = torch.cuda.is_available()
            check_requirements(('onnx', 'onnxruntime-gpu' if cuda else 'onnxruntime'))
            import onnxruntime
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
            session = onnxruntime.InferenceSession(w, providers=providers)
        elif xml:  # OpenVINO
            LOGGER.info(f'Loading {w} for OpenVINO inference...')
            check_requirements(('openvino-dev',))  # requires openvino-dev: https://pypi.org/project/openvino-dev/
            import openvino.inference_engine as ie
            core = ie.IECore()
            if not Path(w).is_file():  # if not *.xml
                w = next(Path(w).glob('*.xml'))  # get *.xml file from *_openvino_model dir
            network = core.read_network(model=w, weights=Path(w).with_suffix('.bin'))  # *.xml, *.bin paths
            executable_network = core.load_network(network, device_name='CPU', num_requests=1)
        elif engine:  # TensorRT
            LOGGER.info(f'Loading {w} for TensorRT inference...')
            import tensorrt as trt  # https://developer.nvidia.com/nvidia-tensorrt-download
            check_version(trt.__version__, '7.0.0', hard=True)  # require tensorrt>=7.0.0
            Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
            logger = trt.Logger(trt.Logger.INFO)
            with open(w, 'rb') as f, trt.Runtime(logger) as runtime:
                model = runtime.deserialize_cuda_engine(f.read())
            bindings = OrderedDict()
            fp16 = False  # default updated below
            for index in range(model.num_bindings):
                name = model.get_binding_name(index)
                dtype = trt.nptype(model.get_binding_dtype(index))
                shape = tuple(model.get_binding_shape(index))
                data = torch.from_numpy(np.empty(shape, dtype=np.dtype(dtype))).to(device)
                bindings[name] = Binding(name, dtype, shape, data, int(data.data_ptr()))
                if model.binding_is_input(index) and dtype == np.float16:
                    fp16 = True
            binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
            context = model.create_execution_context()
            batch_size = bindings['images'].shape[0]
        elif coreml:  # CoreML
            LOGGER.info(f'Loading {w} for CoreML inference...')
            import coremltools as ct
            model = ct.models.MLModel(w)
        else:  # TensorFlow (SavedModel, GraphDef, Lite, Edge TPU)
            if saved_model:  # SavedModel
                LOGGER.info(f'Loading {w} for TensorFlow SavedModel inference...')
                import tensorflow as tf
                keras = False  # assume TF1 saved_model
                model = tf.keras.models.load_model(w) if keras else tf.saved_model.load(w)
            elif pb:  # GraphDef https://www.tensorflow.org/guide/migrate#a_graphpb_or_graphpbtxt
                LOGGER.info(f'Loading {w} for TensorFlow GraphDef inference...')
                import tensorflow as tf

                def wrap_frozen_graph(gd, inputs, outputs):
                    x = tf.compat.v1.wrap_function(lambda: tf.compat.v1.import_graph_def(gd, name=""), [])  # wrapped
                    ge = x.graph.as_graph_element
                    return x.prune(tf.nest.map_structure(ge, inputs), tf.nest.map_structure(ge, outputs))

                gd = tf.Graph().as_graph_def()  # graph_def
                gd.ParseFromString(open(w, 'rb').read())
                frozen_func = wrap_frozen_graph(gd, inputs="x:0", outputs="Identity:0")
            elif tflite or edgetpu:  # https://www.tensorflow.org/lite/guide/python#install_tensorflow_lite_for_python
                try:  # https://coral.ai/docs/edgetpu/tflite-python/#update-existing-tf-lite-code-for-the-edge-tpu
                    from tflite_runtime.interpreter import Interpreter, load_delegate
                except ImportError:
                    import tensorflow as tf
                    Interpreter, load_delegate = tf.lite.Interpreter, tf.lite.experimental.load_delegate,
                if edgetpu:  # Edge TPU https://coral.ai/software/#edgetpu-runtime
                    LOGGER.info(f'Loading {w} for TensorFlow Lite Edge TPU inference...')
                    delegate = {'Linux': 'libedgetpu.so.1',
                                'Darwin': 'libedgetpu.1.dylib',
                                'Windows': 'edgetpu.dll'}[platform.system()]
                    interpreter = Interpreter(model_path=w, experimental_delegates=[load_delegate(delegate)])
                else:  # Lite
                    LOGGER.info(f'Loading {w} for TensorFlow Lite inference...')
                    interpreter = Interpreter(model_path=w)  # load TFLite model
                interpreter.allocate_tensors()  # allocate
                input_details = interpreter.get_input_details()  # inputs
                output_details = interpreter.get_output_details()  # outputs
            elif tfjs:
                raise Exception('ERROR: YOLOv5 TF.js inference is not supported')
        self.__dict__.update(locals())  # assign all variables to self

    def forward(self, im, augment=False, visualize=False, val=False):
        # YOLOv5 MultiBackend inference
        b, ch, h, w = im.shape  # batch, channel, height, width
        if self.pt or self.jit:  # PyTorch
            y = self.model(im) if self.jit else self.model(im, augment=augment, visualize=visualize)
            return y if val else y[0]
        elif self.dnn:  # ONNX OpenCV DNN
            im = im.cpu().numpy()  # torch to numpy
            self.net.setInput(im)
            y = self.net.forward()
        elif self.onnx:  # ONNX Runtime
            im = im.cpu().numpy()  # torch to numpy
            y = self.session.run([self.session.get_outputs()[0].name], {self.session.get_inputs()[0].name: im})[0]
        elif self.xml:  # OpenVINO
            im = im.cpu().numpy()  # FP32
            desc = self.ie.TensorDesc(precision='FP32', dims=im.shape, layout='NCHW')  # Tensor Description
            request = self.executable_network.requests[0]  # inference request
            request.set_blob(blob_name='images', blob=self.ie.Blob(desc, im))  # name=next(iter(request.input_blobs))
            request.infer()
            y = request.output_blobs['output'].buffer  # name=next(iter(request.output_blobs))
        elif self.engine:  # TensorRT
            assert im.shape == self.bindings['images'].shape, (im.shape, self.bindings['images'].shape)
            self.binding_addrs['images'] = int(im.data_ptr())
            self.context.execute_v2(list(self.binding_addrs.values()))
            y = self.bindings['output'].data
        elif self.coreml:  # CoreML
            im = im.permute(0, 2, 3, 1).cpu().numpy()  # torch BCHW to numpy BHWC shape(1,320,192,3)
            im = Image.fromarray((im[0] * 255).astype('uint8'))
            # im = im.resize((192, 320), Image.ANTIALIAS)
            y = self.model.predict({'image': im})  # coordinates are xywh normalized
            if 'confidence' in y:
                box = xywh2xyxy(y['coordinates'] * [[w, h, w, h]])  # xyxy pixels
                conf, cls = y['confidence'].max(1), y['confidence'].argmax(1).astype(np.float)
                y = np.concatenate((box, conf.reshape(-1, 1), cls.reshape(-1, 1)), 1)
            else:
                k = 'var_' + str(sorted(int(k.replace('var_', '')) for k in y)[-1])  # output key
                y = y[k]  # output
        else:  # TensorFlow (SavedModel, GraphDef, Lite, Edge TPU)
            im = im.permute(0, 2, 3, 1).cpu().numpy()  # torch BCHW to numpy BHWC shape(1,320,192,3)
            if self.saved_model:  # SavedModel
                y = (self.model(im, training=False) if self.keras else self.model(im)).numpy()
            elif self.pb:  # GraphDef
                y = self.frozen_func(x=self.tf.constant(im)).numpy()
            else:  # Lite or Edge TPU
                input, output = self.input_details[0], self.output_details[0]
                int8 = input['dtype'] == np.uint8  # is TFLite quantized uint8 model
                if int8:
                    scale, zero_point = input['quantization']
                    im = (im / scale + zero_point).astype(np.uint8)  # de-scale
                self.interpreter.set_tensor(input['index'], im)
                self.interpreter.invoke()
                y = self.interpreter.get_tensor(output['index'])
                if int8:
                    scale, zero_point = output['quantization']
                    y = (y.astype(np.float32) - zero_point) * scale  # re-scale
            y[..., :4] *= [w, h, w, h]  # xywh normalized to pixels

        if isinstance(y, np.ndarray):
            y = torch.tensor(y, device=self.device)
        return (y, []) if val else y

    def warmup(self, imgsz=(1, 3, 640, 640)):
        # Warmup model by running inference once
        if any((self.pt, self.jit, self.onnx, self.engine, self.saved_model, self.pb)):  # warmup types
            if self.device.type != 'cpu':  # only warmup GPU models
                im = torch.zeros(*imgsz, dtype=torch.half if self.fp16 else torch.float, device=self.device)  # input
                for _ in range(2 if self.jit else 1):  #
                    self.forward(im)  # warmup

    @staticmethod
    def model_type(p='path/to/model.pt'):
        # Return model type from model path, i.e. path='path/to/model.onnx' -> type=onnx
        from export import export_formats
        suffixes = list(export_formats().Suffix) + ['.xml']  # export suffixes
        check_suffix(p, suffixes)  # checks
        p = Path(p).name  # eliminate trailing separators
        pt, jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs, xml2 = (s in p for s in suffixes)
        xml |= xml2  # *_openvino_model or *.xml
        tflite &= not edgetpu  # *.tflite
        return pt, jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs

class DetectSTAN(nn.Module):

    def __init__(self, weights='last.pt', device=torch.device('cpu'), dnn=False, data=None, fp16=False):
        super().__init__()
        ckpt = torch.load(weights, map_location='cpu')
        model = ckpt['model']
        self.stride = max(int(model.stride.max()), 32)
        self.names = model.names
        self.model = model.to(device)
        self.model.model_t.eval()
        self.model.model_s.eval()
        self.fp16 = False
        self.maxdet = 300
        for param in self.model.model_t.parameters():
            param.requires_grad = False
        for param in self.model.model_s.parameters():
            param.requires_grad = False
        self.imgsize = (640,640)
    def forward(self, im, augment=False, visualize=False, val=False):
        # YOLOv5 MultiBackend inference
        b, ch, h, w = im.shape  # batch, channel, height, width
        if self.imgsize[0] != h or self.imgsize[1] != w:
            self.imgsize = (h,w)
        (out_all, out_t), ftlist, fslist = self.model(im)
        pred = non_max_suppression(out_all, 0.5, 0.50, None, False, max_det=self.maxdet)

        #dis_out, cos_out, ssim_out, area_out = self.anfeature_visualize(pred, ftlist, fslist)
        #dis_out, cos_out, ssim_out, area_out =  self.anfeature_visualize_crossscale(pred, ftlist, fslist)
        dis_out, cos_out, ssim_out, area_out = self.anfeature_visualize(out_t,ftlist,fslist)
        return pred, dis_out, cos_out, ssim_out, area_out

    def get_expand_boxes(self, scaled_boxes, ratio=1.5):
        exp_boxes = []
        for boxes in scaled_boxes:
            xyxy = boxes[:, :4].clamp(min=0.0, max=1.0)
            cxcy = (xyxy[:, [0, 1]] + xyxy[:, [2, 3]]) / 2
            wh = xyxy[:, [2, 3]] - xyxy[:, [0, 1]]
            xywh = torch.cat((cxcy, wh * ratio), dim=1)
            xywh[:, :2] -= xywh[:, 2:4] / 2
            xywh[:, 2:4] += xywh[:, :2]
            xywh = xywh.clamp(min=0.0, max=1.0)
            exp_box = torch.cat((xywh, boxes[:, 4:]), dim=-1)
            exp_boxes.append(exp_box)
        return exp_boxes

    def anfeature_visualize(self, out_scale, ftlist, fslist):
        h, w = self.imgsize
        cosmaps = []
        dismaps = []
        ssimmaps = []
        origin_areas = []
        for out,ft,fs in zip(out_scale,ftlist,fslist):
            # scaled_detect = out.clone()
            # scaled_detect[:,:4] /= torch.tensor([w,h,w,h],dtype= out.dtype, device = out.device)
            _, scaled_detect = skip_nms(out, 0.5, 0.50, None, max_det=self.maxdet, inputsize=(w, h))
            # scaled_detect = non_max_suppression(out, 0.5, 0.50, None, False, max_det=self.maxdet)
            # scaled_detect = torch.stack(scaled_detect,dim=0)
            # scaled_detect[:,:, :4] /= torch.tensor([w, h, w, h], dtype=scaled_detect.dtype, device=scaled_detect.device)
            exp_detect = self.get_expand_boxes(scaled_detect)

            dismap, cosmap, ssimmap, origin_area = self.get_insfeature_onebatch(exp_detect,ft, fs)
            cosmaps.append(cosmap)
            dismaps.append(dismap)
            ssimmaps.append(ssimmap)
            origin_areas.append(origin_area)
        return self.setformat_blf(dismaps, cosmaps, ssimmaps, origin_areas)

    def anfeature_visualize_crossscale(self, out_detect, ftlist, fslist):
        h, w = self.imgsize
        cosmaps = []
        dismaps = []
        ssimmaps = []
        origin_areas = []
        scaled_detect = torch.stack(out_detect, dim=0)
        scaled_detect[:, :, :4] /= torch.tensor([w, h, w, h], dtype=scaled_detect.dtype, device=scaled_detect.device)
        exp_detect = torch.stack(self.get_expand_boxes(scaled_detect),dim=0)
        ft_maps = []
        fs_maps = []
        for ft,fs in zip(ftlist,fslist):
            ft_map = F.interpolate(ft, size=(int(h), int(w)), mode='bilinear', align_corners=False)
            fs_map = F.interpolate(fs, size=(int(h), int(w)), mode='bilinear', align_corners=False)
            dismap, cosmap, ssimmap, origin_area = self.get_insfeature_onebatch(exp_detect, ft_map, fs_map)
            dismaps.append(dismap)
            cosmaps.append(cosmap)
            ssimmaps.append(ssimmap)
            origin_areas.append(origin_area)
        # ft_maps = torch.cat(ft_maps, dim=1)
        # fs_maps = torch.cat(fs_maps, dim=1)
        # dismap, cosmap, ssimmap, origin_area = self.get_insfeature_onebatch(exp_detect, ft_maps, fs_maps)

        return self.setformat_blf(dismaps, cosmaps, ssimmaps, origin_areas)

    def setformat_blf(self,dismaps, cosmaps, ssimmaps,origin_areas):
        nl = len(dismaps)
        nb = len(dismaps[0])
        dis_out = [[] for i in range(nb)]
        cos_out = [[] for i in range(nb)]
        ssim_out = [[] for i in range(nb)]
        area_out = [[] for i in range(nb)]
        for ni in range(nl):
            for i in range(nb):
                dis_out[i].append(dismaps[ni][i])
                cos_out[i].append(cosmaps[ni][i])
                ssim_out[i].append(ssimmaps[ni][i])
                area_out[i].append(origin_areas[ni][i])
        return dis_out, cos_out, ssim_out, area_out


    def get_insfeature_onebatch(self, boxes, ft_batch, fs_batch):
        nb = len(boxes)
        assert nb == len(ft_batch)
        assert len(ft_batch) == len(fs_batch)
        cosmaps = []
        dismaps = []
        ssimmaps = []
        origin_areas = []
        for i in range(nb):
            dismap, cosmap, ssimmap,origin_area = self.get_instance_feature(boxes[i],ft_batch[i],fs_batch[i])
            cosmaps.append(cosmap)
            dismaps.append(dismap)
            ssimmaps.append(ssimmap)
            origin_areas.append(origin_area)
        return dismaps, cosmaps, ssimmaps, origin_areas

    def get_instance_feature(self, boxes, fs, ft):
        cosmaps = []
        dismaps = []
        ssimmaps = []
        origin_areas = []
        for box in boxes:
            _, h, w = fs.shape
            scale_area = box[:4] * torch.tensor([w, h, w, h]).to(box.device)
            scale_area[:2] = torch.floor(scale_area[:2])
            scale_area[2:4] = torch.ceil(scale_area[2:4])
            h_ = scale_area[[1, 3]].int()
            w_ = scale_area[[0, 2]].int()
            ft_out = ft[:, h_[0]:h_[1], w_[0]:w_[1]]
            fs_out = fs[:, h_[0]:h_[1], w_[0]:w_[1]]
            origin_area = scale_area * (self.imgsize[0]/h)
            # origin_area = scale_area / torch.tensor([w,h,w,h],device = box.device) * \
            #               torch.tensor([self.imgsize[1], self.imgsize[0], self.imgsize[1], self.imgsize[0]]).to(box.device)
            out_w,out_h = (origin_area[2:] - origin_area[:2]).int()
            dissim = get_dissim(ft_out,fs_out)
            cossim = get_cossim(ft_out,fs_out).sum(dim=0)
            ssim = get_ssim(ft_out,fs_out)
            cosmap = (1 - cossim).unsqueeze(dim=0).unsqueeze(dim=0)
            dismap = dissim.unsqueeze(dim=0).unsqueeze(dim=0)
            ssimap = (1 - ssim).unsqueeze(dim=0).unsqueeze(dim=0)
            cosmap = F.interpolate(cosmap, size=(int(out_h),int(out_w)), mode='bilinear', align_corners=False).squeeze(dim=0).squeeze(dim=0)
            dismap = F.interpolate(dismap, size=(int(out_h),int(out_w)), mode='bilinear', align_corners=False).squeeze(dim=0).squeeze(dim=0)
            ssimap = F.interpolate(ssimap, size=(int(out_h),int(out_w)), mode='bilinear', align_corners=False).squeeze(dim=0).squeeze(dim=0)
            cosmaps.append(cosmap.to('cpu'))
            dismaps.append(dismap.to('cpu'))
            ssimmaps.append(ssimap.to('cpu'))
            origin_areas.append(origin_area.to('cpu'))
        return dismaps, cosmaps, ssimmaps, origin_areas

class DetectFireSTPM(nn.Module):
    def __init__(self, weights='yolov5s.pt', device=torch.device('cpu')):
        super().__init__()
        from utils.loss import ComputeAN_Loss
        ckpt = None
        for w in weights if isinstance(weights, list) else [weights]:
            ckpt = torch.load(w)  # load
        model =  FireSTPM(weights=weights,device=device,ckpt= ckpt)
        self.firestpm = model
        self.firestpm.model_s.eval()
        self.stride = model.stride
        self.names = model.names
        self.cal = ComputeAN_Loss(model=model)

    def forward(self, im, augment=False, visualize=False):
        b, ch, h, w = im.shape
        with torch.set_grad_enabled(False):
            tout, sout, detect, out_boxes = self.firestpm(im,test=True)
            #loss = self.cal.train_loss(tout,sout,detect)
            exp_scores = self.cal.test(tout, sout,  out_boxes, (h, w))
            scores = self.cal.test(tout,sout,detect,(h,w))
        return tout, sout, scores, exp_scores, detect,  out_boxes
# layerindexes=[2,4,6, 8,13,17,20,23]

class FireSTPM(nn.Module):
    def __init__(self, weights='yolov5x.pt',weighall=False, device=torch.device('cpu'), data=None,cfg='./models/yolov5s.yaml',
                 layerindexes=[17,20,23],ckpt=None):
        from models.experimental import attempt_download, attempt_load # scoped to avoid circular import
        super().__init__()
        if data:  # data.yaml path (optional)
            with open(data, errors='ignore') as f:
                names = yaml.safe_load(f)['names']  # class names
        if ckpt is None:
            w = attempt_download(str(weights))  if isinstance(weights,str) else weights
            model = attempt_load(weights if isinstance(weights, list) else w, map_location=device,fuse = False)
            stride = max(int(model.stride.max()), 32)  # model stride
            self.names = model.module.names if hasattr(model, 'module') else model.names  # get class names
            self.layerindexes = layerindexes
            for param in model.parameters():
                param.requires_grad = False
            self.model_t = model.eval()
            self.model_t.float()
            self.model_t.to(device)
            nc = self.model_t.yaml['nc']
            from models.yolo import Detect, Model
            self.model_s = Model(cfg, ch=3, nc=nc).to(device)
        else:
            model_t = ckpt['model_t'].float()
            model_s = ckpt['model_s'].float()
            for param in model_t.parameters():
                param.requires_grad = False
            model_t = model_t.eval()
            self.model_t = model_t
            self.model_s = model_s
            self.model_t.to(device)
            self.model_s.to(device)
            stride = max(int(self.model_t.stride.max()), 32)  # model stride
            self.names = model_t.names  # get class names

        #model = attempt_load(weights if isinstance(weights, list) else w, map_location=device)
        #ckpt = torch.load(attempt_download(w), map_location=device)  # load
        #model = (ckpt.get('ema') or ckpt['model']).float()
        #stride = max(int(model.stride.max()), 32)  # model stride

        self.stride = stride
        #names = model.module.names if hasattr(model, 'module') else model.names  # get class names

        self.layerindexes = layerindexes
        self.hook_feature()
        self.feature_t =[]
        self.feature_s =[]
        self.feat_dict_t = {}
        self.feat_dict_s = {}
        self.imgsize = [960,960]
        print('fire_stpm done')

    def hook_feature(self):
        for i in self.layerindexes:
            self.model_t.model[i].register_forward_hook(self.hook_t)
            self.model_s.model[i].register_forward_hook(self.hook_s)

    def hook_s(self,module, input, output):
        self.feature_s.append(output)

    def hook_t(self,module, input, output):
        self.feature_t.append(output)

    def forward(self, im, augment=False, visualize=False, test=False):

        b, ch, h, w = im.shape
        if test:
            with torch.set_grad_enabled(False):
                out_t = self.model_t(im)
                out_s = self.model_s(im)
        else:
            out_t = self.model_t(im)
            out_s = self.model_s(im)

        assert len(self.feature_t) == len(self.layerindexes)
        assert len(self.feature_s) == len(self.layerindexes)
        assert len(self.feature_t) == len(self.feature_s)

        out = out_t[0]

        classes = [2, 5, 7]
        detect,scaled_detect = skip_nms(out, 0.3, 0.50, classes, max_det=500,inputsize=(w,h))
        #detect = non_max_suppression(out, 0.3, 0.50, classes, max_det=1000)
        exp_detect = self.get_expand_boxes(scaled_detect)
        tout_batch, sout_batch = self.setformat_blf(b)
        tout_list = []
        sout_list = []
        out_boxes_batches = []
        anmaps = []
        for bi in range(len(exp_detect)):
            out_boxes = []
            if exp_detect[bi] == [] or len(exp_detect[bi]) == 0:
                tout_list.append([])
                sout_list.append([])
                out_boxes_batches.append([])
                anmaps.append([])
                continue
            tout_ins = []
            sout_ins = []

            for i, box in enumerate(exp_detect[bi]):
                if box == [] or len(box) == 0:
                    continue
                # box_in = box.clone()
                # box_in[:4] /= torch.tensor([w, h, w, h]).to(box_in.device)
                tout_lvls, sout_lvls, exp_box = self.get_insfeature_onebatch(box, tout_batch[bi], sout_batch[bi])
                out_boxes.append(exp_box)
                tout_ins.append(tout_lvls)
                sout_ins.append(sout_lvls)
            tout_list.append(tout_ins)
            sout_list.append(sout_ins)
            out_boxes = torch.stack(out_boxes, dim=0)
            out_boxes[:, :4] *= torch.tensor([w, h, w, h])
            out_boxes_batches.append(out_boxes)


        # for bi in range(len(detect)):
        #     out_boxes = []
        #     if detect[bi] == [] or len(detect[bi]) == 0:
        #         tout_list.append([])
        #         sout_list.append([])
        #         out_boxes_batches.append([])
        #         continue
        #     tout_ins = []
        #     sout_ins = []
        #     for i, box in enumerate(detect[bi]):
        #         if box == [] or len(box) == 0:
        #             continue
        #         box_in = box.clone()
        #         box_in[:4] /= torch.tensor([w, h, w, h]).to(box_in.device)
        #         tout_lvls, sout_lvls, exp_box = self.get_insfeature_onebatch(box_in, tout_batch[bi], sout_batch[bi])
        #         out_boxes.append(exp_box)
        #         tout_ins.append(tout_lvls)
        #         sout_ins.append(sout_lvls)
        #     tout_list.append(tout_ins)
        #     sout_list.append(sout_ins)
        #     out_boxes = torch.stack(out_boxes,dim=0)
        #     out_boxes[:,:4] *= torch.tensor([w,h,w,h])
        #     out_boxes_batches.append(out_boxes)
        self.feature_s = []
        self.feature_t = []

        return tout_list, sout_list, detect,out_boxes_batches


    def forward2(self, im, augment=False, visualize=False, test=False):

        b, ch, h, w = im.shape
        if test:
            with torch.set_grad_enabled(False):
                out_t = self.model_t(im)
                out_s = self.model_s(im)
        else:
            out_t = self.model_t(im)
            out_s = self.model_s(im)

        assert self.feat_dict_t['nums'] == self.feat_dict_s['nums'] == len(self.layerindexes)
        for kt,ks in zip(self.feat_dict_t.keys(),self.feat_dict_s.keys()):
            if ks =='nums' or kt =='nums':
                continue
            self.feat_dict_t[kt] = torch.cat(self.feat_dict_t[kt], dim=1)
            self.feat_dict_s[ks] = torch.cat(self.feat_dict_s[ks], dim=1)
        # assert len(self.feature_t) == len(self.layerindexes)
        # assert len(self.feature_s) == len(self.layerindexes)
        # assert len(self.feature_t) == len(self.feature_s)
        # for kt, ks in zip(self.feat_dict_t,self.feat_dict_s)
        # out = out_t[0]
        out_lvls = out_t[1]
        classes = [2, 5, 7]
        detects = {}
        scaled_detects = {}
        channels = { '8':128,'16':256,'32':512}
        for out in out_lvls:
            out_c = str(channels[str(int(w/np.sqrt(out.shape[1]/3)))])
            detect,scaled_detect = skip_nms(out, 0.3, 0.50, classes, max_det=100, inputsize=(w, h))
            detects[out_c] = detect
            scaled_detects[out_c] = scaled_detect
        keys_t = list(self.feat_dict_t.keys())
        keys_t.remove('nums')
        keys_s = list(self.feat_dict_s.keys())
        keys_s.remove('nums')
        assert keys_t == keys_s
        # toutdict  = []
        # soutdict = {}
        # detectboxes = []
        mask_lvls=[]
        toutlist = []
        soutlist= []
        outboxes = []

        for kt, ks in zip(keys_t, keys_s):
            scaled_box = scaled_detects[kt]
            assert self.feat_dict_s[ks].shape == self.feat_dict_t[ks].shape
            feature_t = self.feat_dict_t[kt]
            feature_s = self.feat_dict_s[ks]
            exp_boxes = self.get_expand_boxes(scaled_box)
            outboxes.append(exp_boxes)
            masks = []

            toutlist.append(feature_t)
            soutlist.append(feature_s)
            for i, (boxes,ft,fs) in enumerate(zip(exp_boxes,feature_t,feature_s)):
                _, fh, fw = ft.shape
                scale_area = boxes[:,:4] * torch.tensor([fw, fh, fw, fh]).to(boxes.device)
                scale_area[:,:2] = torch.floor(scale_area[:,:2])
                scale_area[:,2:4] = torch.ceil(scale_area[:,2:4])
                scale_area = scale_area.int()
                score = boxes[:, 4]
                #l,t,r,b=scale_area[:, 0].min(),scale_area[:, 1].min(),scale_area[:, 2].max(),scale_area[:, 3].max()
                mask = self.get_enclosedbox(scale_area,ft,fs,score)
                masks.append(mask)
                # area_h = scale_area[:,[1, 3]].int()
                # area_w = scale_area[:,[0, 2]].int()


                # area = torch.cat((area_w,area_h),dim=1)[:,[0,2,1,3]]
                # tout.append()
                # tout_list.append(ft[:, h[0]:h[1], w[0]:w[1]])
                # sout_list.append(fs[:, h[0]:h[1], w[0]:w[1]])
            masks = torch.stack(masks, dim=0)
            mask_lvls.append(masks)
        self.feature_s = []
        self.feature_t = []
        return toutlist, soutlist, mask_lvls,outboxes


        #detect,scaled_detect = skip_nms(out, 0.2, 0.50, classes, max_det=500,inputsize=(w,h))
        #detect = non_max_suppression(out, 0.3, 0.50, classes, max_det=1000)
        # exp_detect = self.get_expand_boxes(scaled_detect)
        # tout_batch, sout_batch = self.setformat_blf(b)
        # tout_list = []
        # sout_list = []
        # out_boxes_batches = []


        # for bi in range(len(detect)):
        #     out_boxes = []
        #     if detect[bi] == [] or len(detect[bi]) == 0:
        #         tout_list.append([])
        #         sout_list.append([])
        #         out_boxes_batches.append([])
        #         continue
        #     tout_ins = []
        #     sout_ins = []
        #     for i, box in enumerate(detect[bi]):
        #         if box == [] or len(box) == 0:
        #             continue
        #         box_in = box.clone()
        #         box_in[:4] /= torch.tensor([w, h, w, h]).to(box_in.device)
        #         tout_lvls, sout_lvls, exp_box = self.get_insfeature_onebatch(box_in, tout_batch[bi], sout_batch[bi])
        #         out_boxes.append(exp_box)
        #         tout_ins.append(tout_lvls)
        #         sout_ins.append(sout_lvls)
        #     tout_list.append(tout_ins)
        #     sout_list.append(sout_ins)
        #     out_boxes = torch.stack(out_boxes,dim=0)
        #     out_boxes[:,:4] *= torch.tensor([w,h,w,h])
        #     out_boxes_batches.append(out_boxes)
        # self.feature_s = []
        # self.feature_t = []

        #return tout_list, sout_list, detect,out_boxes_batches



    def setformat_blf(self,batch): #set feature to list formate [batch->[feature levels]
        tout_list = [[] for i in range(batch)]
        sout_list = [[] for i in range(batch)]
        nl = len(self.feature_t)
        for i in range(batch):
            for ni in range(nl):
                tout_list[i].append(self.feature_t[ni][i])
                sout_list[i].append(self.feature_s[ni][i])
        self.feature_t = tout_list
        self.feature_s = sout_list
        return self.feature_t, self.feature_s

    def setformat_blf_dict(self, batch):
        tout_list = [[] for i in range(batch)]
        sout_list = [[] for i in range(batch)]
        channels = {'8': 128, '16': 256, '32': 512}
        for i in range(batch):
            for ni, c in enumerate(channels):
                tout_list[i].append(self.feat_dict_t[c])


    def get_expand_boxes(self,scaled_boxes,ratio = 1.25):
        exp_boxes = []
        for boxes in scaled_boxes:

            xyxy = boxes[:,:4].clamp(min=0.0, max=1.0)
            cxcy = (xyxy[:,[0, 1]] + xyxy[:,[2, 3]]) / 2
            wh = xyxy[:,[2, 3]] - xyxy[:,[0, 1]]
            xywh = torch.cat((cxcy, wh * ratio), dim=1)
            xywh[:,:2] -= xywh[:,2:4] / 2
            xywh[:,2:4] += xywh[:,:2]
            xywh = xywh.clamp(min=0.0, max=1.0)
            exp_box = torch.cat((xywh,boxes[:,4:]) ,dim=-1)
            #boxes[:,:4] = xywh
            exp_boxes.append(exp_box)
        return exp_boxes

    def get_layer_feature(self,index,img, detect):
        tout_list = []
        sout_list = []
        fs_list = self.feature_s[index]
        ft_list = self.feature_t[index]
        _, _, h, w = img.shape
        for i, result in enumerate(detect):
            fs, ft = fs_list[i], ft_list[i]
            boxes = result.clone()
            boxes[:,:4] /= torch.tensor([w,h,w,h]).to(boxes.device)
            tout,sout = self. get_instance_feature(boxes, fs, ft)
            tout_list.append(tout)
            sout_list.append(sout)
        return tout_list, sout_list

    def get_instance_feature(self,boxes, fs, ft):
        tout_list = []
        sout_list = []
        for box in boxes:
            _, h, w = fs.shape
            scale_area = box[ :4] * torch.tensor([w, h, w, h]).to(box.device)
            scale_area[:2] = torch.floor(scale_area[:2])
            scale_area[2:4] = torch.ceil(scale_area[2:4])
            h = scale_area[ [1, 3]].int()
            w = scale_area[[0, 2]].int()
            tout_list.append(ft[:, h[0]:h[1], w[0]:w[1]])
            sout_list.append(fs[:, h[0]:h[1], w[0]:w[1]])
        return  tout_list, sout_list

        # for i,result  in enumerate(detect):
        #     _, _, h, w= img.shape
        #     fs_list = [fk[i] for fk in self.feature_s]
        #     ft_list = [fk[i] for fk in self.feature_t]
        #     mask = torch.zeros(self.feature_s[i].shape)
        #     box = result.clone()
        #     # rect =box.clone()
        #     # rect[:, :2] = torch.floor(rect[:, :2])
        #     # rect[:, 2:4] = torch.ceil(rect[:, 2:4])
        #     # hs = rect[0,[1,3]]
        #     # ws=rect[0,[0,2]]
        #     #
        #     # ims=img[i,:,hs[0]:hs[1],ws[0],ws[1]]
        #     # cv2.imwrite('1.jpg',ims.cpu().numpy())
        #     box[:,:4] /= torch.tensor([w,h,w,h]).to(box.device)
        #     self.scale_feature(box,fs_list,ft_list)
        #
        #     print()
        #
        # self.feature_s = []
        # self.feature_t = []

    def get_feature_instance(self,box,fs,ft):
        _, h, w = fs.shape
        scale_area = box[:4] * torch.tensor([w, h, w, h]).to(box.device)
        scale_area[:2] = torch.floor(scale_area[:2])
        scale_area[2:4] = torch.ceil(scale_area[2:4])
        h = scale_area[[1, 3]].int()
        w = scale_area[[0, 2]].int()
        tout = ft[:, h[0]:h[1], w[0]:w[1]]
        sout = fs[:, h[0]:h[1], w[0]:w[1]]
        return tout, sout

    def get_insfeature_onebatch(self,box,ft_lvls, fs_lvls):
        # nl = len(self.feature_t)
        nl = len(self.feature_t[0])
        tout_inslvls = []
        sout_inslvls = []
        boxes_out = []
        loss = 0
        an_maps = []
        for i in range(nl):
            tout_ins, sout_ins, exp_boxes = self.get_insfeature_onelvl(box, ft_lvls[i], fs_lvls[i])
            tout_inslvls.append(tout_ins)
            sout_inslvls.append(sout_ins)
            boxes_out.append(exp_boxes)
        boxes_out = torch.stack(boxes_out, dim=0)
        box = torch.ones(boxes_out[0].shape)
        box[4:] = boxes_out[0, 4:]
        box[0] = boxes_out[:, 0].min()
        box[1] = boxes_out[:, 1].min()
        box[2] = boxes_out[:, 2].max()
        box[3] = boxes_out[:, 3].max()
        box[:4] = box[:4].clamp(min=0.0,max=1.0)
        return tout_inslvls, sout_inslvls, box

    def get_insfeature_mutilvl(self, box, ft_list, fs_list):
        assert len(ft_list) == len(fs_list)
        nl = len(ft_list)
        tout_lvllist = []
        sout_lvllist = []

        for i in range(nl):
            tout, sout = self.get_insfeature_onelvl(box,ft_list[i],fs_list[i])
            tout_lvllist.append(tout)
            sout_lvllist.append(sout)
        return  tout_lvllist, sout_lvllist

    def get_insfeature_onelvl(self, box,ft,fs,ratio = 1.0):
        assert ft.shape == fs.shape
        _, h, w = fs.shape
        xyxy = box[:4].clamp(min=0.0, max=1.0)
        cxcy = (xyxy[[0,1]] + xyxy[[2,3]])/2
        wh = xyxy[[2,3]] - xyxy[[0,1]]

        xywh = torch.cat((cxcy,wh*ratio),dim=0)
        xywh[:2] -= xywh[2:4]/2
        xywh[2:4] += xywh[:2]

        xywh = xywh.clamp(min=0.0,max=1.0)
        # img_w,img_h = self.imgsize
        # b_area = box[:4] * torch.tensor([img_w,img_h,img_w,img_h],device=box.device)
        # bw = b_area[2].ceil()-b_area[0].floor()
        # bh = b_area[3].ceil() - b_area[1].floor()
        # out_size = [int(bh),int(bw)]

        scale_area = xywh * torch.tensor([w, h, w, h]).to(xywh.device)
        #scale_area = box[:4].clamp(min=0.0,max=1.0) * torch.tensor([w, h, w, h]).to(box.device)
        scale_area[:2] = torch.floor(scale_area[:2])
        scale_area[2:4] = torch.ceil(scale_area[2:4])
        tbh = scale_area[[1, 3]].int()
        lrw = scale_area[[0, 2]].int()

        tout = ft[:, tbh[0]:tbh[1], lrw[0]:lrw[1]]
        sout = fs[:, tbh[0]:tbh[1], lrw[0]:lrw[1]]
        #ins_loss,anmap = self.cal_insloss(tout,sout,out_size)
        scale_area = scale_area / torch.tensor([w, h, w, h]).to(scale_area.device)
        scale_area = torch.cat((scale_area, box[4:]))
        if tout.size(1)==0 or tout.size(2) == 0 or  tout.size(0) == 0:
            print('tout is empty')
        return tout, sout, scale_area

    def cal_insloss(self,ft,fs,out_size):
        ft_norm = F.normalize(ft, p=2)
        fs_norm = F.normalize(fs, p=2)
        _, h, w = fs.shape
        loss = 0.5 * (ft_norm - fs_norm) ** 2
        loss = loss / (h * w)
        a_map = loss.clone().detach()
        loss = loss.sum().cpu()
        a_map = a_map.unsqueeze(dim=0)

        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=False)
        a_map = a_map.squeeze(dim=0)
        a_map = a_map.sum(dim=0)
        a_map = a_map.cpu()
        return loss, a_map
        #a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=False)
        print()
    # get all the instance feature once a time, and remake a feature mask
    def get_allinsfeature_onebatch(self,exp_boxes, ft_lvls,fs_lvls):
        print()

    def scale_feature(self,box, fs_list,ft_list):
        tout_list = []
        sout_list = []
        for i in range(len(fs_list)):
            fs = fs_list[i]
            ft = ft_list[i]
            _,h,w = fs.shape
            scale_area = box[:,:4] * torch.tensor([w,h,w,h]).to(box.device)
            scale_area[:, :2] = torch.floor(scale_area[:, :2])
            scale_area[:, 2:4] = torch.ceil(scale_area[:, 2:4])
            h = scale_area[0, [1, 3]]
            w = scale_area[0, [0, 2]]
            tout_list.append(fs[:,h[0]:h[1],w[0]:w[1]])
            out_s = fs[:,h[0]:h[1],w[0]:w[1]]
            out_t = ft[:,h[0]:h[1],w[0]:w[1]]

        print()

    def get_enclosedbox(self, area, fs,ft,scores): # get the smallestt enclosed box
        #mask = torch.zeros(ft.shape[1], ft.shape[2],device=scores.device)
        #enclosedbox = torch.stack([area[:,0].min(),area[:,1].min(),area[:,2].max(),area[:,3].max()],dim=0)
        scoremask = torch.zeros(ft.shape[1], ft.shape[2], device=scores.device)
        countmask = torch.zeros(ft.shape[1], ft.shape[2], device=scores.device)
        for ((l,t,r,b),score) in zip(area,scores):
            scoremask[t:b, l:r] += torch.ones(scoremask[t:b, l:r].shape,device=scores.device) * score
            countmask[t:b, l:r] += torch.ones(countmask[t:b, l:r].shape, device=scores.device)            #area.T[0], area.T[1], area.T[2], area.T[3]
        countmask[countmask>0] = int(1)
        return countmask

class ForwardHook:
    def __init__(self, hook_dict, layer_index: int):
        self.hook_dict = hook_dict
        self.layer_index = layer_index

    def __call__(self, module, input, output):
       
        self.hook_dict[str(self.layer_index)] = output
     
        return None

class ForwardHook_avg:
    def __init__(self, hook_dict, layer_index: int):
        self.hook_dict = hook_dict
        self.layer_index = layer_index

    def __call__(self, module, input, output):
        output = F.avg_pool2d(output, 3, 1, 1)
        self.hook_dict[str(self.layer_index)] = output
     
        return None


class AN_Patch(nn.Module):
    def __init__(self, weights='yolov5x.pt',weighall=False, device=torch.device('cpu'), data=None,cfg='./models/yolov5x.yaml',
                 layerindexes=[0,2,17,20],ckpt=None,names=[],fp16=False,imgsize=(640, 640),detect = False):
        super().__init__()
        if ckpt is None:
            ckpt = torch.load(weights, map_location='cpu')
            model = ckpt['model']
            self.stride = max(int(model.stride.max()), 32)
            self.names = names
            self.fp16 = fp16
            self.imgsize = (640, 640)
            self.layerindexes = layerindexes
            self.model = model.eval()
            for param in model.parameters():
                param.requires_grad = False
            
            self.model.float()
            self.model.to(device)
            stride = max(int(self.model.stride.max()), 32)  # model stride
            self.names = model.names  # get class names
            self.stride = stride
            self.patch_maker = PatchMaker()
            self.mapper = MeanMapper()
            self.preprocessing =  Preprocessing()
            self.aggregator = Aggregator()
            self.discriminator =  Discriminator().to(device)
        else:
            ckpt = torch.load(ckpt, map_location='cpu')
            model = torch.load(weights, map_location='cpu')['model'].float()
            self.stride = max(int(model.stride.max()), 32)
            self.model = model.eval().to(device)
            self.names = model.names
            ckpt['model'].mapper
            self.patch_maker = PatchMaker()
            self.mapper =MeanMapper()
            self.preprocessing =  Preprocessing()
            self.aggregator = Aggregator()
            self.discriminator =  ckpt['model'].discriminator .to(device)
            self.discriminator .eval()
            # self.fp16 = model.fp16
        for param in self.model.parameters():
                param.requires_grad = False
        self.layerindexes = layerindexes
        self.feat_dict = {}
        self.patch_feat = {}
        self.classes = None
        self.imgsize = imgsize
        self.pool_sizes = [128,64,32,16]
        for i in self.layerindexes:
            forward_hook = ForwardHook(self.feat_dict,i)
            self.model.model[i].register_forward_hook(forward_hook)
        self.detect = detect
        self.device = device
        self.bnc = [80,160,320,640]
        self.bnlist =[nn.BatchNorm2d(c).to(self.device) for c in self.bnc] 
    def forward(self, im, augment=False, visualize=False, save_dir =None,save_name=None):
        b, ch, h, w = im.shape
        detects =  []
        self.feat_dict.clear()
        out = self.model(im)
        detects = non_max_suppression(out , 0.25,  0.50, self.classes, max_det=10000)
        th = 0.95
        loss = 0.0
        for d in detects:
            if len(d) == 0:
                
                continue
            else:
                boxes  = d[:,:4]
                ins_feats = self._roiwithoutup(boxes)
                out_feats = []
                for insf in ins_feats:
                    features = [self.patch_maker.patchify( self.bnlist[i](insf[i][None]),return_spatial_info=True) for i in range(len(self.bnc))]
                    patch_shapes = [f[1] for f in features]
                    features =[f[0] for f in features]
                    ref_num_patches = patch_shapes[0]
                    for i in range(1, len(features)):
                        _features = features[i]
                        patch_dims = patch_shapes[i]
                        _features = _features.reshape(
                            _features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:]
                        )
                        _features = _features.permute(0, -3, -2, -1, 1, 2)
                        perm_base_shape = _features.shape
                        _features = _features.reshape(-1, *_features.shape[-2:])
                        _features = F.interpolate(
                            _features.unsqueeze(1),
                            size=(ref_num_patches[0], ref_num_patches[1]),
                            mode="bilinear",
                            align_corners=False,
                        )
                        _features = _features.squeeze(1)
                        _features = _features.reshape(
                            *perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1]
                        )
                        
                        _features = _features.permute(0, -2, -1, 1, 2, 3)
                        _features = _features.reshape(len(_features), -1, *_features.shape[-3:])
                        features[i] = _features
                    features = [x.reshape(-1, *x.shape[-3:]) for x in features]
                    features = self.preprocessing(features)
                    features = self.aggregator(features)
                    features = features.reshape(b,ref_num_patches[0],ref_num_patches[1],features.shape[-1])
                    emb_dims = features.shape[-1]
                    features = features.reshape(-1,emb_dims )
                    mix_noise = 1
                    noise_idxs = torch.randint(0, mix_noise, torch.Size([features.shape[0]]))
                    noise_one_hot = torch.nn.functional.one_hot(noise_idxs, num_classes=mix_noise).to(self.device) 
                    noise_std=0.05
                    noise = torch.stack([torch.normal(0, noise_std * 1.1**(k), features.shape)
                        for k in range(mix_noise)], dim=1).to(self.device)   # (N, K, C)
                    noise = (noise * noise_one_hot.unsqueeze(-1)).sum(1)
            
                    fake_feats = features + noise
                    out_feats.append(torch.cat([features,fake_feats],dim=0))
                    # out_feats.append(torch.stack([features,fake_feats],dim=-1))
        out_feats = torch.stack(out_feats,dim=0)
        scores = torch.stack([self.discriminator(f) for f in out_feats],dim=0)
        true_scores = scores[:, : len(scores)//2]
        fake_scores = scores[:, len(scores)//2:]
        p_true = (true_scores.detach() >= th).sum(dim=-1) / len(true_scores)
        true_loss = 100*torch.max(torch.zeros_like(true_scores), -true_scores + th) +1e-6
        fake_loss = 100*torch.max(torch.zeros_like(fake_scores), fake_scores + th) +1e-6
        loss = true_loss.mean() + fake_loss.mean()
        return loss, p_true,true_loss.mean(),fake_loss.mean()
      
      
   
    def _roiwithoutup(self,boxes):
        ins_feats = [[] for b in boxes]
        for i in range(len(self.layerindexes)):
            psize = self.pool_sizes[i]
            l = self.layerindexes[i]
            m = MultiScaleRoIAlign([str(l)],psize,-1)
            roi_feats = m(self.feat_dict,[boxes],[self.imgsize])
            for bi in range(boxes.shape[0]):
                ins_feats[bi].append(roi_feats[bi])
        return ins_feats

def roiwithoutup(feat_dict,boxes,layerindexes,pool_sizes,imgsize):
        ins_feats = [[] for b in boxes]
        for i in range(len(layerindexes)):
            psize = pool_sizes[i]
            l = layerindexes[i]
            m = MultiScaleRoIAlign([str(l)],psize,-1)
            # m = MultiScaleRoIAlign([str(l)],psize,2)
            roi_feats = m(feat_dict,[boxes],[imgsize])
            for bi in range(boxes.shape[0]):
                ins_feats[bi].append(roi_feats[bi])
        return ins_feats
    
def roiwithup(feat_dict,boxes,layerindexes,pool_sizes,imgsize):
        ins_feats = []
        psize = pool_sizes[0]
        for l in layerindexes:
            m = MultiScaleRoIAlign([str(l )],psize,-1)
            # m = MultiScaleRoIAlign([str(l )],psize,2)
            ins_feats.append(m(feat_dict,[boxes],[imgsize]))
        ins_feats = torch.cat(ins_feats,dim=1)
        return ins_feats
    
def roi_img(imgs,boxes,pool_sizes,imgsize):
        ins_feats = []
        psize = pool_sizes[0]
        img_dict= {}
        names  = []
        for i,img in enumerate(imgs):
            img_dict[str(i)] = img[None] *255.0
            b = boxes[i][:,:4]
            if len(b) == 0:
                continue
            m = MultiScaleRoIAlign([str(i)],psize,2)
            # m = MultiScaleRoIAlign([str(i)],psize,-1)
            ins_feats.append(m(img_dict,[b],[imgsize]))
            for j in range(len(b)):
                names.append(str(i)+'_'+str(j))

        ins_feats = torch.cat(ins_feats,dim=0)
        return ins_feats,names

class Res_CFA(nn.Module):
    def __init__(self,detect_model=None, ckpt=None,backbone='resnet18',weights=None,device=torch.device('cpu'),data=None,classes=[0,1,2],
                 imgsize= (640,640),detect=False):
        super().__init__()
        self.device = device
        self.gamma_c = 1
        self.gamma_d = 1
        self.alpha = 1e-1
        self.nu = 1e-3
        self.K = 3
        self.J = 3
        self.imgsize = imgsize
        self.pool_sizes = [64,32,16]
        self.feat_dict = {}
        if ckpt :
            ckpt = torch.load(ckpt, map_location='cpu')
            self.detect_model = ckpt['model'].detect_model
            self.r = ckpt['model'].r
            self.C_up = ckpt['model'].C_up 
            self.C_nup = ckpt['model'].C_nup
            self.descriptor = ckpt['model'].descriptor 
            self.cnn = ckpt['model'].cnn
            self.project = ckpt['model'].project
            self.detect_model.eval()
        else:
            self.r   = nn.Parameter(1e-5*torch.ones(1), requires_grad=True)
            self.C_up = 0
            self.C_nup = 0
        
            self.detect_model = detect_model.to(device)
            self.detect_model = self.detect_model.float()
            self.detect_model.eval()
            self.descriptor = Descriptor_ori(self.gamma_d,448,device).to(device)
            self.project  = CoordConv2d(448, 448//self.gamma_d, 1)
        
      
            if backbone =='resnet18':
                self.cnn = resnet18()
            else:
                self.cnn =  wide_resnet50_2()
        self.max_ins_num = 4
        self.stride = max(int(self.detect_model.stride.max()), 32)
        self.cnn.eval()
        self.detect_model.eval()
        self.detect = detect
        self.transform = T.Compose([T.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])])
        self.classes = [0]
    

    def forward(self, x,targets):
        self.feat_dict.clear()
        pred = self.cnn(self.transform (x))
        for i in range(len(pred)):
            self.feat_dict[str(i)] = pred[i]
        if self.detect:
            detects = self.detect_model(x)
            detects = non_max_suppression(detects , 0.75,  0.50, self.classes, max_det=100)
        else:
            detects = targets[:,2:]
            detects[:,:2] -= detects[:,2:]/2
            detects[:,2:] += detects[:,:2]
            detects *= torch.tensor([self.imgsize[0],self.imgsize[1],self.imgsize[0],self.imgsize[1]])
            detects = [detects.to(self.device)]
        batch_id =0
        loss = None
        phi_ps = []
        for d in detects:
            if len(d) == 0:
                batch_id += 1.0
                continue
            else:
                boxes  = d[:,:4]
                wh =  boxes[:,2:] - boxes[:,:2]
                cxcy = boxes[:,:2] + wh/2
                newboxes = torch.cat([cxcy, 1.20*wh],dim=-1)
                newboxes[:,:2] -= newboxes[:,2:]/2
                newboxes[:, 2:] += newboxes[:,:2]
                newboxes = newboxes.clamp(max=self.imgsize[0]-1,min=1.0)
                boxes = newboxes
                ins_feats = roiwithoutup(self.feat_dict,boxes,[0,1,2],self.pool_sizes,self.imgsize)
                
                phi_p = [self.descriptor(feat) for feat in ins_feats]
                phi_p = torch.cat(phi_p,dim=0)
                phi_ps.append(phi_p)
                batch_id += 1.0
        if len(phi_ps) == 0:
            if self.detect:
                return None,None,None
            else:
                return None,None,None
        phi_ps = torch.cat(phi_ps,dim=0)     
        phi_ps = rearrange(phi_ps, 'b c h w -> b (h w) c')
        ni = phi_ps.shape[0]
        if self.detect:
            img_ins,save_names = roi_img(x,[boxes],self.pool_sizes,self.imgsize)
            n_neighbors = self.K
            phi_ps =  phi_ps.detach().cpu()
            if ni> self.max_ins_num:
                select_indexes = torch.randperm(ni)
                select_indexes = select_indexes[:self.max_ins_num]
                phi_ps= phi_ps[select_indexes]
                
                save_names = [save_names[s] for s in  select_indexes]
                img_ins = img_ins[select_indexes]
            features = torch.sum(torch.pow(phi_ps, 2), 2, keepdim=True)    
            centers  = torch.sum(torch.pow(self.C_nup.cpu(), 2), 0, keepdim=True)
            f_c = 2 * torch.matmul(phi_ps, (self.C_nup.cpu()))
            dist = features + centers - f_c
            dist = torch.sqrt(dist)
            dist = (dist.topk(n_neighbors, largest=False).values)
          
            dist = (F.softmin(dist, dim=-1)[:, :, 0]) * dist[:, :, 0]
            dist = dist.unsqueeze(-1)
            score = rearrange(dist, 'b (h w) c -> b c h w', h=self.pool_sizes[0])
            heatmap = score.cpu().detach()
            heatmap = heatmap.mean(dim=1)
            img_ins = img_ins.cpu()
            return heatmap,img_ins,save_names

        if ni> self.max_ins_num:
            select_indexes = torch.randperm(ni)
            select_indexes = select_indexes[:self.max_ins_num]
            phi_ps= phi_ps[select_indexes]

        loss , L_att , L_rep = self._soft_boundary(phi_ps)
        return loss.cpu(), L_att.cpu() ,L_rep.cpu()
        
    def _soft_boundary(self, phi_p):
        features = torch.sum(torch.pow(phi_p, 2), 2, keepdim=True)
        centers  = torch.sum(torch.pow(self.C_nup, 2), 0, keepdim=True)
        f_c      = 2 * torch.matmul(phi_p, (self.C_nup))
        dist     = features + centers - f_c
        n_neighbors = self.K + self.J
        dist     = dist.topk(n_neighbors, largest=False).values

        score = (dist[:, : , :self.K] - self.r**2) 
        L_att = (1/self.nu) * torch.mean(torch.max(torch.zeros_like(score), score))
        
        score = (self.r**2 - dist[:, : , self.J:]) 
        L_rep  = (1/self.nu) * torch.mean(torch.max(torch.zeros_like(score), score - self.alpha))
        #loss = L_att
        loss = L_att + 20*L_rep

        return loss , L_att , L_rep

    def _init_centroid(self, data_loader,save_dir='/mnt/ecc5c6c9-7631-4983-9ba0-1ec98729589b/llc/AN_NEW/ST-CFA/'):
        self.C_up = 0.0
        del self.C_nup
        self.C_nup = 0.0
        self.descriptor = Descriptor_ori(self.gamma_d,448,self.device).to(self.device)
        self.project  = CoordConv2d(448, 448//self.gamma_d, 1).to(self.device)
        pbar = enumerate(data_loader)
        pbar = tqdm(pbar, total=1, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}') 
        count = 0
        transform_x = T.Compose([T.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])])
        for i, (imgs, targets, paths, _) in pbar: 
            imgs = imgs.to(self.device, non_blocking=True).float()/255.0
            
            imgs = transform_x (imgs)
            if len(imgs.shape) == 3:
                imgs = imgs[None]  # expand for batch dim
                
            if len(targets) ==0:
                continue
            self.feat_dict.clear()
            pred = self.cnn(imgs)
            for i in range(len(pred)):
                self.feat_dict[str(i)] = pred[i]
            boxes = targets[:,2:]
            wh =  boxes[:,2:]
            cxcy = boxes[:,:2]
            newboxes = torch.cat([cxcy, 1.20*wh],dim=-1)
            newboxes[:,:2] -= newboxes[:,2:]/2
            newboxes[:, 2:] += newboxes[:,:2]
            newboxes = newboxes.clamp(max=1.0,min=0.0)
            boxes = newboxes
            boxes *= torch.tensor([self.imgsize[0],self.imgsize[1],self.imgsize[0],self.imgsize[1]])
            boxes = boxes.to(self.device)
            layerindexes = [i for i in range(len(pred))]
            ins_feats_nup = roiwithoutup(self.feat_dict,boxes,layerindexes,self.pool_sizes,self.imgsize)
            ins_feats_up = roiwithup(self.feat_dict,boxes,layerindexes,self.pool_sizes,self.imgsize)
            nup_phi_ps = [self.descriptor(feat) for feat in ins_feats_nup]
            up_phi_ps = [self.project(feat[None]) for feat in ins_feats_up]
            # img_ins,save_names = roi_img(imgs ,[boxes],self.pool_sizes,self.imgsize)
            # cv2.imwrite(str(save_dir)+save_names[0]+'.jpg',img_ins[0].cpu().permute(1,2,0).numpy())
            for j in range(len( boxes )):
                nup_phi =   nup_phi_ps[j].detach()
                up_phi =   up_phi_ps[j].detach()
                self.C_up = ((self.C_up * count) + up_phi) / (count+1)
                self.C_nup = ((self.C_nup * count) + nup_phi) / (count+1)
                count += 1
            self.feat_dict.clear()
        self.C_up = rearrange(self.C_up, 'b c h w -> (b h w) c').detach().cpu()
        self.C_nup = rearrange(self.C_nup, 'b c h w -> (b h w) c').detach().cpu()
        if self.gamma_c > 1:
            self.C_up= self.C_up.cpu().detach().numpy()
            self.C_up= KMeans(n_clusters=(self.scale**2)//self.gamma_c, max_iter=3000).fit(self.C_up).cluster_centers_
            self.C_up = torch.Tensor(self.C_up)
        self.C_up = self.C_up.transpose(-1, -2).detach()
        self.C_nup = self.C_nup.transpose(-1, -2).detach()
        self.C_up = nn.Parameter(self.C_up , requires_grad=False).to(self.device)
        self.C_nup = nn.Parameter(self.C_nup , requires_grad=False).to(self.device)
        
        save_pth_C_up =os.path.join(save_dir,'C_up.pth')
        save_pth_C_nup =os.path.join(save_dir,'C_nup.pth')
        save_pth_descriptor = os.path.join(save_dir,'descriptor.pth')
        save_pth_project = os.path.join(save_dir,'project.pth')
        torch.save(self.C_nup, save_pth_C_nup)
        torch.save(self.C_up, save_pth_C_up)
        torch.save(self.descriptor, save_pth_descriptor)
        torch.save(save_pth_project , save_pth_project)
        print('_init_centroid done')


class AN_ST(nn.Module):
    def __init__(self, weights='yolov5x.pt',weighall=False, device=torch.device('cpu'), data=None,cfg='./models/Syolov5x.yaml',
                 layerindexes=[0,2,17,20],ckpt=None,names=[],c_pt =None, fp16=False,imgsize=(640, 640),detect = False):
        super().__init__()
        self.device = device
        self.fp16 = fp16
        self.imgsize = (640, 640)
        from models.yolo import Model
        if ckpt is None:
            weights = weights[0]
            ckpt = torch.load(weights, map_location='cpu')
            self.model_t = ckpt['model']
            nc = self.model_t.yaml['nc']
            self.model_s = Model(cfg, ch=3, nc=nc).to(device)
        else:
            ckpt = torch.load(ckpt, map_location='cpu')
            self.model_t = ckpt['model'].model_t
            nc = self.model_t.yaml['nc']
            self.model_s = ckpt['model'].model_s
            self.model_s = self.model_s.to(device)
        self.model_t = self.model_t.eval()
        self.model_t.float()
        self.model_t.to(device)
        self.model_s.float()
        
        stride = max(int(self.model_t.stride.max()), 32)  # model stride
        self.names = self.model_t.names  # get class names
        self.stride = stride
        self.layerindexes = layerindexes
        for param in self.model_t.parameters():
                param.requires_grad = False
        self.layerindexes = layerindexes
        self.pool_sizes = [128,64,32,16]
        self.feat_dict_t = {}
        self.feat_dict_s = {}
        self.imgsize = imgsize
        self.detect = detect
        for i in self.layerindexes:
            forward_hook_t = ForwardHook(self.feat_dict_t,i)
            self.model_t.model[i].register_forward_hook(forward_hook_t)
            forward_hook_s = ForwardHook(self.feat_dict_s,i)
            self.model_s.model[i].register_forward_hook(forward_hook_s)
        if detect:
            self.classes = [0,1,2,5,6]
        self.loss = nn.MSELoss(reduction='none')
    def forward(self, x,targets):
        self.feat_dict_t.clear()
        self.feat_dict_s.clear()
        pred = self.model_t(x)
        out_s = self.model_s(x)
        if self.detect:
            detects = non_max_suppression(pred , 0.75,  0.50, self.classes, max_det=100)
        else:
            detects = targets[:,2:]
            detects[:,:2] -= detects[:,2:]/2
            detects[:,2:] += detects[:,:2]
            detects *= torch.tensor([self.imgsize[0],self.imgsize[1],self.imgsize[0],self.imgsize[1]])
            detects = [detects.to(self.device)]
        del pred,out_s
        loss = 0
        if self.detect:
            img_ins_list =[]
            scores_list = []
            save_name_list = []

        for d in detects:
            if len(d) == 0:
                continue
            else:
                boxes  = d[:,:4]
                t_ins_feats = roiwithup(self.feat_dict_t,boxes,self.layerindexes,self.pool_sizes,self.imgsize)
                s_ins_feats = roiwithup(self.feat_dict_s,boxes,self.layerindexes,self.pool_sizes,self.imgsize)
                scores = self.loss (t_ins_feats,s_ins_feats).sum(dim=1)
                loss += scores.mean()
                img_ins,save_names = roi_img(x,[boxes],self.pool_sizes,self.imgsize)
                     
        self.feat_dict_t.clear()
        self.feat_dict_s.clear()
        if self.detect:
           return scores.cpu(),img_ins.cpu(),save_names     
        else:
            return loss

class AN_CFA(nn.Module):
    def __init__(self, weights='yolov5x.pt',weighall=False, device=torch.device('cpu'), data=None,cfg='./models/yolov5x.yaml',
                 layerindexes=[0,2,17,20],ckpt=None,names=[],c_pt =None, fp16=False,imgsize=(640, 640),detect = False):
        super().__init__()
        self.nu = 1e-3
        self.scale = None
        self.gamma_c = 1
        self.gamma_d = 1
        self.alpha = 1e-1
        self.K = 3
        self.J = 3
        self.r   = nn.Parameter(1e-5*torch.ones(1), requires_grad=True)
        self.C_up = 0
        self.C_nup = 0
        if c_pt!= None: 
            self.center = torch.load(c_pt).to(device)
        self.device = device
        self.project = None
        self.fp16 = fp16
        self.imgsize = (640, 640)
        
        if ckpt is None:
            if isinstance(weights,list):
                c_w = weights[1]
                des_w = weights[2] 
                # pro_w =  weights[3]
                weights = weights[0]
                ckpt = torch.load(weights, map_location='cpu')
                self.C_nup = torch.load(c_w,map_location='cpu')
                self.C_nup.requires_grad = False
                self.descriptor = torch.load(des_w,map_location='cpu')
                self.descriptor.train()
                self.model = ckpt['model']
                # self.names = names
            else:
                ckpt = torch.load(weights, map_location='cpu')
                # [64,128,256,512] * 1.25
                self.descriptor = Descriptor(self.gamma_d,1200,device).to(device)
                self.project  = CoordConv2d(1200, 1200//self.gamma_d, 1)
                self.model = ckpt['model']
                self.descriptor.train()
        else:
            ckpt = torch.load(ckpt, map_location='cpu')
            self.model = ckpt['model'].model
            self.descriptor = ckpt['model'].descriptor
            self.project =  ckpt['model'].project
            self.C_nup = ckpt['model'].C_nup.to(device)
            self.r   = ckpt['model'].r
            self.C_nup.requires_grad = False
            self.descriptor.train()
            
        self.device = device
        self.model = self.model.eval()
        self.model.float()
        self.model.to(device)
        self.descriptor.to(device)
        if self.project:
            self.project.to(device)
        nc = self.model.yaml['nc']
        stride = max(int(self.model.stride.max()), 32)  # model stride
        self.names = self.model.names  # get class names
        self.stride = stride
        self.layerindexes = layerindexes
        for param in self.model.parameters():
                param.requires_grad = False
        self.layerindexes = layerindexes
        self.pool_sizes = [128,64,32,16]
        self.feat_dict = {}
        self.classes = [0,1,2]
        self.imgsize = imgsize
        self.max_nums = 4
        self.detect = detect
        for i in self.layerindexes:
            forward_hook = ForwardHook(self.feat_dict,i)
            # forward_hook = ForwardHook_avg(self.feat_dict,i)
            self.model.model[i].register_forward_hook(forward_hook)
        if detect:
            self.classes = [0]
            # self.classes = [5,6]

    def _init_centroid(self, data_loader,save_dir='/mnt/ecc5c6c9-7631-4983-9ba0-1ec98729589b/llc/AN_NEW/ST-CFA/'):
        self.C_up = 0.0
        del self.C_nup
        self.C_nup = 0.0
        self.descriptor = Descriptor(self.gamma_d,1200,self.device).to(self.device)
        self.project  = CoordConv2d(1200, 1200//self.gamma_d, 1).to(self.device)

        pbar = enumerate(data_loader)
        pbar = tqdm(pbar, total=1, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}') 
        count = 0
        for i, (imgs, targets, paths, _) in pbar: 
        # for i, (path, imgs,_, _, _) in pbar: 
            imgs = imgs.to(self.device, non_blocking=True).float() / 255.0
            # imgs = torch.from_numpy(imgs).to(self.device).float()/ 255.0
            if len(imgs.shape) == 3:
                imgs = imgs[None]  # expand for batch dim
            
            
            if len(targets) ==0:
                continue
            self.feat_dict.clear()
            pred = self.model(imgs)
            # detects = non_max_suppression(pred , 0.2,  0.50, self.classes, max_det=100)[0]
            # if len(detects)==0:
            #     continue
            # boxes  = detects[:,:4]
            
            boxes = targets[:,2:]
            wh =  boxes[:,2:]
            cxcy = boxes[:,:2]
            newboxes = torch.cat([cxcy, 1.20*wh],dim=-1)
            newboxes[:,:2] -= newboxes[:,2:]/2
            newboxes[:, 2:] += newboxes[:,:2]
            newboxes = newboxes.clamp(max=1.0,min=0.0)
            boxes = newboxes
            boxes *= torch.tensor([self.imgsize[0],self.imgsize[1],self.imgsize[0],self.imgsize[1]])
            boxes = boxes.to(self.device)
            ins_feats_nup = self._roiwithoutup(boxes)
            ins_feats_up = self._roiwithup(boxes)
            nup_phi_ps = [self.descriptor(feat) for feat in ins_feats_nup]
            up_phi_ps = [self.project(feat[None]) for feat in ins_feats_up]
            # img_ins,save_names = self._roi_img(imgs ,[boxes])
            # cv2.imwrite(str(save_dir)+save_names[0]+'.jpg',img_ins[0].cpu().permute(1,2,0).numpy())
            for j in range(len( boxes )):
                nup_phi =   nup_phi_ps[j].detach()
                up_phi =   up_phi_ps[j].detach()
                self.C_up = ((self.C_up * count) + up_phi) / (count+1)
                self.C_nup = ((self.C_nup * count) + nup_phi) / (count+1)
                count += 1
            self.feat_dict.clear()

        self.C_up = rearrange(self.C_up, 'b c h w -> (b h w) c').detach().cpu()
        self.C_nup = rearrange(self.C_nup, 'b c h w -> (b h w) c').detach().cpu()
        if self.gamma_c > 1:
            self.C_up= self.C_up.cpu().detach().numpy()
            self.C_up= KMeans(n_clusters=(self.scale**2)//self.gamma_c, max_iter=3000).fit(self.C_up).cluster_centers_
            self.C_up = torch.Tensor(self.C_up)
        self.C_up = self.C_up.transpose(-1, -2).detach()
        self.C_nup = self.C_nup.transpose(-1, -2).detach()
        self.C_up = nn.Parameter(self.C_up , requires_grad=False).to(self.device)
        self.C_nup = nn.Parameter(self.C_nup , requires_grad=False).to(self.device)
        
        save_pth_C_up =os.path.join(save_dir,'C_up.pth')
        save_pth_C_nup =os.path.join(save_dir,'C_nup.pth')
        save_pth_descriptor = os.path.join(save_dir,'descriptor.pth')
        save_pth_project = os.path.join(save_dir,'project.pth')
        torch.save(self.C_nup, save_pth_C_nup)
        torch.save(self.C_up, save_pth_C_up)
        torch.save(self.descriptor, save_pth_descriptor)
        torch.save(save_pth_project , save_pth_project)
        print('_init_centroid done')

    def _roiwithoutup(self,boxes):
        ins_feats = [[] for b in boxes]
        for i in range(len(self.layerindexes)):
            psize = self.pool_sizes[i]
            l = self.layerindexes[i]
            m = MultiScaleRoIAlign([str(l)],psize,-1)
            #m = MultiScaleRoIAlign([str(l)],psize,2)
            roi_feats = m(self.feat_dict,[boxes],[self.imgsize])
            for bi in range(boxes.shape[0]):
                ins_feats[bi].append(roi_feats[bi])
        return ins_feats
    
    def _roiwithup(self,boxes):
        ins_feats = []
        psize = self.pool_sizes[0]
        for l in self.layerindexes:
            m = MultiScaleRoIAlign([str(l )],psize,-1)
            # m = MultiScaleRoIAlign([str(l )],psize,2)
            ins_feats.append(m(self.feat_dict,[boxes],[self.imgsize]))
        ins_feats = torch.cat(ins_feats,dim=1)
        return ins_feats
    
    def _roi_img(self,imgs,boxes):
        ins_feats = []
        psize = self.pool_sizes[0]
        img_dict= {}
        names  = []
        for i,img in enumerate(imgs):
            img_dict[str(i)] = img[None] *255.0
            b = boxes[i][:,:4]
            if len(b) == 0:
                continue
            # m = MultiScaleRoIAlign([str(i)],psize,2)
            m = MultiScaleRoIAlign([str(i)],psize,-1)
            ins_feats.append(m(img_dict,[b],[self.imgsize]))
            for j in range(len(b)):
                names.append(str(i)+'_'+str(j))

        ins_feats = torch.cat(ins_feats,dim=0)
        return ins_feats,names


    def forward(self, x,targets):
        self.feat_dict.clear()
        pred = self.model(x)
        if self.detect:
            detects = targets
            detects *= torch.tensor([self.imgsize[0],self.imgsize[1],self.imgsize[0],self.imgsize[1]])
            detects = [detects.to(self.device)]
            # detects = non_max_suppression(pred , 0.75,  0.50, self.classes, max_det=100)
        else:
            detects = targets[:,2:]
            detects[:,:2] -= detects[:,2:]/2
            detects[:,2:] += detects[:,:2]
            detects *= torch.tensor([self.imgsize[0],self.imgsize[1],self.imgsize[0],self.imgsize[1]])
            detects = [detects.to(self.device)]
        del pred
        phi_ps = []
        batch_id =0
        loss = None
        
        for d in detects:
            if len(d) == 0:
                batch_id += 1.0
                continue
            else:
                boxes  = d[:,:4]
                wh =  boxes[:,2:] - boxes[:,:2]
                cxcy = boxes[:,:2] + wh/2
                newboxes = torch.cat([cxcy, 1.20*wh],dim=-1)
                newboxes[:,:2] -= newboxes[:,2:]/2
                newboxes[:, 2:] += newboxes[:,:2]
                newboxes = newboxes.clamp(max=self.imgsize[0]-1,min=1.0)
                boxes = newboxes
                ins_feats = self._roiwithoutup(boxes)
                
                phi_p = [self.descriptor(feat) for feat in ins_feats]
                phi_p = torch.cat(phi_p,dim=0)
                phi_ps.append(phi_p)
                batch_id += 1.0
        if len(phi_ps) == 0:
            if self.detect:
                return None,None,None
            else:
                return None,None,None
        phi_ps = torch.cat(phi_ps,dim=0)     
        phi_ps = rearrange(phi_ps, 'b c h w -> b (h w) c')
        ni = phi_ps.shape[0]
        if self.detect:
            img_ins,save_names = self._roi_img(x,[boxes])
            #img_ins,save_names = self._roi_img(x,detects)
            n_neighbors = self.K
            if ni> self.max_nums:
                select_indexes = torch.randperm(ni)
                select_indexes = select_indexes[:self.max_nums]
                phi_ps= phi_ps[select_indexes]
                save_names = save_names[[select_indexes]]
                img_ins = img_ins[select_indexes]
            features = torch.sum(torch.pow(phi_ps, 2), 2, keepdim=True)    
            centers  = torch.sum(torch.pow(self.C_nup, 2), 0, keepdim=True)
            f_c = 2 * torch.matmul(phi_ps, (self.C_nup))
            dist = features + centers - f_c
            dist = torch.sqrt(dist)
            dist = (dist.topk(n_neighbors, largest=False).values)
            #dist = ((F.softmin(dist, dim=-1)) * dist).mean(dim=-1)
            dist = (F.softmin(dist, dim=-1)[:, :, 0]) * dist[:, :, 0]
            dist = dist.unsqueeze(-1)
            score = rearrange(dist, 'b (h w) c -> b c h w', h=self.pool_sizes[0])
            heatmap = score.cpu().detach()
            heatmap = heatmap.mean(dim=1)
            img_ins = img_ins.cpu()
            return heatmap,img_ins,save_names

        if ni> self.max_nums:
            select_indexes = torch.randperm(ni)
            select_indexes = select_indexes[:self.max_nums]
            phi_ps= phi_ps[select_indexes]

        loss , L_att , L_rep = self._soft_boundary(phi_ps)
        # img_ins, save_names = self._roi_img(x,detects)
        return loss.cpu(), L_att.cpu() ,L_rep.cpu()
        
    def _soft_boundary(self, phi_p):
        features = torch.sum(torch.pow(phi_p, 2), 2, keepdim=True)
        centers  = torch.sum(torch.pow(self.C_nup, 2), 0, keepdim=True)
        f_c      = 2 * torch.matmul(phi_p, (self.C_nup))
        dist     = features + centers - f_c
        n_neighbors = self.K + self.J
        dist     = dist.topk(n_neighbors, largest=False).values

        score = (dist[:, : , :self.K] - self.r**2) 
        L_att = (1/self.nu) * torch.mean(torch.max(torch.zeros_like(score), score))
        
        score = (self.r**2 - dist[:, : , self.J:]) 
        L_rep  = (1/self.nu) * torch.mean(torch.max(torch.zeros_like(score), score - self.alpha))
        #loss = L_att
        loss = L_att + L_rep

        return loss , L_att , L_rep



class AN_CFA_V2(nn.Module): 
    def __init__(self, weights='yolov5x.pt',weighall=False, device=torch.device('cpu'), data=None,cfg='./models/yolov5x.yaml',
                 layerindexes=[0,2,17],ckpt=None,names=[],c_pt =None, fp16=False,imgsize=(640, 640),detect = False):
        super().__init__()
        self.nu = 1e-3
        self.scale = None
        self.gamma_c = 1
        self.gamma_d = 1
        self.alpha = 1e-2
        self.K = 3
        self.J = 3
        self.r   = nn.Parameter(1.0*torch.ones(1), requires_grad=True)
        self.C_up = 0
        self.C_nup = 0
        channels = [80,160,320,640]
        pool_sizes = [128,64,32,16]
        
        self.pool_sizes = [ pool_sizes[i] for i in range(len(layerindexes))]
        if c_pt!= None: 
            self.center = torch.load(c_pt).to(device)
        self.device = device
        self.project = None
        self.fp16 = fp16
        self.imgsize = (640, 640)
        self.des_c = 560
        if ckpt is None:
            if isinstance(weights,list):
                self.layerindexes = layerindexes 
                c_w = weights[1]
                des_w = weights[2] 
                weights = weights[0]
                ckpt = torch.load(weights, map_location='cpu')
                self.C_nup = torch.load(c_w,map_location='cpu')
                self.C_nup.requires_grad = False
                self.descriptor = torch.load(des_w,map_location='cpu')
                self.descriptor.train()
                self.model = ckpt['model']
                
                
            else:
                self.layerindexes = layerindexes
                self.channels = [channels[i] for i in range(len(self.layerindexes))]
                self.pool_sizes= [pool_sizes[i] for i in range(len(self.layerindexes))]
                ckpt = torch.load(weights, map_location='cpu')
                des_c = int(torch.tensor(self.channels).sum())
                self.des_c = des_c
                # [64,128,256,512] * 1.25
                self.descriptor = Descriptor(self.gamma_d,des_c,device).to(device)
                self.project  = CoordConv2d(des_c , des_c //self.gamma_d, 1)
                self.model = ckpt['model']
                self.descriptor.train()
        else:
            ckpt = torch.load(ckpt, map_location='cpu')
            self.model = ckpt['model'].model
            self.descriptor = ckpt['model'].descriptor
            self.project =  ckpt['model'].project
            self.C_nup = ckpt['model'].C_nup.to(device)
            self.r   = ckpt['model'].r
            self.C_nup.requires_grad = False
            self.descriptor.train()
            self.layerindexes = ckpt['model'].layerindexes
            self.pool_sizes = ckpt['model'].pool_sizes 
            self.des_c =  ckpt['model'].des_c

        self.device = device
        self.model = self.model.eval()
        self.model.float()
        self.model.to(device)
        self.descriptor.to(device)
        if self.project:
            self.project.to(device)
        nc = self.model.yaml['nc']
        stride = max(int(self.model.stride.max()), 32)  # model stride
        self.names = self.model.names  # get class names
        self.stride = stride
        # self.layerindexes = layerindexes
        for param in self.model.parameters():
                param.requires_grad = False
        
        # self.pool_sizes = [128,64,32]
        # self.channels = [80,160,320]
        self.feat_dict = {}
        self.classes = [0,1,2]
        self.imgsize = imgsize
        self.max_nums = 4
        self.detect = detect
        
        # self.convs_list = [self._make_cbn_block (c).to(self.device) for c in self.channels ]

        for i in self.layerindexes:
            forward_hook = ForwardHook(self.feat_dict,i)
            # forward_hook = ForwardHook_avg(self.feat_dict,i)
            self.model.model[i].register_forward_hook(forward_hook)
        if detect:
            self.classes = [0]
            # self.classes = [5,6]
    
    def _make_cbn_block(self,channel):
        conv = Bottleneck(channel,channel,3,1,1) 
        return conv
        # conv = nn.Conv2d(channel,channel,3,1,1) 
        # bn = nn.BatchNorm2d(channel)
        # act = nn.ReLU()
        # nn.init.kaiming_normal_(conv.weight, mode='fan_out', nonlinearity='relu')
        # nn.init.constant_(bn.weight, 1)
        # nn.init.constant_(bn.bias, 0)
        # return nn.Sequential(conv,bn,act)

    def _init_centroid(self, data_loader,save_dir='/mnt/ecc5c6c9-7631-4983-9ba0-1ec98729589b/llc/AN_NEW/ST-CFA/'):
        self.C_up = 0.0
        del self.C_nup
        self.C_nup = 0.0
        self.descriptor = Descriptor(self.gamma_d,self.des_c,self.device).to(self.device)
        self.project  = CoordConv2d(self.des_c, self.des_c//self.gamma_d, 1).to(self.device)

        pbar = enumerate(data_loader)
        pbar = tqdm(pbar, total=1, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}') 
        count = 0
        for i, (imgs, targets, paths, _) in pbar: 
        # for i, (path, imgs,_, _, _) in pbar:
            imgs = imgs.to(self.device, non_blocking=True).float() / 255.0
            # imgs = torch.from_numpy(imgs).to(self.device).float()/ 255.0
            if len(imgs.shape) == 3:
                imgs = imgs[None]  # expand for batch dim
            if len(targets) ==0:
                continue
            self.feat_dict.clear()
            pred = self.model(imgs)
            del pred
            # detects = non_max_suppression(pred , 0.2,  0.50, self.classes, max_det=100)[0]
            # if len(detects)==0:
            #     continue
            # boxes  = detects[:,:4]
            # boxes = boxes.cpu()
            # wh =  boxes[:,2:] - boxes[:,:2]
            # cxcy = (boxes[:,:2] +  boxes[:,2:])/2
            boxes = targets[:,2:]
            
            wh =  boxes[:,2:]
            cxcy = boxes[:,:2]
            newboxes = torch.cat([cxcy, 1.20*wh],dim=-1)
            newboxes[:,:2] -= newboxes[:,2:]/2
            newboxes[:, 2:] += newboxes[:,:2]
            newboxes = newboxes.clamp(max=1.0,min=0.0)
            newboxes[:,3] = newboxes[:,3].clamp(max=0.781,min=0.0)
            boxes = newboxes
            # newboxes = newboxes.clamp(max=self.imgsize[0]-1.0,min=0.0)
            # boxes = newboxes
            boxes *= torch.tensor([self.imgsize[0],self.imgsize[1],self.imgsize[0],self.imgsize[1]])
            boxes = boxes.to(self.device)
            ins_feats_nup = self._roiwithoutup(boxes)
            nup_phi_ps = [self.descriptor(feat) for feat in ins_feats_nup]
            # img_ins,save_names = self._roi_img(imgs ,[boxes])
            
            # for k in range(len(img_ins)):
            #     cv2.imwrite(str(save_dir)+'/'+str(i)+'_'+save_names[k]+'.jpg',img_ins[k].cpu().permute(1,2,0).numpy()[:,:,[2,1,0]])
            for j in range(len( boxes )):
                nup_phi =   nup_phi_ps[j].detach()
                self.C_nup = ((self.C_nup * count) + nup_phi) / (count+1)
                count += 1
            
            self.feat_dict.clear()

        self.C_nup = rearrange(self.C_nup, 'b c h w -> (b h w) c').detach().cpu()
        if self.gamma_c > 1:
            self.C_up= self.C_up.cpu().detach().numpy()
            self.C_up= KMeans(n_clusters=(self.scale**2)//self.gamma_c, max_iter=3000).fit(self.C_up).cluster_centers_
            self.C_up = torch.Tensor(self.C_up)
        
        self.C_nup = self.C_nup.transpose(-1, -2).detach()
        self.C_nup = nn.Parameter(self.C_nup , requires_grad=False).to(self.device)
        
        
        save_pth_C_nup =os.path.join(save_dir,'C_nup.pth')
        save_pth_descriptor = os.path.join(save_dir,'descriptor.pth')
        
        torch.save(self.C_nup, save_pth_C_nup)
        torch.save(self.descriptor, save_pth_descriptor)
        
        print('_init_centroid done')

    def _roiwithoutup(self,boxes):
        ins_feats = [[] for b in boxes]
        for i in range(len(self.layerindexes)):
            psize = self.pool_sizes[i]
            l = self.layerindexes[i]
            m = MultiScaleRoIAlign([str(l)],psize,-1)
           
            roi_feats = m(self.feat_dict,[boxes],[self.imgsize])
            # roi_feats = self.convs_list[i]((roi_feats))
            for bi in range(boxes.shape[0]):
                ins_feats[bi].append(roi_feats[bi])
        return ins_feats
    
    def _roiwithup(self,boxes):
        ins_feats = []
        psize = self.pool_sizes[0]
        for l in self.layerindexes:
            m = MultiScaleRoIAlign([str(l )],psize,-1)
            ins_feats.append(m(self.feat_dict,[boxes],[self.imgsize]))
        ins_feats = torch.cat(ins_feats,dim=1)
        return ins_feats
    
    def _roi_img(self,imgs,boxes):
        ins_feats = []
        psize = self.pool_sizes[0]
        img_dict= {}
        names  = []
        for i,img in enumerate(imgs):
            img_dict[str(i)] = img[None] *255.0
            b = boxes[i][:,:4]
            if len(b) == 0:
                continue
            m = MultiScaleRoIAlign([str(i)],psize,-1)
            ins_feats.append(m(img_dict,[b],[self.imgsize]))
            for j in range(len(b)):
                names.append(str(i)+'_'+str(j))

        ins_feats = torch.cat(ins_feats,dim=0)
        return ins_feats,names

    def forward(self, x,targets,epoch):
        self.feat_dict.clear()
        pred = self.model(x)
        if self.detect:
            detects = targets
            detects *= torch.tensor([self.imgsize[0],self.imgsize[1],self.imgsize[0],self.imgsize[1]])
            detects = [detects.to(self.device)]
        else:
            detects = targets[:,2:]
            detects[:,:2] -= detects[:,2:]/2
            detects[:,2:] += detects[:,:2]
            detects *= torch.tensor([self.imgsize[0],self.imgsize[1],self.imgsize[0],self.imgsize[1]])
            detects = [detects.to(self.device)]
        del pred
        phi_ps = []
        batch_id =0
        loss = None
        
        for d in detects:
            if len(d) == 0:
                batch_id += 1.0
                continue
            else:
                boxes  = d[:,:4]
                wh =  boxes[:,2:] - boxes[:,:2]
                cxcy = boxes[:,:2] + wh/2
                newboxes = torch.cat([cxcy, 1.20*wh],dim=-1)
                newboxes[:,:2] -= newboxes[:,2:]/2
                newboxes[:, 2:] += newboxes[:,:2]
                newboxes = newboxes.clamp(max=self.imgsize[0]-1,min=0.0)
                newboxes[:,3] = newboxes[:,3].clamp(max=500-1,min=0.0)
                boxes = newboxes
                ins_feats = self._roiwithoutup(boxes)
                
                phi_p = [self.descriptor(feat) for feat in ins_feats]
                phi_p = torch.cat(phi_p,dim=0)
                phi_ps.append(phi_p)
                batch_id += 1.0
        if len(phi_ps) == 0:
            if self.detect:
                return None,None,None
            else:
                return None,None,None
        phi_ps = torch.cat(phi_ps,dim=0)     
        phi_ps = rearrange(phi_ps, 'b c h w -> b (h w) c')
        ni = phi_ps.shape[0]
        if self.detect:
            img_ins,save_names = self._roi_img(x,[boxes])
            #img_ins,save_names = self._roi_img(x,detects)
            n_neighbors = self.K
            if ni> self.max_nums:
                select_indexes = torch.randperm(ni)
                select_indexes = select_indexes[:self.max_nums]
                phi_ps= phi_ps[select_indexes]
                save_names = save_names[[select_indexes]]
                img_ins = img_ins[select_indexes]
            features = torch.sum(torch.pow(phi_ps, 2), 2, keepdim=True)    
            centers  = torch.sum(torch.pow(self.C_nup, 2), 0, keepdim=True)
            f_c = 2 * torch.matmul(phi_ps, (self.C_nup))
            dist = features + centers - f_c
            dist = torch.sqrt(dist)
            dist = (dist.topk(n_neighbors, largest=False).values)
            #dist = ((F.softmin(dist, dim=-1)) * dist).mean(dim=-1)
            dist = (F.softmin(dist, dim=-1)[:, :, 0]) * dist[:, :, 0]
            dist = dist.unsqueeze(-1)
            score = rearrange(dist, 'b (h w) c -> b c h w', h=self.pool_sizes[0])
            heatmap = score.cpu().detach()
            heatmap = heatmap.mean(dim=1)
            img_ins = img_ins.cpu()
            return heatmap,img_ins,save_names

        if ni> self.max_nums:
            select_indexes = torch.randperm(ni)
            select_indexes = select_indexes[:self.max_nums]
            phi_ps= phi_ps[select_indexes]
       
        loss , L_att , L_rep = self._soft_boundary(phi_ps)
        #loss =  (epoch[0] + 1)* L_att + (epoch[1]- epoch[0] -1)* L_rep 
        # loss =  (epoch[1]- epoch[0] -1)* L_att + (epoch[0] + 1)* L_rep 
        return loss.cpu(), L_att.cpu() ,L_rep.cpu()
        # img_ins, save_names = self._roi_img(x,detects)
            
    def _soft_boundary(self, phi_p):
        features = torch.sum(torch.pow(phi_p, 2), 2, keepdim=True)
        centers  = torch.sum(torch.pow(self.C_nup, 2), 0, keepdim=True)
        f_c      = 2 * torch.matmul(phi_p, (self.C_nup))
        dist     = features + centers - f_c
        n_neighbors = self.K + self.J
        dist     = dist.topk(n_neighbors, largest=False).values
        score = (dist[:, : , :self.K] - self.r**2)  
        L_att = (1/self.nu) * torch.mean(torch.max(torch.zeros_like(score), score))
        # L_att = torch.sum(torch.max(torch.zeros_like(score), score))
        score = (self.r**2 - dist[:, : , self.J:]) 
        L_rep  = (1/self.nu) * torch.mean(torch.max(torch.zeros_like(score), score - self.alpha))
        # L_rep  = torch.sum(torch.max(torch.zeros_like(score), score - self.alpha))
        loss = L_att + L_rep
        return loss , L_att , L_rep
       
        


class AN_CFA_ALL(nn.Module):
    def __init__(self, weights='yolov5x.pt',weighall=False, device=torch.device('cpu'), data=None,cfg='./models/yolov5x.yaml',
                 layerindexes=[2,17,20],ckpt=None,names=[],c_pt =None, fp16=False,imgsize=(640, 640),detect = False):
        super().__init__()
        self.nu = 1e-3
        self.scale = None
        self.gamma_c = 1
        self.gamma_d = 1
        self.alpha = 1e-1
        self.K = 3
        self.J = 3
        self.r   = nn.Parameter(1e-5*torch.ones(1), requires_grad=True)
        self.C_up = 0
        self.C_nup = 0
        if c_pt!= None: 
            self.center = torch.load(c_pt).to(device)
        self.device = device
        self.project = None
        self.fp16 = fp16
        self.imgsize = (640, 640)
        
        if ckpt is None:
            if isinstance(weights,list):
                c_w = weights[1]
                des_w = weights[2] 
                # pro_w =  weights[3]
                weights = weights[0]
                ckpt = torch.load(weights, map_location='cpu')
                self.C_nup = torch.load(c_w,map_location='cpu')
                self.C_nup.requires_grad = False
                self.descriptor = torch.load(des_w,map_location='cpu')
                self.descriptor.train()
                self.model = ckpt['model']
                self.cnn = resnet18()
                self.cnn = self.cnn.eval()
                self.cnn = self.cnn.to(device)
       
                # self.names = names
            else:
                ckpt = torch.load(weights, map_location='cpu')
                # [64,128,256,512] * 1.25
                self.descriptor = Descriptor_all(self.gamma_d,1120,device).to(device)
                # self.descriptor = Descriptor_all(self.gamma_d,448,device).to(device)
                self.project  = CoordConv2d(1120, 1120//self.gamma_d, 1)
                self.model = ckpt['model']
                self.descriptor.train()
                self.cnn = resnet18()
                self.cnn = self.cnn.eval()
                self.cnn = self.cnn.to(device)
        else:
            ckpt = torch.load(ckpt, map_location='cpu')
            self.model = ckpt['model'].model
            self.descriptor = ckpt['model'].descriptor
            self.cnn = ckpt['model'].cnn
            self.C_nup = ckpt['model'].C_nup.to(device)
            self.r   = ckpt['model'].r
            self.C_nup.requires_grad = False
            self.descriptor.train()
            
        self.device = device
        self.model = self.model.eval()
        self.model.float()
        self.model.to(device)
        self.descriptor.to(device)
        if self.project:
            self.project.to(device)
        nc = self.model.yaml['nc']
        stride = max(int(self.model.stride.max()), 32)  # model stride
        self.names = self.model.names  # get class names
        self.stride = stride
        self.layerindexes = layerindexes
        for param in self.model.parameters():
                param.requires_grad = False
        self.layerindexes = layerindexes
        self.pool_sizes = [128,64,32,16]
        self.feat_dict = {}
        self.classes = [0,1,2]
        self.imgsize = imgsize
        self.max_nums = 4
        self.detect = detect
        for i in self.layerindexes:
            forward_hook = ForwardHook(self.feat_dict,i)
            self.model.model[i].register_forward_hook(forward_hook)
        if detect:
            self.classes = [0]
            # self.classes = [5,6]

    def _init_centroid(self, data_loader,save_dir='/mnt/ecc5c6c9-7631-4983-9ba0-1ec98729589b/llc/AN_NEW/ST-CFA/'):
        self.C_up = 0.0
        del self.C_nup
        self.C_nup = 0.0
        self.descriptor = Descriptor_all(self.gamma_d,1120,self.device).to(self.device)
        # self.descriptor = Descriptor_all(self.gamma_d,448,self.device).to(self.device)
        self.project  = CoordConv2d(1120, 1120//self.gamma_d, 1).to(self.device)

        pbar = enumerate(data_loader)
        pbar = tqdm(pbar, total=1, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}') 
        count = 0
        for i, (imgs, targets, paths, _) in pbar: 
            imgs = imgs.to(self.device, non_blocking=True).float() / 255.0
            if len(imgs.shape) == 3:
                imgs = imgs[None]  # expand for batch dim
            if len(targets) ==0:
                continue
            self.feat_dict.clear()
            # feat_list= self.cnn(imgs)
            pred = self.model(imgs)
            
            # cv2.imwrite('/mnt/ecc5c6c9-7631-4983-9ba0-1ec98729589b/llc/'+str(i)+'.jpg',np.uint8(imgs[0].permute(1,2,0).cpu().numpy()))
            
            feat_list = [self.feat_dict[str(l)] for l in self.layerindexes]
            nup_phi_ps = self.descriptor(feat_list).detach()
            self.C_nup = ((self.C_nup * count) + nup_phi_ps) / (count+1)
            count += 1
            self.feat_dict.clear()

        self.C_nup = rearrange(self.C_nup, 'b c h w -> (b h w) c').detach().cpu()
        self.C_nup = self.C_nup.transpose(-1, -2).detach()
        self.C_nup = nn.Parameter(self.C_nup , requires_grad=False).to(self.device)
        save_pth_C_nup =os.path.join(save_dir,'C_nup.pth')
        save_pth_descriptor = os.path.join(save_dir,'descriptor.pth')
        torch.save(self.C_nup, save_pth_C_nup)
        torch.save(self.descriptor, save_pth_descriptor)
        print('_init_centroid done')

    def forward(self, x,targets):
        self.feat_dict.clear()
        # feat_list= self.cnn(imgs)
        pred = self.model(x)
        feat_list = [self.feat_dict[str(l)] for l in self.layerindexes]
        phi_ps = self.descriptor(feat_list)
        self.feat_dict.clear()
        phi_ps = rearrange(phi_ps, 'b c h w -> b (h w) c')

        
        if self.detect:
            n_neighbors = self.K
            features = torch.sum(torch.pow(phi_ps, 2), 2, keepdim=True)    
            centers  = torch.sum(torch.pow(self.C_nup, 2), 0, keepdim=True)
            f_c = 2 * torch.matmul(phi_ps, (self.C_nup))
            dist = features + centers - f_c
            dist = torch.sqrt(dist)
            dist = (dist.topk(n_neighbors, largest=False).values)
            
            dist = (F.softmin(dist, dim=-1)[:, :, 0]) * dist[:, :, 0]
            dist = dist.unsqueeze(-1)
            score = rearrange(dist, 'b (h w) c -> b c h w', h=160)
            heatmap = score.cpu().detach()
            heatmap =F.interpolate(heatmap, 640, mode='bilinear',align_corners=True)
            heatmap = heatmap.mean(dim=1)
            
            img = x.cpu()
            return heatmap,img,'0'
        loss , L_att , L_rep = self._soft_boundary(phi_ps)
        return loss.cpu(), L_att.cpu() ,L_rep.cpu()
        
    def _soft_boundary(self, phi_p):
        features = torch.sum(torch.pow(phi_p, 2), 2, keepdim=True)
        centers  = torch.sum(torch.pow(self.C_nup, 2), 0, keepdim=True)
        f_c      = 2 * torch.matmul(phi_p, (self.C_nup))
        dist     = features + centers - f_c
        n_neighbors = self.K + self.J
        dist     = dist.topk(n_neighbors, largest=False).values

        score = (dist[:, : , :self.K] - self.r**2) 
        L_att = (1/self.nu) * torch.mean(torch.max(torch.zeros_like(score), score))
        
        score = (self.r**2 - dist[:, : , self.J:]) 
        L_rep  = (1/self.nu) * torch.mean(torch.max(torch.zeros_like(score), score - self.alpha))
        #loss = L_att
        loss = L_att + L_rep

        return loss , L_att , L_rep

class CoordConv2d(torch.nn.modules.conv .Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                padding=0, dilation=1, groups=1, bias=True, with_r=False, use_cuda=True):
        super(CoordConv2d, self).__init__(in_channels, out_channels, kernel_size,
                                        stride, padding, dilation, groups, bias)
        self.rank = 2
        self.addcoords = AddCoords(self.rank, with_r)
        self.conv = nn.Conv2d(in_channels + self.rank + int(with_r), out_channels,
                            kernel_size, stride, padding, dilation, groups, bias)
    def forward(self, input_tensor):
        out = self.addcoords(input_tensor)
        out = self.conv(out)
        return out

class AddCoords(nn.Module):
    def __init__(self, rank, with_r=False):
        super(AddCoords, self).__init__()
        self.rank = rank
        self.with_r = with_r
        
    def forward(self, input_tensor):
        batch_size_shape, _, dim_y, dim_x = input_tensor.shape
        xx_ones = torch.ones([1, 1, 1, dim_x], dtype=torch.int32)
        yy_ones = torch.ones([1, 1, 1, dim_y], dtype=torch.int32)

        xx_range = torch.arange(dim_y, dtype=torch.int32)
        yy_range = torch.arange(dim_x, dtype=torch.int32)
        xx_range = xx_range[None, None, :, None]
        yy_range = yy_range[None, None, :, None]

        xx_channel = torch.matmul(xx_range, xx_ones)
        yy_channel = torch.matmul(yy_range, yy_ones)

        yy_channel = yy_channel.permute(0, 1, 3, 2)

        xx_channel = xx_channel.float() / (dim_y - 1)
        yy_channel = yy_channel.float() / (dim_x - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size_shape, 1, 1, 1)
        yy_channel = yy_channel.repeat(batch_size_shape, 1, 1, 1)
        xx_channel = xx_channel.to(input_tensor.device)
        yy_channel = yy_channel.to(input_tensor.device)
            
        out = torch.cat([input_tensor, xx_channel, yy_channel], dim=1)

        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_channel - 0.5, 2) + torch.pow(yy_channel - 0.5, 2))
            out = torch.cat([out, rr], dim=1)

        return out

class Descriptor(nn.Module):
    def __init__(self, gamma_d,dim,device):
        super(Descriptor, self).__init__()
        self.layer = CoordConv2d(dim, dim//gamma_d, 1)
        # self.bnc = [80,160,320]
        # self.bnlist  = []
        # for c in self.bnc:
        #     bn = nn.BatchNorm2d(c).to(device) 
        #     self.bnlist.append(bn) 
        #     nn.init.constant_(bn.weight, 1)
        #     nn.init.constant_(bn.bias, 0)
    def forward(self, p):
        sample = None
        count = 0
        for o in p:
            # o =self.bnlist[count](o[None])
            # o = F.relu(o)
            #o =  F.avg_pool2d(o, 3, 1, 1)
            o =  F.avg_pool2d(o[None], 3, 1, 1)
            sample = o if sample is None else torch.cat((sample, F.interpolate(o, sample.size(2), mode='bilinear',align_corners=True)), dim=1)
            # sample = o[None] if sample is None else torch.cat((sample, F.interpolate(o[None], sample.size(2), mode='bilinear',align_corners=True)), dim=1)
           
            count += 1
        phi_p = self.layer(sample)
        return phi_p

class Descriptor_ori(nn.Module):
    def __init__(self, gamma_d,dim,device):
        super( Descriptor_ori, self).__init__()
        self.layer = CoordConv2d(dim, dim//gamma_d, 1)
        # self.bnc = [64,128,256]
        # self.bnlist =[nn.BatchNorm2d(c).to(device) for c in self.bnc] 
    def forward(self, p):
        sample = None
        count = 0
        for o in p:
            # o =self.bnlist[count](o[None])
            # o =  F.avg_pool2d(o, 3, 1, 1)
            o =  F.avg_pool2d(o[None], 3, 1, 1)
            sample = o if sample is None else torch.cat((sample, F.interpolate(o, sample.size(2), mode='bilinear',align_corners=True)), dim=1)
            count += 1
        phi_p = self.layer(sample)
        return phi_p

class Descriptor_all(nn.Module):
    def __init__(self, gamma_d,dim,device):
        super( Descriptor_all, self).__init__()
        self.layer = CoordConv2d(dim, dim//gamma_d, 1)
        # self.bnc = [64,128,256]
        # self.bnlist =[nn.BatchNorm2d(c).to(device) for c in self.bnc] 
    def forward(self, p):
        sample = None
        count = 0
        for o in p:
            o = F.avg_pool2d(o, 3, 1, 1) 
            sample = o if sample is None else torch.cat((sample, F.interpolate(o, sample.size(2), mode='bilinear',align_corners=True)), dim=1)
            count += 1
        phi_p = self.layer(sample)
        return phi_p




class Discriminator(torch.nn.Module):
    def __init__(self, in_planes=1536, n_layers=2, hidden=None):
        super(Discriminator, self).__init__()

        _hidden = in_planes if hidden is None else hidden
        self.body = torch.nn.Sequential()
        for i in range(n_layers-1):
            _in = in_planes if i == 0 else _hidden
            _hidden = int(_hidden // 1.5) if hidden is None else hidden
            self.body.add_module('block%d'%(i+1),
                                 torch.nn.Sequential(
                                     torch.nn.Linear(_in, _hidden),
                                     torch.nn.BatchNorm1d(_hidden),
                                     torch.nn.LeakyReLU(0.2)
                                 ))
        self.tail = torch.nn.Linear(_hidden, 1, bias=False)
        self.apply(init_weight)

    def forward(self,x):
        x = self.body(x)
        x = self.tail(x)
        return x

class Preprocessing(torch.nn.Module):
    def __init__(self, input_dims = [160,320,640,1280], output_dim=1024,size = 160):
        super(Preprocessing, self).__init__()
        self.input_dims = input_dims
        self.output_dim = output_dim
        self.size = 160
        self.preprocessing_modules = torch.nn.ModuleList()
        for input_dim in input_dims:
            module = MeanMapper(output_dim)
            self.preprocessing_modules.append(module)

    def forward(self, features):
        _features = []
        for module, feature in zip(self.preprocessing_modules, features):
            _feature = module(feature)
            # _feature = _feature.reshape(_feature.shape[0]//(self.size **2),self.size,self.size,_feature.shape[1])
            _features.append(module(feature))
        return torch.stack(_features, dim=1)
        # return torch.stack(_features, dim=1)
    
class MeanMapper(torch.nn.Module):
    def __init__(self, preprocessing_dim=1024):
        super(MeanMapper, self).__init__()
        self.preprocessing_dim = preprocessing_dim

    def forward(self, features):
        features = features.reshape(len(features), 1, -1)
        return F.adaptive_avg_pool1d(features, self.preprocessing_dim).squeeze(1)

class Aggregator(torch.nn.Module):
    def __init__(self, target_dim=1536):
        super(Aggregator, self).__init__()
        self.target_dim = target_dim

    def forward(self, features):
        """Returns reshaped and average pooled features."""
        # batchsize x number_of_layers x input_dim -> batchsize x target_dim
        features = features.reshape(len(features), 1, -1)
        features = F.adaptive_avg_pool1d(features, self.target_dim)
        return features.reshape(len(features), -1)
    
class PatchMaker:
    def __init__(self, patchsize=3, top_k=0, stride=1):
        self.patchsize = patchsize
        self.stride = stride
        self.top_k = top_k
        

    def patchify(self, features, return_spatial_info=False):
        """Convert a tensor into a tensor of respective patches.
        Args:
            x: [torch.Tensor, bs x c x w x h]
        Returns:
            x: [torch.Tensor, bs * w//stride * h//stride, c, patchsize,
            patchsize]
        """
        padding = int((self.patchsize - 1) / 2)
        unfolder = torch.nn.Unfold(
            kernel_size=self.patchsize, stride=self.stride, padding=padding, dilation=1
        )
        unfolded_features = unfolder(features)
        number_of_total_patches = []
        for s in features.shape[-2:]:
            n_patches = (
                s + 2 * padding - 1 * (self.patchsize - 1) - 1
            ) / self.stride + 1
            number_of_total_patches.append(int(n_patches))
        unfolded_features = unfolded_features.reshape(
            *features.shape[:2], self.patchsize, self.patchsize, -1
        )
        unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)

        if return_spatial_info:
            return unfolded_features, number_of_total_patches
        return unfolded_features

    def unpatch_scores(self, x, batchsize):
        return x.reshape(batchsize, -1, *x.shape[1:])

    def score(self, x):
        was_numpy = False
        if isinstance(x, np.ndarray):
            was_numpy = True
            x = torch.from_numpy(x)
        while x.ndim > 2:
            x = torch.max(x, dim=-1).values
        if x.ndim == 2:
            if self.top_k > 1:
                x = torch.topk(x, self.top_k, dim=1).values.mean(1)
            else:
                x = torch.max(x, dim=1).values
        if was_numpy:
            return x.numpy()
        return x


class STANNet(nn.Module):
    def __init__(self, weights_t='yolov5x.pt',weights_s=None, weight_all=None, device=torch.device('cpu'),
                 cfg_t='./models/yolov5s.yaml', cfg_s='./models/Syolov5x.yaml',
                 layerindexes=[17, 20, 23], nc=None, anchors=None,imgsize=640,name='yolo'):
        super().__init__()
        self.layerindexes = layerindexes
        config_t = (cfg_t,nc,anchors,device)
        config_s = (cfg_s,nc,anchors,device)
        model_t = self.make_teachernet(name,  weights_t, config_t)
        model_s = self.make_studentnet(name, config_s)
        self.names = model_t.module.names if hasattr(model_t, 'module') else model_t.names  # get class names
        # if weight_all is None:
        #     t_ckpt = torch.load(weights_t, map_location='cpu')
        #     from models.yolo import Model
        #     from utils.general import intersect_dicts
        #     model_t = Model(cfg_t, ch=3, nc=nc, anchors=anchors).to(device)
        #     exclude = ['anchor'] if (cfg_t or anchors) else []  # exclude keys
        #     csd = t_ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
        #     csd = intersect_dicts(csd, model_t.state_dict(), exclude=exclude)  # intersect
        #     model_t.load_state_dict(csd, strict=False)  # load

        #     if type == 'test':
        #         for param in model_t.parameters():
        #             param.requires_grad = False
        #         self.model_t = model_t.eval()
        #     else:
        #         self.model_t = model_t
        #     self.model_t.float()
        #     self.model_t.to(device)
        #     nc = self.model_t.yaml['nc']
        #     self.model_s = Model(cfg_s, ch=3, nc=nc).to(device)
        self.model_t = model_t.float()
        self.model_s = model_s.float()
        self.model_t.to(device)
        self.model_s.to(device)
        self.names = model_t.names  # get class names
        nc = self.model_t.yaml['nc']
        self.nc = nc
        self.stride = model_t.stride
        self.layerindexes = layerindexes
        self.imgsize = [imgsize,imgsize] if isinstance(imgsize, int) else imgsize
        print('STANNet done')


    def forward(self, im):
        out_t, feature_t = self.model_t(im, layers=self.layerindexes)
        out_s, feature_s = self.model_s(im, layers=self.layerindexes)
        assert len(feature_t) == len(feature_s)
        assert len(feature_t) == len(self.layerindexes)
        assert len(feature_s) == len(self.layerindexes)

        return (out_t, [t.detach() for t in feature_t], feature_s)

    def make_teachernet(self,name,weights,config):
        model = None
        if name == 'yolo':
            cfg,nc,anchors,device = config
            model = self.make_yolo(cfg,weights,nc,anchors,device)
        return model

    def make_studentnet(self,name,config):
        model = None
        if name =='yolo':
            cfg, nc, anchors, device = config
            model = self.make_yolo(cfg=cfg,weights=None,nc=nc,anchors=anchors,device=device)
        return model

    def make_yolo(self,cfg,weights,nc, anchors,device):
        from models.yoloan import Model
        from utils.general import intersect_dicts

        if weights:
            ckpt = torch.load(weights, map_location='cpu')
            model = Model(cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=anchors).to(device)  # create
            exclude = ['anchor'] if (cfg or anchors) else []  # exclude keys
            csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
            csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # intersect
            model.load_state_dict(csd, strict=False)  # load
        else:
            model = Model(cfg, ch=3, nc=nc, anchors=anchors).to(device)  # create
        return model

class MyDetect(nn.Module):
    def __init__(self, weights='last.pt', device=torch.device('cpu'), dnn=False, data=None, fp16=False,layerindexes=[0,2,4,6]):
        super().__init__()
        ckpt = torch.load(weights, map_location='cpu')
        
        model = ckpt['model'].float()
        self.stride = max(int(model.stride.max()), 32)
        self.model = None
        self.names = model.names
        model = model.to(device)
        self.model = model
        self.model.eval()
        self.fp16 = fp16
        self.imgsize = (640, 640)
        self.features = []
        self.layerindexes= layerindexes
        self.hook()
    def hook(self):
        for i in self.layerindexes:
            self.model.model[i].register_forward_hook(self._hook)
    def _hook(self,module, input, output):
        self.features.append(output)

  
    
    def forward(self, im, augment=False, visualize=False, val=False):
        b, ch, h, w = im.shape  # batch, channel, height, width
        if self.imgsize[0] != h or self.imgsize[1] != w:
            self.imgsize = (h, w)
        y = self.model(im)
        features = self.features
        self.features = []
        return y
        return (y[0],y[2])
        
class PreModel(nn.Module):
    def __init__(self, weights='last.pt', device=torch.device('cpu'), dnn=False, data=None, fp16=False):
        super().__init__()
        ckpt = torch.load(weights, map_location='cpu')
        model = ckpt['model'].float()
        self.stride = max(int(model.stride.max()), 32)
        self.model = None
        self.names = model.names
        model = model.to(device)
        self.model = model
        self.model.eval()
        self.fp16 = fp16
    def forward(self, im, augment=False, visualize=False, val=False):
        y = self.model(im)

class AutoShape(nn.Module):
    # YOLOv5 input-robust model wrapper for passing cv2/np/PIL/torch inputs. Includes preprocessing, inference and NMS
    conf = 0.25  # NMS confidence threshold
    iou = 0.45  # NMS IoU threshold
    agnostic = False  # NMS class-agnostic
    multi_label = False  # NMS multiple labels per box
    classes = None  # (optional list) filter by class, i.e. = [0, 15, 16] for COCO persons, cats and dogs
    max_det = 1000  # maximum number of detections per image
    amp = False  # Automatic Mixed Precision (AMP) inference

    def __init__(self, model):
        super().__init__()
        LOGGER.info('Adding AutoShape... ')
        copy_attr(self, model, include=('yaml', 'nc', 'hyp', 'names', 'stride', 'abc'), exclude=())  # copy attributes
        self.dmb = isinstance(model, DetectMultiBackend)  # DetectMultiBackend() instance
        self.pt = not self.dmb or model.pt  # PyTorch model
        self.model = model.eval()

    def _apply(self, fn):
        # Apply to(), cpu(), cuda(), half() to model tensors that are not parameters or registered buffers
        self = super()._apply(fn)
        if self.pt:
            m = self.model.model.model[-1] if self.dmb else self.model.model[-1]  # Detect()
            m.stride = fn(m.stride)
            m.grid = list(map(fn, m.grid))
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))
        return self

    @torch.no_grad()
    def forward(self, imgs, size=640, augment=False, profile=False):
        # Inference from various sources. For height=640, width=1280, RGB images example inputs are:
        #   file:       imgs = 'data/images/zidane.jpg'  # str or PosixPath
        #   URI:             = 'https://ultralytics.com/images/zidane.jpg'
        #   OpenCV:          = cv2.imread('image.jpg')[:,:,::-1]  # HWC BGR to RGB x(640,1280,3)
        #   PIL:             = Image.open('image.jpg') or ImageGrab.grab()  # HWC x(640,1280,3)
        #   numpy:           = np.zeros((640,1280,3))  # HWC
        #   torch:           = torch.zeros(16,3,320,640)  # BCHW (scaled to size=640, 0-1 values)
        #   multiple:        = [Image.open('image1.jpg'), Image.open('image2.jpg'), ...]  # list of images

        t = [time_sync()]
        p = next(self.model.parameters()) if self.pt else torch.zeros(1)  # for device and type
        autocast = self.amp and (p.device.type != 'cpu')  # Automatic Mixed Precision (AMP) inference
        if isinstance(imgs, torch.Tensor):  # torch
            with amp.autocast(enabled=autocast):
                return self.model(imgs.to(p.device).type_as(p), augment, profile)  # inference

        # Pre-process
        n, imgs = (len(imgs), imgs) if isinstance(imgs, list) else (1, [imgs])  # number of images, list of images
        shape0, shape1, files = [], [], []  # image and inference shapes, filenames
        for i, im in enumerate(imgs):
            f = f'image{i}'  # filename
            if isinstance(im, (str, Path)):  # filename or uri
                im, f = Image.open(requests.get(im, stream=True).raw if str(im).startswith('http') else im), im
                im = np.asarray(exif_transpose(im))
            elif isinstance(im, Image.Image):  # PIL Image
                im, f = np.asarray(exif_transpose(im)), getattr(im, 'filename', f) or f
            files.append(Path(f).with_suffix('.jpg').name)
            if im.shape[0] < 5:  # image in CHW
                im = im.transpose((1, 2, 0))  # reverse dataloader .transpose(2, 0, 1)
            im = im[..., :3] if im.ndim == 3 else np.tile(im[..., None], 3)  # enforce 3ch input
            s = im.shape[:2]  # HWC
            shape0.append(s)  # image shape
            g = (size / max(s))  # gain
            shape1.append([y * g for y in s])
            imgs[i] = im if im.data.contiguous else np.ascontiguousarray(im)  # update
        shape1 = [make_divisible(x, self.stride) if self.pt else size for x in np.array(shape1).max(0)]  # inf shape
        x = [letterbox(im, new_shape=shape1, auto=False)[0] for im in imgs]  # pad
        x = np.ascontiguousarray(np.array(x).transpose((0, 3, 1, 2)))  # stack and BHWC to BCHW
        x = torch.from_numpy(x).to(p.device).type_as(p) / 255  # uint8 to fp16/32
        t.append(time_sync())

        with amp.autocast(enabled=autocast):
            # Inference
            y = self.model(x, augment, profile)  # forward
            t.append(time_sync())

            # Post-process
            y = non_max_suppression(y if self.dmb else y[0], self.conf, iou_thres=self.iou, classes=self.classes,
                                    agnostic=self.agnostic, multi_label=self.multi_label, max_det=self.max_det)  # NMS
            for i in range(n):
                scale_coords(shape1, y[i][:, :4], shape0[i])

            t.append(time_sync())
            return Detections(imgs, y, files, t, self.names, x.shape)

class Detections:
    # YOLOv5 detections class for inference results
    def __init__(self, imgs, pred, files, times=(0, 0, 0, 0), names=None, shape=None):
        super().__init__()
        d = pred[0].device  # device
        gn = [torch.tensor([*(im.shape[i] for i in [1, 0, 1, 0]), 1, 1], device=d) for im in imgs]  # normalizations
        self.imgs = imgs  # list of images as numpy arrays
        self.pred = pred  # list of tensors pred[0] = (xyxy, conf, cls)
        self.names = names  # class names
        self.files = files  # image filenames
        self.times = times  # profiling times
        self.xyxy = pred  # xyxy pixels
        self.xywh = [xyxy2xywh(x) for x in pred]  # xywh pixels
        self.xyxyn = [x / g for x, g in zip(self.xyxy, gn)]  # xyxy normalized
        self.xywhn = [x / g for x, g in zip(self.xywh, gn)]  # xywh normalized
        self.n = len(self.pred)  # number of images (batch size)
        self.t = tuple((times[i + 1] - times[i]) * 1000 / self.n for i in range(3))  # timestamps (ms)
        self.s = shape  # inference BCHW shape

    def display(self, pprint=False, show=False, save=False, crop=False, render=False, save_dir=Path('')):
        crops = []
        for i, (im, pred) in enumerate(zip(self.imgs, self.pred)):
            s = f'image {i + 1}/{len(self.pred)}: {im.shape[0]}x{im.shape[1]} '  # string
            if pred.shape[0]:
                for c in pred[:, -1].unique():
                    n = (pred[:, -1] == c).sum()  # detections per class
                    s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                if show or save or render or crop:
                    annotator = Annotator(im, example=str(self.names))
                    for *box, conf, cls in reversed(pred):  # xyxy, confidence, class
                        label = f'{self.names[int(cls)]} {conf:.2f}'
                        if crop:
                            file = save_dir / 'crops' / self.names[int(cls)] / self.files[i] if save else None
                            crops.append({'box': box, 'conf': conf, 'cls': cls, 'label': label,
                                          'im': save_one_box(box, im, file=file, save=save)})
                        else:  # all others
                            annotator.box_label(box, label, color=colors(cls))
                    im = annotator.im
            else:
                s += '(no detections)'

            im = Image.fromarray(im.astype(np.uint8)) if isinstance(im, np.ndarray) else im  # from np
            if pprint:
                LOGGER.info(s.rstrip(', '))
            if show:
                im.show(self.files[i])  # show
            if save:
                f = self.files[i]
                im.save(save_dir / f)  # save
                if i == self.n - 1:
                    LOGGER.info(f"Saved {self.n} image{'s' * (self.n > 1)} to {colorstr('bold', save_dir)}")
            if render:
                self.imgs[i] = np.asarray(im)
        if crop:
            if save:
                LOGGER.info(f'Saved results to {save_dir}\n')
            return crops

    def print(self):
        self.display(pprint=True)  # print results
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {tuple(self.s)}' %
                    self.t)

    def show(self):
        self.display(show=True)  # show results

    def save(self, save_dir='runs/detect/exp'):
        save_dir = increment_path(save_dir, exist_ok=save_dir != 'runs/detect/exp', mkdir=True)  # increment save_dir
        self.display(save=True, save_dir=save_dir)  # save results

    def crop(self, save=True, save_dir='runs/detect/exp'):
        save_dir = increment_path(save_dir, exist_ok=save_dir != 'runs/detect/exp', mkdir=True) if save else None
        return self.display(crop=True, save=save, save_dir=save_dir)  # crop results

    def render(self):
        self.display(render=True)  # render results
        return self.imgs

    def pandas(self):
        # return detections as pandas DataFrames, i.e. print(results.pandas().xyxy[0])
        new = copy(self)  # return copy
        ca = 'xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class', 'name'  # xyxy columns
        cb = 'xcenter', 'ycenter', 'width', 'height', 'confidence', 'class', 'name'  # xywh columns
        for k, c in zip(['xyxy', 'xyxyn', 'xywh', 'xywhn'], [ca, ca, cb, cb]):
            a = [[x[:5] + [int(x[5]), self.names[int(x[5])]] for x in x.tolist()] for x in getattr(self, k)]  # update
            setattr(new, k, [pd.DataFrame(x, columns=c) for x in a])
        return new

    def tolist(self):
        # return a list of Detections objects, i.e. 'for result in results.tolist():'
        r = range(self.n)  # iterable
        x = [Detections([self.imgs[i]], [self.pred[i]], [self.files[i]], self.times, self.names, self.s) for i in r]
        # for d in x:
        #    for k in ['imgs', 'pred', 'xyxy', 'xyxyn', 'xywh', 'xywhn']:
        #        setattr(d, k, getattr(d, k)[0])  # pop out of list
        return x

    def __len__(self):
        return self.n

class Classify(nn.Module):
    # Classification head, i.e. x(b,c1,20,20) to x(b,c2)
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.aap = nn.AdaptiveAvgPool2d(1)  # to x(b,c1,1,1)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g)  # to x(b,c2,1,1)
        self.flat = nn.Flatten()

    def forward(self, x):
        z = torch.cat([self.aap(y) for y in (x if isinstance(x, list) else [x])], 1)  # cat if list
        return self.flat(self.conv(z))  # flatten to x(b,c2)

