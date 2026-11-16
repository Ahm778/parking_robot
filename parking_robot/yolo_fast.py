"""YOLOv5n pour le robot réel (Jetson Nano) : TensorRT sur la puce graphique, ou OpenCV DNN en secours.

Les deux moteurs ont la même interface que YoloOnnx de vision_node.py :
    yolo = charger_yolo('weights')            # TensorRT si weights/yolov5n_fp16.engine existe, sinon OpenCV
    for x1, y1, x2, y2, conf, cls in yolo(image_bgr): ...

Le moteur TensorRT se fabrique une fois, sur la Jetson elle-même (il dépend de la carte et de la version) :
    /usr/src/tensorrt/bin/trtexec --onnx=weights/yolov5n.onnx --saveEngine=weights/yolov5n_fp16.engine --fp16

Pas de pycuda (long à compiler sur la Jetson) : la mémoire de la puce graphique est gérée directement par
libcudart via ctypes. Compatible Python 3.6 (JetPack 4.6).
"""
import ctypes
import os

import cv2
import numpy as np

COCO = {0: 'pieton', 2: 'voiture', 5: 'bus', 7: 'utilitaire'}


def postprocess(out, w, h, size, conf_min=0.25, nms=0.45):
    """Sortie YOLOv5 (N x 85 : cx, cy, bw, bh, objectness, 80 classes) -> boîtes dans l'image d'origine."""
    scores = out[:, 4:5] * out[:, 5:]
    cls = scores.argmax(1)
    conf = scores.max(1)
    keep = (conf > conf_min) & np.isin(cls, list(COCO))
    out, cls, conf = out[keep], cls[keep], conf[keep]
    sx, sy = w / size, h / size
    boxes = [[float((cx - bw / 2) * sx), float((cy - bh / 2) * sy), float(bw * sx), float(bh * sy)]
             for cx, cy, bw, bh in out[:, :4]]
    if not boxes:
        return []
    idx = np.array(cv2.dnn.NMSBoxes(boxes, conf.tolist(), conf_min, nms)).flatten()
    return [(boxes[i][0], boxes[i][1], boxes[i][0] + boxes[i][2], boxes[i][1] + boxes[i][3],
             float(conf[i]), int(cls[i])) for i in idx]


class YoloOpenCV:
    """OpenCV DNN sur le processeur (OpenCV >= 4.5). Environ 145 ms par image sur la Jetson Nano 2 Go."""
    nom = 'OpenCV DNN (processeur)'

    def __init__(self, onnx_path, size=320, conf=0.25, nms=0.45):
        self.net = cv2.dnn.readNetFromONNX(onnx_path)
        self.size, self.conf, self.nms = size, conf, nms

    def __call__(self, img):
        h, w = img.shape[:2]
        self.net.setInput(cv2.dnn.blobFromImage(img, 1 / 255.0, (self.size, self.size), swapRB=True, crop=False))
        return postprocess(self.net.forward()[0], w, h, self.size, self.conf, self.nms)


def _cudart():
    for name in ('libcudart.so', '/usr/local/cuda/lib64/libcudart.so'):
        try:
            lib = ctypes.CDLL(name)
            break
        except OSError:
            lib = None
    if lib is None:
        raise OSError('libcudart introuvable (CUDA absent ?)')
    lib.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    lib.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    lib.cudaFree.argtypes = [ctypes.c_void_p]
    return lib


class YoloTensorRT:
    """TensorRT (FP16) sur la puce graphique de la Jetson."""
    nom = 'TensorRT FP16 (puce graphique)'
    H2D, D2H = 1, 2                                     # cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost

    def __init__(self, engine_path, conf=0.25, nms=0.45):
        import tensorrt as trt                          # fourni par JetPack (python3-libnvinfer)
        self.cuda = _cudart()
        self.conf, self.nms = conf, nms
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError('moteur TensorRT illisible : refaites-le avec trtexec sur cette Jetson')
        self.ctx = self.engine.create_execution_context()
        self.host, self.dev = [], []
        for i in range(self.engine.num_bindings):
            shape = tuple(self.engine.get_binding_shape(i))
            arr = np.empty(shape, dtype=trt.nptype(self.engine.get_binding_dtype(i)))
            ptr = ctypes.c_void_p()
            self._check(self.cuda.cudaMalloc(ctypes.byref(ptr), arr.nbytes))
            self.host.append(arr)
            self.dev.append(ptr)
            if self.engine.binding_is_input(i):
                self.inp, self.size = i, shape[2]
            else:
                self.out = i
        self.bindings = [int(p.value) for p in self.dev]

    @staticmethod
    def _check(err):
        if err != 0:
            raise RuntimeError('erreur CUDA %d' % err)

    def __call__(self, img):
        h, w = img.shape[:2]
        blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (self.size, self.size), swapRB=True, crop=False)
        hin, hout = self.host[self.inp], self.host[self.out]
        np.copyto(hin, blob.astype(hin.dtype, copy=False))
        self._check(self.cuda.cudaMemcpy(self.dev[self.inp], hin.ctypes.data, hin.nbytes, self.H2D))
        if not self.ctx.execute_v2(self.bindings):
            raise RuntimeError('échec de l\'inférence TensorRT')
        self._check(self.cuda.cudaMemcpy(hout.ctypes.data, self.dev[self.out], hout.nbytes, self.D2H))
        return postprocess(hout.reshape(hout.shape[-2], hout.shape[-1]).astype(np.float32),
                           w, h, self.size, self.conf, self.nms)

    def __del__(self):
        for p in getattr(self, 'dev', []):
            try:
                self.cuda.cudaFree(p)
            except Exception:
                pass


def charger_yolo(weights_dir, moteur='auto', size=320, conf=0.25):
    """moteur : 'auto' (TensorRT si possible), 'tensorrt' ou 'opencv'."""
    engine = os.path.join(weights_dir, 'yolov5n_fp16.engine')
    if moteur in ('auto', 'tensorrt') and os.path.exists(engine):
        try:
            return YoloTensorRT(engine, conf=conf)
        except Exception as e:                          # pas de TensorRT, moteur d'une autre carte, etc.
            if moteur == 'tensorrt':
                raise
            print('TensorRT indisponible (%s) : passage à OpenCV' % e)
    elif moteur == 'tensorrt':
        raise FileNotFoundError(engine + ' absent : lancez trtexec (voir l\'en-tête de yolo_fast.py)')
    return YoloOpenCV(os.path.join(weights_dir, 'yolov5n.onnx'), size=size, conf=conf)
