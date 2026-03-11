import  os
import  math
import  cv2
import  numpy as np
import  onnxruntime as ort
import  torch
from    torch import nn
import  torch.nn.functional as F
from    torchvision import transforms

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")

def _weights_path(filename):
    return os.path.join(WEIGHTS_DIR, filename)

def _load_checkpoint(path, map_location="cpu"):
    # PyTorch >=2.6 defaults torch.load(..., weights_only=True), which breaks
    # legacy checkpoints containing objects like OmegaConf DictConfig.
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        # Older torch versions do not support the weights_only argument.
        return torch.load(path, map_location=map_location)

# def preprocess_face_image(img, face_bbox):
#     x, y, x1, y1,p = face_bbox
#     w, h = x1-x, y1-y
#     # Expand the bounding box by 96/112 pixels in all directions
#     x1 = round(max(0, x - int(w * 96 / 112)))
#     y1 = round(max(0, y - int(h * 96 / 112)))
#     x2 = round(min(img.shape[1], x + w + int(w * 96 / 112)))
#     y2 = round(min(img.shape[0], y + h + int(h * 96 / 112)))
#     # Extract the expanded face region
#     face_region = img[y1:y2, x1:x2]
#     # Symmetrically expand the shorter side to match the longer side
#     h, w = face_region.shape[:2]
#     if h > w:
#         padding = (h - w) // 2
#         face_region = cv2.copyMakeBorder(face_region, 0, 0, padding, padding, cv2.BORDER_REFLECT)
#     else:
#         padding = (w - h) // 2
#         face_region = cv2.copyMakeBorder(face_region, padding, padding, 0, 0, cv2.BORDER_REFLECT)
#     # If the image is still not square, symmetrically pad with 127 to make it square
#     h, w = face_region.shape[:2]
#     if h != w:
#         padding = abs(h - w) // 2
#         if h > w:
#             face_region = cv2.copyMakeBorder(face_region, 0, 0, padding, padding, cv2.BORDER_CONSTANT, value=127)
#         else:
#             face_region = cv2.copyMakeBorder(face_region, padding, padding, 0, 0, cv2.BORDER_CONSTANT, value=127)
#     # Resize the image to 128x128
#     face_region = cv2.resize(face_region, (128, 128))
#     # Center crop the image to 112x112
#     face_region = face_region[8:120, 8:120]
#     return face_region

# resize image for face detector
def resizeImage(img, input_size=(640,640)):
        if img is None or not hasattr(img, "shape") or img.size == 0:
            return None, None, None
        input_size = [(i+127) & 0xffff80 for i in input_size]
        im_ratio = float(img.shape[0]) / img.shape[1]
        model_ratio = float(input_size[1]) / input_size[0]
        if im_ratio>model_ratio:
            new_height = input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = input_size[0]
            new_height = int(new_width * im_ratio)
        det_scale = float(new_height) / img.shape[0]
        resized_img = cv2.resize(img, (new_width, new_height))
        det_img = np.zeros( (input_size[1], input_size[0], 3), dtype=np.uint8 )
        x0=(input_size[0]-new_width)//2
        y0=(input_size[1]-new_height)//2

        det_img[y0:y0+new_height, x0:x0+new_width, :] = resized_img
        return det_img,det_scale,np.array((x0,y0))




class YOLOv8_face:
    def __init__(self,model_file =None, conf_thres=0.4, iou_thres=0.5,device='AUTO',engine='onnx'):
        if model_file is None:
            model_file = _weights_path("yolov8n-face.onnx")
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres
        # Initialize model
        self.engine=engine
        providers=['CUDAExecutionProvider','CPUExecutionProvider']
        if device=='CPU': providers=providers[1:]
        self.session = ort.InferenceSession(model_file, providers=providers)
        outputs = self.session.get_outputs()
        self.output_names = [o.name for o in outputs]
        self.input_name   = self.session.get_inputs()[0].name
        self.reg_max        = 16
        self.project = np.arange(self.reg_max)
        self.strides = (8, 16, 32)
        self.center_cache={}
        self.input_std=255
    def distance2bbox(self,points, distance):
        max_shape = (self.input_height, self.input_width)
        x1 = points[:, 0] - distance[:, 0]
        y1 = points[:, 1] - distance[:, 1]
        x2 = points[:, 0] + distance[:, 2]
        y2 = points[:, 1] + distance[:, 3]
        if max_shape is not None:
            x1 = np.clip(x1, 0, max_shape[1])
            y1 = np.clip(y1, 0, max_shape[0])
            x2 = np.clip(x2, 0, max_shape[1])
            y2 = np.clip(y2, 0, max_shape[0])
        return np.stack([x1, y1, x2, y2], axis=-1)

    def get_anchors(self, width,height, grid_cell_offset=0.5):
        key = width+height*1024
        if not key in self.center_cache: 
            feats_hw = [(math.ceil(height / self.strides[i]), math.ceil(width / self.strides[i])) for i in range(len(self.strides))]
            """Generate anchors from features."""
            anchor_points = {}
            for i, stride in enumerate(self.strides):
                h,w = feats_hw[i]
                x = np.arange(0, w) + grid_cell_offset  # shift x
                y = np.arange(0, h) + grid_cell_offset  # shift y
                sx, sy = np.meshgrid(x, y)
                anchor_points[stride] = np.stack((sx, sy), axis=-1).reshape(-1, 2)
            self.center_cache[key]=anchor_points
        return self.center_cache[key]

    def softmax(self, x, axis=1):
        x_exp = np.exp(x)
        x_sum = np.sum(x_exp, axis=axis, keepdims=True)
        s = x_exp / x_sum
        return s
    
    def __call__(self, input_img,det_scale,origin,threshold=None):
        self.input_width =input_img.shape[1]
        self.input_height=input_img.shape[0]
        self.anchors=self.get_anchors(self.input_width,self.input_height, grid_cell_offset=0.5)
        blob = cv2.dnn.blobFromImage(input_img, 1.0/self.input_std,swapRB=True)
        if self.engine == 'onnx':
            onnx_outs = self.session.run(self.output_names, {self.input_name : blob})
            outputs=[onnx_outs[1],onnx_outs[2],onnx_outs[0]]
        else:    
            ov_outs = self.Model(blob)
            outputs=[ov_outs[self.Model.output(1)],ov_outs[self.Model.output(2)],ov_outs[self.Model.output(0)]]

        # Perform inference on the image
        det_bboxes, det_conf, landmarks = self.post_process(outputs,threshold or self.conf_threshold)
        if det_bboxes is not None:
            boxes = np.hstack((det_bboxes, det_conf.reshape(-1,1)))
            boxes[:,:2] -= origin
            boxes[:,:4] /= det_scale
            boxes[:,2:4]+= boxes[:,:2]
            landmarks = landmarks.reshape((-1, 5, 3))[:,:,:2]
            landmarks -=origin
            landmarks /=det_scale
            return boxes, landmarks
        return np.array([]),np.array([])

    def post_process(self, preds,threshold):
        bboxes, scores, landmarks = [], [], []
        for i, pred in enumerate(preds):
            stride = int(self.input_height/pred.shape[2])
            pred = pred.transpose((0, 2, 3, 1))
            
            box = pred[..., :self.reg_max * 4]
            cls = 1 / (1 + np.exp(-pred[..., self.reg_max * 4:-15])).reshape((-1,1))
            kpts = pred[..., -15:].reshape((-1,15)) ### x1,y1,score1, ..., x5,y5,score5

            tmp = box.reshape(-1, 4, self.reg_max)
            bbox_pred = self.softmax(tmp, axis=-1)
            bbox_pred = np.dot(bbox_pred, self.project).reshape((-1,4))

            bbox = self.distance2bbox(self.anchors[stride], bbox_pred) * stride
            kpts[:, 0::3] = (kpts[:, 0::3] * 2.0 + (self.anchors[stride][:, 0].reshape((-1,1)) - 0.5)) * stride
            kpts[:, 1::3] = (kpts[:, 1::3] * 2.0 + (self.anchors[stride][:, 1].reshape((-1,1)) - 0.5)) * stride
            kpts[:, 2::3] = 1 / (1+np.exp(-kpts[:, 2::3]))

            bboxes.append(bbox)
            scores.append(cls)
            landmarks.append(kpts)

        bboxes = np.concatenate(bboxes, axis=0)
        scores = np.concatenate(scores, axis=0)
        landmarks = np.concatenate(landmarks, axis=0)
    
        bboxes_wh = bboxes.copy()
        bboxes_wh[:, 2:4] = bboxes[:, 2:4] - bboxes[:, 0:2]  ####xywh
        # classIds = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)  ####max_class_confidence
        
        mask = confidences>threshold
        bboxes_wh = bboxes_wh[mask]  ###合理使用广播法则
        if len(bboxes_wh) > 0:
            confidences = confidences[mask]
            # classIds = classIds[mask]
            landmarks = landmarks[mask]
            
            indices = cv2.dnn.NMSBoxes(bboxes_wh.tolist(), confidences.tolist(), threshold,
                                    self.iou_threshold).flatten()
            # indices2 = nms(bboxes_wh,self.iou_threshold, confidences)
            if len(indices) > 0:
                mlvl_bboxes = bboxes_wh[indices]
                confidences = confidences[indices]
                # classIds = classIds[indices]
                landmarks = landmarks[indices]
                return mlvl_bboxes, confidences, landmarks
        
        return None,None,None

def add_face_margin(x, y, w, h, margin=0.5):
    x_marign = int(w * margin / 2)
    y_marign = int(h * margin / 2)
    x1 = x - x_marign
    x2 = x + w + x_marign
    y1 = y - y_marign
    y2 = y + h + y_marign
    return x1, x2, y1, y2
    
def crop_from_5landmarks(img,res,margin):
        if margin==0: return img
        x_list = res[:,0]
        y_list = res[:,1]
        x, y = round(min(x_list)), round(min(y_list))
        w, h = round(max(x_list)) - x, round(max(y_list)) - y
        side = w if w > h else h
        x1, x2, y1, y2 = add_face_margin(x, y, side, side, margin=margin)
        max_h, max_w = img.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(max_w, x2)
        y2 = min(max_h, y2)
        img = img[y1:y2, x1:x2]
        return img

class DKGModule(nn.Module):
    def __init__(self, k, inplanes, planes, m=4, padding=None, stride=1):
        """
            k (int): The size of the kernel.
            inplanes (int): The number of input channels.
            planes (int): The number of output channels.
            m (int, optional): The channel reduction rate. Defaults to 4.
            padding (int, optional): The padding value for convolution. Defaults to None.
            stride (int, optional): The stride value for convolution. Defaults to 1.
        """
        super(DKGModule, self).__init__()
        self.k = k
        self.channel = inplanes 
        self.group = self.channel // 2
        # cov1
        self.conv = nn.Conv2d(self.channel, self.channel // m, 1, padding=0, bias=True)
        self.pad = padding
        self.stride = stride
        # conv2'
        self.conv_k = nn.Conv2d(1, 1, 1, padding=0, bias=True)
        # conv2
        self.conv_kernel = nn.Conv2d(1, k*k, 1, padding=0, bias=True)

        # conv4
        self.conv_static = nn.Conv2d(self.channel // 2, self.channel // 2, kernel_size=3, dilation=1, padding=1, bias=True)
        self.fuse = nn.Conv2d(self.channel, planes, 1, padding=0, bias=True)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        N, C, H, W = x.shape                         # [B * C * H * W]
        # print('x.shape', x.shape)
        x1 = x[:, :int(C/2), :, :]                   # [B * C/2 * H * W]
        x2 = x[:, int(C/2):, :, :]                   # [B * C/2 * H * W]
        
        # Kernel Generator----------------------------------------------
        # conv1 + avg_pool
        g = self.avg_pool(x1)                       # [B * C/2 * 1 * 1]
        g_perm = g.permute(0, 2, 1, 3).contiguous() # [B * 1 * C/2 * 1]
        # conv2
        kernel = self.conv_kernel(g_perm)           # [B * k^2 * C/2 * 1]
        kernel = kernel.permute(0, 3, 2, 1)         # [B * 1 * C/2 * k^2]
       
        f_list = torch.split(x1, 1, 0)              # [1 * C/2 * H * W]
        g_list = torch.split(kernel, 1, 0)          # [1 * 1 * C/2 * k^2]
        # Instance-wise Interaction-------------------------------------
        out = []
        for i in range(N):
            f_one = f_list[i] # [1* C/2 * H * W]
            g_one = g_list[i] # [1 * 1 * C/2 * k^2]
            # Dynamic Kenerl with conv2'
            g_k = self.conv_k(g_one)                                    # [1 * 1 * C/2 * k^2]
            g_k = g_k.reshape(g_k.size(2), g_k.size(1), self.k, self.k) # [C/2 * 1 * k * k]

            # Padding
            if self.pad is None:
                padding = ((self.k-1) // 2, (self.k-1) // 2, (self.k-1) // 2, (self.k-1) // 2)
            else:
                padding = (self.pad, self.pad, self.pad, self.pad)

            f_one = F.pad(input=f_one, pad=padding, mode='constant', value=0) # [1* C/2 * H * W]

            # Dynamic Kernel Interaction
            o = F.conv2d(input=f_one, weight=g_k, stride=self.stride, groups=self.group)
            out.append(o)

        # Output of Keneral Generator branch
        y_res = torch.cat(out, dim=0)  # [B * C/2 * H * W]
        
        y_out = self.conv_static(x2)   # [B * C/2 * H * W]

        y_out = torch.cat([y_res, y_out], dim=1) # [B * C * H * W] 
        y_out = self.fuse(y_out)                 # [B * C' * H * W]

        return y_out
    
def conv3x3(in_channels, out_channels, stride=1, padding=1, bias=True):
    '''
        the cnn module with specific parameters
        Args:
            in_channels (int): the channel numbers of input features
            out_channels (int): the channel numbers of output features
            stride (int): the stride paramters of Conv2d
            padding (int): the padding parameters of Conv2D
            bias (bool): the bool parameters of Conv2d
    '''
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=bias)



# New implementation of basic blocks------------------------------------------------------------
class Conv_block_gate(nn.Module):
    def __init__(self, in_channels, out_channels, padding, dkg_flag, model_initial='kaiming'):
        '''
            Args:
                in_channels (int): the channel numbers of input features
                out_channels (int): the channel numbers of output features
                dkg_flag (bool):
                    'True' allows the DKG module
                    'False' does not use the DKG module
                model_initial (str):
                    'kaiming' allow the Conv_block to use 'kaiming' methods to initialize the networks
        '''
        super(Conv_block_gate, self).__init__()
        self.model_initial = model_initial
        self.dkg_flag = dkg_flag
        if self.dkg_flag==True:
            self.conv = DKGModule(3, in_channels, out_channels, m=16, padding=1, stride=1)
        else:     
            self.conv = conv3x3(in_channels, out_channels)
        self.norm = nn.InstanceNorm2d(out_channels) 
        self.relu = nn.ReLU(inplace=True)
        self.in_channels = in_channels
        self.out_channels = out_channels
        # model initial
        # init_weights(self.conv, init_type=self.model_initial)

    def forward(self, x):
        # Nomal branch of conb+bn+relu
        out = self.conv(x)
        out = self.norm(out)
        out = self.relu(out)
        # print(x.shape)
            
        return out

class Basic_block_gate(nn.Module):
    def __init__(self, in_channels, out_channels, padding, dkg_flag, model_initial='kaiming'):
        '''
            Basic_block contains three Conv_block

            Args:
                in_channels (int): the channel numbers of input features
                out_channels (int): the channel numbers of output features
                padding (int): the padding parameters of conv block
                dkg_flag (bool):
                    'True' allows the DKG module
                    'False' does not use the DKG module
                model_initial:
                    'kaiming' allow the Conv_block to use 'kaiming' methods to initialize the networks
        '''
        super(Basic_block_gate, self).__init__()
        self.model_initial = model_initial
        self.dkg_flag = dkg_flag
        self.padding = padding
        if self.dkg_flag==True:
            self.conv_block1_gate = Conv_block_gate(in_channels, 128, 0, False, self.model_initial)
            self.conv_block2_gate = Conv_block_gate(128, 196, self.padding, True, self.model_initial)
            self.conv_block3_gate = Conv_block_gate(196, out_channels, 0, False, self.model_initial)
        else:
            self.conv_block1_gate = Conv_block_gate(in_channels, 128, 0, False, self.model_initial)
            self.conv_block2_gate = Conv_block_gate(128, 196, 0, False, self.model_initial)
            self.conv_block3_gate = Conv_block_gate(196, out_channels, 0, False, self.model_initial)   
        self.max_pool = nn.MaxPool2d(2)

    def forward(self, input):
        # print("conv1")
        input = self.conv_block1_gate(input)
        # print("conv2")
        input = self.conv_block2_gate(input)
        # print("conv3")
        input = self.conv_block3_gate(input)
        input = self.max_pool(input)
        return input

class FeatExtractor(nn.Module):
    def __init__(self, dkg_flag, in_channels=6, model_initial='kaiming'):
        '''
            Args:
                dkg_flag (bool):
                    'True' allows the DKG module
                    'False' does not use the DKG module
                in_channels (int): the channel numbers of input features
                model_initial:
                    'kaiming' allow the Conv_block to use 'kaiming' methods to initialize the networks
        '''
        super(FeatExtractor, self).__init__()
        self.model_initial = model_initial
        self.dkg_flag = dkg_flag
        self.inc = Conv_block_gate(in_channels, 64, 0, False, self.model_initial)
        self.down1 = Basic_block_gate(64, 128, 1, self.dkg_flag, self.model_initial)
        self.down2 = Basic_block_gate(128, 128, 1, self.dkg_flag, self.model_initial)
        self.down3 = Basic_block_gate(128, 128, 1, self.dkg_flag, self.model_initial)

    def cal_cat_feat(self, x1):
        x1 = self.inc(x1)
        x1_1 = self.down1(x1)
        x1_2 = self.down2(x1_1)
        x1_3 = self.down3(x1_2)

        re_x1_1 = F.adaptive_avg_pool2d(x1_1, 32)
        re_x1_2 = F.adaptive_avg_pool2d(x1_2, 32)
        catfeat = torch.cat([re_x1_1, re_x1_2, x1_3],1)

        return catfeat

    def forward(self, input):

        x1 = self.cal_cat_feat(input)
        fea_x1_x1 = x1
            
        # get outputs
        outputs = {}
        outputs["cat_feat"] = x1
        outputs["out"] = fea_x1_x1

        return outputs

class FeatEmbedder(nn.Module):
    def __init__(self, dkg_flag, in_channels=384, model_initial='kaiming'):
        '''
            Args:
                dkg_flag (bool):
                    'True' allows the DKG module
                    'False' does not use the DKG module
                in_channels (int): the channel numbers of input features
                model_initial:
                    'kaiming' allow the Conv_block to use 'kaiming' methods to initialize the networks
        '''
        super(FeatEmbedder, self).__init__()
        self.model_initial = model_initial
        self.dkg_flag = dkg_flag
        if self.dkg_flag==True:
            self.conv_block1 = Conv_block_gate(in_channels, 128, False, self.model_initial)
            self.conv_block2 = Conv_block_gate(128, 256, False, self.model_initial)
            self.conv_block3 = Conv_block_gate(256, 512, False, self.model_initial)   
        else:
            self.conv_block1 = Conv_block_gate(in_channels, 128, False, self.model_initial)
            self.conv_block2 = Conv_block_gate(128, 256, False, self.model_initial)
            self.conv_block3 = Conv_block_gate(256, 512, False, self.model_initial)        	
        self.max_pool = nn.MaxPool2d(2)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, 2)

    def forward(self, x):

        # Normal brach for Feature Classifier
        x = self.conv_block1(x)
        x = self.max_pool(x)
        x= self.conv_block2(x)
        x = self.max_pool(x)
        x = self.conv_block3(x) # torch.Size([0, 13])
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
            
        x = self.fc(x) # torch.Size([12, 2])
            
        # get outputs
        outputs = {}
        outputs["out"] = x

        return outputs



class Framework(nn.Module):
    def __init__(self, total_dkg_flag, style_dim, base_style_num, concentration_coeff, in_channels=6,mid_channels=384,model_initial='kaiming',clusters=2):
        '''
            Args:
                total_dkg_flag (bool):
                    'True' allows the DKG module
                    'False' does not use the DKG module
                style_dim (int): The dimension of the style vector.
                base_style_num (int): The number of base styles.
                concentration_coeff (float): The concentration coefficient for the Dirichlet distribution.
                in_channels (int): the channel numbers of input features
                mid_channels (int): the channel numbers of middle features
                model_initial:
                    'kaiming' allow the Conv_block to use 'kaiming' methods to initialize the networks
        '''
        super(Framework, self).__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.model_initial = model_initial
        self.total_dkg_flag = total_dkg_flag
        self.style_dim = style_dim
        self.base_style_num = base_style_num
        self.concentration_coeff = concentration_coeff

        self.FeatExtractor = FeatExtractor(dkg_flag=self.total_dkg_flag, 
                                           in_channels=self.in_channels, model_initial=self.model_initial)
        self.Classifier = FeatEmbedder(dkg_flag=False, in_channels=self.mid_channels)
    def forward(self, x):
        outputs_catfeat = self.FeatExtractor(x)
        outputs_catcls = self.Classifier(outputs_catfeat["out"])
        return outputs_catcls

class aFaceDetect:
    def __init__(self):
        self.detector   = YOLOv8_face(_weights_path("yolov8n-face.onnx"))
    def __call__(self,image):
        det_img,det_scale,origin=resizeImage(image)
        if det_img is None:
            return [[], []]
        bboxes, landmarks = self.detector(det_img,det_scale,origin,0.4)
        return [[],[]] if len(landmarks)==0 else [bboxes,landmarks]

class aSpoof:
    def __init__(self,model_name='ICM2O',threshold=0.1):
        self.device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.crop       = 0.7 if model_name in ['ICM2O','IOM2C'] else 0
        self.threshold  = threshold
        checkpoint      = _load_checkpoint(_weights_path(f'{model_name}.pth.tar'), map_location='cpu')
        # transform       = {'image_size': 256, 'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5]}
        transform       = checkpoint['args'].transform
        model_defs      = checkpoint['args'].model
        state_dict      = checkpoint['state_dict']
        self.model      = Framework(**model_defs['params'])
        self.model.load_state_dict(state_dict,strict=False)
        self.model      = self.model.to(self.device)
        self.transform  = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize([transform['image_size']]*2),
            transforms.ToTensor(),
            transforms.Normalize(mean = transform['mean'], std = transform['std'])
        ])
        self.model.eval()

    def __call__(self,image,bbox,landmarks): # image RGB 
        image=crop_from_5landmarks(image,landmarks,self.crop)
        with torch.no_grad():
            tensor = self.transform(image).to(self.device)
            tensor = tensor.unsqueeze(0)  # add batch dimension (batch size = 1)
            outputs_catcls = self.model(tensor) # RGB
            spoof_prob = torch.softmax(outputs_catcls["out"], dim=1)[0,1]
            spoof_prob =  float(spoof_prob.cpu())
            spoof =  spoof_prob>self.threshold
            return spoof,spoof_prob,image

class aSpoofONNX:
    def __init__(self,model_name='modelrgb',threshold=0.5):
        model_file = _weights_path(f'{model_name}.onnx')
        providers=['CUDAExecutionProvider','CPUExecutionProvider']
        self.session         = ort.InferenceSession(model_file, providers=providers)
        self.inputs          = self.session.get_inputs()
        self.outputs         = self.session.get_outputs()
        self.input_names     = [i.name for i in self.inputs]
        self.output_names    = [o.name for o in self.outputs]
        self.device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.detector        = YOLOv8_face(_weights_path("yolov8n-face.onnx"))
        self.crop            = 1.5
        self.threshold       = threshold
    def __call__(self,image,bbox,landmarks): # image RGB 
        image=crop_from_5landmarks(image,landmarks,self.crop)
        image=cv2.resize(image,(112,112))
        # image = preprocess_face_image(image,bbox)
        blob = cv2.dnn.blobFromImage(image, 1.0/255,mean=0,swapRB=False)
        onnx_outs = self.session.run(self.output_names, {self.input_names[0] : blob})
        spoof_prob=onnx_outs[0][0][0]
        spoof     = spoof_prob>self.threshold
        return spoof,spoof_prob,image

class aCelebASpoof:
    def __init__(self,threshold=0.5):
        import  torch
        from    tsn_predict import TSNPredictor as CelebASpoofDetector
        self.threshold=threshold
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model=CelebASpoofDetector(self.device)
        self.crop=3
    def __call__(self,image,bbox,landmarks): # image RGB 
        # image=crop_from_5landmarks(image,landmarks,self.crop)
        spoof_prob = self.model.predict(image[np.newaxis, ...])[0][1] # prob: [real,cpoof]
        spoof      = spoof_prob>self.threshold
        return spoof,spoof_prob,image
       

def find_best_threshold(spoof_probs,min_reals_perc=50):
    best_thre,best_count = 0.5,0
    spoof_probs     = np.array(spoof_probs)
    candidate_thres = list(np.unique(np.sort(spoof_probs[:,0])))
    total_real      = len([prob for prob in spoof_probs if prob[1]==0 ])
    for thre in candidate_thres:
        count         = len([prob for prob in spoof_probs if int(prob[0] > thre)==prob[1] ])
        count_real    = len([prob for prob in spoof_probs if prob[1]==0 and prob[0] <= thre ])
        if  count > best_count and count_real>=(1-min_reals_perc/100)*total_real:
            best_count = count
            best_thre = thre
    count_spoof   = len([prob for prob in spoof_probs if prob[1] and prob[0] > best_thre ])
    err_real  = 1-(best_count-count_spoof)/(total_real or 1)
    err_spoof  = 1-count_spoof/(spoof_probs.shape[0]-total_real)
    return best_thre,err_real,err_spoof

if __name__ == '__main__':
    import random
    from tqdm import tqdm
    mode=5

    # folders=[r'f:\datasets\anti-spoofing\MSU-MFSD\scene01\real_frames',r'f:\datasets\anti-spoofing\MSU-MFSD\scene01\attack_frames']
    folders=[r'c:\datasets\CelebA_Spoof\LCC_FASD\val\real',r'c:\datasets\CelebA_Spoof\LCC_FASD\val\spoof']
    # folders=[r'c:\datasets\CelebA_Spoof\LCC_FASD\train\real',r'c:\datasets\CelebA_Spoof\LCC_FASD\train\spoof']
    ModelD = aFaceDetect()
    if mode==1:
        Model=aSpoof('ICM2O',threshold=0.9980)        # 10% 60.9210 Threshold: 0.9991 20% 41.1890 Threshold: 0.9980 30% 25.3207 Threshold: 0.9954 40% 13.615  Threshold: 0.9871  50% 5.9737  Threshold: 0.9432
    elif mode==2:
        Model=aSpoof('IOM2C',threshold=9944)          # 10% 61.126  Threshold: 0.9993 20% 28.545  Threshold: 0.9944 30% 20.806  Threshold: 0.9867 40% 14.363  Threshold: 0.9499  50% 8.0999  Threshold: 0.63560
    elif mode==3:
        Model=aSpoofONNX('modelrgb',threshold=0.2808) # 10% 31.785% Threshold: 0.4629 20% 20.530% Threshold: 0.2808 30% 12.208% Threshold: 0.1555 40% 6.7358% Threshold: 0.08302 50% 4.5046% Threshold: 0.05533
    elif mode==4:
        import SASF
        Model=SASF.aSASF(threshold=0.0094)               # 10% 23%     Threshold: 0.11 20% 13%     Threshold: 0.021 30% 10%     Threshold: 0.006 40%  7%     Threshold: 0.0026 50% 5%      Threshold: 0.00121
    elif mode==5:
        from tsn_predict import TSNPredictor as CelebASpoofDetector
        Model=aCelebASpoof(threshold=0.998)

    # Model=aSpoof('OCI2M',threshold=1-0.1)
    # Model=aSpoof('OCM2I',threshold=1-0.4)
    
    files=[]
    for folder in folders:
        for root, d, f in os.walk(folder):
            if (len(files)&127)==0: print(root,len(files),end='\r')
            for file in f:
                if os.path.splitext(file)[-1].lower() in ['.png','.jpg']:
                    files.append([root,file])
    random.seed(123)
    random.shuffle(files)
    spoof_probs=[]
    total,nerrs=[0,0],[0,0]
    calc_thre=0
    for root,file in  tqdm(files):
        spoof=-1
        token=root.split(os.sep)[-1]
        if 'real' in token or 'live' in token:
            spoof+=1
        if 'spoof' in token or 'attack' in token:
            spoof+=2
        assert spoof==0 or spoof==1
        path = root+'/'+file
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bboxes,landmarks = ModelD(image)
        if len(landmarks)!=1 : 
            continue

        spoof1,spoof_prob,cimg = Model(image,bboxes[0],landmarks[0])
        if not spoof_prob is None :
            total[spoof]+=1
            spoof_probs+=[[spoof_prob,spoof]]
            if spoof1!=spoof:
                nerrs[spoof]+=1
                if calc_thre and random.random()<0.1:
                    threshold,err_real,err_spoof=find_best_threshold(spoof_probs)
                    Model.threshold=threshold
                    print(f'\n{err_real*100}% {err_spoof*100}% Threshold: {threshold}')
                else:
                    print(f'\n{nerrs[0]*100/(total[0] or 1)}% {nerrs[1]*100/(total[1] or 1)}%')
    for p in range(10,90,10):
        threshold,err_real,err_spoof=find_best_threshold(spoof_probs,p)
        print(f'{err_real*100}% {err_spoof*100}% Threshold: {threshold}')
