from PIL import Image as Img
import torch
import os
import os.path as osp
import torch.nn.functional as F
import numpy as np
from optimizer_loss import Optimizer
from models.model_stages import BiSeNet
from cityscapes import CityScapes
import torchvision.transforms as transforms

from transform import *
from torch.utils.data import DataLoader
from tqdm import tqdm
import time

def save_pic(out,origin_pth,out_pth):
    np.random.seed(4659)
    origin = Img.open(origin_pth)
    W, H = origin.size
    
    logits = F.interpolate(out,size=[H,W],mode='bilinear',align_corners=True)
    probs = torch.softmax(logits,dim=1)
    preds = torch.argmax(probs,dim =1,keepdim=False).squeeze(0)
    pred_np = np.array(preds.cpu(),dtype=np.uint8)
    img = Img.fromarray(pred_np,mode='L')
    mapp = np.random.randint(0,255,768)
    img.putpalette(list(mapp))
    name = origin_pth.split('/')[-1]
    img.save(osp.join(out_pth,name))
    
    
val_pth = 'D:/Workspace/data/cityscapes/leftImg8bit/val/frankfurt'
imgs = os.listdir(val_pth)
imgpth = [osp.join(val_pth,img) for img in imgs]
img0 = Img.open(imgpth[0])
use_boundary_16 = False
use_boundary_8 = True
use_boundary_4 = False
use_boundary_2 = False
n_classes = 19
backbone = 'STDCNet813'

dsval = CityScapes('D:/Workspace/data/cityscapes', mode='val')
dl = DataLoader(dsval,
                    batch_size = 5,
                    shuffle = False,
                    num_workers = 1,
                    drop_last = False)

diter = enumerate(tqdm(dl))
for i,(imgs,label) in diter:
    N, C, H, W = imgs.size()
    time.sleep(3)
    print(N, C, H, W)
    print(label.size())