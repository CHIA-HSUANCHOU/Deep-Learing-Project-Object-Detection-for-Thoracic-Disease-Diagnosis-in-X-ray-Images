#!/usr/bin/env python
# coding: utf-8

# # Object detection on medical image
# - Dataset format
# - Model pipeline

# In[ ]:


get_ipython().system('pip install pydicom')


# In[ ]:


get_ipython().system('pip install scikit-multilearn')


# In[ ]:


get_ipython().system('pip install grad-cam')


# In[ ]:


get_ipython().system('pip install opencv-python')


# In[ ]:


# import libraries

# basic
import warnings
warnings.filterwarnings('ignore')

import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import random
import pydicom
import itertools
import cv2
import numpy as np
import pandas as pd
import math
from tqdm.notebook import tqdm
from sklearn.preprocessing import MultiLabelBinarizer
from skmultilearn.model_selection import iterative_train_test_split
# visualization
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
# PyTorch
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection import FasterRCNN
from torchvision.transforms import v2
from torchvision import tv_tensors
from torchvision.tv_tensors import BoundingBoxes

# object detection
get_ipython().system('pip install pycocotools')
import pycocotools
from pycocotools.coco import COCO

# object detection
import json
from skimage.measure import label as sk_label
from skimage.measure import regionprops as sk_regions

import pytorch_grad_cam
from pytorch_grad_cam import EigenCAM, AblationCAM
from pytorch_grad_cam.ablation_layer import AblationLayerFasterRCNN
from pytorch_grad_cam.utils.model_targets import FasterRCNNBoxScoreTarget
from pytorch_grad_cam.utils.reshape_transforms import fasterrcnn_reshape_transform
from pytorch_grad_cam.utils.image import show_cam_on_image


# 為了使用 COCO API 來評估模型成效，我們會需要用到以下五個檔案

# In[ ]:


get_ipython().system('wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/engine.py')
get_ipython().system('wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/utils.py')
get_ipython().system('wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/coco_utils.py')
get_ipython().system('wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/coco_eval.py')
get_ipython().system('wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/transforms.py')

from engine import evaluate


# In[ ]:


from google.colab import drive
drive.mount('/content/drive')

data_path = '/content/drive/MyDrive/processed_images'
print("目錄內容：", os.listdir(data_path))  # 檢查檔案是否存在


# In[ ]:


## TODO: Prepare your own information
class config:

    ## roots for training & valid
    root = "/content/drive/MyDrive/processed_images/train"
    info_root = "/content/drive/MyDrive/processed_images/train"
    save_root = "/content/drive/MyDrive/working/"

    ## for test images
    test_root = "/content/drive/MyDrive/processed_images/test"
    info_root_test = "/content/drive/MyDrive/processed_images/test"

    num_classes = 8 #(for fasterrcnn: background + # of classes): 1+7=8

    batch_size = 4
    epochs = 20
    weight_decay = 1e-7
    lr = 1e-3
    momentum = 0.9 ###SGD
    seed = 42
    workers = 4
    categories = ['normal', 'aortic_curvature', 'aortic_atherosclerosis_calcification',
                  'cardiac_hypertrophy', 'intercostal_pleural_thickening', 'lung_field_infiltration',
                  'degenerative_joint_disease_of_the_thoracic_spine', 'scoliosis']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[ ]:


def seed_everything(seed):

    random.seed(seed) # Set Python random seed
    np.random.seed(seed) # Set NumPy random seed
    torch.manual_seed(seed) # Set PyTorch random seed for CPU and GPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Set PyTorch deterministic operations for cudnn backend
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(config.seed)


# ---
# 
# # Read data information
# 
# 我們可以利用 `pycocotools` 這個套件來讀取 .json 檔案中的資料
# ## Categories
# 
# 包含所有類別的 dictionary ( 不含 background ) ，每個 dictionary 中需要 2 個 key :
# 
# * `id` : 類別編號
# * `name` : 類別名稱

# In[ ]:


annfile = config.info_root + "/train.json"
#annfile = '/content/drive/MyDrive/train.json'
coco = COCO(annfile)   ###object(dataloader比較方便)
coco.cats


# ## Images
# 
# 影像相關資訊，一個 dictionary 含一張影像，內有 4 個 key :
# 
# * `file_name` : 影像路徑
# * `height` : 影像高度
# * `width` : 影像寬度
# * `id` : 影像編號 ( unique  )

# In[ ]:


coco.loadImgs(0)


# ## Annotations
# 
# 標註相關資訊，一個 dictionary 只包含一個 annotation ( bounding box ) ，共有 7 個 key :
# 
# * `id` : 該 annotation 的編號
# * `image_id` : 該 bounding box 所屬影像的編號
# * `category_id` : 該 bounding box 所屬類別的編號
# * `bbox` : bounding box 的標註資訊，格式為 $[\text{xmin}, \text{ymin}, \text{width}, \text{height}]$。$\text{xmin}$ 和 $\text{ymin}$ 表示 bounding box 左上角在影像上的座標，$\text{width}$ 和 $\text{height}$ 則為 bounding box 的寬跟高
# * `area` : 每個 bounding box 所圍出的面積。
# * `iscrowd` : 是單一物件 ( 0 ) 或一組物件 ( 1 )。segmentation 時使用，此處直接設為 0 即可
# * `segmentation` : segmentation 時使用，可忽略

# In[ ]:


ann_ids = coco.getAnnIds(imgIds = 0)
coco.loadAnns(ann_ids)


# In[ ]:


###目前id較原本的sample多一
ann_ids = coco.getAnnIds(imgIds = 155)
coco.loadAnns(ann_ids)


# In[ ]:


del coco


# ---
# 
# # Data augmentation
# 
# 
# 

# In[ ]:


##Augmentation?No
class medTransform:
    def __init__(self, train=False):
        if train:
            self.transforms = v2.Compose(
                [
                    v2.ToImage(), ## Used while using PIL image
                    v2.ToDtype(torch.float32, scale=True),
                ]
            )
        else:
            self.transforms = v2.Compose(
                [
                    v2.ToImage(), ## Used while using PIL image
                    v2.ToDtype(torch.float32, scale=True),
                ]
            )

    def __call__(self, x, bboxes):
        if isinstance(x, torch.Tensor):
            height, width = x.shape[-2], x.shape[-1]  # (C, H, W) format
        else:  # Assuming x is a PIL Image
            width, height = x.size
        ## Loading format is COCO bboxes[x,y,w,h](吃左上XY和右下的XY)
        bboxes = tv_tensors.BoundingBoxes(bboxes, format="XYWH", canvas_size=(height,width))
        return self.transforms(x, bboxes)


# ---
# 
# # Dataset
# 
# 在 Dataset 的部分，我們需要回傳的東西有兩項：image 和 target。
# 
# image 與先前作業沒有太大差異，只有讀取方式有所不同。至於 target 則是一個 dictionary，裡面需包含 5 個 key：
# 
# 1. `boxes`：該影像中所有 bounding box 的標註，格式為 $[\text{xmin}, \text{ymin}, \text{xmax}, \text{ymax}]$。$\text{xmin}$ 和 $\text{ymin}$ 表示 bounding box 左上角在影像上的座標，$\text{xmax}$ 和 $\text{ymax}$ 則表示 bounding box 右下角在影像上的座標
# 2. `labels`：每個 bounding box 所對應的疾病類別
# 3. `image_id`：影像編號
# 4. `area`：每個 bounding box 所圍出的面積。
# 5. `iscrowd`：是單一物件 ( 0 ) 或一組物件 ( 1 )。segmentation 時使用，此處直接設為 0 即可

# In[ ]:


class MedDataset(Dataset):

    def __init__(self, root, info_root, split, transforms = None):
        self.split = split
        self.root = root
        self.info_root = info_root
        self.transforms = transforms
        self.coco = COCO(os.path.join(self.info_root, f"{self.split}.json"))
        self.ids = list(sorted(self.coco.imgs.keys()))

    def get_image(self, img_id: int):
        image_path = os.path.join(self.root,self.coco.loadImgs(img_id)[0]['file_name'])
        image = Image.open(image_path).convert("RGB")
        return image

    def get_annotation(self, img_id: int):
        return self.coco.loadAnns(self.coco.getAnnIds(img_id))

    def __getitem__(self, index):
        normal = False
        img_id = self.ids[index]
        image = self.get_image(img_id)
        annotation = self.get_annotation(img_id)

        bboxes = [a['bbox']  for a in annotation]
        category_ids = [a['category_id']  for a in annotation]
        if bboxes == []:
            normal = True

        if self.transforms:
            image, bboxes = self.transforms(image, bboxes)

        def reformat_bboxes(boxes):
            return [[val[0], val[1], val[0] + val[2], val[1] + val[3]] for val in boxes]

        if normal != True:
            ## Recall that the original format is COCO
            bboxes = reformat_bboxes(bboxes)

        def create_target(bboxes, normal):
            if normal:
                return {
                    'boxes': torch.zeros((0, 4), dtype=torch.float32),  # Empty boxes
                    'labels': torch.tensor(category_ids, dtype=torch.int64),
                    'image_id': img_id,
                    'area': torch.zeros((0,), dtype=torch.float32),  # Empty areas
                    'iscrowd': torch.zeros((0,), dtype=torch.int64),  # Empty tensor for iscrowd
                }
            else:

                return {
                    'boxes': torch.tensor(bboxes),
                    'labels': torch.tensor(category_ids, dtype=torch.int64),
                    'image_id': img_id,
                    'area': torch.tensor([(bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) for bbox in bboxes], dtype=torch.float32),
                    'iscrowd': torch.tensor([a['iscrowd'] for a in annotation], dtype=torch.int64)
                }

        targets = create_target(bboxes,normal)
        return image, targets

    def __len__(self):
        return len(self.ids)


# ## Collate_fn
# 
# 用於 dataloader。由於 object detection 讀取 data 的方式與先前的 classification 和 segmentation 有所不同，因此需自定義 `collate_fn`。 <br>
# 

# In[ ]:


def collate_fn(batch: list[torch.tensor, dict]):
    return tuple(zip(*batch))


# In[ ]:


def plot_image_with_boxes(image_tensor, boxes_dict):
    image_np = image_tensor.permute(1, 2, 0).numpy()
    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(image_np)
    for box in boxes_dict['boxes']:
        # Extract coordinates (x0, y0, x1, y1)
        x0, y0, x1, y1 = box
        # Calculate the height as (y0 - y1) since y0 is the top and y1 is the bottom
        height = y1 - y0
        # Create a rectangle patch with (x0, y0) as the top-left corner
        rect = patches.Rectangle((x0, y0), x1 - x0, height, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    plt.show()


# In[ ]:


train_dataset = MedDataset(root = config.root, info_root = config.info_root, split = "train", transforms = medTransform(train=True))
train_loader = DataLoader(train_dataset, batch_size = config.batch_size, shuffle = True,num_workers=4, collate_fn = collate_fn)


# In[ ]:


a,b = train_dataset.__getitem__(1)
plot_image_with_boxes(a,b)


# In[ ]:


a,b = train_dataset.__getitem__(155)
print(b)
plot_image_with_boxes(a,b)


# ---
# 
# # Model: Faster R-CNN
# 
# 這邊使用 torchvision 中內建的 Faster R-CNN 模型，並加載預訓練權重，但要記得更改 predictor 的類別數量為 8 類 ( 含 background，也就是 normal ) ，如下所示：

# In[ ]:


def fasterrcnn(num_classes):
    model = models.detection.fasterrcnn_resnet50_fpn(weights='COCO_V1')
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = None
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


# In[ ]:


model = fasterrcnn(config.num_classes)


# 模型架構如下：

# ---
# 
# # Training
# 
# 在 PyTorch 的 Faster R-CNN 這個模型中，我們不須再自行定義 loss function，因為在 `model.train()` 下，`model(images, targets)` 會自動回傳訓練的 loss，其包含以下四種損失：
# 
# 1. `loss_classifier`：分類器之損失
# 2. `loss_box_reg`：bounding box regressor 之損失
# 3. `loss_rpn_box_reg`：RPN 中 bounding box regressor 之損失
# 4. `loss_objectness`：RPN 中分類器之損失，此分類器用以判斷 bounding box 中是否包含物體
# 
# 總損失為這四種 loss 的總和。

# In[ ]:


def train_one_epoch(model, train_loader, optimizer, epoch, device):
    model.train()

    train_loss = []
    train_loss_dict = []

    lr_scheduler = None

    for images, targets in tqdm(train_loader):
        images = [image.to(device) for image in images]
        targets = [{k: (torch.tensor(v,device=device) if not isinstance(v, torch.Tensor) else v.to(device)) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        batch_loss_value = losses.item()
        batch_loss_dict = {k: v.item() for k, v in loss_dict.items()}

        train_loss.append(batch_loss_value)
        train_loss_dict.append(batch_loss_dict)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

    train_loss = np.mean(train_loss)
    train_loss_dict = pd.DataFrame(train_loss_dict).mean()
    train_loss_classifier = train_loss_dict.loss_classifier
    train_loss_box_reg = train_loss_dict.loss_box_reg
    train_loss_rpn_box_reg = train_loss_dict.loss_rpn_box_reg
    train_loss_objectness = train_loss_dict.loss_objectness

    return train_loss, train_loss_classifier, train_loss_box_reg, train_loss_rpn_box_reg, train_loss_objectness


# ---
# 
# # Validation
# 
# 在此模型中，若設定 `model.eval()`，只會返回預測的 bounding box、confidence score 和該 bounding box 的 label。
# 
# 為了取得 validation set 的 loss 以選出最好的模型，這裡我在進行 validation 時使用 `model.train()`。如果要這麼做，需要把模型中的 batch normalization 和 dropout 的係數固定住，但因 Faster R-CNN 中不含 dropout 層，所以只需凍結 batch normalization 層的參數。

# In[ ]:


def validation(model, val_loader, device):
    model.train()
    #model.eval()
    for m in model.modules():
        if isinstance(m, torchvision.ops.Conv2dNormActivation):
            m.eval()
        if isinstance(m, torchvision.ops.FrozenBatchNorm2d):
            m.eval()
        if isinstance(m, torch.nn.BatchNorm2d):
            m.eval()
    val_loss = []
    val_loss_dict = []
    with torch.no_grad():
        for images, targets in tqdm(val_loader):
            images = [image.to(device) for image in images]
            targets = [{k: (torch.tensor(v,device=device) if not isinstance(v, torch.Tensor) else v.to(device)) for k, v in t.items()} for t in targets]

            loss = model(images, targets)
            total_loss = sum(l for l in loss.values())

            loss_value = total_loss.item()
            loss_dict = {k: v.item() for k, v in loss.items()}

            val_loss.append(loss_value)
            val_loss_dict.append(loss_dict)

    val_loss = np.mean(val_loss)

    val_loss_dict = pd.DataFrame(val_loss_dict).mean()
    val_loss_classifier = val_loss_dict.loss_classifier
    val_loss_box_reg = val_loss_dict.loss_box_reg
    val_loss_rpn_box_reg = val_loss_dict.loss_rpn_box_reg
    val_loss_objectness = val_loss_dict.loss_objectness

    return val_loss, val_loss_classifier, val_loss_box_reg, val_loss_rpn_box_reg, val_loss_objectness


# In[ ]:


del model
del train_dataset, train_loader


# In[ ]:


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ---
# 
# # Main

# ##SGD batch size=8

# In[ ]:


def train_one_epoch(model, train_loader, optimizer, epoch, device):
    model.train()

    train_loss = []
    train_loss_dict = []

    lr_scheduler = None

    for images, targets in tqdm(train_loader):
        images = [image.to(device) for image in images]
        targets = [{k: (torch.tensor(v,device=device) if not isinstance(v, torch.Tensor) else v.to(device)) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        batch_loss_value = losses.item()
        batch_loss_dict = {k: v.item() for k, v in loss_dict.items()}

        train_loss.append(batch_loss_value)
        train_loss_dict.append(batch_loss_dict)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

    train_loss = np.mean(train_loss)
    train_loss_dict = pd.DataFrame(train_loss_dict).mean()
    train_loss_classifier = train_loss_dict.loss_classifier
    train_loss_box_reg = train_loss_dict.loss_box_reg
    train_loss_rpn_box_reg = train_loss_dict.loss_rpn_box_reg
    train_loss_objectness = train_loss_dict.loss_objectness

    return train_loss, train_loss_classifier, train_loss_box_reg, train_loss_rpn_box_reg, train_loss_objectness


# In[ ]:


def main():

    seed_everything(config.seed)
    g = torch.Generator()
    g.manual_seed(config.seed)

    train_dataset = MedDataset(root = config.root, info_root = config.info_root, split = "train", transforms = medTransform(train=True))
    val_dataset = MedDataset(root = config.root, info_root = config.info_root, split = "val",  transforms = medTransform(train=False))

    train_loader = DataLoader(train_dataset, batch_size = config.batch_size, shuffle = True,
                              num_workers=config.workers, collate_fn = collate_fn,pin_memory=True
                             )
    val_loader = DataLoader(val_dataset, batch_size = config.batch_size, shuffle = False,
                            num_workers=config.workers, worker_init_fn=seed_worker,
                            generator=g, collate_fn = collate_fn,pin_memory=True
                           )


    device = config.device
    model =  fasterrcnn(num_classes = config.num_classes)
    model.to(device)

    parameters = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.SGD(parameters, lr = config.lr, momentum = config.momentum, nesterov = True, weight_decay = config.weight_decay)

    #scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=0)

    best_val_loss = float("inf")
    best_map50 = 0.0
    history = {
        "train": {
            "loss": [],
            "loss_classifier": [],
            "loss_box_reg": [],
            "loss_rpn_box_reg": [],
            "loss_objectness": []
        },
        "val": {
            "loss": [],
            "loss_classifier": [],
            "loss_box_reg": [],
            "loss_rpn_box_reg": [],
            "loss_objectness": []
        },
        "map50":{
            "train":[],
            "valid":[],
        }
    }
    best_idx = 0
    print('start')
    for epoch in range(config.epochs):
        print()
        train_loss, train_loss_classifier, train_loss_box_reg, train_loss_rpn_box_reg, train_loss_objectness = train_one_epoch(
            model, train_loader, optimizer, epoch, device,
        )

        val_loss, val_loss_classifier, val_loss_box_reg, val_loss_rpn_box_reg, val_loss_objectness = validation(
            model, val_loader, device
        )

        ## Training
        history["train"]["loss"].append(train_loss)
        history["train"]["loss_classifier"].append(train_loss_classifier)
        history["train"]["loss_box_reg"].append(train_loss_box_reg)
        history["train"]["loss_rpn_box_reg"].append(train_loss_rpn_box_reg)
        history["train"]["loss_objectness"].append(train_loss_objectness)
        ## Validation
        history["val"]["loss"].append(val_loss)
        history["val"]["loss_classifier"].append(val_loss_classifier)
        history["val"]["loss_box_reg"].append(val_loss_box_reg)
        history["val"]["loss_rpn_box_reg"].append(val_loss_rpn_box_reg)
        history["val"]["loss_objectness"].append(val_loss_objectness)


        print(f'Epoch: {epoch+1}/{config.epochs} | LR: {optimizer.state_dict()["param_groups"][0]["lr"]:.6f}')

        print("*****Training*****")
        print(f'Loss: {train_loss:.4f} | Classifier Loss: {train_loss_classifier:.4f} | Box Reg Loss: {train_loss_box_reg:.4f} | RPN Box Reg Loss: {train_loss_rpn_box_reg:.4f} | Objectness Loss: {train_loss_objectness:.4f}')
        train_evaluator = evaluate(model, train_loader, device = device)
        print("*****Validation*****")
        print(f'Loss: {val_loss:.4f} | Classifier Loss: {val_loss_classifier:.4f} | Box Reg Loss: {val_loss_box_reg:.4f} | RPN Box Reg Loss: {val_loss_rpn_box_reg:.4f} | Objectness Loss: {val_loss_objectness:.4f}')
        valid_evaluator = evaluate(model, val_loader, device = device)

        train_map50 = train_evaluator.coco_eval['bbox'].stats[1]
        valid_map50 = valid_evaluator.coco_eval['bbox'].stats[1]
        print("*****Training*****")
        print(f'MAP: {train_map50:.4f}')
        print("*****Validationg*****")
        print(f'MAP: {valid_map50:.4f}')
        history["map50"]["train"].append(train_map50)
        history["map50"]["valid"].append(valid_map50)

        ## TODO save your model

        if valid_map50 > best_map50:
            save_file = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "args": config
            }
            best_idx=epoch
            best_map50 = valid_map50
            torch.save(save_file, os.path.join(config.save_root,"final.pth"))
        #scheduler.step()
    print(f'Best epoch in {best_idx+1}')


    ## Evaluation result
    plt.figure(figsize = (12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(config.epochs), history["map50"]["train"], label = 'Training map50')
    plt.plot(range(config.epochs), history["map50"]["valid"], label = 'Validation map50')
    plt.xlabel('Epoch')
    plt.ylabel('map')
    plt.legend()
    plt.title('Training and Validation map50')
    plt.show()

    plt.figure(figsize = (12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(config.epochs), history["train"]["loss"], label = 'Training Loss')
    plt.plot(range(config.epochs), history["val"]["loss"], label = 'Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss Curves')
    plt.show()

    plt.figure(figsize = (12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(config.epochs), history["train"]["loss_classifier"], label = 'Training Classifier Loss')
    plt.plot(range(config.epochs), history["val"]["loss_classifier"], label = 'Validation Classifier Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Classifier Loss')
    plt.legend()
    plt.title('Training and Validation Classifier Loss Curves')
    plt.show()

    plt.figure(figsize = (12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(config.epochs), history["train"]["loss_box_reg"], label = 'Training Box Reg Loss')
    plt.plot(range(config.epochs), history["val"]["loss_box_reg"], label = 'Validation Box Reg Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Box Reg Loss')
    plt.legend()
    plt.title('Training and Validation Box Reg Loss Curves')
    plt.show()

    plt.figure(figsize = (12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(config.epochs), history["train"]["loss_rpn_box_reg"], label = 'Training RPN Box Reg Loss')
    plt.plot(range(config.epochs), history["val"]["loss_rpn_box_reg"], label = 'Validation RPN Box Reg Loss')

    plt.xlabel('Epoch')
    plt.ylabel('RPN Box Reg Loss')
    plt.legend()
    plt.title('Training and Validation RPN Box Reg Loss Curves')
    plt.show()

    plt.figure(figsize = (12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(config.epochs), history["train"]["loss_objectness"], label = 'Training Objectness Loss')
    plt.plot(range(config.epochs), history["val"]["loss_objectness"], label = 'Validation Objectness Loss')

    plt.xlabel('Epoch')
    plt.ylabel('Objectness Loss')
    plt.legend()
    plt.title('Training and Validation Objectness Loss Curves')
    plt.show()


# In[ ]:


## IMAGENET 3
if __name__ == "__main__":
    main()


# ##AdamW+Onecycle

# In[ ]:


def train_one_epoch(model, train_loader, optimizer, lr_scheduler,  epoch, device):
    model.train()

    train_loss = []
    train_loss_dict = []

    #lr_scheduler = None

    for images, targets in tqdm(train_loader):
        images = [image.to(device) for image in images]
        targets = [{k: (torch.tensor(v,device=device) if not isinstance(v, torch.Tensor) else v.to(device)) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        batch_loss_value = losses.item()
        batch_loss_dict = {k: v.item() for k, v in loss_dict.items()}

        train_loss.append(batch_loss_value)
        train_loss_dict.append(batch_loss_dict)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

    train_loss = np.mean(train_loss)
    train_loss_dict = pd.DataFrame(train_loss_dict).mean()
    train_loss_classifier = train_loss_dict.loss_classifier
    train_loss_box_reg = train_loss_dict.loss_box_reg
    train_loss_rpn_box_reg = train_loss_dict.loss_rpn_box_reg
    train_loss_objectness = train_loss_dict.loss_objectness

    return train_loss, train_loss_classifier, train_loss_box_reg, train_loss_rpn_box_reg, train_loss_objectness


# In[ ]:


##AdamW OneSample
def main():

    seed_everything(config.seed)
    g = torch.Generator()
    g.manual_seed(config.seed)

    train_dataset = MedDataset(root = config.root, info_root = config.info_root, split = "train", transforms = medTransform(train=True))
    val_dataset = MedDataset(root = config.root, info_root = config.info_root, split = "val",  transforms = medTransform(train=False))

    train_loader = DataLoader(train_dataset, batch_size = config.batch_size, shuffle = True,
                              num_workers=config.workers, collate_fn = collate_fn,pin_memory=True
                             )
    val_loader = DataLoader(val_dataset, batch_size = config.batch_size, shuffle = False,
                            num_workers=config.workers, worker_init_fn=seed_worker,
                            generator=g, collate_fn = collate_fn,pin_memory=True
                           )


    device = config.device
    model =  fasterrcnn(num_classes = config.num_classes)
    model.to(device)

    parameters = [p for p in model.parameters() if p.requires_grad]

    # Use AdamW optimizer
    optimizer = torch.optim.AdamW(parameters, lr=config.lr, weight_decay=config.weight_decay)
    # Use OneSampleLR scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=config.lr,  # Maximum learning rate
    steps_per_epoch=len(train_loader),  # Number of steps per epoch
    epochs=config.epochs,  # Total number of epochs
    anneal_strategy='cos',  # Cosine annealing for learning rate decay
    pct_start=0.3  # Proportion of the cycle spent increasing learning rate
)
    best_val_loss = float("inf")
    best_map50 = 0.0
    history = {
        "train": {
            "loss": [],
            "loss_classifier": [],
            "loss_box_reg": [],
            "loss_rpn_box_reg": [],
            "loss_objectness": []
        },
        "val": {
            "loss": [],
            "loss_classifier": [],
            "loss_box_reg": [],
            "loss_rpn_box_reg": [],
            "loss_objectness": []
        },
        "map50":{
            "train":[],
            "valid":[],
        }
    }
    best_idx = 0
    print('start')
    for epoch in range(config.epochs):
        print()
        train_loss, train_loss_classifier, train_loss_box_reg, train_loss_rpn_box_reg, train_loss_objectness = train_one_epoch(
            model, train_loader, optimizer, scheduler, epoch, device,
        )

        val_loss, val_loss_classifier, val_loss_box_reg, val_loss_rpn_box_reg, val_loss_objectness = validation(
            model, val_loader, device
        )

        ## Training
        history["train"]["loss"].append(train_loss)
        history["train"]["loss_classifier"].append(train_loss_classifier)
        history["train"]["loss_box_reg"].append(train_loss_box_reg)
        history["train"]["loss_rpn_box_reg"].append(train_loss_rpn_box_reg)
        history["train"]["loss_objectness"].append(train_loss_objectness)
        ## Validation
        history["val"]["loss"].append(val_loss)
        history["val"]["loss_classifier"].append(val_loss_classifier)
        history["val"]["loss_box_reg"].append(val_loss_box_reg)
        history["val"]["loss_rpn_box_reg"].append(val_loss_rpn_box_reg)
        history["val"]["loss_objectness"].append(val_loss_objectness)


        print(f'Epoch: {epoch+1}/{config.epochs} | LR: {optimizer.state_dict()["param_groups"][0]["lr"]:.6f}')

        print("*****Training*****")
        print(f'Loss: {train_loss:.4f} | Classifier Loss: {train_loss_classifier:.4f} | Box Reg Loss: {train_loss_box_reg:.4f} | RPN Box Reg Loss: {train_loss_rpn_box_reg:.4f} | Objectness Loss: {train_loss_objectness:.4f}')
        train_evaluator = evaluate(model, train_loader, device = device)
        print("*****Validation*****")
        print(f'Loss: {val_loss:.4f} | Classifier Loss: {val_loss_classifier:.4f} | Box Reg Loss: {val_loss_box_reg:.4f} | RPN Box Reg Loss: {val_loss_rpn_box_reg:.4f} | Objectness Loss: {val_loss_objectness:.4f}')
        valid_evaluator = evaluate(model, val_loader, device = device)

        train_map50 = train_evaluator.coco_eval['bbox'].stats[1]
        valid_map50 = valid_evaluator.coco_eval['bbox'].stats[1]
        print("*****Training*****")
        print(f'MAP: {train_map50:.4f}')
        print("*****Validationg*****")
        print(f'MAP: {valid_map50:.4f}')
        history["map50"]["train"].append(train_map50)
        history["map50"]["valid"].append(valid_map50)

        ## TODO save your model

        if valid_map50 > best_map50:
            save_file = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "args": config
            }
            best_idx=epoch
            best_map50 = valid_map50
            torch.save(save_file, os.path.join(config.save_root,"final.pth"))
        #scheduler.step()
    print(f'Best epoch in {best_idx+1}')


    ## Evaluation result
    plt.figure(figsize = (12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(config.epochs), history["map50"]["train"], label = 'Training map50')
    plt.plot(range(config.epochs), history["map50"]["valid"], label = 'Validation map50')
    plt.xlabel('Epoch')
    plt.ylabel('map')
    plt.legend()
    plt.title('Training and Validation map50')
    plt.show()

    plt.figure(figsize = (12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(config.epochs), history["train"]["loss"], label = 'Training Loss')
    plt.plot(range(config.epochs), history["val"]["loss"], label = 'Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss Curves')
    plt.show()

    plt.figure(figsize = (12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(config.epochs), history["train"]["loss_classifier"], label = 'Training Classifier Loss')
    plt.plot(range(config.epochs), history["val"]["loss_classifier"], label = 'Validation Classifier Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Classifier Loss')
    plt.legend()
    plt.title('Training and Validation Classifier Loss Curves')
    plt.show()

    plt.figure(figsize = (12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(config.epochs), history["train"]["loss_box_reg"], label = 'Training Box Reg Loss')
    plt.plot(range(config.epochs), history["val"]["loss_box_reg"], label = 'Validation Box Reg Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Box Reg Loss')
    plt.legend()
    plt.title('Training and Validation Box Reg Loss Curves')
    plt.show()

    plt.figure(figsize = (12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(config.epochs), history["train"]["loss_rpn_box_reg"], label = 'Training RPN Box Reg Loss')
    plt.plot(range(config.epochs), history["val"]["loss_rpn_box_reg"], label = 'Validation RPN Box Reg Loss')

    plt.xlabel('Epoch')
    plt.ylabel('RPN Box Reg Loss')
    plt.legend()
    plt.title('Training and Validation RPN Box Reg Loss Curves')
    plt.show()

    plt.figure(figsize = (12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(config.epochs), history["train"]["loss_objectness"], label = 'Training Objectness Loss')
    plt.plot(range(config.epochs), history["val"]["loss_objectness"], label = 'Validation Objectness Loss')

    plt.xlabel('Epoch')
    plt.ylabel('Objectness Loss')
    plt.legend()
    plt.title('Training and Validation Objectness Loss Curves')
    plt.show()


# In[ ]:


## IMAGENET 3
if __name__ == "__main__":
    main()


# In[ ]:


class MedInferenceDataset(Dataset):
    def __init__(self, root, transforms=None):
        """
        初始化資料集，只處理圖片檔案。

        :param root: 圖片資料夾路徑
        :param transforms: 圖片處理方法（可選）
        """
        self.root = root
        self.transforms = transforms
        self.image_files = sorted(
            [f for f in os.listdir(self.root) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        )

    def get_image(self, index):
        """
        讀取指定索引的圖片。

        :param index: 圖片索引
        :return: PIL 圖片物件
        """
        image_path = os.path.join(self.root, self.image_files[index])
        image = Image.open(image_path).convert("RGB")

        return image

    def __getitem__(self, index):
        image = self.get_image(index)
        #print(f"Fetching image at index: {index}")
        if self.transforms:
            image = self.transforms(image)

        target = {
            'boxes': torch.zeros((0, 4), dtype=torch.float32),  # 空的 bounding box
            'labels': torch.tensor([], dtype=torch.int64),      # 空的標籤
            'image_id': torch.tensor([index]),                 # 圖片 ID
            'size': image.shape[1:]  # 圖片原始尺寸 (高度, 寬度)
        }
        #'size': image.size[::-1],  # 圖片原始尺寸 (高度, 寬度)
        #'size': image.shape[1:]
        #print(f"Image{image}")
        #print(f"Target{target}")
        return image, target

    def __len__(self):
        """
        返回資料集中圖片的數量。
        """
        return len(self.image_files)


# In[ ]:


def get_transform():

    transform = v2.Compose(
                [
                    v2.ToImage(), ## Used while using PIL image
                    #v2.ConvertBoundingBoxFormat(tv_tensors.BoundingBoxFormat.XYXY),
                    v2.ToDtype(torch.float32, scale=True),

                ])

    return transform


# In[ ]:


def predict(input_tensor, model, device, detection_threshold):
    outputs = model(input_tensor)
    pred_classes = [config.categories[i] for i in outputs[0]['labels'].cpu().numpy()]
    pred_labels = outputs[0]['labels'].cpu().numpy()
    pred_scores = outputs[0]['scores'].detach().cpu().numpy()
    pred_bboxes = outputs[0]['boxes'].detach().cpu().numpy()

    boxes, classes, labels, indices,scores = [], [], [], [], []
    for index in range(len(pred_scores)):
        if pred_scores[index] >= detection_threshold:
            boxes.append(pred_bboxes[index].astype(np.int32))
            classes.append(pred_classes[index])
            labels.append(pred_labels[index])
            indices.append(index)
            scores.append(pred_scores[index])
    boxes = np.int32(boxes)

    return boxes, classes, labels, indices, scores

COLORS = np.random.uniform(0, 255, size=(len(config.categories), 3))

def draw_boxes(boxes, labels, classes, image):
    for i, box in enumerate(boxes):
        # Convert RGB to BGR for OpenCV
        color = COLORS[labels[i]].astype(int)[::-1]

        # Draw the bounding box
        cv2.rectangle(
            image,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            color.tolist(), 8
        )

        # Increase font size and thickness for label
        font_scale = 4 # Increase this value for larger font
        thickness = 10     # Increase thickness for better visibility

        # Add class label as text
        cv2.putText(image, classes[i],
                    (int(box[0]), int(box[1]) - 10),  # Adjust text position
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    color.tolist(),
                    thickness,
                    lineType=cv2.LINE_AA)
    return image


# In[ ]:


def run_predictions(model, test_loader, device, detection_threshold):
    model.eval()
    all_predictions = []

    with torch.no_grad():
        for batch_idx, (images2, targets) in enumerate(test_loader):
                image_data = images2[0]
                input_tensor = image_data.unsqueeze(0).to(config.device)
                boxes, classes, labels, indices, scores = predict(
                    input_tensor, model, device, detection_threshold
                )

                target_dict = targets[0]
                img_id = target_dict["image_id"].item()
                img_size = target_dict["size"]

            # 取得圖片的寬度和高度
                img_height, img_width = img_size[0], img_size[1]

            # 正規化邊界框
                for j, box in enumerate(boxes):
                    xmin, ymin, xmax, ymax = box

                # 根據圖片的寬度和高度進行正規化
                    xmin_norm = xmin / img_width
                    ymin_norm = ymin / img_height
                    xmax_norm = xmax / img_width
                    ymax_norm = ymax / img_height

                # 存正規化後的預測結果
                    all_predictions.append({
                        "ID": img_id,
                        "category": classes[j],
                        "score": scores[j],
                        "xmin": xmin_norm,
                        "xmax": xmax_norm,
                        "ymin": ymin_norm,
                        "ymax": ymax_norm,
                    })

    return all_predictions


# In[ ]:


# 測試資料集與 DataLoader
test_dataset = MedInferenceDataset(root=config.test_root, transforms=get_transform())
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,num_workers=config.workers, collate_fn=collate_fn)
model = fasterrcnn(num_classes = config.num_classes)
state_dict = torch.load("/content/drive/MyDrive/working/final.pth")
model.load_state_dict(state_dict["model"])
model.to(config.device)


# In[ ]:


predictions = run_predictions(model, test_loader, config.device,0.5)
# 建立 DataFrame 並根據 test_ids 排序
results_df = pd.DataFrame(predictions)
results_df["ID"] = results_df["ID"].astype(int) + 1
results_df = results_df.sort_values(by="ID")
#results_df.head(10)


# In[ ]:


test_csv_df = pd.read_csv('/content/drive/MyDrive/hwk05_data/test.csv')
test_csv_df['filename_id'] = test_csv_df['Filename'].str.extract(r'(\d+)').astype(int)


# In[ ]:


merged_df = results_df.merge(test_csv_df[['filename_id', 'ID']], left_on='ID', right_on='filename_id', how='left')
# 可以刪除不再需要的欄位
merged_df = merged_df.drop(columns=['filename_id', 'ID_x'])
# 將 ID_y 移到第一列並將名稱更改為 ID
merged_df = merged_df.rename(columns={'ID_y': 'ID'})  # 重命名 ID_y 為 ID
# 將 ID 列移動到最前面
merged_df = merged_df[['ID'] + [col for col in merged_df.columns if col != 'ID']]
merged_df.head(10)


# In[ ]:


merged_df.tail(10)


# In[ ]:


merged_df.shape


# In[ ]:


from google.colab import files
# 保存 CSV 文件到指定路徑
merged_df.to_csv('/content/AdamWOneCyclelr10-3_batch4.csv', index=False, header = True)
# 下載文件到本地機器
files.download('/content/AdamWOneCyclelr10-3_batch4.csv')


# In[ ]:


def plot_ablation_cam_images(transforms, model):

    rows, cols = 4, 2
    fig = plt.figure(figsize = (10, 20))
    grid = plt.GridSpec(rows, cols)
    best_ckpt = torch.load("/content/drive/MyDrive/working/final.pth", map_location = config.device)
    model.load_state_dict(best_ckpt["model"])
    model.eval().to(config.device)
    target_layers = [model.backbone]

    cam = AblationCAM(model,
                      target_layers,
                      reshape_transform = fasterrcnn_reshape_transform,
                      ablation_layer = AblationLayerFasterRCNN(),
                      ratio_channels_to_ablate = 1.0)

    for i in range(rows * cols):

        all_images = os.listdir(os.path.join(config.root, config.categories[i]))
        image_path = os.path.join(config.root, config.categories[i], all_images[0])
        image = Image.open(image_path).convert("RGB")
        input_tensor = transforms(image)
        input_tensor = input_tensor.to(config.device)
        input_tensor = input_tensor.unsqueeze(0)
        image = np.array(image)
        image_float_np = np.float32(image) / 255

        boxes, classes, labels, indices, scores = predict(input_tensor, model, config.device, 0.5)
        image = draw_boxes(boxes, labels, classes, image)
        targets = [FasterRCNNBoxScoreTarget(labels = labels, bounding_boxes = boxes)]

        grayscale_cam = cam(input_tensor, targets = targets)
        grayscale_cam = grayscale_cam[0, :]
        cam_image = show_cam_on_image(image_float_np, grayscale_cam, use_rgb = True)
        image_with_bounding_boxes = draw_boxes(boxes, labels, classes, cam_image)

        categories = fig.add_subplot(grid[i])
        categories.set_axis_off()

        gs = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec = grid[i])

        ax = fig.add_subplot(gs[0])
        ax.imshow(image_with_bounding_boxes)
        ax.set_title(f"{config.categories[i]}")
        ax.axis("off")

    fig.patch.set_facecolor('white')
    fig.suptitle("AblationCAM Images of 8 categories\n", fontweight = 'bold', size = 16)
    fig.tight_layout()


# In[ ]:


plot_ablation_cam_images(transforms = get_transform(), model = fasterrcnn(config.num_classes))


# In[ ]:


import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import os
from PIL import Image
from matplotlib.colors import ListedColormap
from matplotlib.colors import TABLEAU_COLORS


# In[ ]:


def plot_image_with_boxes_test(df, test_csv_path, id_column, bbox_columns, image_dir):
    """
    Plots images with bounding boxes for each unique ID in the dataframe, with all boxes for the same ID on one image.

    Args:
        df (pd.DataFrame): DataFrame containing bounding box information.
        test_csv_path (str): Path to the test.csv file.
        id_column (str): Column name in df corresponding to the ID.
        bbox_columns (list of str): Column names for bounding box coordinates ["xmin", "xmax", "ymin", "ymax"].
        image_dir (str): Path to the directory containing processed test images.
        scale (float): Scale factor for resizing the image (0 < scale <= 1).
    """
    # Load test.csv
    test_csv = pd.read_csv("/content/drive/MyDrive/processed_images/test.csv")

    scale = 0.2
    unique_ids = df[id_column].unique()
    num_images = len(unique_ids)

    # Create subplots
    fig, axes = plt.subplots(
        nrows=(num_images + 2) // 3, ncols=3, figsize=(15, 5 * ((num_images + 2) // 3))
    )
    axes = axes.flatten()

    # Create a color map for categories
    categories = df['category'].unique()
    category_colors = {category: plt.cm.get_cmap('tab10')(i % 10) for i, category in enumerate(categories)}

    for ax, img_id in zip(axes, unique_ids):
        img_data = df[df[id_column] == img_id]

        # Find the corresponding dcm name in test.csv
        dcm_name = test_csv.loc[test_csv['ID'] == img_id, "Filename"].values[0]
        dcm_name = os.path.splitext(dcm_name)[0]

        # Construct the image path
        image_path = os.path.join(image_dir, f"{dcm_name}.jpg")

        # Check if the image file exists
        if not os.path.exists(image_path):
            ax.set_title(f"Image {dcm_name} not found.")
            ax.axis('off')
            continue

        # Open the image
        image = Image.open(image_path)
        width, height = image.size

        # Resize the image based on scale
        image = image.resize((int(width * scale), int(height * scale)))
        width, height = image.size

        # Plot the image
        ax.imshow(image)

        # Draw bounding boxes
        for i, row in img_data.iterrows():
            xmin, xmax, ymin, ymax = [row[col] for col in bbox_columns]
            bounding_width = (xmax - xmin) * width
            bounding_height = (ymax - ymin) * height

            # Get the color for the category
            category = row.get("category", "Unknown")
            color = category_colors.get(category, 'gray')  # Default to gray if category is not in the map

            rect = patches.Rectangle(
                (xmin * width, ymin * height),
                bounding_width,
                bounding_height,
                linewidth=2,
                edgecolor=color,
                facecolor='none'
            )
            ax.add_patch(rect)

            # Display the category text near the bounding box
            ax.text(
                xmin * width,
                ymin * height - 10,
                category,
                color='white',
                fontsize=8,
                bbox=dict(facecolor=color, alpha=0.5)
            )

        ax.set_title(f"ID: {img_id}")
        ax.axis('off')

    # Turn off any unused subplots
    for ax in axes[num_images:]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()


# Example usage
plot_image_with_boxes_test(merged_df[:10], "test.csv", "ID", ["xmin", "xmax", "ymin", "ymax"], "/content/drive/MyDrive/processed_images/test")

