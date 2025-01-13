# Deep Learning Project：Object Detection
---
Project: Object Detection for Thoracic Disease Diagnosis in X-ray Images

Author: CHOU CHIA-HSUAN

Date: 2025-12-30

Course: Deep Learning in Medical Image Analysis

---
## 1. Objective

This project aims to apply the **Faster R-CNN model** to **detect and annotate potential thoracic diseases** from chest X-ray images, enhancing diagnostic efficiency.

## 2. Data Background

- **Source:** Chest X-ray images provided by E-Da Hospital (DICOM format).
- **Dataset:**
  - **Training set:** 451 images covering 8 disease categories:
    - Normal (正常)
    - Aortic atherosclerosis calcification (主動脈硬鈣化)
    - Aortic curvature (主動脈彎曲)
    - Lung field infiltration (肺浸潤)
    - Degenerative joint disease of the thoracic spine (胸椎退化性關節病變)
    - Scoliosis (脊椎側彎)
    - Intercostal pleural thickening (肺肋膜增厚)
    - Cardiac hypertrophy (心臟肥大)
  - **Test set:** 113 images.
  - Due to patient privacy, photos will not be displayed.
## 3. Analysis Workflow

### (1) Data Preprocessing

- Applied intensity log-transformation and simplest color balance.
- Converted images to JPG format with annotated abnormal regions.
- Created corresponding JSON files to store bounding box information.

### (2) Model and Architecture

- **Model: Faster R-CNN**
- **Feature Extraction Backbone:** ResNet-50 with FPN for enhanced multi-level feature learning.
- **Optimizer:** AdamW with OneCycle Scheduler.

### (3) Model Evaluation Metrics

- **Loss Functions:**
  - Classifier loss
  - Bounding box regression loss
  - Bounding box regression loss about RPN
  - Bounding box object loss
- **Performance Metrics:**
  - Intersection over Union (IoU)
  - Mean Average Precision (mAP)

## 4. Results

### (1) Model Analysis with Ablation CAM

Using **Ablation CAM** to analyze the model's focus areas for disease detection, we observed:

- The attention regions for **aortic calcification**, **aortic curvature**, and **cardiac hypertrophy** align well with medical understanding.
- The model's ability to differentiate diseases like **thoracic degenerative joint disease**, **apical pleural thickening**, and **scoliosis** still requires improvement.

### (2) Mapping X-ray Images to Disease Locations

Mapping patient X-rays to corresponding disease locations indicates that **improvements are needed** for detecting conditions such as **scoliosis** and **lung field infiltration**.





# 深度學習:物件偵測
---
title: "深度學習專案：物件偵測"

author: "周佳萱"

date: "2025-12-30"

course: 深度學習於醫學影像分析

---

# 專案：使用物件偵測技術檢測胸部X光影像中的常見胸腔疾病

## 1. 目標

本研究主要目的是應用 **Faster R-CNN 模型**與胸部X光影像，**檢測並標註潛在的胸腔疾病**，以提升診斷效率。

## 2. 資料背景

- **來源：** 義大醫院提供的胸部 X 光影像資料。
- **資料集：**
  - **訓練集：** 451 張影像，涵蓋 8 種疾病類別：
    - 正常
    - 主動脈硬鈣化 (Aortic atherosclerosis calcification)
    - 主動脈彎曲 (Aortic curvature)
    - 肺浸潤 (Lung field infiltration)
    - 胸椎退化性關節病變 (Degenerative joint disease of the thoracic spine)
    - 脊椎側彎 (Scoliosis)
    - 肋膜增厚 (Intercostal pleural thickening)
    - 心臟肥大 (Cardiac hypertrophy)
  - **測試集：** 113 張影像。

## 3. 分析流程

### (1) 資料前處理

- 應用intensity log-transformation 跟 simplest color balance algorithm 的處理（圖二），目的是為了轉換影像型態及調整 L 和 R字樣的亮度，讓圖片更易於辨識。
- 將影像轉換為 JPG 格式，並標註異常區域。
- 建立對應的 JSON 檔案以存儲邊界框資訊。

### (2) 模型與架構

- **Model: Faster R-CNN**
- **Feature Extraction Backbone:** ResNet-50 結合 FPN 提升多層次特徵學習。
- **Optimizer：** AdamW，並採用 OneCycle Scheduler。

### (3) 模型評估指標

- **損失函數：**
  - Classifier loss
  - Bounding box regression loss
  - Bounding box regression loss about RPN
  - Bounding box object loss 

**預測指標：**
  - IoU (Intersection over Union)
  - mAP (Mean Average Precision)

## 4. 結果

### (1) 使用 Ablation CAM 分析模型

利用 **Ablation CAM** 分析模型在疾病檢測中的關注區域，結果顯示：

- 對於 **主動脈硬鈣化**、**主動脈彎曲** 和 **心臟肥大** 的關注範圍與醫學認知一致。
- 對於 **胸椎退化性關節病變**、**肺尖肋膜增厚** 和 **脊椎側彎** 等疾病的區分能力仍需提升。

### (2) 繪製 X 光影像與疾病位置的對應圖

將病患 X 光影像與疾病位置對應後發現，**在檢測脊椎側彎與肺浸潤等疾病，仍需要進一步改進。**

