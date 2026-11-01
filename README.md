# Deepseek融合yolo的行人车辆多目标检测系统  |行人检测 车辆检测 多目标检测 YOLO DeepSeek9165期

## 系统概述

本系统基于 **YOLO 深度学习模型**，融合 **DeepSeek、Qwen 大语言模型**与 **Pytorch、SpringBoot、Flask、Vue3** 等技术，构建了一套支持**图片、视频、摄像头**三种检测方式的行人车辆多目标检测平台。系统支持 AI 智能分析，可导出 PDF 检测报告，适配智慧交通、安防监控及城市管理场景。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/bec44f5b5b474680ae873d17f699d502.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/c9baaabb38d940e6b1f8803741396cdd.png)

## 系统核心信息

| 项目 | 详细说明 |
| :--- | :--- |
| **检测类别** | 共 **2** 类：`行人`、`车辆` |
| **核心技术** | YOLO + DeepSeek/Qwen + Pytorch + SpringBoot + Flask + Vue3 |
| **检测模式** | 图片检测、视频检测、摄像头实时检测 |
| **系统架构** | 前后端分离（Vue3 + SpringBoot + Flask + YOLO） |
| **应用价值** | 1. **智慧交通**：支撑路面行人车辆流量统计与违规行为分析。<br>2. **安防监控**：实时监测重点区域的人车动态，辅助安全预警。<br>3. **AI 分析**：结合大语言模型生成检测报告与治理建议。<br>4. **算法研发**：为多目标检测、多模态融合等视觉任务提供完整系统框架。 |


![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/4a2124d681b04e56b7792165466d9e64.png)

## 技术栈

| 层次 | 技术选型 |
| :--- | :--- |
| **深度学习** | YOLO、Pytorch |
| **算法服务** | Flask |
| **前端框架** | Vue3 |
| **后端框架** | SpringBoot |
| **AI 分析** | DeepSeek、Qwen |


![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/3250f75911f84fec8bd838de76ce3216.png)

## 系统功能

**1. 多模态检测支持**
- **图片检测**：支持单张与批量图片上传检测
- **视频检测**：上传视频文件，逐帧分析并输出标注视频
- **摄像头实时检测**：连接 USB 或网络摄像头，实时输出检测结果

**2. AI 智能分析**
- 结合 DeepSeek 和 Qwen 进行智能分析
- 提供详细检测建议与治理方案
- 支持导出检测结果为 PDF

**3. 检测报告**
- 实时检测结果展示
- 详细分析报告生成
- 目标位置、类别、置信度等信息展示

**4. 用户交互**
- 界面友好，操作直观
- 支持批量检测和单张检测
- 无需命令行操作
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/c3f8a72b543d45c28d7429a4412682f9.png)

## 交付内容

1. **详细系统代码**：前端、后端、YOLO 算法、AI 分析模块
2. **本地部署教程**：环境配置与运行指导
3. **实时视频、摄像头、图片检测**：支持三种输入方式
4. **代码简洁，有注释**：便于理解和二次开发

## 快速开始（代码示例）

**1. 数据集配置文件 (`data.yaml`)**
```yaml
train: ./dataset/images/train
val: ./dataset/images/val
nc: 2
names: ['person', 'vehicle']
```

**2. YOLO 训练代码**
```python
from ultralytics import YOLO

model = YOLO('yolov8m.pt')
model.train(
    data='data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    device=0
)
```

**3. Flask 推理与 AI 分析服务**
```python
from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import requests

app = Flask(__name__)
model = YOLO('runs/detect/train/weights/best.pt')

@app.route('/detect', methods=['POST'])
def detect():
    file = request.files['image']
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    results = model(img)
    detections = []
    for box in results[0].boxes:
        detections.append({
            'class': model.names[int(box.cls[0])],
            'confidence': float(box.conf[0]),
            'bbox': box.xyxy[0].tolist()
        })
    # 调用 DeepSeek/Qwen 生成分析建议
    ai_suggestion = analyze_with_llm(detections)
    return jsonify({'detections': detections, 'suggestion': ai_suggestion})
```

**4. Vue3 前端检测页面**
```vue
<template>
  <div class="detector-container">
    <el-upload action="/api/detection/image" :on-success="handleSuccess">
      <el-button type="primary">上传图片</el-button>
    </el-upload>
    <div v-if="resultImage" class="result-area">
      <img :src="resultImage" alt="Detection Result" />
      <el-table :data="detections" border>
        <el-table-column prop="class" label="类别" />
        <el-table-column prop="confidence" label="置信度" />
      </el-table>
      <el-button @click="exportPDF">导出 PDF 报告</el-button>
    </div>
  </div>
</template>
```

## 标签
`行人检测` `车辆检测` `多目标检测` `YOLO` `DeepSeek` `智慧交通` `目标检测` `深度学习` `计算机视觉`
