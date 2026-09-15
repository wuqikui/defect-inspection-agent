# -*- coding: utf-8 -*-
"""
视觉检测子包
============
* sliding_window  固定方形滑窗规划 / 切片 / 坐标复原
* nms             跨滑窗重复检测框的非极大值抑制
* cv_detector     离线经典 CV 缺陷检测（无视觉模型时的兜底）
* vlm_detector    多模态大模型候选框裁决
* segmenter       OpenCV ROI 轮廓提取（SAM 已弃用）
* annotator       在原图上绘制框 / 掩膜 / 标签
* detector        检测器门面：编排上述组件
"""
