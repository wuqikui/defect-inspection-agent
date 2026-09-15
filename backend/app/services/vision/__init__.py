# -*- coding: utf-8 -*-
"""
视觉检测子包
============
* sliding_window  固定方形滑窗规划 / 切片 / 坐标复原
* nms             跨滑窗重复检测框的非极大值抑制
* cv_detector     离线经典 CV 缺陷检测（无视觉模型时的兜底）
* vlm_detector    多模态大模型滑窗检测
* segmenter       SAM 精确分割（不可用时 OpenCV 轮廓兜底）
* annotator       在原图上绘制框 / 掩膜 / 标签
* detector        检测器门面：编排上述组件
"""
