# -*- coding: utf-8 -*-
"""
一次性模型导出脚本（运行时不需要本脚本与 torch，仅更新模型时重跑）
================================================================
把两个 CPU 友好的深度模型导出为 ONNX 并存入 backend/models/，
作为项目文件随仓库分发 —— 其它主机克隆后即可直接运行，无需下载：

1. resnet18_features.onnx —— ResNet18（ImageNet 预训练）特征提取器
   输出 layer2 (128ch, H/8) 与 layer3 (256ch, H/16) 两级特征图，
   供 PatchCore 哨兵做 patch 特征异常扫描。
   来源：torchvision resnet18（IMAGENET1K_V1，首次运行自动下载）。

2. yolov8s-worldv2.onnx —— YOLO-World-S 开放词汇零样本检测器
   导出时通过 set_classes 把 app/services/vision/yolo_labels.py
   中定义的缺陷类别文本嵌入【内联】进检测头，运行时无需 CLIP。
   来源：ultralytics yolov8s-worldv2.pt（首次运行自动下载）。

用法（一次性依赖 torch / torchvision / ultralytics）：
  cd backend
  .conda-env/python.exe scripts/prepare_models.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

MODELS_DIR = BACKEND_DIR / "models"


def export_resnet18_features() -> Path:
    """导出 ResNet18 双特征输出（layer2 + layer3）ONNX。"""
    import torch
    import torchvision

    backbone = torchvision.models.resnet18(
        weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1
    )
    # 裁剪到 layer3：PatchCore 只需要 H/8 与 H/16 两级 patch 特征
    stem = torch.nn.Sequential(
        backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool
    )

    class FeatureBackbone(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.stem = stem
            self.layer1 = backbone.layer1
            self.layer2 = backbone.layer2
            self.layer3 = backbone.layer3

        def forward(self, x: torch.Tensor):
            x = self.stem(x)
            x = self.layer1(x)
            f2 = self.layer2(x)               # (1,128,H/8,W/8)
            f3 = self.layer3(f2)              # (1,256,H/16,W/16)
            return f2, f3

    net = FeatureBackbone().eval()
    dummy = torch.randn(1, 3, 448, 448)
    out_path = MODELS_DIR / "resnet18_features.onnx"
    torch.onnx.export(
        net,
        dummy,
        str(out_path),
        input_names=["images"],
        output_names=["feat_layer2", "feat_layer3"],
        dynamic_axes={
            "images": {2: "h", 3: "w"},
            "feat_layer2": {2: "h", 3: "w"},
            "feat_layer3": {2: "h", 3: "w"},
        },
        opset_version=13,
        do_constant_folding=True,
        dynamo=False,  # 传统 TorchScript 导出器：权重内联单文件，opset 精确可控
    )
    return out_path


def export_yolo_world() -> Path:
    """导出内联了缺陷类别文本嵌入的 YOLO-World-S ONNX。

    注意：ultralytics 在 set_classes 后直接 export 会丢失文本嵌入
    （导出器内部重新加载权重），必须先 save() 定制模型再重载导出
    （官方 open-vocabulary 离线导出标准流程）。
    """
    from ultralytics import YOLOWorld

    from app.services.vision.yolo_labels import prompt_list

    model = YOLOWorld("yolov8s-worldv2.pt")   # 首次自动下载 ~13MB
    model.set_classes(prompt_list())          # 文本嵌入内联进检测头

    custom_pt = MODELS_DIR / "_yolov8s-world-custom.pt"
    model.save(str(custom_pt))                # 固化定制嵌入
    exporter = YOLOWorld(str(custom_pt))      # 重载：确保嵌入随权重加载
    exported = exporter.export(
        format="onnx",
        imgsz=640,
        opset=13,
        simplify=True,
        dynamic=False,
        half=False,
        device="cpu",
    )
    custom_pt.unlink(missing_ok=True)
    out_path = MODELS_DIR / "yolov8s-worldv2.onnx"
    src = Path(exported)
    if src.resolve() != out_path.resolve():
        shutil.move(str(src), out_path)
    return out_path


def main() -> int:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    p1 = MODELS_DIR / "resnet18_features.onnx"
    p2 = MODELS_DIR / "yolov8s-worldv2.onnx"
    if p1.exists() and p1.stat().st_size > 10_000_000:
        print(f"[1/2] 已存在，跳过：{p1}")
    else:
        print("[1/2] 导出 ResNet18 特征提取器 …")
        p1 = export_resnet18_features()
        print(f"      完成：{p1}  ({p1.stat().st_size / 1e6:.1f} MB)")
    if p2.exists() and p2.stat().st_size > 10_000_000:
        print(f"[2/2] 已存在，跳过：{p2}")
    else:
        print("[2/2] 导出 YOLO-World-S（内联缺陷标签）…")
        p2 = export_yolo_world()
        print(f"      完成：{p2}  ({p2.stat().st_size / 1e6:.1f} MB)")
    print("\n全部模型已就绪，位于 backend/models/。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
