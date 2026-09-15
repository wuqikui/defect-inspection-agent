# -*- coding: utf-8 -*-
"""
PatchCore 哨兵：全图异常扫描（CPU 友好）
========================================
角色：检测管线第一级"哨兵"，在 CPU 上对整图快速算出 patch 级异常热力图，
过滤 100% 正常区域，仅输出【高危异常区域框】供后续动态切片与定位。

实现要点（PatchCore 思想的轻量自参照版，纯 numpy + onnxruntime）：

1. 分条带：长边 > band 的超长图切成 ≤band 的方形条带，每条带独立
   resize 到 sentinel_input×sentinel_input（长图因此不被过度压缩）；
2. 特征：ResNet18 ONNX 输出 layer2(128ch,H/8) + layer3(256ch,H/16)，
   layer3 最近邻上采样后拼接成 384 维 patch 特征（每 patch 约对应
   原图 8×sentinel 缩放比的一片区域）；
3. 记忆库：取【产品区域内】全部 patch 作为候选，按 coreset_sampling_ratio
   （默认 0.01，控内存）随机下采样，再剔除距离库中心最远的少量离群
   patch（防止缺陷特征混入"正常"库，自参照方案的关键一步）；
4. 打分：每个 patch 与记忆库的最大余弦相似度 → 异常分 = 1 - sim；
5. 出框：异常图高斯平滑 → 自适应阈值（μ+Nσ 与 P99 取大，绝对下限兜底）
   → 连通域 → 映射回原图坐标，按分数取 top-K 高危区域。

模型文件：backend/models/resnet18_features.onnx（由 scripts/prepare_models.py
导出，随项目分发；缺失时上层自动退回旧版盲滑窗管线）。
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import onnxruntime as ort

from app.config import settings

# ImageNet 归一化常数
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)


@dataclass
class SentinelResult:
    """哨兵扫描结果。"""

    regions: List[Dict[str, Any]] = field(default_factory=list)
    # 调试/自测信息
    stats: Dict[str, Any] = field(default_factory=dict)


class PatchCoreSentinel:
    """PatchCore 哨兵（线程安全的惰性单例，见 get_sentinel）。"""

    def __init__(self, model_path: Path):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = max(1, (os.cpu_count() or 4) // 2)
        opts.log_severity_level = 3
        self._session = ort.InferenceSession(
            str(model_path), opts, providers=["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name
        self._out_names = [o.name for o in self._session.get_outputs()]
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # 对外主入口
    # ------------------------------------------------------------------
    def scan(self, image_rgb: np.ndarray) -> SentinelResult:
        """
        对整图执行异常扫描。

        :param image_rgb: 原图 RGB uint8 ndarray
        :return: SentinelResult（regions 为原图坐标高危框，按分数降序）
        """
        t0 = time.perf_counter()
        H, W = image_rgb.shape[:2]
        size = settings.sentinel_input_size

        patches: List[np.ndarray] = []       # (384,) 已 L2 归一化
        centers: List[List[float]] = []      # patch 中心原图坐标 [cx, cy]
        band_count = 0

        for (x0, y0, x1, y1) in self._plan_bands(W, H):
            band_count += 1
            p_list, c_list = self._scan_band(
                image_rgb, x0, y0, x1, y1, size, W, H
            )
            if p_list is not None:
                patches.append(p_list)
                centers.append(c_list)

        stats: Dict[str, Any] = {
            "elapsed_s": round(time.perf_counter() - t0, 3),
            "bands": band_count,
            "patches": 0,
            "bank": 0,
            "threshold": 0.0,
        }
        if not patches:
            return SentinelResult([], stats)

        feats = np.concatenate(patches, axis=0)          # (N,384)
        cents = np.concatenate(centers, axis=0)          # (N,2)
        stats["patches"] = int(feats.shape[0])

        bank = self._build_bank(feats)
        stats["bank"] = int(bank.shape[0])
        if bank.shape[0] < 8:
            # 库过小说明图内几乎没有有效产品区域
            return SentinelResult([], stats)

        # k=1 最近邻余弦距离 = 异常分
        with self._lock:
            sim = feats @ bank.T                         # (N,M)
        dist = 1.0 - sim.max(axis=1)
        # 遥测：距离分布中位数（裁判以"区域分 − 中位数"的幅度判真伪）
        stats["dist_median"] = round(float(np.median(dist)), 4)

        regions = self._extract_regions(
            dist, cents, W, H, stats
        )
        stats["elapsed_s"] = round(time.perf_counter() - t0, 3)
        return SentinelResult(regions, stats)

    # ------------------------------------------------------------------
    # 内部步骤
    # ------------------------------------------------------------------
    @staticmethod
    def _plan_bands(W: int, H: int) -> List[List[int]]:
        """把图切成 ≤band 的矩形条带（返回 [x0,y0,x1,y1] 原图坐标）。"""
        band = settings.sentinel_band_size
        if max(W, H) <= band:
            return [[0, 0, W, H]]
        bands: List[List[int]] = []
        if H >= W:  # 竖切条带
            n = int(np.ceil(H / band))
            step = H / n
            for i in range(n):
                bands.append([0, int(i * step), W, min(H, int((i + 1) * step))])
        else:       # 横切条带
            n = int(np.ceil(W / band))
            step = W / n
            for i in range(n):
                bands.append([int(i * step), 0, min(W, int((i + 1) * step)), H])
        return bands

    def _scan_band(
        self, image_rgb: np.ndarray,
        x0: int, y0: int, x1: int, y1: int, size: int, W: int, H: int,
    ):
        """单条带：前向取特征 + 收集产品区域 patch。返回 (feats, centers)。"""
        band = image_rgb[y0:y1, x0:x1]
        bh, bw = band.shape[:2]
        resized = cv2.resize(band, (size, size), interpolation=cv2.INTER_AREA)

        f2, f3 = self._forward(resized)              # (128,s/8,s/8) (256,s/16,s/16)
        gh = f2.shape[1]
        f3_up = np.repeat(np.repeat(f3, 2, axis=1), 2, axis=2)
        feat = np.concatenate([f2, f3_up], axis=0)   # (384,gh,gh)
        c, gh2, gw2 = feat.shape
        flat = feat.reshape(c, -1).T                 # (gh*gw,384)
        norm = np.linalg.norm(flat, axis=1, keepdims=True) + 1e-6
        flat /= norm

        # patch 中心 → 原图坐标
        sx, sy = bw / gw2, bh / gh2
        jj, ii = np.meshgrid(np.arange(gw2), np.arange(gh2))
        cx = x0 + (jj.ravel() + 0.5) * sx
        cy = y0 + (ii.ravel() + 0.5) * sy

        # 产品区域过滤：patch 中心灰度需高于背景地板（排除纯黑边等）
        gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
        small = cv2.resize(gray, (gw2, gh2), interpolation=cv2.INTER_AREA)
        valid = small.ravel() > settings.sentinel_bg_gray
        # 图像边缘剔除：卷积零填充使贴边 patch 的特征系统性偏高，
        # 是干净图边缘误报的主要来源（中心不在边缘缓冲带内才保留）
        m = settings.sentinel_border_px
        valid &= (cx > m) & (cx < W - m) & (cy > m) & (cy < H - m)
        if not valid.any():
            return None, None
        return flat[valid], np.stack([cx[valid], cy[valid]], axis=1)

    def _forward(self, rgb: np.ndarray):
        """ONNX 前向：RGB uint8 → (layer2, layer3) float32 特征图。"""
        x = rgb.astype(np.float32) / 255.0
        x = (x - _MEAN) / _STD
        x = x.transpose(2, 0, 1)[None]               # (1,3,s,s)
        with self._lock:
            f2, f3 = self._session.run(self._out_names, {self._input_name: x})
        return f2[0], f3[0]

    @staticmethod
    def _build_bank(feats: np.ndarray) -> np.ndarray:
        """
        构建自参照记忆库（两遍式，去除缺陷特征污染）：

        1. 粗库：coreset 随机下采样（比例来自配置，默认 0.01，下限 64 条）；
        2. 正常核：以粗库计算全图 patch 的最近邻距离，取距离最小的前
           bank_quantile（默认 50%）patch 作为"正常核心"——
           缺陷 patch 距离大、天然被排除在核心之外；
        3. 终库：在正常核心上再做一次 coreset 下采样 + 剔除距库中心
           最远的 5% 离群 patch。
        """
        n = feats.shape[0]

        def _coreset(pool: np.ndarray) -> np.ndarray:
            m = max(64, int(pool.shape[0] * settings.sentinel_coreset_ratio))
            m = min(m, settings.sentinel_max_bank, pool.shape[0])
            rng = np.random.default_rng(20260915)
            idx = rng.choice(pool.shape[0], size=m, replace=False)
            bank = pool[idx]
            # 离群清理：剔除与库中心（中位数向量）距离最远的 5%
            center = np.median(bank, axis=0, keepdims=True)
            center /= np.linalg.norm(center) + 1e-6
            d = 1.0 - bank @ center.T
            keep = max(8, int(bank.shape[0] * 0.95))
            order = np.argsort(d.ravel())[:keep]
            return np.ascontiguousarray(bank[order])

        bank0 = _coreset(feats)
        if bank0.shape[0] < 8:
            return bank0
        # 正常核心：与粗库最近的 50% patch
        d_all = 1.0 - feats @ bank0.T
        d_all = d_all.max(axis=1)
        cutoff = np.quantile(d_all, settings.sentinel_bank_quantile)
        core = feats[d_all <= cutoff]
        if core.shape[0] < 64:
            core = feats
        return _coreset(core)

    def _extract_regions(
        self, dist: np.ndarray, cents: np.ndarray,
        W: int, H: int, stats: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """异常分 → 自适应阈值 → 连通域 → 原图坐标高危框（top-K）。"""
        thr = max(
            float(np.percentile(dist, settings.sentinel_threshold_percentile)),
            settings.sentinel_abs_floor,
        )
        stats["threshold"] = round(float(thr), 4)

        hot = dist >= thr
        if not hot.any():
            return []
        # patch 分数栅格化：粒度 ≈ 单个 patch 对应的原图像素数
        patch_px = max(8.0, float(settings.sentinel_patch_px))
        gw = max(1, int(np.ceil(W / patch_px)))
        gh = max(1, int(np.ceil(H / patch_px)))
        heat = np.zeros((gh, gw), np.float32)
        gi = np.clip((cents[:, 1] / patch_px).astype(int), 0, gh - 1)
        gj = np.clip((cents[:, 0] / patch_px).astype(int), 0, gw - 1)
        np.maximum.at(heat, (gi, gj), dist)

        # 掩膜直接在【未模糊】热图上阈值化：小缺陷往往只占 1~2 个栅格，
        # 若先做高斯模糊再阈值化，孤岛热点会被邻域稀释到阈值以下而漏检；
        # 膨胀仅 1 次（±1 栅格）：合并相邻热点并适度外扩，又不至于把
        # 小图上分散的热点桥接成触发 max_frac 上限的巨型区域。
        mask = (heat >= thr).astype(np.uint8) * 255
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)

        n, _, stat_arr, _ = cv2.connectedComponentsWithStats(mask, 8)
        min_side = settings.sentinel_min_region_px
        max_frac = settings.sentinel_max_region_frac
        regions: List[Dict[str, Any]] = []
        for i in range(1, n):
            rx, ry, rw, rh, area = stat_arr[i]
            # 栅格坐标 → 原图坐标
            bx0, by0 = rx * patch_px, ry * patch_px
            bw, bh = rw * patch_px, rh * patch_px
            if min(bw, bh) < min_side:
                continue
            if bw * bh > W * H * max_frac:
                continue  # 过大区域多为整体光照异常，非局部缺陷
            # 打分用原热图（未模糊）区域内峰值
            cell = heat[ry : ry + rh, rx : rx + rw]
            score = float(cell.max())
            regions.append(
                {
                    "bbox": [
                        int(max(0, bx0)),
                        int(max(0, by0)),
                        int(min(W - max(0, bx0), bw)),
                        int(min(H - max(0, by0), bh)),
                    ],
                    "score": round(score, 4),
                }
            )
        regions.sort(key=lambda r: r["score"], reverse=True)
        return regions[: settings.sentinel_max_regions]


# ----------------------------------------------------------------------
# 惰性单例
# ----------------------------------------------------------------------
_sentinel: Optional[PatchCoreSentinel] = None
_sentinel_lock = threading.Lock()


def sentinel_model_path() -> Path:
    return settings.models_dir / "resnet18_features.onnx"


def get_sentinel() -> Optional[PatchCoreSentinel]:
    """模型文件存在则返回哨兵单例，否则 None（上层退回旧管线）。"""
    global _sentinel
    if _sentinel is not None:
        return _sentinel
    path = sentinel_model_path()
    if not path.exists():
        return None
    with _sentinel_lock:
        if _sentinel is None:
            try:
                _sentinel = PatchCoreSentinel(path)
            except Exception:  # noqa: BLE001
                return None
    return _sentinel
