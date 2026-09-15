# -*- coding: utf-8 -*-
"""
多模态缺陷检测智能体 —— 后端应用包
=================================
模块划分：
    config      全局配置（环境变量 / .env）
    core        异常体系、文件安全校验等横切能力
    db          SQLite 持久化（文档元数据 / 规则 / 冲突 / 检测历史）
    schemas     Pydantic 请求/响应数据模型
    services    业务服务（文档解析、RAG、LLM、冲突、视觉检测……）
    api         FastAPI 路由层
"""
__version__ = "1.0.0"
