# -*- coding: utf-8 -*-
"""
统一异常体系
============
* ``AppException``：业务异常基类，携带错误码与 HTTP 状态码，message 可直接展示给前端
* 各具体子类覆盖“文档数量超限 / 文件格式不支持 / 模型调用失败 / 状态冲突”等场景
* ``register_exception_handlers``：在 FastAPI 上注册统一处理器，
  保证前端永远收到结构一致的 JSON：``{"error": {"code", "message"}}``
"""
from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


class AppException(Exception):
    """所有可预期业务异常的基类。"""

    def __init__(self, message: str, code: str = "bad_request", status_code: int = 400):
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code


class ValidationError(AppException):
    """输入参数 / 文件校验失败。"""

    def __init__(self, message: str):
        super().__init__(message, code="validation_error", status_code=422)


class DocumentLimitError(AppException):
    """规则文档数量超过上限。"""

    def __init__(self, limit: int):
        super().__init__(
            f"规则文档数量已达上限（最多 {limit} 个），请先删除旧文档后再上传。",
            code="document_limit_exceeded",
            status_code=409,
        )


class UnsupportedFileError(AppException):
    """文件扩展名 / MIME 不在白名单内。"""

    def __init__(self, message: str = "不支持的文件格式。"):
        super().__init__(message, code="unsupported_file", status_code=415)


class FileTooLargeError(AppException):
    """文件大小超过限制。"""

    def __init__(self, message: str):
        super().__init__(message, code="file_too_large", status_code=413)


class NotFoundError(AppException):
    """资源不存在（文档 / 冲突 / 任务 / 历史记录）。"""

    def __init__(self, message: str = "资源不存在。"):
        super().__init__(message, code="not_found", status_code=404)


class ConflictStateError(AppException):
    """操作与当前状态冲突，例如：仍存在未解决冲突却发起检测。"""

    def __init__(self, message: str):
        super().__init__(message, code="conflict_state", status_code=409)


class ModelCallError(AppException):
    """大模型 / 分割模型调用失败（网络、鉴权、限流等）。"""

    def __init__(self, message: str):
        super().__init__(message, code="model_call_failed", status_code=502)


class ImageProcessError(AppException):
    """图像解码 / 处理失败（文件损坏、像素过高等）。"""

    def __init__(self, message: str):
        super().__init__(message, code="image_process_error", status_code=422)


def register_exception_handlers(app: FastAPI) -> None:
    """
    在 FastAPI 应用上注册统一异常处理：
    1. 业务异常 → 按其 status_code 返回结构化 JSON；
    2. 兜底异常 → 500，且不向前端泄露堆栈细节。
    """

    @app.exception_handler(AppException)
    async def _handle_app_exception(_: Request, exc: AppException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": exc.code, "message": exc.message}},
        )

    @app.exception_handler(Exception)
    async def _handle_unknown(_: Request, exc: Exception) -> JSONResponse:  # noqa: BLE001
        # 服务端打印完整异常链便于排障，前端只收到友好提示
        import traceback

        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "internal_error",
                    "message": "服务器内部错误，请稍后重试或联系管理员。",
                }
            },
        )
