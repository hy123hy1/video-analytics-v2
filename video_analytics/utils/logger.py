"""
日志系统配置
统一的日志管理，支持分级控制和结构化输出
"""
import logging
import sys
import os
from datetime import datetime
from typing import Optional


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    enable_console: bool = True,
    enable_file: bool = False,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    设置日志系统

    Args:
        level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: 日志文件路径
        enable_console: 是否输出到控制台
        enable_file: 是否输出到文件
        format_string: 自定义格式字符串

    Returns:
        Logger: 根日志记录器
    """
    # 转换日志级别
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # 默认格式
    if format_string is None:
        format_string = (
            "%(asctime)s - "
            "%(name)s - "
            "%(levelname)s - "
            "[%(filename)s:%(lineno)d] - "
            "%(message)s"
        )

    formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")

    # 获取根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # 清除现有处理器
    root_logger.handlers.clear()

    # 控制台处理器
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # 文件处理器
    if enable_file and log_file:
        # 确保日志目录存在
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    获取命名日志记录器

    Args:
        name: 模块名称，建议使用 __name__

    Returns:
        Logger: 配置好的日志记录器
    """
    return logging.getLogger(name)


class ContextAdapter(logging.LoggerAdapter):
    """
    带上下文的日志适配器
    自动添加 camera_id 等上下文信息
    """

    def process(self, msg, kwargs):
        # 添加上下文信息到日志消息
        context_str = " ".join(f"[{k}={v}]" for k, v in self.extra.items())
        if context_str:
            return f"{context_str} {msg}", kwargs
        return msg, kwargs


def get_context_logger(name: str, **context) -> logging.LoggerAdapter:
    """
    获取带上下文的日志记录器

    Example:
        logger = get_context_logger(__name__, camera_id="cam_001", algo="intrusion")
        logger.info("Event detected")  # 输出: [camera_id=cam_001] [algo=intrusion] Event detected
    """
    logger = logging.getLogger(name)
    return ContextAdapter(logger, context)


# 预定义的日志级别快捷方法
def debug(msg, *args, **kwargs):
    """输出 DEBUG 级别日志"""
    logging.debug(msg, *args, **kwargs)


def info(msg, *args, **kwargs):
    """输出 INFO 级别日志"""
    logging.info(msg, *args, **kwargs)


def warning(msg, *args, **kwargs):
    """输出 WARNING 级别日志"""
    logging.warning(msg, *args, **kwargs)


def error(msg, *args, **kwargs):
    """输出 ERROR 级别日志"""
    logging.error(msg, *args, **kwargs)


def exception(msg, *args, **kwargs):
    """输出 EXCEPTION 级别日志（自动包含堆栈）"""
    logging.exception(msg, *args, **kwargs)


def critical(msg, *args, **kwargs):
    """输出 CRITICAL 级别日志"""
    logging.critical(msg, *args, **kwargs)
