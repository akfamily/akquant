import logging
import sys
from typing import Optional, Union

# Default format: Time | Level | Message
DEFAULT_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class Logger:
    r"""
    akquant 日志封装.

    :description: 提供控制台与文件日志的快捷配置
    """

    _instance = None

    def __init__(self) -> None:
        """Initialize the Logger."""
        self._logger = logging.getLogger("akquant")
        self._logger.setLevel(logging.INFO)
        self._handlers: dict[str, logging.Handler] = {}  # key -> handler

        # Add default console handler if not present
        if not self._logger.handlers:
            self.enable_console()

    @classmethod
    def get_logger(cls) -> logging.Logger:
        """Get the singleton logger instance."""
        if cls._instance is None:
            cls._instance = Logger()
        return cls._instance._logger

    def set_level(self, level: Union[str, int]) -> None:
        r"""
        设置日志等级.

        :param level: 日志等级字符串或整数 (DEBUG/INFO/WARNING/ERROR/CRITICAL)
        :type level: str | int
        """
        self._logger.setLevel(level)

    def enable_console(self, format_str: str = DEFAULT_FORMAT) -> None:
        r"""
        启用控制台日志.

        :param format_str: 日志格式字符串
        :type format_str: str
        """
        if "console" in self._handlers:
            return

        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(format_str, datefmt=DATE_FORMAT))
        self._logger.addHandler(handler)
        self._handlers["console"] = handler

    def disable_console(self) -> None:
        r"""禁用控制台日志."""
        if "console" in self._handlers:
            self._logger.removeHandler(self._handlers["console"])
            del self._handlers["console"]

    def enable_file(
        self, filename: str, format_str: str = DEFAULT_FORMAT, mode: str = "a"
    ) -> None:
        r"""
        启用文件日志.

        :param filename: 日志文件路径
        :type filename: str
        :param format_str: 日志格式字符串
        :type format_str: str
        :param mode: 文件打开模式 ('a' 追加 或 'w' 覆写)
        :type mode: str
        """
        # Remove existing file handler if path matches (simple check)
        key = f"file_{filename}"
        if key in self._handlers:
            return

        handler = logging.FileHandler(filename, mode=mode, encoding="utf-8")
        handler.setFormatter(logging.Formatter(format_str, datefmt=DATE_FORMAT))
        self._logger.addHandler(handler)
        self._handlers[key] = handler


# Global helper functions
def get_logger() -> logging.Logger:
    r"""
    获取全局 logger 实例.

    :return: 已初始化的 logger
    :rtype: logging.Logger
    """
    return Logger.get_logger()


def set_log_level(level: Union[str, int]) -> None:
    r"""
    设置全局日志等级.

    :param level: 日志等级字符串或整数
    :type level: str | int
    """
    Logger.get_logger().setLevel(level)


def register_logger(
    filename: Optional[str] = None, console: bool = True, level: str = "INFO"
) -> None:
    r"""
    日志一体化配置.

    :param filename: 日志文件路径，提供则写入文件
    :type filename: str, optional
    :param console: 是否输出到控制台
    :type console: bool
    :param level: 日志等级 ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
    :type level: str
    """
    logger_manager = Logger._instance or Logger()
    Logger._instance = logger_manager

    logger_manager.set_level(level.upper())

    if console:
        logger_manager.enable_console()
    else:
        logger_manager.disable_console()

    if filename:
        logger_manager.enable_file(filename)
