"""
第 3 章：金融数据获取与处理.

本示例演示了量化交易中数据工程的核心步骤：
1. 获取数据：从 AKShare 获取 A 股历史日线数据
2. 数据清洗：将原始数据转换为 AKQuant 所需的标准格式
3. 数据存储：将清洗后的数据保存为高性能的 Parquet 格式
4. 数据读取：从本地加载数据进行验证

这些步骤是构建任何量化策略的基石。
"""

import os
from pathlib import Path

import akshare as ak
import numpy as np
import pandas as pd


def generate_mock_raw_data(length: int = 240) -> pd.DataFrame:
    """断网兜底：生成与 AKShare 新浪日线接口同列名的原始日线数据.

    保持与 ``stock_zh_a_daily`` 相同的列结构（含 ``amount`` 等多余字段），
    使后续筛选与清洗流程在离线时也能完整演示。
    """
    np.random.seed(7)
    dates = pd.date_range(start="2023-01-03", periods=length, freq="B")
    close = 7.0 + np.cumsum(np.random.randn(length) * 0.06)
    open_ = close + np.random.randn(length) * 0.03
    high = np.maximum(open_, close) + np.abs(np.random.randn(length) * 0.05)
    low = np.minimum(open_, close) - np.abs(np.random.randn(length) * 0.05)
    volume = np.random.uniform(1e8, 5e8, length)
    return pd.DataFrame(
        {
            "date": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "amount": volume * close,
        }
    )


def to_sina_symbol(symbol: str) -> str:
    """把 6 位 A 股代码转换为新浪接口需要的带市场前缀代码.

    :param symbol: 6 位代码 (如 "600000")
    :return: 带前缀代码 (如 "sh600000")
    """
    if symbol.startswith(("6", "9")):
        return f"sh{symbol}"
    if symbol.startswith(("4", "8")):
        return f"bj{symbol}"
    return f"sz{symbol}"


def fetch_and_clean_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    获取 A 股日线数据并清洗为 AKQuant 标准格式.

    :param symbol: 股票代码 (如 "600000")
    :param start_date: 开始日期 (如 "20230101")
    :param end_date: 结束日期 (如 "20231231")
    :return: 清洗后的 DataFrame
    """
    print(f"正在获取 {symbol} 的历史数据 ({start_date}-{end_date})...")

    # 1. 获取数据
    # 使用新浪数据源的 stock_zh_a_daily 接口获取历史行情（东方财富接口
    # stock_zh_a_hist 近期常被断开连接，这里统一用新浪源）
    # 新浪接口的代码需要带市场前缀，如 "sh600000"
    # adjust="qfq" 表示前复权，这是回测中最常用的复权方式
    try:
        df = ak.stock_zh_a_daily(
            symbol=to_sina_symbol(symbol),
            start_date=start_date,
            end_date=end_date,
            adjust="qfq",
        )
    except Exception as e:  # noqa: BLE001 - 示例容错：网络/依赖异常一律回退
        print(f"AKShare 获取失败（{e}），改用本地合成数据。")
        df = generate_mock_raw_data()

    if df.empty:
        print("警告: 获取到的数据为空，改用本地合成数据。")
        df = generate_mock_raw_data()

    # 2. 字段映射
    # 标准列名: date, open, high, low, close, volume
    # 新浪接口已返回英文列名，无需重命名；若换用返回中文列名的数据源
    # （如东方财富 stock_zh_a_hist），在这里用 rename 映射即可，例如
    # df.rename(columns={"日期": "date", "开盘": "open", "收盘": "close"})
    # 注意：新浪接口的 volume 单位是“股”，东方财富接口是“手”（100 股）

    # 3. 格式转换与清洗
    # 转换日期列为 pandas datetime 类型
    df["date"] = pd.to_datetime(df["date"])

    # 筛选必要的列，去除多余字段 (如成交额、换手率等，除非策略需要)
    required_cols = ["date", "open", "high", "low", "close", "volume"]
    df = df[required_cols]

    # 添加 symbol 列 (多标的回测时必需)
    df["symbol"] = symbol

    # 处理缺失值 (简单的丢弃策略)
    df = df.dropna()

    # 按日期升序排序
    df = df.sort_values("date").reset_index(drop=True)

    print(f"数据清洗完成，共 {len(df)} 条记录")
    return df  # type: ignore


def save_to_parquet(df: pd.DataFrame, file_path: str) -> None:
    """
    将 DataFrame 保存为 Parquet 格式.

    Parquet 是一种高性能列式存储格式，读写速度远快于 CSV。
    """
    # 确保父目录存在
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(file_path)
    print(f"数据已保存至: {file_path}")


def load_from_parquet(file_path: str) -> pd.DataFrame:
    """从 Parquet 文件读取数据."""
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return pd.DataFrame()

    df = pd.read_parquet(file_path)
    print(f"从本地加载数据成功，共 {len(df)} 条记录")
    return df


if __name__ == "__main__":
    # 配置参数
    SYMBOL = "600000"  # 浦发银行
    START_DATE = "20230101"
    END_DATE = "20231231"
    DATA_DIR = "data"  # 数据存储目录

    # 1. 获取并清洗数据
    df_clean = fetch_and_clean_data(SYMBOL, START_DATE, END_DATE)

    if not df_clean.empty:
        # 打印前 5 行预览
        print("\n数据预览 (Head):")
        print(df_clean.head())

        # 打印数据类型信息
        print("\n数据信息 (Info):")
        df_clean.info()

        # 2. 保存数据
        file_path = f"{DATA_DIR}/{SYMBOL}.parquet"
        save_to_parquet(df_clean, file_path)

        # 3. 验证读取
        print("\n正在验证读取...")
        df_loaded = load_from_parquet(file_path)

        # 简单验证
        assert len(df_clean) == len(df_loaded)
        print("验证通过：读取的数据与原始数据一致")
