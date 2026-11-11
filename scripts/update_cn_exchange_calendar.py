"""重新生成内置的中国交易所休市日表 ``src/calendar/cn_exchange_closed.txt``.

数据取自 AKShare ``tool_trade_date_hist_sina()``(上交所交易日历)。文件只存
"覆盖范围内休市的工作日": 周末从不开市, 其余工作日都是交易日。

每年年底交易所公布下一年休市安排后运行一次, 然后 ``uvx maturin develop``
重新编译并跑 ``tests/test_builtin_calendar.py``::

    uv run --no-sync python scripts/update_cn_exchange_calendar.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

TARGET = (
    Path(__file__).resolve().parents[1] / "src" / "calendar" / "cn_exchange_closed.txt"
)

HEADER = """\
# 中国交易所(沪深及期货交易所)休市的工作日。
# 由 scripts/update_cn_exchange_calendar.py 生成。
# 覆盖范围内: 周末不开市; 周一至周五中, 本表列出的日期休市, 其余都是交易日。
# 覆盖范围之外按"周一至周五"推算。请勿手工编辑。
"""


def main() -> int:
    """抓取交易日历并写出休市日表."""
    import akshare as ak

    trade = pd.DatetimeIndex(
        pd.to_datetime(ak.tool_trade_date_hist_sina()["trade_date"])
    )
    trade = trade.sort_values().unique()
    if len(trade) < 8000:
        print(f"交易日数量异常: {len(trade)}", file=sys.stderr)
        return 1
    weekend = trade[trade.weekday >= 5]
    if len(weekend):
        # 文件格式假定周末从不开市; 出现周末交易日时需要先扩展格式
        print(f"出现周末交易日, 当前格式无法表示: {list(weekend[:5])}", file=sys.stderr)
        return 1
    first, last = trade[0], trade[-1]
    closed = pd.bdate_range(first, last).difference(trade)
    lines = [HEADER.rstrip("\n"), f"coverage {first.date()} {last.date()}"]
    lines.extend(str(day.date()) for day in closed)
    TARGET.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    span = f"{first.date()} ~ {last.date()}"
    print(f"{TARGET.name}: 覆盖 {span}, 休市工作日 {len(closed)} 个")
    return 0


if __name__ == "__main__":
    sys.exit(main())
