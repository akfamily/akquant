import pandas as pd

ts = 1675638000000000000
print(f"TS: {ts}")
print(f"UTC: {pd.to_datetime(ts, unit='ns', utc=True)}")
print(f"BJ: {pd.to_datetime(ts, unit='ns', utc=True).tz_convert('Asia/Shanghai')}")
