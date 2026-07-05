"""成本/手数配置收敛: _cost_config 单一真源, 费率只读, lot_size 可写, pickle 保真."""

import pickle

import pytest
from akquant.strategy import Strategy


def _bare() -> Strategy:
    return Strategy.__new__(Strategy)


def test_cost_config_defaults_present() -> None:
    """构造即带 _cost_config 默认(percent 0, lot 1)."""
    s = _bare()
    assert s._cost_config["commission_policy"] == {"type": "percent", "value": 0.0}
    assert s._cost_config["lot_size"] == 1
    assert s.commission_rate == 0.0
    assert s.min_commission == 0.0
    assert s.stamp_tax_rate == 0.0
    assert s.transfer_fee_rate == 0.0


def test_fee_properties_are_read_only() -> None:
    """费率 5 项写入报错, 信息指向 run_backtest."""
    s = _bare()
    for attr in (
        "commission_rate",
        "commission_policy",
        "min_commission",
        "stamp_tax_rate",
        "transfer_fee_rate",
    ):
        with pytest.raises(AttributeError, match="run_backtest"):
            setattr(s, attr, 0.001)


def test_lot_size_is_writable() -> None:
    """lot_size 可写, 写入 _cost_config; 支持 int 与 dict."""
    s = _bare()
    s.lot_size = 1000
    assert s.lot_size == 1000
    assert s._cost_config["lot_size"] == 1000
    s.lot_size = {"600000.SH": 100, "DEFAULT": 1}
    assert s.lot_size["600000.SH"] == 100


def test_commission_rate_derives_from_policy() -> None:
    """commission_rate 从 policy 派生: percent 取 value, 否则 0.0(与引擎一致)."""
    s = _bare()
    s._inject_cost_config(commission_policy={"type": "percent", "value": 0.0003})
    assert s.commission_rate == 0.0003
    s._inject_cost_config(commission_policy={"type": "per_unit", "value": 3.0})
    assert s.commission_rate == 0.0  # 非 percent -> 0.0
    assert s.commission_policy == {"type": "per_unit", "value": 3.0}


def test_inject_cost_config_none_does_not_override() -> None:
    """_inject_cost_config(None) 不覆盖既有值(维持 lot_size None 守卫语义)."""
    s = _bare()
    s.lot_size = 100
    s._inject_cost_config(min_commission=5.0, lot_size=None)
    assert s.lot_size == 100  # 未被 None 覆盖
    assert s.min_commission == 5.0


def test_pickle_roundtrip_preserves_cost_config() -> None:
    """_cost_config 入 pickle, 反序列化保真; 费率仍只读、lot 仍可写."""
    s = _bare()
    s._inject_cost_config(
        commission_policy={"type": "percent", "value": 0.0005}, min_commission=5.0
    )
    s.lot_size = 100
    s2 = pickle.loads(pickle.dumps(s))
    assert s2.commission_rate == 0.0005
    assert s2.min_commission == 5.0
    assert s2.lot_size == 100
    with pytest.raises(AttributeError):
        s2.commission_rate = 0.1
    s2.lot_size = 200
    assert s2.lot_size == 200


def test_setstate_migrates_old_flat_snapshot() -> None:
    """旧 snapshot(无 _cost_config, 费率/lot 为裸键)经 __setstate__ 迁移."""
    s = _bare()
    # 模拟旧版实例的 __dict__: 扁平费率/lot 键, 无 _cost_config
    old_state = {
        "commission_rate": 0.0003,  # 旧派生量, 迁移后应被清掉
        "commission_policy": {"type": "percent", "value": 0.0003},
        "min_commission": 5.0,
        "stamp_tax_rate": 0.001,
        "transfer_fee_rate": 0.00002,
        "lot_size": 100,
    }
    s.__setstate__(dict(old_state))
    assert s._cost_config["commission_policy"] == {"type": "percent", "value": 0.0003}
    assert s.commission_rate == 0.0003  # 从 policy 派生
    assert s.min_commission == 5.0
    assert s.stamp_tax_rate == 0.001
    assert s.transfer_fee_rate == 0.00002
    assert s.lot_size == 100
    # 扁平死键已从实例 __dict__ 清掉
    assert "commission_rate" not in s.__dict__
    assert "lot_size" not in s.__dict__
    # 费率仍只读、lot 仍可写
    with pytest.raises(AttributeError):
        s.commission_rate = 0.1
    s.lot_size = 200
    assert s.lot_size == 200
