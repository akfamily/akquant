use super::result::BacktestResult;
use super::tracker::TradeTracker;
use super::types::{ClosedTrade, TradePnL};
use crate::model::{OrderSide, PositionEffect, Trade};
use chrono::{TimeZone, Utc};
use chrono_tz::Tz;
use rust_decimal::Decimal;
use std::collections::HashMap;

fn create_trade(
    symbol: &str,
    side: OrderSide,
    quantity: Decimal,
    price: Decimal,
    commission: Decimal,
) -> Trade {
    Trade {
        id: "trade_id".to_string(),
        order_id: "order_id".to_string(),
        symbol: symbol.to_string(),
        side,
        quantity,
        price,
        commission,
        timestamp: 0,
        bar_index: 0,
        position_effect: PositionEffect::Auto,
        owner_strategy_id: None,
    }
}

fn analyze_trades(
    trades: Vec<Trade>,
    current_prices: Option<HashMap<String, Decimal>>,
) -> (TradePnL, Vec<ClosedTrade>) {
    analyze_trades_with_multiplier(trades, current_prices, Decimal::ONE)
}

fn analyze_trades_with_multiplier(
    trades: Vec<Trade>,
    current_prices: Option<HashMap<String, Decimal>>,
    multiplier: Decimal,
) -> (TradePnL, Vec<ClosedTrade>) {
    let mut tracker = TradeTracker::new();
    for trade in trades {
        tracker.process_trade(&trade, multiplier, None, None, Decimal::ZERO);
    }
    (
        tracker.calculate_pnl(current_prices),
        tracker.closed_trades.to_vec(),
    )
}

#[test]
fn test_trade_analyzer_long_profit() {
    // Buy 100 @ 10, Sell 100 @ 12
    let t1 = create_trade(
        "AAPL",
        OrderSide::Buy,
        Decimal::from(100),
        Decimal::from(10),
        Decimal::ZERO,
    );
    let t2 = create_trade(
        "AAPL",
        OrderSide::Sell,
        Decimal::from(100),
        Decimal::from(12),
        Decimal::ZERO,
    );

    let (pnl, _) = analyze_trades(vec![t1, t2], None);

    assert_eq!(pnl.total_closed_trades, 1);
    assert_eq!(pnl.won_count, 1);
    assert_eq!(pnl.gross_pnl, 200.0); // (12-10)*100
    assert_eq!(pnl.win_rate, 100.0);
}

#[test]
fn test_trade_analyzer_short_loss() {
    // Sell 100 @ 10, Buy 100 @ 12
    let t1 = create_trade(
        "AAPL",
        OrderSide::Sell,
        Decimal::from(100),
        Decimal::from(10),
        Decimal::ZERO,
    );
    let t2 = create_trade(
        "AAPL",
        OrderSide::Buy,
        Decimal::from(100),
        Decimal::from(12),
        Decimal::ZERO,
    );

    let (pnl, _) = analyze_trades(vec![t1, t2], None);

    assert_eq!(pnl.total_closed_trades, 1);
    assert_eq!(pnl.lost_count, 1);
    assert_eq!(pnl.gross_pnl, -200.0); // (10-12)*100
    assert_eq!(pnl.loss_rate, 100.0);
}

#[test]
fn test_trade_analyzer_respects_contract_multiplier_for_closed_trade() {
    let t1 = create_trade(
        "RB2310",
        OrderSide::Buy,
        Decimal::ONE,
        Decimal::from(4000),
        Decimal::ZERO,
    );
    let t2 = create_trade(
        "RB2310",
        OrderSide::Sell,
        Decimal::ONE,
        Decimal::from(4010),
        Decimal::ZERO,
    );

    let (pnl, closed_trades) =
        analyze_trades_with_multiplier(vec![t1, t2], None, Decimal::from(10));

    assert_eq!(pnl.gross_pnl, 100.0);
    assert_eq!(pnl.net_pnl, 100.0);
    assert_eq!(closed_trades.len(), 1);
    assert_eq!(closed_trades[0].pnl, 100.0);
    assert_eq!(closed_trades[0].net_pnl, 100.0);
    assert!((closed_trades[0].return_pct - 0.25).abs() < 1e-9);
}

#[test]
fn test_trade_analyzer_fifo() {
    // Buy 100 @ 10
    // Buy 100 @ 12
    // Sell 150 @ 11
    let t1 = create_trade(
        "AAPL",
        OrderSide::Buy,
        Decimal::from(100),
        Decimal::from(10),
        Decimal::ZERO,
    );
    let t2 = create_trade(
        "AAPL",
        OrderSide::Buy,
        Decimal::from(100),
        Decimal::from(12),
        Decimal::ZERO,
    );
    let t3 = create_trade(
        "AAPL",
        OrderSide::Sell,
        Decimal::from(150),
        Decimal::from(11),
        Decimal::ZERO,
    );

    let (pnl, _) = analyze_trades(vec![t1, t2, t3], None);

    assert_eq!(pnl.gross_pnl, 50.0);
    assert_eq!(pnl.total_closed_trades, 2);
}

#[test]
fn test_unrealized_pnl() {
    // Buy 100 @ 10
    let t1 = create_trade(
        "AAPL",
        OrderSide::Buy,
        Decimal::from(100),
        Decimal::from(10),
        Decimal::ZERO,
    );

    let mut prices = HashMap::new();
    prices.insert("AAPL".to_string(), Decimal::from(15));

    let (pnl, _) = analyze_trades(vec![t1], Some(prices));

    assert_eq!(pnl.total_closed_trades, 0);
    assert_eq!(pnl.unrealized_pnl, 500.0); // (15-10)*100
}

#[test]
fn test_unrealized_pnl_respects_contract_multiplier() {
    let t1 = create_trade(
        "RB2310",
        OrderSide::Buy,
        Decimal::ONE,
        Decimal::from(4000),
        Decimal::ZERO,
    );

    let mut prices = HashMap::new();
    prices.insert("RB2310".to_string(), Decimal::from(4010));

    let (pnl, _) = analyze_trades_with_multiplier(vec![t1], Some(prices), Decimal::from(10));

    assert_eq!(pnl.total_closed_trades, 0);
    assert_eq!(pnl.unrealized_pnl, 100.0);
}

#[test]
fn test_max_drawdown_logic() {
    let empty_pnl = TradeTracker::new().calculate_pnl(None);

    // Case 1: Standard Drawdown
    // 100 -> 120 -> 90 -> 110
    let equity_curve = vec![
        (1, Decimal::from(100)),
        (2, Decimal::from(120)), // Peak
        (3, Decimal::from(90)),  // Drawdown: (120-90)/120 = 0.25
        (4, Decimal::from(110)),
    ];

    let result = BacktestResult::calculate(crate::analysis::CalculatorInput {
        equity_curve_decimal: equity_curve,
        cash_curve_decimal: vec![],
        margin_curve_decimal: vec![],
        snapshots: vec![],
        timezone_name: Some("UTC".to_string()),
        timezone_offset: 0,
        trade_pnl: empty_pnl.clone(),
        trades: vec![],
        initial_cash: Decimal::from(100),
        orders: vec![],
        executions: vec![],
        liquidation_audits: vec![],
        days_per_year: 252.0,
        risk_free_rate: 0.0,
    });
    assert_eq!(result.metrics.max_drawdown, 0.25);

    // Case 2: No Drawdown (Monotonic Increase)
    // 100 -> 110 -> 120
    let equity_curve_2 = vec![
        (1, Decimal::from(100)),
        (2, Decimal::from(110)),
        (3, Decimal::from(120)),
    ];
    let result_2 = BacktestResult::calculate(crate::analysis::CalculatorInput {
        equity_curve_decimal: equity_curve_2,
        cash_curve_decimal: vec![],
        margin_curve_decimal: vec![],
        snapshots: vec![],
        timezone_name: Some("UTC".to_string()),
        timezone_offset: 0,
        trade_pnl: empty_pnl.clone(),
        trades: vec![],
        initial_cash: Decimal::from(100),
        orders: vec![],
        executions: vec![],
        liquidation_audits: vec![],
        days_per_year: 252.0,
        risk_free_rate: 0.0,
    });
    assert_eq!(result_2.metrics.max_drawdown, 0.0);

    // Case 3: Immediate Drawdown
    // 100 -> 80 -> 90
    let equity_curve_3 = vec![
        (1, Decimal::from(100)), // Peak
        (2, Decimal::from(80)),  // Drawdown: (100-80)/100 = 0.2
        (3, Decimal::from(90)),
    ];
    let result_3 = BacktestResult::calculate(crate::analysis::CalculatorInput {
        equity_curve_decimal: equity_curve_3,
        cash_curve_decimal: vec![],
        margin_curve_decimal: vec![],
        snapshots: vec![],
        timezone_name: Some("UTC".to_string()),
        timezone_offset: 0,
        trade_pnl: empty_pnl.clone(),
        trades: vec![],
        initial_cash: Decimal::from(100),
        orders: vec![],
        executions: vec![],
        liquidation_audits: vec![],
        days_per_year: 252.0,
        risk_free_rate: 0.0,
    });
    assert_eq!(result_3.metrics.max_drawdown, 0.2);

    // Case 4: Multiple Peaks
    // 100 -> 90 (0.1) -> 100 -> 110 -> 55 (0.5) -> 110
    let equity_curve_4 = vec![
        (1, Decimal::from(100)),
        (2, Decimal::from(90)), // DD 0.1
        (3, Decimal::from(100)),
        (4, Decimal::from(110)), // New Peak
        (5, Decimal::from(55)),  // DD (110-55)/110 = 0.5
        (6, Decimal::from(110)),
    ];
    let result_4 = BacktestResult::calculate(crate::analysis::CalculatorInput {
        equity_curve_decimal: equity_curve_4,
        cash_curve_decimal: vec![],
        margin_curve_decimal: vec![],
        snapshots: vec![],
        timezone_name: Some("UTC".to_string()),
        timezone_offset: 0,
        trade_pnl: empty_pnl.clone(),
        trades: vec![],
        initial_cash: Decimal::from(100),
        orders: vec![],
        executions: vec![],
        liquidation_audits: vec![],
        days_per_year: 252.0,
        risk_free_rate: 0.0,
    });
    assert_eq!(result_4.metrics.max_drawdown, 0.5);
}

#[test]
fn test_ulcer_index_logic() {
    let empty_pnl = TradeTracker::new().calculate_pnl(None);

    // Equity Curve: [100, 100, 90, 90, 100]
    // DDs: [0, 0, 0.1, 0.1, 0]
    // Squares: [0, 0, 0.01, 0.01, 0]
    // Sum = 0.02
    // Mean = 0.02 / 5 = 0.004
    // UI = sqrt(0.004) = 0.0632455532

    let equity_curve = vec![
        (1, Decimal::from(100)),
        (2, Decimal::from(100)),
        (3, Decimal::from(90)),
        (4, Decimal::from(90)),
        (5, Decimal::from(100)),
    ];

    let result = BacktestResult::calculate(crate::analysis::CalculatorInput {
        equity_curve_decimal: equity_curve,
        cash_curve_decimal: vec![],
        margin_curve_decimal: vec![],
        snapshots: vec![],
        timezone_name: Some("UTC".to_string()),
        timezone_offset: 0,
        trade_pnl: empty_pnl.clone(),
        trades: vec![],
        initial_cash: Decimal::from(100),
        orders: vec![],
        executions: vec![],
        liquidation_audits: vec![],
        days_per_year: 252.0,
        risk_free_rate: 0.0,
    });
    let expected_ui = 0.004f64.sqrt();
    assert!((result.metrics.ulcer_index - expected_ui).abs() < 1e-9);
}

#[test]
fn test_calmar_uses_raw_drawdown_ratio_not_pct() {
    let empty_pnl = TradeTracker::new().calculate_pnl(None);
    let equity_curve = vec![
        (0, Decimal::from(100)),
        (15_768_000_000_000_000, Decimal::from(110)),
        (31_536_000_000_000_000, Decimal::from(105)),
    ];

    let result = BacktestResult::calculate(crate::analysis::CalculatorInput {
        equity_curve_decimal: equity_curve,
        cash_curve_decimal: vec![],
        margin_curve_decimal: vec![],
        snapshots: vec![],
        timezone_name: Some("UTC".to_string()),
        timezone_offset: 0,
        trade_pnl: empty_pnl,
        trades: vec![],
        initial_cash: Decimal::from(100),
        orders: vec![],
        executions: vec![],
        liquidation_audits: vec![],
        days_per_year: 252.0,
        risk_free_rate: 0.0,
    });

    let expected_raw_calmar = result.metrics.annualized_return / result.metrics.max_drawdown;
    let wrong_pct_calmar = result.metrics.annualized_return / result.metrics.max_drawdown_pct;

    assert!((result.metrics.calmar_ratio - expected_raw_calmar).abs() < 1e-12);
    assert!((result.metrics.calmar_ratio - wrong_pct_calmar).abs() > 1e-3);
}

#[test]
fn test_daily_metrics_resample_by_local_timezone_day() {
    let empty_pnl = TradeTracker::new().calculate_pnl(None);
    let tz: Tz = "Asia/Shanghai".parse().unwrap();
    let local_ns = |year, month, day, hour, minute| {
        tz.with_ymd_and_hms(year, month, day, hour, minute, 0)
            .single()
            .unwrap()
            .with_timezone(&Utc)
            .timestamp()
            * 1_000_000_000
    };

    let equity_curve = vec![
        (local_ns(2024, 1, 1, 23, 0), Decimal::from(100)),
        (local_ns(2024, 1, 2, 2, 0), Decimal::from(110)),
        (local_ns(2024, 1, 2, 23, 0), Decimal::from(121)),
        (
            local_ns(2024, 1, 3, 2, 0),
            Decimal::from_str_exact("133.1").unwrap(),
        ),
    ];

    let result = BacktestResult::calculate(crate::analysis::CalculatorInput {
        equity_curve_decimal: equity_curve,
        cash_curve_decimal: vec![],
        margin_curve_decimal: vec![],
        snapshots: vec![],
        timezone_name: Some("Asia/Shanghai".to_string()),
        timezone_offset: 8 * 3600,
        trade_pnl: empty_pnl,
        trades: vec![],
        initial_cash: Decimal::from(100),
        orders: vec![],
        executions: vec![],
        liquidation_audits: vec![],
        days_per_year: 252.0,
        risk_free_rate: 0.0,
    });

    let expected_returns = [0.21_f64, 0.10_f64];
    let mean = expected_returns.iter().sum::<f64>() / expected_returns.len() as f64;
    let variance = expected_returns
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / (expected_returns.len() - 1) as f64;
    let expected_volatility = variance.sqrt() * (252.0f64).sqrt();

    assert!((result.metrics.volatility - expected_volatility).abs() < 1e-12);
}

/// 构造一段每日权益曲线（日收益 [0.21, 0.10]），用指定的 days_per_year / risk_free_rate 计算结果。
fn build_sharpe_fixture(days_per_year: f64, risk_free_rate: f64) -> BacktestResult {
    let empty_pnl = TradeTracker::new().calculate_pnl(None);
    let day_ns = 86_400_000_000_000_i64;
    let equity_curve = vec![
        (0, Decimal::from(100)),
        (day_ns, Decimal::from(121)),
        (2 * day_ns, Decimal::from_str_exact("133.1").unwrap()),
    ];

    BacktestResult::calculate(crate::analysis::CalculatorInput {
        equity_curve_decimal: equity_curve,
        cash_curve_decimal: vec![],
        margin_curve_decimal: vec![],
        snapshots: vec![],
        timezone_name: Some("UTC".to_string()),
        timezone_offset: 0,
        trade_pnl: empty_pnl,
        trades: vec![],
        initial_cash: Decimal::from(100),
        orders: vec![],
        executions: vec![],
        liquidation_audits: vec![],
        days_per_year,
        risk_free_rate,
    })
}

#[test]
fn test_sharpe_uses_arithmetic_mean_not_cagr() {
    let dpy = 252.0_f64;
    let result = build_sharpe_fixture(dpy, 0.0);

    // 期望：分子为日收益算术均值年化 (mean * dpy)，与分母 std * sqrt(dpy) 口径匹配。
    let returns = [0.21_f64, 0.10_f64];
    let mean = returns.iter().sum::<f64>() / returns.len() as f64;
    let variance =
        returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (returns.len() - 1) as f64;
    let std = variance.sqrt();
    let expected_sharpe = (mean * dpy) / (std * dpy.sqrt());

    assert!((result.metrics.sharpe_ratio - expected_sharpe).abs() < 1e-9);

    // 反向断言：旧的 CAGR 口径 (annualized_return / volatility) 在这种短回测下会爆炸，
    // 新口径必须显著不同于它，以防回归到 PR #305 修复前的行为。
    let old_cagr_sharpe = result.metrics.annualized_return / result.metrics.volatility;
    assert!((result.metrics.sharpe_ratio - old_cagr_sharpe).abs() > 1.0);
}

#[test]
fn test_days_per_year_scales_annualization() {
    let sharpe_252 = build_sharpe_fixture(252.0, 0.0).metrics.sharpe_ratio;
    let sharpe_365 = build_sharpe_fixture(365.0, 0.0).metrics.sharpe_ratio;

    // sharpe = mean * sqrt(dpy) / std，故比值应为 sqrt(365/252)。
    let expected_ratio = (365.0_f64 / 252.0_f64).sqrt();
    assert!((sharpe_365 / sharpe_252 - expected_ratio).abs() < 1e-9);
}

#[test]
fn test_risk_free_rate_lowers_sharpe() {
    let dpy = 252.0_f64;
    let rf = 0.5_f64; // 年化无风险利率
    let base = build_sharpe_fixture(dpy, 0.0);
    let with_rf = build_sharpe_fixture(dpy, rf);

    // rf=0 与默认行为一致，保证基线不动。
    let returns = [0.21_f64, 0.10_f64];
    let mean = returns.iter().sum::<f64>() / returns.len() as f64;
    let variance =
        returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (returns.len() - 1) as f64;
    let annualized_vol = variance.sqrt() * dpy.sqrt();

    // Sharpe 下降量应恰为 rf / annualized_volatility。
    let expected_drop = rf / annualized_vol;
    assert!(
        (base.metrics.sharpe_ratio - with_rf.metrics.sharpe_ratio - expected_drop).abs() < 1e-9
    );
}
