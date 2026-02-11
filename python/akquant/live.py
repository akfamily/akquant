# -*- coding: utf-8 -*-
import threading
import time
from typing import Any, List, Optional, Type

from akquant import Bar, DataFeed, Engine, Instrument, Strategy
from akquant.gateway.ctp import CTPMarketGateway


class LiveRunner:
    """
    Live/Paper Trading Runner.

    Encapsulates the boilerplate code for setting up the engine, data feed,
    instruments, and gateways for live or paper trading.
    """

    def __init__(
        self,
        strategy_cls: Type[Strategy],
        instruments: List[Instrument],
        md_front: str,
        td_front: Optional[str] = None,
        broker_id: str = "",
        user_id: str = "",
        password: str = "",
        app_id: str = "",
        auth_code: str = "",
        use_aggregator: bool = True,
    ):
        """
        Initialize the LiveRunner.

        :param strategy_cls: The strategy class to run.
        :param instruments: List of instruments to trade.
        :param md_front: CTP Market Data Front URL.
        :param td_front: CTP Trade Front URL (optional).
        :param broker_id: CTP Broker ID (optional).
        :param user_id: CTP User ID (optional).
        :param password: CTP Password (optional).
        :param app_id: CTP App ID (optional).
        :param auth_code: CTP Auth Code (optional).
        :param use_aggregator: Whether to use BarAggregator (default True).
        """
        self.strategy_cls = strategy_cls
        self.instruments = instruments
        self.md_front = md_front
        self.td_front = td_front
        self.broker_id = broker_id
        self.user_id = user_id
        self.password = password
        self.app_id = app_id
        self.auth_code = auth_code
        self.use_aggregator = use_aggregator

        self.feed = DataFeed.create_live()  # type: ignore
        self.engine = Engine()

    def run(
        self,
        cash: float = 1_000_000.0,
        show_progress: bool = False,
        duration: Optional[str] = None,
    ) -> None:
        """
        Run the live/paper trading session.

        :param cash: Initial cash (default 1,000,000).
        :param show_progress: Whether to show progress bar (default False).
        :param duration: Optional run duration string (e.g., "1m", "1h", "60s").
                         If set, strategy will stop after this duration.
        """
        print("[LiveRunner] Configuring Engine...")
        self.engine.add_data(self.feed)
        self.engine.set_cash(cash)

        for instrument in self.instruments:
            self.engine.add_instrument(instrument)

        self.engine.use_china_futures_market()
        # Force continuous session for simulation/paper trading often needed
        self.engine.set_force_session_continuous(True)

        # Start CTP Market Gateway
        print("[LiveRunner] Starting CTP Market Gateway...")
        symbols = [inst.symbol for inst in self.instruments]
        md_gateway = CTPMarketGateway(
            self.feed, self.md_front, symbols, self.use_aggregator
        )
        md_thread = threading.Thread(target=md_gateway.start, daemon=True)
        md_thread.start()

        # Optional: Start Trader Gateway if credentials provided
        if self.td_front and self.user_id:
            print(
                "[LiveRunner] Trader Gateway credentials provided, but Trader "
                "Gateway is not fully integrated in this runner yet."
            )

        time.sleep(2.0)

        # Create Strategy Instance
        strategy_instance = self.strategy_cls()

        # Apply duration limit if specified
        if duration:
            print(f"[LiveRunner] Auto-stop enabled: {duration}")
            self._apply_time_limit(strategy_instance, duration)

        print("[LiveRunner] Running Strategy (Press Ctrl+C to stop)...")
        try:
            self.engine.run(strategy_instance, show_progress=show_progress)
        except KeyboardInterrupt:
            print("\n[LiveRunner] Stopping by User (or Duration Limit)...")
        except Exception as e:
            print(f"\n[LiveRunner] Stopping due to Error: {e}")
            import traceback

            traceback.print_exc()
        finally:
            self._print_summary()

    def _apply_time_limit(self, strategy: Strategy, duration_str: str) -> None:
        """Inject time check into strategy methods."""
        import re

        # Parse duration
        duration_sec = 0
        match = re.match(r"^(\d+)([smh]?)$", duration_str)
        if match:
            val, unit = match.groups()
            val = int(val)
            if unit == "s" or unit == "":
                duration_sec = val
            elif unit == "m":
                duration_sec = val * 60
            elif unit == "h":
                duration_sec = val * 3600
        else:
            print(
                f"[LiveRunner] Warning: Invalid duration format '{duration_str}', "
                "ignoring."
            )
            return

        start_time = time.time()

        # Patch on_bar
        original_on_bar = strategy.on_bar

        def wrapped_on_bar(bar: Bar) -> None:
            if time.time() - start_time > duration_sec:
                raise KeyboardInterrupt(f"Duration {duration_str} reached")
            original_on_bar(bar)

        # Use setattr to bypass mypy method assignment check
        setattr(strategy, "on_bar", wrapped_on_bar)

        # Patch on_tick if it exists/is overridden
        if hasattr(strategy, "on_tick"):
            original_on_tick = strategy.on_tick

            def wrapped_on_tick(tick: Any) -> None:
                if time.time() - start_time > duration_sec:
                    raise KeyboardInterrupt(f"Duration {duration_str} reached")
                original_on_tick(tick)

            setattr(strategy, "on_tick", wrapped_on_tick)

    def _print_summary(self) -> None:
        try:
            results = self.engine.get_results()
            print("\n" + "=" * 50)
            print("TRADING SUMMARY (Manual Stop)")
            print("=" * 50)
            print(f"Total Return: {results.metrics.total_return_pct:.2%}")
            print(f"Annualized Return: {results.metrics.annualized_return:.2%}")
            print(f"Max Drawdown: {results.metrics.max_drawdown_pct:.2%}")
            print(f"Sharpe Ratio: {results.metrics.sharpe_ratio:.4f}")
            print(f"Win Rate: {results.metrics.win_rate:.2%}")
            print(f"Total Trades: {len(results.trades)}")
            print("=" * 50)

            # Print Current Positions if available
            if results.snapshots:
                last_snapshots = results.snapshots[-1][1]
                print("\nCurrent Positions:")
                has_pos = False
                for s in last_snapshots:
                    if abs(s.quantity) > 0:
                        print(f"  {s.symbol}: {s.quantity}")
                        has_pos = True
                if not has_pos:
                    print("  (None)")
        except Exception as e:
            print(f"Error generating summary: {e}")
