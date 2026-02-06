from akquant import Strategy


class TestStrategy(Strategy):
    """Test strategy for verifying super().__init__()."""

    def __init__(self) -> None:
        """Initialize."""
        super().__init__()
        print("super().__init__() called successfully")


s = TestStrategy()
print(f"Strategy initialized: {s}")
