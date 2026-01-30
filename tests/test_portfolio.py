from akquant import Portfolio


def test_portfolio_initialization() -> None:
    """Test Portfolio initialization."""
    p = Portfolio(100000.0)
    assert p.cash == 100000.0
    assert len(p.positions) == 0
    assert len(p.available_positions) == 0


def test_portfolio_get_position_empty() -> None:
    """Test get_position on empty portfolio."""
    p = Portfolio(100000.0)
    assert p.get_position("AAPL") == 0.0
    assert p.get_available_position("AAPL") == 0.0


def test_portfolio_repr() -> None:
    """Test string representation."""
    p = Portfolio(100000.0)
    assert "Portfolio(cash=100000.00, positions_count=0)" in str(p)


def test_portfolio_readonly() -> None:
    """Verify that we cannot modify cash directly if no setter is exposed."""
    p = Portfolio(100000.0)
    # This should fail if there is no setter, or succeed if there is one.
    # Based on Rust code, there is no setter for cash.
    try:
        p.cash = 200000.0
    except AttributeError:
        pass  # Expected
    except TypeError:
        pass  # Expected
    else:
        # If it succeeded, check if it actually changed
        # (it shouldn't if it's a getter only)
        # Or maybe PyO3 generates a setter? No, explicit #[getter] usually
        # implies read-only unless #[setter] is present.
        # However, if it's a public field in a pyclass, it might be writable
        # if not handled by getter.
        # In Rust: pub cash: Decimal. But #[pyclass] makes struct opaque
        # unless fields are #[pyo3(get)].
        # But here we use explicit #[getter] fn get_cash.
        # So it should be read-only.
        assert p.cash == 100000.0  # Should remain unchanged or raise error
