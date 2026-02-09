# Contributing to AKQuant

First off, thanks for taking the time to contribute! ❤️

All types of contributions are welcome:
- 🐛 Bug reports
- 💡 Feature requests
- 📖 Documentation improvements
- 🔧 Code contributions

## Development Setup

1. **Prerequisites**:
   - Rust (latest stable)
   - Python 3.10+
   - `maturin`

2. **Clone and Install**:
   ```bash
   git clone https://github.com/your-username/akquant.git
   cd akquant
   # Install dev dependencies
   pip install -e ".[dev,ml,plot]"
   # Build Rust extension in development mode
   maturin develop
   ```

3. **Running Tests**:
   ```bash
   pytest tests/
   ```

4. **Code Quality**:
   We use `ruff` for linting and `mypy` for type checking.
   ```bash
   ruff check .
   mypy .
   ```

## Pull Request Process

1. Fork the repository.
2. Create a new branch for your changes.
3. Make sure your changes follow the [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide and have proper type hints.
4. Add tests for new features or bug fixes.
5. Ensure all tests pass.
6. Submit a Pull Request with a clear description of the changes.

## Documentation

If you modify the source code, please update the corresponding documentation in the `docs/` folder.

## License

By contributing, you agree that your contributions will be licensed under its MIT License.
