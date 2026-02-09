# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.13] - 2026-02-09

### Added
- Incremental learning support for `SklearnAdapter` (via `partial_fit`) and `PyTorchAdapter` (via weight reset control).
- `incremental` parameter in `ValidationConfig`.
- Updated `ml_guide.md` with new features and clarified API signatures.

### Changed
- `PyTorchAdapter` now defaults to `incremental=False` for strict Walk-Forward Validation.

## [0.1.12] - Previous Release
- Basic implementation of `BarAggregator`.
- Rust-based performance optimizations.
- Zero-copy data access via PyO3.
