# Changelog

All notable changes to VioraTalk - MindWave Mate will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.0.1] - 2025-08-17

### Added
- Initial project structure setup
- Basic Poetry configuration with Python 3.11.9 support
- Development environment configuration files
  - `.gitignore` for Python projects
  - `.pre-commit-config.yaml` for code quality
  - `pyproject.toml` with development dependencies
- Documentation structure
  - `README.md` with bilingual (Japanese/English) project description
  - `docs/requirements.md` with system requirements
  - `docs/architecture.md` with high-level architecture overview
  - `docs/CONTRIBUTING.md` with contribution guidelines
- Internationalization foundation
  - `messages/ja/errors.yaml` for Japanese error messages
  - `messages/en/errors.yaml` for English error messages
- Testing foundation
  - `tests/test_import.py` with basic import tests
- MIT License

### Project Structure
- Established modular directory structure following VioraTalk specifications
- Created placeholder directories for future implementation:
  - `src/vioratalk/` for main application code
  - `tests/` for test suites
  - `docs/` for documentation
  - `scripts/` for utility scripts
  - `user_settings/` for user configurations
  - `messages/` for i18n support

### Development Setup
- Configured Poetry for dependency management
- Set up development tools (Black, isort, flake8, mypy, pytest, pre-commit)
- Established Git repository with initial commit
- Created GitHub private repository for version control

[Unreleased]: https://github.com/MizuiroDeep/vioratalk-mindwave-mate/compare/v0.0.1...HEAD
[0.0.1]: https://github.com/MizuiroDeep/vioratalk-mindwave-mate/releases/tag/v0.0.1