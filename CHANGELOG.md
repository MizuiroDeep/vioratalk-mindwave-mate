# Changelog

All notable changes to VioraTalk - MindWave Mate will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.0] - 2025-08-29

### Added
- **Real Engine Implementations (Phase 4)**
  - `FasterWhisperEngine` for local speech-to-text with GPU/CPU auto-selection
  - `GeminiEngine` for LLM integration with BYOK (Bring Your Own Key) support
  - `Pyttsx3Engine` for cross-platform text-to-speech synthesis
  - Comprehensive error handling with retry mechanisms
  - Engine switching mechanism with fallback support

### Features
- **Production-Ready Speech Processing**
  - FasterWhisper integration with automatic model downloading
  - Support for multiple Whisper model sizes (tiny, base, small, medium, large)
  - Language auto-detection for 99+ languages
  - GPU acceleration when CUDA is available
  
- **Advanced LLM Integration**
  - Gemini API integration with streaming support
  - Conversation history management
  - Temperature and max_tokens configuration
  - Automatic retry with exponential backoff
  
- **Cross-Platform TTS**
  - Windows SAPI, macOS NSSpeechSynthesizer, Linux espeak support
  - Voice selection and rate/volume/pitch control
  - Iterative synthesis for memory efficiency

### Technical Details
- **Testing Excellence**
  - 743 total tests (742 passing, 99.9% success rate)
  - Test coverage: 78.17% (target 65-70% exceeded)
  - Individual engine coverage: FasterWhisper 79%, Gemini 84%, Pyttsx3 77%
  - 8 integration tests with 100% success rate
  - Comprehensive manual test scripts for real-world validation

- **Code Quality Improvements**
  - Applied ruff fixes to 89 files (63 issues auto-fixed)
  - Applied Black formatting to 26 files
  - Organized imports with isort in 24 files
  - Maintained code consistency across entire codebase

### Fixed
- APIError constructor parameter issues in exception handling
- Mock vs real engine test compatibility issues
- Pyttsx3 engine reuse problems (engine recreation on each call)
- Integration test concurrent processing issues

### Changed
- Updated from mock engines to real production engines
- Enhanced error handling with specific error codes (E1xxx-E3xxx)
- Improved test infrastructure for real engine validation
- Refined engine initialization patterns for better reliability

### Infrastructure
- Added comprehensive integration tests in tests/integration/
- Created manual test scripts in scripts/manual_tests/feature/
- Established BYOK pattern for API key management
- Prepared foundation for Phase 5 CLI implementation

## [0.3.0] - 2025-08-23

### Added
- **Mock Engine Implementations (Phase 3)**
  - `MockSTTEngine` for simulating speech-to-text functionality
  - `MockLLMEngine` for simulating language model responses
  - `MockTTSEngine` for simulating text-to-speech synthesis
  - Character-specific response patterns in MockLLMEngine
  - WAV audio data generation in MockTTSEngine
  - Error simulation modes for all mock engines

### Features
- **Testing Infrastructure Enhancement**
  - 99 unit tests for mock engines
  - 16 integration tests for engine coordination
  - 4 manual test scenarios with performance validation
  - Total: 548 tests with 86.51% coverage
  - Average response time: 0.329 seconds (SLA: <3 seconds)

- **Mock Engine Capabilities**
  - **STT**: Language detection, alternatives generation, confidence scores
  - **LLM**: Streaming responses, character personalities (Aoi, Haru, Yui), conversation history
  - **TTS**: Voice parameters, style management, realistic WAV generation
  - Error mode simulation for resilience testing
  - Configurable response delays for performance testing

### Technical Details
- **Code Quality Improvements**
  - Applied Black formatting to 13 files
  - Organized imports with isort in 12 files
  - Fixed 28 code issues with ruff
  - Maintained consistent code style across mock implementations

- **Test Coverage**
  - Unit tests: 99/99 passing (100%)
  - Integration tests: 16/16 passing (100%)
  - Manual tests: 4/4 scenarios passing (100%)
  - Overall coverage: 86.51% (target: 60% exceeded)

### Fixed
- Exception error code alignment with specification (E1000→E1001, E3000→E3001)
- Mock engine import paths corrected
- Character class export in tests/mocks/__init__.py
- Voice ID format standardization (ja-JP-Female-1 format)

### Changed
- Updated version from 0.2.0 to 0.3.0
- Enhanced mock engines to support Phase 1-3 integration
- Improved test infrastructure for comprehensive validation

### Infrastructure
- Mock engines placed in tests/mocks/ directory
- Maintained separation between test mocks and production code
- Prepared foundation for Phase 4 real engine implementations

## [0.2.0] - 2025-08-22

### Added
- **DialogueManager Implementation (Phase 2)**
  - `DialogueManager` with conversation state management
  - `DialogueConfig` for dialogue settings and configuration
  - `DialogueTurn` data structure for conversation history
  - `ConversationContext` for managing dialogue flow
  - `ConversationState` enum for dialogue states
  - Integration with `VioraTalkEngine`
  - `MockCharacterManager` for testing character interactions

### Features
- **Conversation Management**
  - Maximum turns configuration (default: 10)
  - Temperature settings for response variation
  - Response timeout handling
  - Conversation history tracking
  - Character-aware dialogue system preparation

- **Testing Infrastructure**
  - 433 unit tests (all passing)
  - 22 integration tests (all passing)
  - Total: 455 tests with 86.25% coverage
  - Mock character system for testing
  - Comprehensive test suite for dialogue components

### Technical Details
- **Code Quality Improvements**
  - Applied Black code formatting across all files
  - Organized imports with isort
  - Fixed 47 code style issues with ruff
  - Consistent code style throughout the project

- **Integration**
  - DialogueManager integrated with VioraTalkEngine
  - Configuration management through DialogueConfig
  - Prepared for future CharacterManager integration
  - Mock implementations for testing

### Fixed
- Test-implementation inconsistencies resolved (12 issues)
- Import organization across all modules
- Code style violations corrected
- Test fixtures properly initialized

### Infrastructure
- Enhanced test coverage from 86.52% to 86.25% (stabilized)
- Added dialogue-specific error handling
- Implemented conversation state management
- Prepared for Phase 3 mock engine implementations

## [0.1.0] - 2025-08-20

### Added
- **Core Foundation Implementation (Phase 0 & Phase 1)**
  - Project structure with Poetry dependency management
  - `VioraTalkEngine` as the main orchestration component
  - `ConfigManager` for YAML-based configuration management
  - `LoggerManager` with JSON formatting and debug mode
  - `I18nManager` for internationalization (ja/en support)
  - `AutoSetupManager` for component verification
  - `BackgroundServiceManager` for service lifecycle management
  - `ModelDownloadManager` for ML model management
  - Base classes and type definitions

### Features
- **System Architecture**
  - 8-step initialization flow with state management
  - Component state tracking (NOT_INITIALIZED, INITIALIZING, READY, ERROR, TERMINATED)
  - Asynchronous initialization support
  - Comprehensive error handling with error codes
  - First-run detection with marker file system

- **Testing Infrastructure**
  - 305 unit tests (all passing)
  - 20 integration tests (all passing)
  - Test coverage: 86.52% (target: 60% exceeded)
  - Mock implementations for testing
  - Fixture-based test architecture

### Technical Details
- **Dependencies**
  - Python 3.11.9 with pyenv
  - Poetry 2.1.3 for dependency management
  - Core: apscheduler, pyyaml, typing-extensions
  - Testing: pytest, pytest-asyncio, pytest-cov
  - Code quality: black, isort, flake8, ruff

### Infrastructure
- GitHub repository initialized
- CI/CD preparation with GitHub Actions compatibility
- Comprehensive documentation structure
- Semantic versioning adopted
- Keep a Changelog format adopted

## [0.0.1] - 2025-08-01

### Added
- Initial project setup
- Basic project structure
- Poetry configuration
- README with project vision

[Unreleased]: https://github.com/MizuiroDeep/vioratalk-mindwave-mate/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/MizuiroDeep/vioratalk-mindwave-mate/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/MizuiroDeep/vioratalk-mindwave-mate/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/MizuiroDeep/vioratalk-mindwave-mate/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/MizuiroDeep/vioratalk-mindwave-mate/compare/v0.0.1...v0.1.0
[0.0.1]: https://github.com/MizuiroDeep/vioratalk-mindwave-mate/releases/tag/v0.0.1