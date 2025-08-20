# VioraTalk - MindWave Mate

[![Release](https://img.shields.io/github/v/release/MizuiroDeep/vioratalk-mindwave-mate)](https://github.com/MizuiroDeep/vioratalk-mindwave-mate/releases)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![Coverage](https://img.shields.io/badge/coverage-86.52%25-brightgreen.svg)](https://github.com/MizuiroDeep/vioratalk-mindwave-mate)
[![Tests](https://img.shields.io/badge/tests-305%20passed-success.svg)](https://github.com/MizuiroDeep/vioratalk-mindwave-mate)

An AI voice assistant that brings natural conversation to your desktop.

AIキャラクターと自然に会話できるデスクトップ音声アシスタント

[English](#english) | [日本語](#japanese)

---

## 📌 Latest Release

**[v0.1.0 - Core Foundation](https://github.com/MizuiroDeep/vioratalk-mindwave-mate/releases/tag/v0.1.0)** (2025-08-20)

We've completed the core foundation with 9 essential components, achieving 86.52% test coverage across 305 tests. This establishes a solid base for the upcoming dialogue system.

詳細: [Changelog](CHANGELOG.md) | [Release Notes](https://github.com/MizuiroDeep/vioratalk-mindwave-mate/releases/tag/v0.1.0)

---

## English

### About This Project

VioraTalk is a personal learning project where I'm developing an AI voice assistant with the help of Claude AI. As someone relatively new to Python, I'm building this step by step, focusing on creating something genuinely useful while learning best practices along the way.

The goal is simple: create a desktop assistant that feels natural to talk to, with characters that have their own personalities and can remember past conversations.

### Features

VioraTalk aims to provide:

- **Push-to-Talk voice interaction** - Press Space to talk, just like a walkie-talkie
- **Multiple AI models** - Works with Gemini, Claude, ChatGPT, and local models via Ollama
- **Character personalities** - Each character has unique traits and speaking styles
- **Memory system** - Characters remember your previous conversations
- **Offline capability** - Can run entirely on your machine with local models
- **Bilingual support** - Full Japanese and English interface

### Current Status

This is a work in progress. Phase 1 (core foundation) is complete, and I'm now working on Phase 2 (dialogue system). The project follows a structured development plan, but I'm taking my time to learn and implement things properly.

If you're interested in following along or want to see how a Python beginner tackles a complex project with AI assistance, you're welcome to star the repository or join the discussions.

### Requirements

- Windows 11 (Windows 10 support planned)
- Python 3.11.9 or higher
- 8GB RAM minimum (16GB recommended for smoother operation)
- A microphone if you want to use voice input

### Installation

```bash
# Clone the repository
git clone https://github.com/MizuiroDeep/vioratalk-mindwave-mate.git
cd vioratalk-mindwave-mate

# Install dependencies
poetry install

# Run tests to make sure everything works
poetry run pytest

# The main application will be available from Phase 2
poetry run python -m vioratalk  # Coming soon
```

### Development Setup

If you want to explore the code or run the test suite:

```bash
# Install development dependencies
poetry install --with dev

# Run the test suite
poetry run pytest --cov=vioratalk

# Check code quality
poetry run black src/ tests/
poetry run isort src/ tests/
poetry run flake8 src/ tests/
```

### Project Structure

The codebase is organized to be modular and maintainable:

```
vioratalk-mindwave-mate/
├── src/vioratalk/          # Main application code
│   ├── core/              # Core components (Phase 1 ✓)
│   ├── services/          # Background services
│   ├── infrastructure/    # System utilities
│   └── configuration/     # Settings management
├── tests/                  # Comprehensive test suite
├── docs/                   # Documentation
├── messages/              # Translations (ja/en)
└── user_settings/         # User configuration files
```

### Development Roadmap

**Completed:**
- ✅ Phase 0: Project setup (v0.0.1)
- ✅ Phase 1: Core foundation (v0.1.0)

**In Progress:**
- 🚧 Phase 2: Dialogue system

**Upcoming:**
- Phase 3-4: Speech recognition and AI integration
- Phase 5-6: Conversation management and testing
- Phase 7-9: Character personalities and memory
- Phase 10-11: Desktop GUI
- Phase 12-14: Advanced features and optimization

### API Keys

VioraTalk uses a BYOK (Bring Your Own Key) approach. You'll need to get your own API keys from:

- **Gemini**: [Google AI Studio](https://makersuite.google.com/app/apikey)
- **Claude**: [Anthropic Console](https://console.anthropic.com/)
- **ChatGPT**: [OpenAI Platform](https://platform.openai.com/)

Local models through Ollama don't require API keys.

### Available Characters

The free version will include three characters:
- **Aoi** - Cheerful and energetic
- **Haru** - Calm and intellectual  
- **Yui** - Kind and empathetic

Additional characters are planned for the Pro version.

### Contributing

This is primarily a personal learning project, and I'm not currently accepting pull requests. However, I'd love to hear from you! Feel free to:

- ⭐ Star the repository if you find it interesting
- 💬 Open a discussion if you have ideas or questions
- 🐛 Report bugs (I'll do my best to fix them)
- 💡 Suggest features for future consideration

Since I'm still learning Python, responses might take some time, but I appreciate all feedback.

### Testing

The project maintains high test coverage:

```bash
# Run all tests
poetry run pytest

# Run with coverage report
poetry run pytest --cov=vioratalk --cov-report=html
```

Current metrics: 305 tests, 86.52% coverage

### License

MIT License - see [LICENSE](LICENSE) for details.

### Contact

- **GitHub**: [@MizuiroDeep](https://github.com/MizuiroDeep)
- **Issues**: [Bug Reports](https://github.com/MizuiroDeep/vioratalk-mindwave-mate/issues)
- **Discussions**: [Ideas & Questions](https://github.com/MizuiroDeep/vioratalk-mindwave-mate/discussions)

---

## Japanese

### このプロジェクトについて

VioraTalkは、私がClaude AIの助けを借りながら開発している音声アシスタントプロジェクトです。Pythonはまだ勉強中ですが、実用的なものを作りながら、しっかりとした開発手法を学んでいます。

目指しているのは、自然に話しかけられるデスクトップアシスタント。キャラクターごとに個性があり、過去の会話も覚えていてくれる、そんなアプリケーションです。

### 機能

VioraTalkの特徴：

- **Push-to-Talk音声入力** - スペースキーを押している間だけ音声認識
- **複数のAIモデル対応** - Gemini、Claude、ChatGPT、Ollamaでのローカル実行
- **個性的なキャラクター** - それぞれ異なる性格と話し方
- **記憶システム** - 過去の会話を覚えている
- **オフライン動作** - ローカルモデルで完全オフライン動作可能
- **日英対応** - インターフェースは日本語と英語に対応

### 開発状況

現在、Phase 1（基盤実装）が完了し、Phase 2（対話システム）の開発に入っています。計画的に進めていますが、学習しながらの開発なので、じっくり時間をかけて実装しています。

Python初心者がAIと一緒に複雑なプロジェクトに挑戦する過程に興味がある方は、ぜひスターやディスカッションにご参加ください。

### 必要環境

- Windows 11（Windows 10対応は予定）
- Python 3.11.9以上
- メモリ8GB以上（16GB推奨）
- 音声入力を使う場合はマイク

### インストール方法

```bash
# リポジトリのクローン
git clone https://github.com/MizuiroDeep/vioratalk-mindwave-mate.git
cd vioratalk-mindwave-mate

# 依存関係のインストール
poetry install

# テストを実行して動作確認
poetry run pytest

# メインアプリケーション（Phase 2から利用可能）
poetry run python -m vioratalk  # 開発中
```

### 開発環境のセットアップ

コードを確認したり、テストを実行したい場合：

```bash
# 開発用依存関係をインストール
poetry install --with dev

# テストスイートの実行
poetry run pytest --cov=vioratalk

# コード品質チェック
poetry run black src/ tests/
poetry run isort src/ tests/
poetry run flake8 src/ tests/
```

### プロジェクト構成

メンテナンスしやすいモジュール構造になっています：

```
vioratalk-mindwave-mate/
├── src/vioratalk/          # アプリケーション本体
│   ├── core/              # コアコンポーネント (Phase 1 ✓)
│   ├── services/          # バックグラウンドサービス
│   ├── infrastructure/    # システムユーティリティ
│   └── configuration/     # 設定管理
├── tests/                  # テストスイート
├── docs/                   # ドキュメント
├── messages/              # 翻訳ファイル (ja/en)
└── user_settings/         # ユーザー設定
```

### 開発ロードマップ

**完了:**
- ✅ Phase 0: プロジェクト準備 (v0.0.1)
- ✅ Phase 1: 基盤実装 (v0.1.0)

**進行中:**
- 🚧 Phase 2: 対話システム

**今後の予定:**
- Phase 3-4: 音声認識とAI統合
- Phase 5-6: 会話管理とテスト
- Phase 7-9: キャラクター性格と記憶
- Phase 10-11: デスクトップGUI
- Phase 12-14: 拡張機能と最適化

### APIキー

VioraTalkはBYOK（Bring Your Own Key）方式です。各サービスのAPIキーをご自身で取得してください：

- **Gemini**: [Google AI Studio](https://makersuite.google.com/app/apikey)
- **Claude**: [Anthropic Console](https://console.anthropic.com/)
- **ChatGPT**: [OpenAI Platform](https://platform.openai.com/)

Ollamaでのローカル実行にはAPIキーは不要です。

### 利用可能なキャラクター

無料版では3体のキャラクターを提供予定：
- **あおい** - 明るく元気な性格
- **はる** - 落ち着いた知的な性格
- **ゆい** - 優しく共感的な性格

Pro版では追加キャラクターを予定しています。

### 貢献について

これは個人の学習プロジェクトのため、現在プルリクエストは受け付けていません。ただし、以下は大歓迎です：

- ⭐ 興味を持っていただけたらスターをお願いします
- 💬 アイデアや質問があればディスカッションを開いてください
- 🐛 バグを見つけたら報告してください（できる限り修正します）
- 💡 将来的な機能の提案も歓迎です

Pythonを勉強中のため返信に時間がかかることもありますが、すべてのフィードバックに感謝しています。

### テスト

高いテストカバレッジを維持しています：

```bash
# 全テストの実行
poetry run pytest

# カバレッジレポート付きで実行
poetry run pytest --cov=vioratalk --cov-report=html
```

現在の指標: 305テスト、カバレッジ86.52%

### ライセンス

MITライセンス - 詳細は[LICENSE](LICENSE)をご覧ください。

### 連絡先

- **GitHub**: [@MizuiroDeep](https://github.com/MizuiroDeep)
- **Issues**: [バグ報告](https://github.com/MizuiroDeep/vioratalk-mindwave-mate/issues)
- **Discussions**: [アイデア・質問](https://github.com/MizuiroDeep/vioratalk-mindwave-mate/discussions)

---

<div align="center">

Building an AI voice assistant, one step at a time.

一歩ずつ、AIボイスアシスタントを作っています。

© 2025 VioraTalk Project

</div>