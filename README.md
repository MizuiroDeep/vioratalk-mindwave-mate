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

## English

### About This Project

VioraTalk is a personal learning project where I'm developing an AI voice assistant with Claude AI's help. As someone still learning Python, I'm building this step by step, focusing on creating something practical while learning proper development practices.

The goal is to create a desktop assistant that feels natural to talk to, with characters that have their own personalities and can remember past conversations.

### Project Vision & Roadmap

This project follows a transparent development approach:

**Current Status:**
- Phase 1 (Core Foundation) completed ✅
- Phase 2 (Dialogue System) in progress 🚧
- Learning and developing in public

**Development Philosophy:**
- **Forever Free Core**: All essential features will always be free and open source
- **BYOK Model**: Bring Your Own Key - you control your API costs
- **Learning First**: Code quality improves as I learn

**Future Plans:**
While the core features will always remain free, I'm planning to develop optional extended features in the future (Phase 13+) to sustain long-term development. The free version will always be fully functional.

### Features

VioraTalk offers:

- **Voice Interaction** - Push-to-Talk voice input using the spacebar
- **Multiple AI Support** - Works with Gemini, Claude, ChatGPT, and Ollama (local)
- **Character System** - Each character has unique personality and speaking style
- **Memory System** - Remembers past conversations within a session
- **Offline Capability** - Full offline operation with local models
- **Bilingual Support** - Interface available in Japanese and English

### Feature Availability

| Feature | Free (Forever) | Extended (Future) |
|---------|---------------|-------------------|
| Core voice chat | ✅ Full | ✅ Full |
| GUI application | ✅ Full | ✅ Full |
| Basic characters (3) | ✅ | ✅ |
| Additional characters | ❌ | ✅ More variety |
| Memory duration | Session | Extended |
| Emotion types | Basic | More nuanced |
| Commercial use | ❌ | ✅ |
| Support | Community | Priority |

**Note**: The free version will always remain fully functional. Extended features are optional additions, not restrictions.

### Requirements

- Windows 11 (Windows 10 support planned)
- Python 3.11.9 or higher
- 8GB RAM minimum (16GB recommended)
- Microphone for voice input

### Installation

```bash
# Clone the repository
git clone https://github.com/MizuiroDeep/vioratalk-mindwave-mate.git
cd vioratalk-mindwave-mate

# Install dependencies
poetry install

# Run the application
poetry run python -m vioratalk
```

### Configuration (BYOK - Bring Your Own Key)

VioraTalk uses a BYOK model - you provide your own API keys:

1. **Initial Setup**: The setup wizard will guide you on first launch
2. **API Keys**: You'll need to obtain your own keys from:
   - **Gemini**: [Google AI Studio](https://makersuite.google.com/app/apikey)
   - **Claude**: [Anthropic Console](https://console.anthropic.com/)
   - **ChatGPT**: [OpenAI Platform](https://platform.openai.com/)
   - **Ollama**: No key needed (runs locally)
3. **Cost Control**: You manage your own API usage and costs directly

### Usage

1. Launch the application
2. Choose your character
3. Select your AI model
4. Hold spacebar to talk (Push-to-Talk)
5. Release to send your message

### Available Characters

The free version includes three characters:
- **Aoi** - Cheerful and energetic assistant
- **Haru** - Calm and thoughtful companion
- **Yui** - Kind and supportive friend

### Contributing

This is primarily a personal learning project. While I appreciate interest in contributing:

- ⭐ **Stars** are welcome and encouraging!
- 💬 **Discussions** for ideas and questions are great
- 🐛 **Bug reports** help me learn and improve
- 🔀 **Pull requests** are generally not accepted (I need to understand all code myself)

Please see [CONTRIBUTING.md](docs/CONTRIBUTING.md) for details.

### Testing

The project maintains comprehensive test coverage:

```bash
# Run all tests
poetry run pytest

# Run with coverage report
poetry run pytest --cov=vioratalk --cov-report=html
```

Current status: 305 tests, 86.52% coverage

### Frequently Asked Questions

**Q: Will the free version always be available?**
Yes. The core features will always be free and open source. This is a firm commitment.

**Q: Why BYOK (Bring Your Own Key)?**
This gives you full control over your costs and data. No subscription fees, no surprises.

**Q: When will extended features be available?**
They're planned for Phase 13+ (late 2025/early 2026), but the timeline may change as I'm learning.

**Q: Can I use this commercially?**
The free version is for personal use. Commercial use will require a future commercial license.

**Q: Why aren't you accepting PRs?**
As a learning project, I need to understand every line of code. I appreciate the interest though!

### License

MIT License - see [LICENSE](LICENSE) for details.

### Contact

- **GitHub**: [@MizuiroDeep](https://github.com/MizuiroDeep)
- **Issues**: [Bug Reports](https://github.com/MizuiroDeep/vioratalk-mindwave-mate/issues)
- **Discussions**: [Ideas & Questions](https://github.com/MizuiroDeep/vioratalk-mindwave-mate/discussions)

---

## Japanese

### このプロジェクトについて

VioraTalkは、私がClaude AIの助けを借りながら開発している音声アシスタントプロジェクトです。Pythonはまだ勉強中ですが、実用的なものを作りながら、適切な開発手法を学んでいます。

目指しているのは、自然に話しかけられるデスクトップアシスタント。キャラクターごとに個性があり、過去の会話も覚えていてくれる、そんなアプリケーションです。

### プロジェクトのビジョンとロードマップ

このプロジェクトは透明な開発アプローチを採用しています：

**現在の状況：**
- Phase 1（基盤実装）完了 ✅
- Phase 2（対話システム）開発中 🚧
- 学習しながら公開開発

**開発方針：**
- **永続的無料提供**：基本機能は永続的に無料でオープンソース
- **BYOKモデル**：API利用料は自己管理で透明性確保
- **学習優先**：コード品質は学習とともに向上

**将来の計画：**
基本機能は常に無料で提供し続けますが、長期的な開発を維持するため、将来的に（Phase 13以降）オプションの拡張機能を計画しています。無料版は常に完全に機能します。

### 機能

VioraTalkの特徴：

- **音声対話** - スペースキーを押している間だけ音声認識（Push-to-Talk）
- **複数AI対応** - Gemini、Claude、ChatGPT、Ollama（ローカル）に対応
- **キャラクターシステム** - それぞれ個性的な性格と話し方
- **記憶システム** - セッション内での会話を記憶
- **オフライン動作** - ローカルモデルで完全オフライン動作可能
- **日英対応** - インターフェースは日本語と英語に対応

### 機能の提供範囲

| 機能 | 無料版（永続） | 拡張版（将来） |
|------|--------------|---------------|
| 基本音声対話 | ✅ 完全提供 | ✅ 完全提供 |
| GUIアプリケーション | ✅ 完全提供 | ✅ 完全提供 |
| 基本キャラクター（3体） | ✅ | ✅ |
| 追加キャラクター | ❌ | ✅ より多様に |
| 記憶の持続期間 | セッション中 | より長期間 |
| 感情表現の種類 | 基本的 | より豊かに |
| 商用利用 | ❌ | ✅ |
| サポート | コミュニティ | 優先対応 |

**注記**：無料版は常に完全に機能します。拡張機能は制限ではなく、オプションの追加です。

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

# アプリケーションの起動
poetry run python -m vioratalk
```

### 設定（BYOK - 自分のAPIキーを使用）

VioraTalkはBYOKモデルを採用 - APIキーは自分で用意します：

1. **初期設定**：初回起動時にセットアップウィザードが案内
2. **APIキーの取得**：以下から各自で取得してください：
   - **Gemini**: [Google AI Studio](https://makersuite.google.com/app/apikey)
   - **Claude**: [Anthropic Console](https://console.anthropic.com/)
   - **ChatGPT**: [OpenAI Platform](https://platform.openai.com/)
   - **Ollama**: 不要（ローカル実行）
3. **コスト管理**：API利用料は直接管理で透明性確保

### 使い方

1. アプリケーションを起動
2. キャラクターを選択
3. AIモデルを選択
4. スペースキーを押しながら話す（Push-to-Talk）
5. 離すとメッセージ送信

### 利用可能なキャラクター

無料版には3体のキャラクターが含まれます：
- **あおい** - 明るく元気なアシスタント
- **はる** - 落ち着いた思慮深い相棒
- **ゆい** - 優しくサポートしてくれる友達

### 貢献について

これは主に個人の学習プロジェクトです。貢献への関心は嬉しいですが：

- ⭐ **スター**は歓迎です！励みになります
- 💬 **ディスカッション**でのアイデアや質問は大歓迎
- 🐛 **バグ報告**は学習と改善に役立ちます
- 🔀 **プルリクエスト**は基本的に受け付けていません（すべてのコードを自分で理解する必要があるため）

詳細は[CONTRIBUTING.md](docs/CONTRIBUTING.md)をご覧ください。

### テスト

プロジェクトは包括的なテストカバレッジを維持しています：

```bash
# すべてのテストを実行
poetry run pytest

# カバレッジレポート付きで実行
poetry run pytest --cov=vioratalk --cov-report=html
```

現在の状況：305個のテスト、カバレッジ86.52%

### よくある質問

**Q: 無料版は永続的に利用できますか？**
はい。基本機能は永続的に無料でオープンソースです。これは確固たる約束です。

**Q: なぜBYOK（自分のキーを使用）なのですか？**
コストとデータを完全に自己管理できます。サブスクリプション料金も、予期しない請求もありません。

**Q: 拡張機能はいつ利用可能になりますか？**
Phase 13以降（2025年後半〜2026年前半）を予定していますが、学習中のため時期は変動する可能性があります。

**Q: 商用利用はできますか？**
無料版は個人利用向けです。商用利用には将来的に商用ライセンスが必要になります。

**Q: なぜPRを受け付けないのですか？**
学習プロジェクトとして、すべてのコードを理解する必要があるためです。関心を持っていただけることは嬉しいです！

### ライセンス

MITライセンス - 詳細は[LICENSE](LICENSE)をご覧ください。

### 連絡先

- **GitHub**: [@MizuiroDeep](https://github.com/MizuiroDeep)
- **Issues**: [バグ報告](https://github.com/MizuiroDeep/vioratalk-mindwave-mate/issues)
- **Discussions**: [アイデアと質問](https://github.com/MizuiroDeep/vioratalk-mindwave-mate/discussions)

---

<div align="center">

🌟 **Thank you for your interest in VioraTalk!** 🌟

VioraTalkに興味を持っていただきありがとうございます！

</div>