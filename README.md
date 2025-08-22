# VioraTalk - MindWave Mate

[![Release](https://img.shields.io/github/v/release/MizuiroDeep/vioratalk-mindwave-mate)](https://github.com/MizuiroDeep/vioratalk-mindwave-mate/releases)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![Coverage](https://img.shields.io/badge/coverage-86.25%25-brightgreen.svg)](https://github.com/MizuiroDeep/vioratalk-mindwave-mate)
[![Tests](https://img.shields.io/badge/tests-455%20passed-success.svg)](https://github.com/MizuiroDeep/vioratalk-mindwave-mate)

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
- Phase 2 (Dialogue System) completed ✅
- Phase 3 (Mock Implementation) in progress 🚧
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

Current status: 455 tests, 86.25% coverage

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

---

## Japanese

### このプロジェクトについて

VioraTalkは、私がClaude AIの助けを借りて開発しているAI音声アシスタントの個人学習プロジェクトです。Pythonを学習中の身として、実用的なものを作りながら適切な開発手法を学ぶことに焦点を当てて、一歩ずつ構築しています。

目標は、独自の個性を持ち、過去の会話を記憶できるキャラクターと自然に話せるデスクトップアシスタントを作ることです。

### プロジェクトビジョンとロードマップ

このプロジェクトは透明性のある開発アプローチを採用しています：

**現在の状況：**
- Phase 1（コア基盤）完了 ✅
- Phase 2（対話システム）完了 ✅
- Phase 3（Mock実装）進行中 🚧
- 公開しながら学習・開発中

**開発理念：**
- **永久無料コア**: すべての基本機能は永久に無料でオープンソース
- **BYOKモデル**: あなた自身のAPIキーを使用 - コストを完全に管理
- **学習優先**: 学びながらコード品質を向上

**将来の計画：**
コア機能は常に無料のまま維持されますが、長期的な開発を維持するために、将来的（Phase 13以降）にオプションの拡張機能を開発する予定です。無料版は常に完全に機能します。

### 機能

VioraTalkの提供機能：

- **音声対話** - スペースバーを使用したプッシュトゥトーク音声入力
- **複数AI対応** - Gemini、Claude、ChatGPT、Ollama（ローカル）に対応
- **キャラクターシステム** - それぞれ独自の個性と話し方を持つキャラクター
- **記憶システム** - セッション内で過去の会話を記憶
- **オフライン対応** - ローカルモデルによる完全オフライン動作
- **バイリンガル対応** - 日本語と英語のインターフェース

### 機能の提供範囲

| 機能 | 無料版（永久） | 拡張版（将来） |
|------|--------------|---------------|
| コア音声チャット | ✅ 完全機能 | ✅ 完全機能 |
| GUIアプリケーション | ✅ 完全機能 | ✅ 完全機能 |
| 基本キャラクター（3体） | ✅ | ✅ |
| 追加キャラクター | ❌ | ✅ より多様 |
| 記憶の持続期間 | セッション内 | 拡張 |
| 感情タイプ | 基本 | より繊細 |
| 商用利用 | ❌ | ✅ |
| サポート | コミュニティ | 優先サポート |

**注記**: 無料版は常に完全に機能します。拡張機能は制限ではなく、オプションの追加機能です。

### 必要環境

- Windows 11（Windows 10対応予定）
- Python 3.11.9以上
- RAM 8GB以上（16GB推奨）
- 音声入力用マイク

### インストール

```bash
# リポジトリのクローン
git clone https://github.com/MizuiroDeep/vioratalk-mindwave-mate.git
cd vioratalk-mindwave-mate

# 依存関係のインストール
poetry install

# アプリケーションの実行
poetry run python -m vioratalk
```

### 設定（BYOK - 自分のキーを使用）

VioraTalkはBYOKモデルを採用 - あなた自身のAPIキーを提供します：

1. **初期設定**: 初回起動時にセットアップウィザードがガイドします
2. **APIキー**: 以下から自分のキーを取得する必要があります：
   - **Gemini**: [Google AI Studio](https://makersuite.google.com/app/apikey)
   - **Claude**: [Anthropic Console](https://console.anthropic.com/)
   - **ChatGPT**: [OpenAI Platform](https://platform.openai.com/)
   - **Ollama**: キー不要（ローカル動作）
3. **コスト管理**: API使用量とコストを直接管理できます

### 使い方

1. アプリケーションを起動
2. キャラクターを選択
3. AIモデルを選択
4. スペースバーを押しながら話す（プッシュトゥトーク）
5. 離してメッセージを送信

### 利用可能なキャラクター

無料版には3つのキャラクターが含まれます：
- **あおい** - 明るく元気なアシスタント
- **はる** - 落ち着いた思慮深い相棒
- **ゆい** - 優しくサポートしてくれる友人

### 貢献について

これは主に個人の学習プロジェクトです。貢献への関心には感謝しますが：

- ⭐ **スター**は歓迎で励みになります！
- 💬 **ディスカッション**でのアイデアや質問は素晴らしいです
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

現在の状況: 455テスト、86.25%カバレッジ

### よくある質問

**Q: 無料版は常に利用可能ですか？**
はい。コア機能は常に無料でオープンソースです。これは確固たる約束です。

**Q: なぜBYOK（自分のキーを使用）なのですか？**
これによりコストとデータを完全に管理できます。サブスクリプション料金なし、予期しない請求なし。

**Q: 拡張機能はいつ利用可能になりますか？**
Phase 13以降（2025年後半/2026年前半）を予定していますが、学習中のためタイムラインは変更される可能性があります。

**Q: 商用利用はできますか？**
無料版は個人使用向けです。商用利用には将来的な商用ライセンスが必要になります。

**Q: なぜPRを受け付けないのですか？**
学習プロジェクトとして、すべてのコード行を理解する必要があります。でも関心を持っていただけることには感謝しています！

### ライセンス

MITライセンス - 詳細は[LICENSE](LICENSE)をご覧ください。