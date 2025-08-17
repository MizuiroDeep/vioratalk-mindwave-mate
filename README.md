# VioraTalk - MindWave Mate

AIキャラクターとの自然な音声対話を実現するデスクトップアシスタント

An AI Voice Assistant with Emotional Expression

## 🌟 特徴 / Features

- 🎤 **音声対話** - Push-to-Talk方式による自然な会話
- 🤖 **複数AI対応** - Gemini, Claude, ChatGPT, Ollama (ローカルLLM)
- 😊 **感情表現** - キャラクターが感情を持って応答
- 🧠 **記憶システム** - 継続的な対話が可能
- 🔌 **オフライン対応** - インターネット接続なしでも動作

## 📋 必要環境 / Requirements

- Windows 11
- Python 3.11.9+
- マイク（音声入力用）

## 🚀 インストール / Installation

```bash
# リポジトリのクローン
git clone https://github.com/MizuiroDeep/vioratalk-mindwave-mate.git
cd vioratalk-mindwave-mate

# 依存関係のインストール
poetry install

# アプリケーションの起動
poetry run python -m vioratalk
```

## 🔧 設定 / Configuration

初回起動時に自動セットアップウィザードが起動します。

### APIキーの設定（BYOK方式）

各AIサービスのAPIキーは自分で用意する必要があります：

- **Gemini**: [Google AI Studio](https://makersuite.google.com/app/apikey)
- **Claude**: [Anthropic Console](https://console.anthropic.com/)
- **ChatGPT**: [OpenAI Platform](https://platform.openai.com/)
- **Ollama**: ローカル実行のため不要

## 📖 使い方 / Usage

1. アプリケーションを起動
2. キャラクターを選択
3. Spaceキーを押しながら話す（Push-to-Talk）
4. AIキャラクターが音声で応答

## 🎭 利用可能なキャラクター / Available Characters

### 無料版
- 001_aoi - 明るく元気な性格
- 002_haru - 落ち着いた知的な性格  
- 003_yui - 優しく共感的な性格

### Pro版（予定）
- 004_pro_mira - プロフェッショナルアシスタント
- その他多数

## 🛠️ 開発状況 / Development Status

現在Phase 0（プロジェクト準備）実施中

- [x] Phase 0: プロジェクト準備
- [ ] Phase 1: コア基盤の実装
- [ ] Phase 2: 対話システム構築
- [ ] Phase 3-4: 基本機能実装
- [ ] Phase 5-6: 会話管理とテスト
- [ ] Phase 7-9: キャラクター実装
- [ ] Phase 10: GUI実装
- [ ] Phase 11-14: 拡張機能

## 📝 ライセンス / License

MIT License - 詳細は[LICENSE](LICENSE)ファイルを参照

## 🤝 貢献 / Contributing

貢献を歓迎します！詳細は[CONTRIBUTING.md](docs/CONTRIBUTING.md)を参照してください。

## 📧 連絡先 / Contact

- GitHub: [@MizuiroDeep](https://github.com/MizuiroDeep)
- Issues: [GitHub Issues](https://github.com/MizuiroDeep/vioratalk-mindwave-mate/issues)

---

© 2025 VioraTalk Team. All rights reserved.
```