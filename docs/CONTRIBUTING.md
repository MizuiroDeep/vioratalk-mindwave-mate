# Contributing to VioraTalk

We welcome contributions to VioraTalk! / VioraTalkへの貢献を歓迎します！

## How to Contribute / 貢献方法

### 1. Fork & Clone
```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR-USERNAME/vioratalk-mindwave-mate.git
cd vioratalk-mindwave-mate
```

### 2. Create a Branch / ブランチ作成
```bash
git checkout -b feature/your-feature-name
```

### 3. Make Changes / 変更作業
- Follow PEP 8 style guide
- Write tests for new features
- Update documentation

### 4. Commit / コミット
```bash
# Use conventional commits format
git commit -m "feat: add new feature"
git commit -m "fix: resolve issue"
git commit -m "docs: update README"
```

### 5. Push & Create Pull Request
```bash
git push origin feature/your-feature-name
```

## Coding Standards / コーディング規約

- **Python**: PEP 8準拠
- **Docstrings**: Google Style
- **Type Hints**: 必須
- **Tests**: pytest使用

## Commit Message Format / コミットメッセージ

```
<type>(<scope>): <subject>

<body>

<footer>
```

Types:
- `feat`: 新機能
- `fix`: バグ修正
- `docs`: ドキュメント
- `style`: フォーマット
- `refactor`: リファクタリング
- `test`: テスト
- `chore`: その他

## Questions? / 質問

Please open an issue on GitHub!
```