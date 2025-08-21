# Security Policy / セキュリティポリシー

## Supported Versions / サポート対象バージョン

As this is a personal learning project in active development, only the latest version receives security updates.

これは開発中の個人学習プロジェクトのため、最新バージョンのみがセキュリティ更新の対象です。

| Version | Supported / サポート |
| ------- | -------------------- |
| 0.1.x   | ✅ Yes / はい |
| < 0.1   | ❌ No / いいえ |

## Reporting a Vulnerability / 脆弱性の報告

### How to Report / 報告方法

If you discover a security vulnerability, please report it through one of these methods:

セキュリティ脆弱性を発見した場合は、以下のいずれかの方法で報告してください：

1. **GitHub Security Advisory** (Preferred / 推奨)
   - Go to the Security tab in the repository
   - Click "Report a vulnerability"
   - リポジトリのSecurityタブにアクセス
   - 「Report a vulnerability」をクリック

2. **Private Issue** (Alternative / 代替)
   - Create a new issue with `[SECURITY]` in the title
   - Include details but avoid public disclosure
   - タイトルに`[SECURITY]`を含めて新しいissueを作成
   - 詳細を含めますが、公開情報は避けてください

### What to Include / 含めるべき情報

Please provide:
以下の情報を提供してください：

- Description of the vulnerability / 脆弱性の説明
- Steps to reproduce / 再現手順
- Potential impact / 潜在的な影響
- Suggested fix (if any) / 修正案（あれば）
- Your contact information (optional) / 連絡先（任意）

## Response Approach / 対応方針

As a solo developer working on this project in my spare time, I will make every effort to respond promptly:

個人開発者として余暇に取り組んでいるため、可能な限り迅速に対応するよう努めます：

- **Initial response**: As soon as possible / 初期対応：可能な限り早く
- **Assessment**: Based on severity and complexity / 評価：深刻度と複雑さに基づく
- **Fix development**: Priority given to critical issues / 修正開発：重要な問題を優先
- **Communication**: Regular updates on progress / コミュニケーション：進捗の定期的な更新

I appreciate your understanding that response times may vary depending on the issue complexity and my learning curve.

問題の複雑さと私の学習進度により対応時間が変動することをご理解いただければ幸いです。

## Security Considerations / セキュリティに関する考慮事項

### Current Status / 現在の状況

This project is in early development (Phase 1-2) and should be considered:

このプロジェクトは初期開発段階（Phase 1-2）であり、以下の点にご注意ください：

- **Experimental**: Not yet suitable for production use / 実験的：本番環境での使用には適していません
- **Learning-focused**: Security best practices are being learned and applied gradually / 学習重視：セキュリティのベストプラクティスを段階的に学習・適用中
- **BYOK Model**: API keys are user-managed, reducing some security responsibilities / BYOKモデル：APIキーはユーザー管理のため、一部のセキュリティ責任が軽減

### Security Improvements in Progress / セキュリティ改善の取り組み

I'm continuously working to enhance security:

セキュリティの向上に継続的に取り組んでいます：

1. **Secure Configuration**: Improving configuration management / 安全な設定：設定管理の改善
2. **Input Validation**: Strengthening validation mechanisms / 入力検証：検証メカニズムの強化
3. **Error Handling**: Refining error messages for better security / エラー処理：セキュリティ向上のためのメッセージ改善

## Security Best Practices for Users / ユーザーのためのセキュリティベストプラクティス

To use VioraTalk safely:
VioraTalkを安全に使用するために：

### API Key Management / APIキー管理

- Never commit API keys to version control / APIキーをバージョン管理にコミットしない
- Use environment variables when possible / 可能な場合は環境変数を使用
- Regularly rotate your API keys / 定期的にAPIキーを更新
- Monitor your API usage and billing / API使用量と請求を監視

### General Precautions / 一般的な注意事項

- Run only on trusted networks / 信頼できるネットワークでのみ実行
- Keep the software updated / ソフトウェアを最新に保つ
- Review code changes before updating / 更新前にコード変更を確認
- Report suspicious behavior immediately / 不審な動作は直ちに報告

## Learning Together / 一緒に学ぶ

As this is a learning project, I welcome educational discussions about security:

これは学習プロジェクトのため、セキュリティに関する教育的な議論を歓迎します：

- Security best practices suggestions / セキュリティベストプラクティスの提案
- Code review feedback / コードレビューのフィードバック
- Educational resources / 教育リソース
- Constructive criticism / 建設的な批判

Your patience and understanding as I learn to handle security properly is greatly appreciated.

セキュリティを適切に扱えるよう学習する中で、皆様の忍耐とご理解に深く感謝します。

## Acknowledgments / 謝辞

I'm grateful to anyone who takes the time to:

以下の時間を割いてくださる方々に感謝します：

- Report vulnerabilities responsibly / 責任を持って脆弱性を報告
- Provide security guidance / セキュリティガイダンスを提供
- Share educational resources / 教育リソースを共有
- Help improve the project's security / プロジェクトのセキュリティ改善を支援

Contributors who report valid security issues will be acknowledged here (with permission):

有効なセキュリティ問題を報告してくださった方は、許可を得てここに記載します：

- *No reports yet / まだ報告はありません*

---

## Important Notice / 重要なお知らせ

This is an early-stage learning project. While I strive to follow security best practices, please:

これは初期段階の学習プロジェクトです。セキュリティのベストプラクティスに従うよう努めていますが：

- Use in development environments only / 開発環境でのみ使用してください
- Evaluate risks before any production use / 本番使用前にリスクを評価してください
- Keep your API keys secure / APIキーを安全に管理してください
- Report any security concerns / セキュリティ上の懸念があれば報告してください

Your understanding and support in improving security together is greatly appreciated.

一緒にセキュリティを改善していくご理解とサポートに深く感謝します。

---

<div align="center">

**Thank you for helping make VioraTalk more secure!**

**VioraTalkをより安全にするためのご協力に感謝します！**

</div>