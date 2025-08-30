#!/usr/bin/env python3
"""
pyttsx3日本語問題調査スクリプト

前身プロジェクトで確認されていたpyttsx3の日本語処理問題を体系的に調査する。
句読点、日本語特有文字、その他のパターンでのエラー発生条件を特定し、
回避策を検討する。

開発規約書 v1.12準拠
"""

import logging
import sys
import traceback
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

try:
    import pyttsx3
except ImportError:
    print("ERROR: pyttsx3がインストールされていません。")
    print("実行: poetry add pyttsx3")
    sys.exit(1)


# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("pyttsx3_investigation.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


class TestResult(Enum):
    """テスト結果の種別"""

    SUCCESS = "成功"
    ERROR = "エラー"
    WARNING = "警告"
    SKIPPED = "スキップ"


@dataclass
class TestCase:
    """テストケースの定義"""

    name: str
    text: str
    category: str
    expected_issue: Optional[str] = None


@dataclass
class TestReport:
    """テスト結果レポート"""

    case: TestCase
    result: TestResult
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    workaround: Optional[str] = None
    details: Dict[str, any] = field(default_factory=dict)


class Pyttsx3Investigator:
    """pyttsx3の日本語問題を調査するクラス"""

    def __init__(self) -> None:
        """初期化処理"""
        self.engine: Optional[pyttsx3.Engine] = None
        self.test_cases: List[TestCase] = self._create_test_cases()
        self.reports: List[TestReport] = []
        self.available_voices: List[Dict[str, str]] = []

    def _create_test_cases(self) -> List[TestCase]:
        """テストケースの作成

        Returns:
            テストケースのリスト
        """
        return [
            # 基本動作確認
            TestCase("基本_ひらがな", "こんにちは", "基本動作"),
            TestCase("基本_カタカナ", "コンニチハ", "基本動作"),
            TestCase("基本_漢字", "今日は", "基本動作"),
            TestCase("基本_英数字", "Hello123", "基本動作"),
            # 句読点問題（前身プロジェクトで確認済み）
            TestCase("句点のみ", "こんにちは。", "句読点", "エラー発生の可能性"),
            TestCase("読点のみ", "こんにちは、", "句読点", "エラー発生の可能性"),
            TestCase("句読点混在", "こんにちは。今日は、良い天気です。", "句読点", "エラー発生の可能性"),
            TestCase("連続句点", "はい。。。", "句読点", "エラー発生の可能性"),
            TestCase("連続読点", "えーっと、、、", "句読点", "エラー発生の可能性"),
            # 日本語特殊文字（前身プロジェクトで確認済み）
            TestCase("かぎ括弧", "「こんにちは」", "特殊文字", "エラー発生の可能性"),
            TestCase("二重かぎ括弧", "『こんにちは』", "特殊文字", "エラー発生の可能性"),
            TestCase("中点", "東京・大阪", "特殊文字", "エラー発生の可能性"),
            TestCase("長音符", "コーヒー", "特殊文字"),
            TestCase("感嘆符", "すごい！", "特殊文字"),
            TestCase("疑問符", "本当？", "特殊文字"),
            TestCase("波線", "そう〜", "特殊文字"),
            TestCase("三点リーダー", "そうですね…", "特殊文字"),
            # 文字種混在
            TestCase("日英混在", "今日はGoodな天気", "混在"),
            TestCase("全角半角混在", "ＡＢＣとABC", "混在"),
            TestCase("記号混在", "価格：¥1,000", "混在"),
            # エッジケース
            TestCase("空文字列", "", "エッジケース"),
            TestCase("スペースのみ", "   ", "エッジケース"),
            TestCase("改行含む", "こんにちは\n今日は", "エッジケース"),
            TestCase("タブ含む", "こんにちは\t今日は", "エッジケース"),
            TestCase("長文", "これは長いテキストのテストです。" * 20, "エッジケース"),
            # Unicode特殊文字
            TestCase("絵文字", "こんにちは😊", "Unicode", "エラー発生の可能性"),
            TestCase("特殊記号", "※注意事項", "Unicode"),
            TestCase("丸数字", "①②③", "Unicode"),
        ]

    def initialize_engine(self) -> bool:
        """pyttsx3エンジンの初期化

        Returns:
            初期化成功の可否
        """
        try:
            logger.info("pyttsx3エンジンを初期化中...")
            self.engine = pyttsx3.init()

            # 利用可能な音声を取得
            voices = self.engine.getProperty("voices")
            for voice in voices:
                voice_info = {
                    "id": voice.id,
                    "name": voice.name,
                    "languages": getattr(voice, "languages", []),
                    "gender": getattr(voice, "gender", "unknown"),
                }
                self.available_voices.append(voice_info)
                logger.info(f"利用可能な音声: {voice.name} ({voice.id})")

            # 日本語音声を探す
            japanese_voice = self._find_japanese_voice()
            if japanese_voice:
                self.engine.setProperty("voice", japanese_voice["id"])
                logger.info(f"日本語音声を設定: {japanese_voice['name']}")
            else:
                logger.warning("日本語音声が見つかりません。デフォルト音声を使用します。")

            # 基本プロパティ設定
            self.engine.setProperty("rate", 150)  # 読み上げ速度
            self.engine.setProperty("volume", 0.8)  # 音量

            logger.info("エンジン初期化完了")
            return True

        except Exception as e:
            logger.error(f"エンジン初期化失敗: {e}")
            logger.error(traceback.format_exc())
            return False

    def _find_japanese_voice(self) -> Optional[Dict[str, str]]:
        """日本語音声を探す

        Returns:
            日本語音声の情報、見つからない場合はNone
        """
        for voice in self.available_voices:
            # 日本語キーワードを含む音声を探す
            keywords = ["japanese", "japan", "jp", "ja", "haruka", "zira"]
            voice_str = f"{voice['name']} {voice['id']}".lower()

            if any(keyword in voice_str for keyword in keywords):
                return voice

        return None

    def run_test(self, test_case: TestCase) -> TestReport:
        """個別テストケースの実行

        Args:
            test_case: テストケース

        Returns:
            テスト結果レポート
        """
        logger.info(f"テスト実行: {test_case.name} - '{test_case.text}'")
        report = TestReport(case=test_case, result=TestResult.SKIPPED)

        if not self.engine:
            report.result = TestResult.ERROR
            report.error_message = "エンジンが初期化されていません"
            return report

        # 空文字列やスペースのみの場合はスキップ
        if not test_case.text or test_case.text.isspace():
            report.result = TestResult.WARNING
            report.error_message = "空またはスペースのみのテキスト"
            logger.warning(f"  ⚠ 警告: {test_case.name} - 空のテキストはスキップ")
            return report

        try:
            # テキストを音声合成キューに追加
            self.engine.say(test_case.text)

            # 実際に音声合成を実行（音声は出力しない設定も可能）
            self.engine.runAndWait()

            report.result = TestResult.SUCCESS
            logger.info(f"  ✓ 成功: {test_case.name}")

        except UnicodeDecodeError as e:
            report.result = TestResult.ERROR
            report.error_type = "UnicodeDecodeError"
            report.error_message = str(e)
            report.workaround = "文字エンコーディングの変換が必要"
            logger.error(f"  ✗ Unicode エラー: {e}")

        except RuntimeError as e:
            report.result = TestResult.ERROR
            report.error_type = "RuntimeError"
            report.error_message = str(e)
            report.workaround = "エンジンの再初期化が必要な可能性"
            logger.error(f"  ✗ ランタイムエラー: {e}")

        except Exception as e:
            report.result = TestResult.ERROR
            report.error_type = type(e).__name__
            report.error_message = str(e)
            logger.error(f"  ✗ 予期しないエラー: {e}")
            logger.debug(traceback.format_exc())

        return report

    def test_workarounds(self, failed_case: TestCase) -> List[TestReport]:
        """失敗したケースの回避策をテスト

        Args:
            failed_case: 失敗したテストケース

        Returns:
            回避策のテスト結果リスト
        """
        workaround_reports = []
        original_text = failed_case.text

        # 回避策1: 句読点をスペースに置換
        if any(char in original_text for char in ["。", "、"]):
            modified_text = original_text.replace("。", " ").replace("、", " ")
            workaround_case = TestCase(
                name=f"{failed_case.name}_句読点スペース置換",
                text=modified_text,
                category="回避策",
            )
            report = self.run_test(workaround_case)
            report.workaround = "句読点をスペースに置換"
            workaround_reports.append(report)

        # 回避策2: 句読点を削除
        if any(char in original_text for char in ["。", "、", "「」", "『』"]):
            modified_text = (
                original_text.replace("。", "")
                .replace("、", "")
                .replace("「", "")
                .replace("」", "")
                .replace("『", "")
                .replace("』", "")
            )
            workaround_case = TestCase(
                name=f"{failed_case.name}_特殊文字削除", text=modified_text, category="回避策"
            )
            report = self.run_test(workaround_case)
            report.workaround = "特殊文字を削除"
            workaround_reports.append(report)

        # 回避策3: ASCII文字のみに変換（最終手段）
        try:
            ascii_text = original_text.encode("ascii", "ignore").decode("ascii")
            if ascii_text:
                workaround_case = TestCase(
                    name=f"{failed_case.name}_ASCII変換", text=ascii_text, category="回避策"
                )
                report = self.run_test(workaround_case)
                report.workaround = "ASCII文字のみに変換"
                workaround_reports.append(report)
        except Exception:
            pass

        return workaround_reports

    def run_investigation(self) -> None:
        """調査の実行"""
        logger.info("=" * 60)
        logger.info("pyttsx3 日本語問題調査を開始します")
        logger.info("=" * 60)

        # エンジン初期化
        if not self.initialize_engine():
            logger.error("エンジン初期化に失敗したため、調査を中止します")
            return

        # 各テストケースを実行
        failed_cases = []
        for test_case in self.test_cases:
            report = self.run_test(test_case)
            self.reports.append(report)

            if report.result == TestResult.ERROR:
                failed_cases.append(test_case)

        # 失敗したケースの回避策をテスト
        if failed_cases:
            logger.info("\n" + "=" * 60)
            logger.info("回避策のテスト")
            logger.info("=" * 60)

            for failed_case in failed_cases:
                workaround_reports = self.test_workarounds(failed_case)
                self.reports.extend(workaround_reports)

        # 結果のサマリーを出力
        self.print_summary()

        # 推奨事項を出力
        self.print_recommendations()

    def print_summary(self) -> None:
        """調査結果のサマリーを出力"""
        logger.info("\n" + "=" * 60)
        logger.info("調査結果サマリー")
        logger.info("=" * 60)

        # カテゴリ別の集計
        category_stats: Dict[str, Dict[str, int]] = {}
        for report in self.reports:
            category = report.case.category
            if category not in category_stats:
                category_stats[category] = {"success": 0, "error": 0, "warning": 0, "skipped": 0}

            if report.result == TestResult.SUCCESS:
                category_stats[category]["success"] += 1
            elif report.result == TestResult.ERROR:
                category_stats[category]["error"] += 1
            elif report.result == TestResult.WARNING:
                category_stats[category]["warning"] += 1
            else:
                category_stats[category]["skipped"] += 1

        # カテゴリ別結果を出力
        print("\n【カテゴリ別結果】")
        print("-" * 40)
        for category, stats in category_stats.items():
            total = sum(stats.values())
            success_rate = (stats["success"] / total * 100) if total > 0 else 0
            print(f"{category:15} : 成功 {stats['success']:3d} / エラー {stats['error']:3d} " f"(成功率: {success_rate:.1f}%)")

        # エラーパターンの分析
        error_patterns: Dict[str, List[str]] = {}
        for report in self.reports:
            if report.result == TestResult.ERROR:
                error_type = report.error_type or "Unknown"
                if error_type not in error_patterns:
                    error_patterns[error_type] = []
                error_patterns[error_type].append(report.case.text)

        if error_patterns:
            print("\n【エラーパターン分析】")
            print("-" * 40)
            for error_type, texts in error_patterns.items():
                print(f"{error_type}:")
                for text in texts[:5]:  # 最初の5件まで表示
                    print(f"  - '{text}'")

        # 成功した回避策
        successful_workarounds = [r for r in self.reports if r.case.category == "回避策" and r.result == TestResult.SUCCESS]

        if successful_workarounds:
            print("\n【有効な回避策】")
            print("-" * 40)
            for report in successful_workarounds:
                print(f"  ✓ {report.workaround}")

    def print_recommendations(self) -> None:
        """推奨事項を出力"""
        logger.info("\n" + "=" * 60)
        logger.info("推奨事項")
        logger.info("=" * 60)

        recommendations = []

        # エラーパターンから推奨事項を生成
        has_punctuation_error = any(
            r.result == TestResult.ERROR and r.case.category == "句読点" for r in self.reports
        )

        has_special_char_error = any(
            r.result == TestResult.ERROR and r.case.category == "特殊文字" for r in self.reports
        )

        if has_punctuation_error:
            recommendations.append("【句読点処理】音声合成前に句読点をスペースまたは改行に置換することを推奨")

        if has_special_char_error:
            recommendations.append("【特殊文字処理】かぎ括弧等の特殊文字は事前に除去または置換を推奨")

        if not self._find_japanese_voice():
            recommendations.append("【日本語音声】Windows日本語音声パック（Microsoft Haruka等）のインストールを推奨")

        recommendations.append("【フォールバック】pyttsx3でエラーが発生した場合、他のTTSエンジンへの切り替えを実装")

        recommendations.append("【文字正規化】音声合成前にテキストの正規化処理を実装することを強く推奨")

        print("\n【Pyttsx3Engine実装への推奨事項】")
        print("-" * 40)
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")

        # 実装例のコード提示
        print("\n【推奨される前処理実装例】")
        print("-" * 40)
        print("""
def preprocess_text_for_tts(text: str) -> str:
    \"\"\"pyttsx3用のテキスト前処理\"\"\"
    # 句読点の置換
    text = text.replace("。", " ").replace("、", " ")
    
    # 特殊文字の除去
    special_chars = ["「」", "『』", "【】", "〈〉", "《》"]
    for chars in special_chars:
        text = text.replace(chars[0], " ").replace(chars[1], " ")
    
    # 連続スペースを単一スペースに
    text = " ".join(text.split())
    
    return text.strip()
        """)

    def cleanup(self) -> None:
        """クリーンアップ処理"""
        if self.engine:
            try:
                # 残っているキューをクリア
                self.engine.stop()
                # エンジンを適切に終了
                del self.engine
                self.engine = None
                logger.info("エンジンを停止しました")
            except Exception as e:
                logger.error(f"エンジン停止時エラー: {e}")


def main() -> None:
    """メイン処理"""
    investigator = Pyttsx3Investigator()

    try:
        investigator.run_investigation()
    except KeyboardInterrupt:
        logger.info("\n調査を中断しました")
    except Exception as e:
        logger.error(f"予期しないエラー: {e}")
        logger.error(traceback.format_exc())
    finally:
        investigator.cleanup()

        # ログファイルの場所を表示
        print("\n" + "=" * 60)
        print("詳細なログは pyttsx3_investigation.log を参照してください")
        print("=" * 60)


if __name__ == "__main__":
    main()
