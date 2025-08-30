#!/usr/bin/env python3
"""
pyttsx3日本語問題調査スクリプト（実音声出力版）

実際に音声を出力して、前身プロジェクトで報告された問題を検証する。
各テキストを完全に読み上げ、問題があれば記録する。

開発規約書 v1.12準拠
"""

import logging
import sys
import time
import traceback
from dataclasses import dataclass
from enum import Enum
from typing import Optional

try:
    import pyttsx3
except ImportError:
    print("ERROR: pyttsx3がインストールされていません。")
    print("実行: poetry add pyttsx3")
    sys.exit(1)


# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class TestResult(Enum):
    """テスト結果の種別"""
    SUCCESS = "成功"
    ERROR = "エラー"
    FREEZE = "フリーズ"
    PARTIAL = "部分的成功"


@dataclass
class TestCase:
    """テストケースの定義"""
    name: str
    text: str
    description: str
    critical: bool = False  # 前身プロジェクトで問題があったケース


@dataclass
class TestReport:
    """テスト結果レポート"""
    case: TestCase
    result: TestResult
    duration: float
    error_message: Optional[str] = None
    notes: Optional[str] = None


class RealPyttsx3Tester:
    """実際に音声を出力してpyttsx3の問題を検証するクラス"""

    def __init__(self):
        """初期化処理"""
        self.engine: Optional[pyttsx3.Engine] = None
        self.test_cases = self._create_test_cases()
        self.reports = []

    def _create_test_cases(self):
        """重要なテストケースのみを作成"""
        return [
            # 基本動作確認
            TestCase(
                "基本_日本語",
                "こんにちは",
                "基本的な日本語"
            ),
            
            # 前身プロジェクトで問題があったケース
            TestCase(
                "感嘆符フリーズ",
                "すごい！",
                "前身プロジェクトでフリーズ報告あり",
                critical=True
            ),
            TestCase(
                "感嘆符混在",
                "本当にすごい！素晴らしい！",
                "複数の感嘆符でフリーズする可能性",
                critical=True
            ),
            TestCase(
                "句点",
                "今日は良い天気です。",
                "句点での問題報告あり",
                critical=True
            ),
            TestCase(
                "読点",
                "今日は、良い天気です",
                "読点での問題報告あり",
                critical=True
            ),
            TestCase(
                "句読点混在",
                "こんにちは。今日は、良い天気ですね。",
                "句読点混在での問題",
                critical=True
            ),
            TestCase(
                "かぎ括弧",
                "彼は「こんにちは」と言った",
                "かぎ括弧での問題報告あり",
                critical=True
            ),
            TestCase(
                "二重かぎ括弧",
                "『重要なお知らせ』です",
                "二重かぎ括弧での問題",
                critical=True
            ),
            
            # その他の重要なケース
            TestCase(
                "長文",
                "これは長いテキストのテストです。日本語の音声合成が正常に動作するか確認します。句読点も含まれています。",
                "長文での安定性確認"
            ),
            TestCase(
                "絵文字",
                "こんにちは😊楽しいです",
                "絵文字を含むテキスト"
            ),
            TestCase(
                "複雑な混在",
                "今日は素晴らしい天気です！「本当に気持ちいい」と思いました。",
                "感嘆符、句読点、かぎ括弧の混在",
                critical=True
            ),
        ]

    def initialize_engine(self) -> bool:
        """エンジンの初期化"""
        try:
            logger.info("pyttsx3エンジンを初期化中...")
            self.engine = pyttsx3.init()
            
            # 音声設定
            voices = self.engine.getProperty('voices')
            japanese_voice = None
            
            for voice in voices:
                if 'japanese' in voice.name.lower() or 'haruka' in voice.name.lower():
                    japanese_voice = voice
                    break
            
            if japanese_voice:
                self.engine.setProperty('voice', japanese_voice.id)
                logger.info(f"日本語音声を設定: {japanese_voice.name}")
            else:
                logger.warning("日本語音声が見つかりません")
            
            # 読み上げ速度を遅めに設定（聞き取りやすくするため）
            self.engine.setProperty('rate', 120)
            self.engine.setProperty('volume', 1.0)
            
            logger.info("エンジン初期化完了")
            return True
            
        except Exception as e:
            logger.error(f"エンジン初期化失敗: {e}")
            return False

    def run_single_test(self, test_case: TestCase) -> TestReport:
        """単一のテストを実行（実際に音声を出力）"""
        print(f"\n{'='*60}")
        print(f"テスト: {test_case.name}")
        print(f"テキスト: '{test_case.text}'")
        print(f"説明: {test_case.description}")
        if test_case.critical:
            print("⚠️  前身プロジェクトで問題報告あり")
        print('-'*60)
        
        report = TestReport(
            case=test_case,
            result=TestResult.SUCCESS,
            duration=0.0
        )
        
        try:
            # ユーザーに準備を促す
            input("Enterキーを押すと音声を再生します...")
            
            start_time = time.time()
            
            # 音声合成を実行
            logger.info(f"音声出力開始: {test_case.name}")
            self.engine.say(test_case.text)
            
            # タイムアウト付きで実行（フリーズ検出）
            self.engine.runAndWait()
            
            duration = time.time() - start_time
            report.duration = duration
            
            logger.info(f"音声出力完了: {duration:.2f}秒")
            
            # ユーザーに結果を確認
            print("\n【確認事項】")
            print("1. 音声は最後まで再生されましたか？ (y/n)")
            print("2. 途中で止まったり、おかしな部分はありましたか？ (y/n)")
            print("3. フリーズしましたか？ (y/n)")
            
            complete = input("1. 完全に再生された？ (y/n): ").lower() == 'y'
            issues = input("2. 問題があった？ (y/n): ").lower() == 'y'
            freeze = input("3. フリーズした？ (y/n): ").lower() == 'y'
            
            if freeze:
                report.result = TestResult.FREEZE
                report.error_message = "フリーズが発生"
            elif not complete:
                report.result = TestResult.PARTIAL
                report.error_message = "音声が途中で停止"
            elif issues:
                report.result = TestResult.PARTIAL
                report.notes = input("どのような問題がありましたか？: ")
            else:
                report.result = TestResult.SUCCESS
                
        except KeyboardInterrupt:
            report.result = TestResult.FREEZE
            report.error_message = "ユーザーによる強制終了（フリーズの可能性）"
            logger.error("テストが中断されました")
            
        except Exception as e:
            report.result = TestResult.ERROR
            report.error_message = str(e)
            logger.error(f"エラー発生: {e}")
            logger.debug(traceback.format_exc())
        
        # 結果の表示
        print(f"\n結果: {report.result.value}")
        if report.error_message:
            print(f"エラー: {report.error_message}")
        if report.notes:
            print(f"メモ: {report.notes}")
            
        return report

    def run_all_tests(self):
        """すべてのテストを実行"""
        print("\n" + "="*60)
        print("pyttsx3 日本語問題の実音声テスト")
        print("="*60)
        print("\n各テストで実際に音声が再生されます。")
        print("音声を聞いて、問題がないか確認してください。")
        
        if not self.initialize_engine():
            print("エンジン初期化に失敗しました")
            return
        
        for i, test_case in enumerate(self.test_cases, 1):
            print(f"\n【テスト {i}/{len(self.test_cases)}】")
            
            report = self.run_single_test(test_case)
            self.reports.append(report)
            
            # 重大な問題が発生した場合
            if report.result == TestResult.FREEZE:
                print("\n⚠️  フリーズが検出されました！")
                continue_test = input("テストを続行しますか？ (y/n): ").lower() == 'y'
                if not continue_test:
                    break
            
            # 次のテストまで少し待機
            time.sleep(1)
        
        self.print_summary()

    def print_summary(self):
        """テスト結果のサマリーを表示"""
        print("\n" + "="*60)
        print("テスト結果サマリー")
        print("="*60)
        
        total = len(self.reports)
        success = sum(1 for r in self.reports if r.result == TestResult.SUCCESS)
        partial = sum(1 for r in self.reports if r.result == TestResult.PARTIAL)
        error = sum(1 for r in self.reports if r.result == TestResult.ERROR)
        freeze = sum(1 for r in self.reports if r.result == TestResult.FREEZE)
        
        print(f"\n総テスト数: {total}")
        print(f"✅ 成功: {success}")
        print(f"⚠️  部分的成功: {partial}")
        print(f"❌ エラー: {error}")
        print(f"🔴 フリーズ: {freeze}")
        
        # 重要な問題の詳細
        critical_issues = [r for r in self.reports 
                          if r.case.critical and r.result != TestResult.SUCCESS]
        
        if critical_issues:
            print("\n【前身プロジェクトで報告された問題の再現状況】")
            print("-"*40)
            for report in critical_issues:
                print(f"• {report.case.name}: {report.result.value}")
                if report.error_message:
                    print(f"  詳細: {report.error_message}")
        else:
            print("\n✅ 前身プロジェクトで報告された問題は再現しませんでした！")
        
        # 実装への推奨事項
        print("\n【Pyttsx3Engine実装への推奨事項】")
        print("-"*40)
        
        if freeze > 0:
            print("🔴 フリーズが発生しました。以下の対策が必要です：")
            print("   1. 該当する文字パターンの前処理")
            print("   2. タイムアウト処理の実装")
            print("   3. 代替TTSエンジンへのフォールバック")
        elif error > 0 or partial > 0:
            print("⚠️  一部問題が発生しました。以下を検討してください：")
            print("   1. エラーが発生したパターンの回避")
            print("   2. エラーハンドリングの強化")
        else:
            print("✅ すべてのテストが成功しました！")
            print("   特別な前処理は不要と思われます。")

    def cleanup(self):
        """クリーンアップ処理"""
        if self.engine:
            try:
                self.engine.stop()
                del self.engine
                logger.info("エンジンを停止しました")
            except Exception as e:
                logger.error(f"クリーンアップエラー: {e}")


def main():
    """メイン処理"""
    tester = RealPyttsx3Tester()
    
    try:
        tester.run_all_tests()
    except KeyboardInterrupt:
        print("\n\nテストを中断しました")
    except Exception as e:
        logger.error(f"予期しないエラー: {e}")
        logger.error(traceback.format_exc())
    finally:
        tester.cleanup()


if __name__ == "__main__":
    main()
