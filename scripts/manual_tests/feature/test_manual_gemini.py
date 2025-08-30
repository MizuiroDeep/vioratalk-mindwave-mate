#!/usr/bin/env python3
"""
手動テスト: GeminiEngine実動作確認（Phase 4仕様準拠版）
実行方法: python scripts/manual_tests/feature/test_manual_gemini.py

Phase 4では会話履歴機能は実装範囲外のため、ステートレスなテストのみ実施。
レート制限を考慮し、実使用に近い間隔でテストを実行。

テスト戦略ガイドライン v1.7準拠
開発規約書 v1.12準拠
"""

import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import argparse

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

try:
    from vioratalk.core.llm.gemini_engine import GeminiEngine, AVAILABLE_MODELS, DEFAULT_MODEL
    from vioratalk.core.llm.base import LLMConfig
    from vioratalk.core.exceptions import (
        LLMError,
        APIError,
        RateLimitError,
        AuthenticationError,
        ConfigurationError,
        ModelNotFoundError
    )
    from vioratalk.infrastructure.credential_manager import APIKeyManager
    from vioratalk.utils.logger_manager import LoggerManager
except ImportError as e:
    print(f"Error: 必要なモジュールをインポートできません: {e}")
    print("プロジェクトルートが正しく設定されているか確認してください。")
    sys.exit(1)


# ============================================================================
# 定数定義
# ============================================================================

# レート制限対策：リクエスト間の待機時間（秒）
REQUEST_INTERVAL = 3.0  # Gemini APIのレート制限を考慮

# テストシナリオ定義
TEST_SCENARIOS = {
    "basic": {
        "name": "基本動作テスト",
        "tests": [
            {
                "prompt": "こんにちは。今日はいい天気ですね。",
                "expected_keywords": ["こんにちは", "天気"],
                "max_tokens": 100,
                "description": "日本語の挨拶と応答"
            },
            {
                "prompt": "What is the capital of Japan?",
                "expected_keywords": ["Tokyo", "東京"],
                "max_tokens": 50,
                "description": "英語での事実質問"
            },
            {
                "prompt": "1から10までの数字を足すといくつになりますか？計算過程も教えてください。",
                "expected_keywords": ["55"],
                "max_tokens": 200,
                "description": "計算問題と説明"
            }
        ]
    },
    "creative": {
        "name": "創造的生成テスト",
        "tests": [
            {
                "prompt": "春をテーマにした俳句を1つ作ってください。",
                "expected_keywords": None,  # 創造的な内容なので特定のキーワードは期待しない
                "max_tokens": 100,
                "temperature": 0.9,
                "description": "創造的な日本語生成（高温度）"
            },
            {
                "prompt": "AIアシスタントの未来について、3つのポイントで説明してください。",
                "expected_keywords": None,
                "max_tokens": 300,
                "temperature": 0.7,
                "description": "構造化された説明文生成"
            }
        ]
    },
    "multilingual": {
        "name": "多言語対応テスト",
        "tests": [
            {
                "prompt": "Hello! 日本語で返答してください。あなたの機能を簡単に説明してください。",
                "expected_keywords": ["アシスタント", "お手伝い", "質問"],
                "max_tokens": 150,
                "description": "言語切り替え指示"
            },
            {
                "prompt": "「ありがとう」を英語、中国語、韓国語で教えてください。",
                "expected_keywords": ["Thank", "谢谢", "감사"],
                "max_tokens": 100,
                "description": "多言語翻訳"
            }
        ]
    },
    "system_prompt": {
        "name": "システムプロンプトテスト",
        "tests": [
            {
                "prompt": "今日の天気はどうですか？",
                "system_prompt": "あなたは気象予報士です。専門的な用語を使って説明してください。",
                "expected_keywords": None,
                "max_tokens": 150,
                "description": "専門的な役割設定"
            },
            {
                "prompt": "プログラミングについて教えてください。",
                "system_prompt": "あなたは5歳の子供に説明する先生です。とても簡単な言葉を使ってください。",
                "expected_keywords": None,
                "max_tokens": 150,
                "description": "簡単な説明への変換"
            },
            {
                "prompt": "おすすめの本を教えてください。",
                "system_prompt": "あなたは図書館司書です。ジャンルごとに整理して紹介してください。",
                "expected_keywords": None,
                "max_tokens": 200,
                "description": "構造化された応答"
            }
        ]
    }
}

# パフォーマンス基準値（秒）
PERFORMANCE_TARGETS = {
    "short_response": 2.0,   # 短い応答
    "medium_response": 3.0,  # 中程度の応答
    "long_response": 5.0,    # 長い応答
    "streaming_start": 1.5,  # ストリーミング開始
}


# ============================================================================
# テストレポートクラス
# ============================================================================

class TestReport:
    """テスト結果レポート管理"""
    
    def __init__(self):
        self.timestamp = datetime.now()
        self.results = {
            "configuration": {},
            "scenarios": {},
            "performance": {},
            "errors": [],
            "summary": {}
        }
        
    def add_configuration(self, config: Dict[str, Any]):
        """設定情報を追加"""
        self.results["configuration"] = config
        
    def add_scenario_result(self, scenario_name: str, result: Dict[str, Any]):
        """シナリオ結果を追加"""
        if scenario_name not in self.results["scenarios"]:
            self.results["scenarios"][scenario_name] = []
        self.results["scenarios"][scenario_name].append(result)
        
    def add_performance(self, metric: str, value: float):
        """パフォーマンス測定値を追加"""
        if metric not in self.results["performance"]:
            self.results["performance"][metric] = []
        self.results["performance"][metric].append(value)
        
    def add_error(self, error: Dict[str, Any]):
        """エラー情報を追加"""
        self.results["errors"].append(error)
        
    def generate_summary(self):
        """サマリーを生成"""
        total_tests = 0
        passed_tests = 0
        
        for scenario_results in self.results["scenarios"].values():
            for test in scenario_results:
                total_tests += 1
                if test.get("success", False):
                    passed_tests += 1
                    
        # パフォーマンスサマリー
        perf_summary = {}
        for metric, values in self.results["performance"].items():
            if values:
                perf_summary[metric] = {
                    "avg": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values)
                }
                
        self.results["summary"] = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "success_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0,
            "error_count": len(self.results["errors"]),
            "performance_summary": perf_summary,
            "test_duration": (datetime.now() - self.timestamp).total_seconds()
        }
        
    def save_json(self, filepath: Path):
        """JSON形式で保存"""
        self.generate_summary()
        
        # 日時をISO形式に変換
        def json_encoder(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            raise TypeError(f"Type {type(obj)} not serializable")
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2, default=json_encoder)
            
    def print_summary(self):
        """コンソールにサマリーを表示"""
        self.generate_summary()
        
        print("\n" + "=" * 60)
        print("GeminiEngine テスト結果サマリー（Phase 4仕様準拠版）")
        print("=" * 60)
        
        summary = self.results["summary"]
        print(f"\n[全体結果]")
        print(f"  総テスト数: {summary['total_tests']}")
        print(f"  成功: {summary['passed_tests']}")
        print(f"  成功率: {summary['success_rate']:.1f}%")
        print(f"  エラー: {summary['error_count']}")
        print(f"  実行時間: {summary['test_duration']:.1f}秒")
        
        # シナリオ別結果
        print(f"\n[シナリオ別結果]")
        for scenario_name, tests in self.results["scenarios"].items():
            passed = sum(1 for t in tests if t.get("success", False))
            print(f"  {scenario_name}: {passed}/{len(tests)} 成功")
            
        # パフォーマンス
        if summary["performance_summary"]:
            print(f"\n[パフォーマンス]")
            for metric, stats in summary["performance_summary"].items():
                print(f"  {metric}:")
                print(f"    平均: {stats['avg']:.2f}秒")
                print(f"    最速: {stats['min']:.2f}秒 / 最遅: {stats['max']:.2f}秒")


# ============================================================================
# テスタークラス
# ============================================================================

class GeminiManualTester:
    """GeminiEngine手動テストツール（Phase 4仕様準拠）"""
    
    def __init__(self, debug: bool = False):
        self.engine: Optional[GeminiEngine] = None
        self.api_key_manager = APIKeyManager()
        self.logger_manager = LoggerManager()
        self.logger = self.logger_manager.get_logger("gemini_tester")
        self.report = TestReport()
        self.debug = debug
        
        if debug:
            import logging
            logging.basicConfig(level=logging.DEBUG)
            
    def check_api_key_configuration(self) -> Dict[str, Any]:
        """APIキー設定状況の確認"""
        print("\n[APIキー設定確認中...]")
        
        config_info = {"available": False}
        
        # 環境変数チェック
        import os
        if os.environ.get("GEMINI_API_KEY"):
            config_info["source"] = "環境変数"
            config_info["available"] = True
            print("✓ 環境変数: 設定済み")
        else:
            print("✗ 環境変数: 未設定")
            
        # YAMLファイルチェック
        yaml_paths = [
            Path.home() / ".vioratalk" / "api_keys.yaml",
            Path("user_settings/api_keys.yaml")
        ]
        
        for yaml_path in yaml_paths:
            if yaml_path.exists() and not config_info["available"]:
                try:
                    import yaml
                    with open(yaml_path, 'r', encoding='utf-8') as f:
                        data = yaml.safe_load(f)
                        if data and "llm" in data and "gemini_api_key" in data["llm"]:
                            config_info["source"] = str(yaml_path)
                            config_info["available"] = True
                            print(f"✓ {yaml_path}: 設定済み")
                            break
                except Exception:
                    pass
                    
        if not config_info["available"]:
            print("\n⚠️ APIキーが設定されていません")
            print("以下のいずれかの方法で設定してください：")
            print("1. 環境変数: export GEMINI_API_KEY='your-key'")
            print("2. YAMLファイル: user_settings/api_keys.yaml")
            
        return config_info
        
    async def initialize_engine(self, model: Optional[str] = None) -> bool:
        """エンジンの初期化"""
        try:
            print("\n[GeminiEngine初期化中...]")
            
            api_key = self.api_key_manager.get_api_key('gemini')
            if not api_key:
                raise ConfigurationError("Gemini APIキーが設定されていません")
                
            config = LLMConfig(
                engine="gemini",
                model=model or DEFAULT_MODEL,
                temperature=0.7,
                max_tokens=1000
            )
            
            self.engine = GeminiEngine(config=config, api_key=api_key)
            await self.engine.initialize()
            
            print(f"✓ 初期化成功")
            print(f"  モデル: {self.engine.current_model}")
            print(f"  最大トークン: {self.engine.get_max_tokens()}")
            
            return True
            
        except Exception as e:
            print(f"✗ 初期化失敗: {e}")
            self.report.add_error({
                "type": "initialization",
                "error": str(e)
            })
            return False
            
    async def test_scenario(self, scenario_name: str, scenario_data: Dict) -> bool:
        """シナリオテストの実行"""
        print(f"\n[{scenario_data['name']}]")
        print("-" * 40)
        
        scenario_success = True
        
        for i, test in enumerate(scenario_data["tests"], 1):
            print(f"\nテスト {i}: {test['description']}")
            print(f"プロンプト: {test['prompt'][:50]}...")
            
            # レート制限対策の待機
            if i > 1:
                print(f"  (待機中: {REQUEST_INTERVAL}秒...)")
                await asyncio.sleep(REQUEST_INTERVAL)
                
            try:
                start_time = time.perf_counter()
                
                # テスト実行
                response = await self.engine.generate(
                    prompt=test["prompt"],
                    system_prompt=test.get("system_prompt"),
                    temperature=test.get("temperature", 0.7),
                    max_tokens=test.get("max_tokens", 150)
                )
                
                elapsed = time.perf_counter() - start_time
                
                # 結果の評価
                test_result = {
                    "description": test["description"],
                    "success": bool(response.content),
                    "response_length": len(response.content) if response.content else 0,
                    "elapsed_time": elapsed
                }
                
                # キーワードチェック（指定されている場合）
                if test.get("expected_keywords") and response.content:
                    found_keywords = []
                    for keyword in test["expected_keywords"]:
                        if keyword in response.content:
                            found_keywords.append(keyword)
                    
                    test_result["keywords_found"] = found_keywords
                    test_result["keywords_match"] = len(found_keywords) > 0
                    
                    if found_keywords:
                        print(f"✓ キーワード検出: {', '.join(found_keywords)}")
                    else:
                        print(f"⚠ 期待されるキーワードが見つかりません")
                        
                # パフォーマンス評価
                if test.get("max_tokens", 0) <= 100:
                    target = PERFORMANCE_TARGETS["short_response"]
                elif test.get("max_tokens", 0) <= 200:
                    target = PERFORMANCE_TARGETS["medium_response"]
                else:
                    target = PERFORMANCE_TARGETS["long_response"]
                    
                if elapsed <= target:
                    print(f"✓ 応答時間: {elapsed:.2f}秒 (目標: {target}秒以内)")
                else:
                    print(f"⚠ 応答時間: {elapsed:.2f}秒 (目標: {target}秒超過)")
                    
                # レスポンス内容の表示（常に最初の150文字を表示）
                if response.content:
                    display_length = 150 if not self.debug else 300
                    content_preview = response.content[:display_length]
                    if len(response.content) > display_length:
                        content_preview += "..."
                    print(f"応答内容: {content_preview}")
                    
                self.report.add_scenario_result(scenario_name, test_result)
                self.report.add_performance(f"{scenario_name}_{i}", elapsed)
                
                if not test_result["success"]:
                    scenario_success = False
                    
            except Exception as e:
                print(f"✗ エラー: {e}")
                self.report.add_scenario_result(scenario_name, {
                    "description": test["description"],
                    "success": False,
                    "error": str(e)
                })
                self.report.add_error({
                    "scenario": scenario_name,
                    "test": test["description"],
                    "error": str(e)
                })
                scenario_success = False
                
        return scenario_success
        
    async def test_streaming(self) -> bool:
        """ストリーミング生成テスト"""
        print("\n[ストリーミング生成テスト]")
        print("-" * 40)
        
        try:
            prompt = "日本の四季について、それぞれの特徴を簡潔に説明してください。"
            print(f"プロンプト: {prompt}")
            
            chunks = []
            start_time = time.perf_counter()
            first_chunk_time = None
            
            print("生成中: ", end="", flush=True)
            async for chunk in self.engine.stream_generate(
                prompt=prompt,
                max_tokens=300,
                temperature=0.7
            ):
                if first_chunk_time is None:
                    first_chunk_time = time.perf_counter() - start_time
                    
                chunks.append(chunk)
                print(".", end="", flush=True)
                
            total_time = time.perf_counter() - start_time
            full_response = "".join(chunks)
            
            print(f"\n✓ ストリーミング完了")
            print(f"  最初のチャンク: {first_chunk_time:.2f}秒")
            print(f"  合計時間: {total_time:.2f}秒")
            print(f"  チャンク数: {len(chunks)}")
            print(f"  生成文字数: {len(full_response)}")
            
            self.report.add_performance("streaming_first_chunk", first_chunk_time)
            self.report.add_performance("streaming_total", total_time)
            
            return True
            
        except Exception as e:
            print(f"✗ ストリーミングエラー: {e}")
            self.report.add_error({
                "type": "streaming",
                "error": str(e)
            })
            return False
            
    async def test_error_handling(self) -> bool:
        """エラーハンドリングテスト"""
        print("\n[エラーハンドリングテスト]")
        print("-" * 40)
        
        test_cases = [
            {
                "name": "無効なモデル",
                "action": lambda: self.engine.set_model("invalid-model"),
                "expected_error": ModelNotFoundError,
                "is_async": False
            },
            {
                "name": "超長文プロンプト",
                "action": lambda: self.engine.generate(
                    prompt="これは非常に長いプロンプトです。" * 10000,
                    max_tokens=10
                ),
                "expected_error": (LLMError, APIError),
                "is_async": True
            }
        ]
        
        success = True
        
        for test in test_cases:
            print(f"\nテスト: {test['name']}")
            
            try:
                if test["is_async"]:
                    await test["action"]()
                else:
                    test["action"]()
                    
                print(f"⚠ エラーが発生しませんでした")
                success = False
                
            except test["expected_error"] as e:
                print(f"✓ 期待通りのエラー: {e.__class__.__name__}")
                if hasattr(e, 'error_code'):
                    print(f"  エラーコード: {e.error_code}")
                    
            except Exception as e:
                print(f"✗ 予期しないエラー: {e}")
                success = False
                
            # レート制限対策
            await asyncio.sleep(REQUEST_INTERVAL)
            
        return success
        
    async def run_all_tests(self, scenarios: Optional[List[str]] = None):
        """すべてのテストを実行"""
        
        # APIキー確認
        config_info = self.check_api_key_configuration()
        if not config_info["available"]:
            print("\nテストを中止します。APIキーを設定してください。")
            return False
            
        self.report.add_configuration(config_info)
        
        # エンジン初期化
        if not await self.initialize_engine():
            return False
            
        # 実行するシナリオを決定
        if scenarios:
            test_scenarios = {k: v for k, v in TEST_SCENARIOS.items() if k in scenarios}
        else:
            test_scenarios = TEST_SCENARIOS
            
        # 各シナリオテスト
        print("\n" + "=" * 60)
        print("シナリオテスト開始")
        print("=" * 60)
        
        for scenario_name, scenario_data in test_scenarios.items():
            await self.test_scenario(scenario_name, scenario_data)
            
        # ストリーミングテスト
        await asyncio.sleep(REQUEST_INTERVAL)
        await self.test_streaming()
        
        # エラーハンドリングテスト
        await asyncio.sleep(REQUEST_INTERVAL)
        await self.test_error_handling()
        
        # クリーンアップ
        if self.engine:
            await self.engine.cleanup()
            
        # レポート出力
        self.report.print_summary()
        
        # JSONレポート保存
        report_dir = Path("logs")
        report_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = report_dir / f"gemini_test_{timestamp}.json"
        self.report.save_json(report_file)
        print(f"\n詳細レポート保存: {report_file}")
        
        return self.report.results["summary"]["success_rate"] >= 70


# ============================================================================
# エントリーポイント
# ============================================================================

async def main():
    """メイン処理"""
    
    parser = argparse.ArgumentParser(
        description="GeminiEngine手動テストツール（Phase 4仕様準拠版）",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=list(TEST_SCENARIOS.keys()),
        help="実行するシナリオを指定"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="デバッグモード（詳細ログと応答内容を300文字まで表示）"
    )
    
    parser.add_argument(
        "--show-response",
        action="store_true",
        help="応答内容を表示（デフォルトで150文字まで表示）"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=f"使用するモデル（デフォルト: {DEFAULT_MODEL}）"
    )
    
    args = parser.parse_args()
    
    # テスター作成と実行
    tester = GeminiManualTester(debug=args.debug)
    
    try:
        success = await tester.run_all_tests(scenarios=args.scenarios)
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\n\nテストを中断しました。")
        return 130
        
    except Exception as e:
        print(f"\n予期しないエラー: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    # Windows環境でのイベントループポリシー設定
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
    # メイン実行
    exit_code = asyncio.run(main())
    sys.exit(exit_code)