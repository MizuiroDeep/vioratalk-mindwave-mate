"""pytest共通設定ファイル

テスト実装ガイド v1.3準拠
テスト戦略ガイドライン v1.7準拠
"""

import sys
import logging
from pathlib import Path
import pytest
import asyncio

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))


# =============================================================================
# ログ設定
# =============================================================================

@pytest.fixture(scope="session", autouse=True)
def setup_test_logging():
    """テスト実行時のログ設定
    
    テスト実装ガイド v1.3 セクション5.1準拠
    """
    # テスト用のログレベル設定
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 特定のロガーのレベル調整
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("pytest").setLevel(logging.WARNING)


# =============================================================================
# 非同期テスト設定
# =============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """非同期テスト用のイベントループ
    
    非同期処理実装ガイド v1.0準拠
    """
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# =============================================================================
# パスフィクスチャ
# =============================================================================

@pytest.fixture(scope="session")
def project_root_path() -> Path:
    """プロジェクトルートパス"""
    return project_root


@pytest.fixture(scope="session")
def src_path(project_root_path) -> Path:
    """srcディレクトリパス"""
    return project_root_path / "src"


@pytest.fixture(scope="session")
def test_data_path(project_root_path) -> Path:
    """テストデータディレクトリパス"""
    test_data = project_root_path / "tests" / "fixtures"
    test_data.mkdir(parents=True, exist_ok=True)
    return test_data


# =============================================================================
# テスト用設定
# =============================================================================

@pytest.fixture
def test_config() -> dict:
    """テスト用の基本設定
    
    テスト実装ガイド v1.3 セクション5.1準拠
    """
    return {
        "general": {
            "app_name": "VioraTalk Test",
            "version": "0.0.1",
            "language": "ja"
        },
        "stt": {
            "engine": "mock",
            "model_size": "base"
        },
        "llm": {
            "engine": "mock",
            "timeout": 30
        },
        "tts": {
            "engine": "mock",
            "speaker_id": 1
        }
    }


# =============================================================================
# Phase 1用フィクスチャ
# =============================================================================

@pytest.fixture
def sample_error_codes() -> dict:
    """エラーコードのサンプル
    
    エラーハンドリング指針 v1.20準拠
    """
    return {
        "config": "E0001",
        "init": "E0100",
        "component": "E1000",
        "stt": "E1001",
        "llm": "E2001",
        "tts": "E3001",
        "character": "E4001",
        "memory": "E4201",
        "resource": "E5300",
        "network": "E5200",
        "service": "E7000",
        "setup": "E8000",
        "humanlike": "E9000"
    }


# =============================================================================
# テストマーカー
# =============================================================================

def pytest_configure(config):
    """pytestのカスタムマーカーを定義
    
    テスト戦略ガイドライン v1.7準拠
    """
    config.addinivalue_line(
        "markers", "unit: 単体テスト"
    )
    config.addinivalue_line(
        "markers", "integration: 統合テスト"
    )
    config.addinivalue_line(
        "markers", "e2e: エンドツーエンドテスト"
    )
    config.addinivalue_line(
        "markers", "slow: 実行に時間がかかるテスト"
    )
    config.addinivalue_line(
        "markers", "manual: 手動確認が必要なテスト"
    )
    config.addinivalue_line(
        "markers", "phase(n): 特定のPhaseでのテスト"
    )


# =============================================================================
# カバレッジ設定
# =============================================================================

@pytest.fixture(scope="session", autouse=True)
def coverage_requirements():
    """Phase別カバレッジ要件
    
    テスト戦略ガイドライン v1.7 セクション14準拠
    """
    return {
        "phase_0": 40,  # 初期段階
        "phase_1": 60,  # 基礎実装
        "phase_2": 60,
        "phase_3": 60,
        "phase_4": 65,
        "phase_5": 65,
        "phase_6": 70,
        "phase_7": 70,
        "phase_8": 75,  # Phase 8での目標
        "phase_9": 75,
        "phase_10": 80,  # 最終目標
        "phase_11": 80,
        "phase_12": 80,
        "phase_13": 80,
        "phase_14": 80
    }