"""
VioraTalk 環境設定と定数定義

プロジェクト全体で使用される定数、パス定義、環境変数名などを管理。
Phase 1では基本的な設定のみ定義し、Phase毎に拡張していく。

プロジェクト構造説明書 v2.25準拠
開発規約書 v1.12準拠
"""

import os
from pathlib import Path
from typing import Any, Dict

# ============================================================================
# バージョン情報
# ============================================================================

VERSION = "0.1.0"
PHASE = 1
BUILD_DATE = "2025-01-29"
AUTHOR = "MizuiroDeep"

# ============================================================================
# パス定義
# ============================================================================

# プロジェクトルートディレクトリ（settings.pyから4階層上）
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

# 主要ディレクトリ
DATA_DIR = PROJECT_ROOT / "data"
USER_SETTINGS_DIR = PROJECT_ROOT / "user_settings"
MESSAGES_DIR = PROJECT_ROOT / "messages"
LOGS_DIR = PROJECT_ROOT / "logs"
TESTS_DIR = PROJECT_ROOT / "tests"

# データディレクトリ内のサブディレクトリ
MODELS_DIR = DATA_DIR / "models"
CACHE_DIR = DATA_DIR / "cache"
CONVERSATIONS_DIR = DATA_DIR / "conversations"
MEMORY_DIR = DATA_DIR / "memory"

# 設定ファイルパス
DEFAULT_CONFIG_PATH = USER_SETTINGS_DIR / "config.yaml"
API_KEYS_PATH = USER_SETTINGS_DIR / "api_keys.yaml"
LICENSE_PATH = USER_SETTINGS_DIR / "license.dat"

# マーカーファイル
SETUP_COMPLETED_MARKER = DATA_DIR / ".setup_completed"
FIRST_RUN_MARKER = DATA_DIR / ".first_run"

# ============================================================================
# 環境変数名
# ============================================================================

# 設定関連
ENV_CONFIG_PATH = "VIORATALK_CONFIG_PATH"
ENV_LOG_LEVEL = "VIORATALK_LOG_LEVEL"
ENV_DEBUG_MODE = "VIORATALK_DEBUG"

# APIキー関連（BYOK: Bring Your Own Key）
ENV_API_KEY_PREFIX = "VIORATALK_API_"
ENV_CLAUDE_API_KEY = "VIORATALK_API_CLAUDE"
ENV_GEMINI_API_KEY = "VIORATALK_API_GEMINI"
ENV_OPENAI_API_KEY = "VIORATALK_API_OPENAI"

# デバイス関連
ENV_CUDA_VISIBLE_DEVICES = "CUDA_VISIBLE_DEVICES"
ENV_FORCE_CPU = "VIORATALK_FORCE_CPU"

# ============================================================================
# デフォルト設定値
# ============================================================================

# 言語設定
DEFAULT_LANGUAGE = "ja"
SUPPORTED_LANGUAGES = ["ja", "en"]

# ログ設定
DEFAULT_LOG_LEVEL = "INFO"
LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
LOG_FORMAT = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
LOG_ROTATION = "10 MB"
LOG_RETENTION = "30 days"

# タイムアウト設定（秒）
DEFAULT_API_TIMEOUT = 30
DEFAULT_CONNECTION_TIMEOUT = 10
DEFAULT_READ_TIMEOUT = 60

# リトライ設定
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 1.0  # 秒
DEFAULT_RETRY_BACKOFF = 2.0  # 指数バックオフの係数

# メモリ制限
MAX_MEMORY_MB = 2048  # 最大メモリ使用量（MB）
MAX_CACHE_SIZE_MB = 500  # キャッシュサイズ上限（MB）

# ============================================================================
# Phase 1機能制限
# ============================================================================

PHASE_1_FEATURES = {
    "dialogue_enabled": False,  # 対話機能は無効
    "stt_enabled": False,  # 音声認識は無効
    "llm_enabled": False,  # LLMは無効
    "tts_enabled": False,  # 音声合成は無効
    "character_enabled": False,  # キャラクター機能は無効
    "memory_enabled": False,  # 記憶システムは無効
    "emotion_enabled": False,  # 感情分析は無効
    "gui_enabled": False,  # GUIは無効
    "plugin_enabled": False,  # プラグインは無効
    "auto_setup": True,  # 自動セットアップは有効
    "background_service": False,  # バックグラウンドサービスは無効
}

# ============================================================================
# エンジン設定
# ============================================================================

# STTエンジン
DEFAULT_STT_ENGINE = "faster-whisper"
SUPPORTED_STT_ENGINES = ["faster-whisper"]  # Phase 1では1つのみ

# LLMエンジン
DEFAULT_LLM_ENGINE = "gemini"
SUPPORTED_LLM_ENGINES = ["gemini", "claude", "chatgpt", "ollama"]

# TTSエンジン
DEFAULT_TTS_ENGINE = "pyttsx3"
SUPPORTED_TTS_ENGINES = ["pyttsx3", "edge-tts", "windows-sapi", "aivisspeech"]

# ============================================================================
# モデル設定
# ============================================================================

# Whisperモデル
WHISPER_MODEL_SIZES = ["tiny", "base", "small", "medium", "large"]
DEFAULT_WHISPER_MODEL = "base"

# LLMモデル
LLM_MODELS = {
    "gemini": ["gemini-1.5-flash", "gemini-1.5-pro"],
    "claude": ["claude-3-haiku", "claude-3-sonnet", "claude-3-opus"],
    "chatgpt": ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"],
    "ollama": ["llama3", "mistral", "phi3"],
}

# ============================================================================
# キャラクター設定
# ============================================================================

# キャラクターID形式
CHARACTER_ID_PATTERN = r"^\d{3}_[a-z]+$"  # 例: 001_aoi
PRESET_CHARACTER_RANGE = range(1, 100)  # 001-099: プリセット
USER_CHARACTER_RANGE = range(100, 1000)  # 100-999: ユーザー作成

# デフォルトキャラクター
DEFAULT_CHARACTER_ID = "001_aoi"

# ============================================================================
# ネットワーク設定
# ============================================================================

# プロキシ設定（環境変数から取得）
HTTP_PROXY = os.getenv("HTTP_PROXY", "")
HTTPS_PROXY = os.getenv("HTTPS_PROXY", "")
NO_PROXY = os.getenv("NO_PROXY", "localhost,127.0.0.1")

# API設定
API_BASE_URLS = {
    "claude": "https://api.anthropic.com",
    "gemini": "https://generativelanguage.googleapis.com",
    "openai": "https://api.openai.com",
}

# ============================================================================
# ファイルサイズ制限
# ============================================================================

MAX_CONFIG_FILE_SIZE = 1 * 1024 * 1024  # 1MB
MAX_LOG_FILE_SIZE = 10 * 1024 * 1024  # 10MB
MAX_AUDIO_FILE_SIZE = 100 * 1024 * 1024  # 100MB
MAX_MODEL_FILE_SIZE = 5 * 1024 * 1024 * 1024  # 5GB

# ============================================================================
# セキュリティ設定
# ============================================================================

# APIキーのマスク表示
API_KEY_MASK_PATTERN = "sk-...{last4}"
API_KEY_MIN_LENGTH = 20

# ファイル権限（Unix系）
DEFAULT_FILE_PERMISSIONS = 0o644
DEFAULT_DIR_PERMISSIONS = 0o755
SENSITIVE_FILE_PERMISSIONS = 0o600  # APIキーファイルなど

# ============================================================================
# ヘルパー関数
# ============================================================================


def get_env_or_default(env_name: str, default: Any = None) -> Any:
    """環境変数を取得、存在しない場合はデフォルト値を返す

    Args:
        env_name: 環境変数名
        default: デフォルト値

    Returns:
        Any: 環境変数の値またはデフォルト値
    """
    return os.getenv(env_name, default)


def ensure_directories() -> None:
    """必要なディレクトリを作成

    アプリケーション起動時に呼び出して、
    必要なディレクトリ構造を確保する。
    """
    directories = [
        DATA_DIR,
        USER_SETTINGS_DIR,
        MESSAGES_DIR,
        LOGS_DIR,
        MODELS_DIR,
        CACHE_DIR,
        CONVERSATIONS_DIR,
        MEMORY_DIR,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def get_phase_features(phase: int = PHASE) -> Dict[str, bool]:
    """指定されたPhaseで利用可能な機能を取得

    Args:
        phase: Phase番号（デフォルト: 現在のPhase）

    Returns:
        Dict[str, bool]: 機能の有効/無効を示す辞書
    """
    if phase == 1:
        return PHASE_1_FEATURES.copy()

    # Phase 2以降は順次実装
    # Phase 2: dialogue_enabled, stt_enabled, llm_enabled, tts_enabled = True
    # Phase 3: character_enabled, memory_enabled = True
    # ...

    return PHASE_1_FEATURES.copy()


def is_feature_enabled(feature_name: str) -> bool:
    """指定された機能が現在のPhaseで有効か確認

    Args:
        feature_name: 機能名

    Returns:
        bool: 有効な場合True
    """
    features = get_phase_features()
    return features.get(feature_name, False)


def mask_api_key(api_key: str) -> str:
    """APIキーをマスク表示用に変換

    Args:
        api_key: マスクするAPIキー

    Returns:
        str: マスクされたAPIキー（例: "sk-...abcd"）
    """
    if not api_key or len(api_key) < 8:
        return "***"

    return f"{api_key[:3]}...{api_key[-4:]}"
