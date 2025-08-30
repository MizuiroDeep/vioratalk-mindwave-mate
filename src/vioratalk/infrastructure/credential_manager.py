"""APIキーと認証情報の安全な管理

BYOK（Bring Your Own Key）方式でAPIキーを管理。
Phase 4では基本実装、Phase 8-9で完全実装予定。

エラーハンドリング指針 v1.20準拠
セキュリティ実装ガイド v1.5準拠
設定ファイル完全仕様書 v1.2準拠
インターフェース定義書 v1.34準拠（IAPIKeyManager簡易実装）
開発規約書 v1.12準拠
"""

import logging
import os
from pathlib import Path
from typing import Dict, Optional

import yaml

from ..core.exceptions import ConfigurationError

logger = logging.getLogger(__name__)


class APIKeyManager:
    """APIキーの安全な管理（BYOK対応）

    Phase 4実装：環境変数とYAMLファイルからの読み込み
    将来実装：OSキーリング、暗号化保存（Phase 8で実装予定）

    優先順位：
        1. 環境変数（最優先）
        2. ~/.vioratalk/api_keys.yaml（ホームディレクトリ）
        3. user_settings/api_keys.yaml（プロジェクトディレクトリ）

    Attributes:
        keys: 読み込まれたAPIキーの辞書
        _cache_valid_until: キャッシュ有効期限（将来実装用）

    Note:
        インターフェース定義書v1.34のIAPIKeyManagerの簡易実装。
        Phase 9で完全なインターフェースに準拠予定。
    """

    SERVICE_NAME = "VioraTalk"

    # サポートするサービス一覧（セキュリティ実装ガイド v1.5準拠）
    SUPPORTED_SERVICES = {
        # LLM
        "claude",
        "gemini",
        "openai",
        "chatgpt",
        # STT
        "faster_whisper",
        # TTS（pyttsx3はローカルなのでAPIキー不要）
        "edge_tts",
        "aivisspeech",
        # その他
        "ollama_base_url",
    }

    # 環境変数マッピング（設定ファイル完全仕様書 v1.2準拠）
    ENV_MAPPING = {
        "claude": "CLAUDE_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "openai": "OPENAI_API_KEY",
        "chatgpt": "OPENAI_API_KEY",  # OpenAIと同じ
        "ollama_base_url": "OLLAMA_BASE_URL",
        "faster_whisper": "FASTER_WHISPER_API_KEY",  # 将来用
        "edge_tts": "EDGE_TTS_API_KEY",  # 将来用
        "aivisspeech": "AIVISSPEECH_API_KEY",  # 将来用
    }

    def __init__(self):
        """APIKeyManagerの初期化

        Note:
            VioraTalkComponentは継承しない（インフラ層のため）
        """
        self.keys: Dict[str, str] = {}
        self._cache_valid_until: Optional[float] = None
        self._load_api_keys()
        logger.debug(f"APIKeyManager initialized with {len(self.keys)} keys")

    def _load_api_keys(self) -> None:
        """APIキーを環境変数とYAMLから読み込み

        優先順位（低→高）：
            1. user_settings/api_keys.yaml
            2. ~/.vioratalk/api_keys.yaml
            3. 環境変数（最優先）
        """
        # 1. プロジェクトディレクトリのYAML
        project_yaml = Path("user_settings/api_keys.yaml")
        if project_yaml.exists():
            logger.debug(f"Loading API keys from {project_yaml}")
            self._load_from_yaml(project_yaml)

        # 2. ホームディレクトリのYAML
        home_yaml = Path.home() / ".vioratalk" / "api_keys.yaml"
        if home_yaml.exists():
            logger.debug(f"Loading API keys from {home_yaml}")
            self._load_from_yaml(home_yaml)

        # 3. 環境変数（最優先）
        self._load_from_environment()

    def _load_from_yaml(self, yaml_path: Path) -> None:
        """YAMLファイルからAPIキーを読み込み

        Args:
            yaml_path: YAMLファイルのパス

        YAMLフォーマット（設定ファイル完全仕様書 v1.2準拠）:
            llm:
              claude_api_key: "sk-..."
              gemini_api_key: "..."
              openai_api_key: "..."
              ollama_base_url: "http://localhost:11434"
        """
        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}

            # LLMキー
            llm_keys = data.get("llm", {})
            for service in ["claude", "gemini", "openai"]:
                key = llm_keys.get(f"{service}_api_key")
                if key and isinstance(key, str) and key.strip():
                    self.keys[service] = key.strip()
                    logger.debug(f"Loaded {service} key from {yaml_path.name}")

            # OpenAI キーは ChatGPT でも使用
            if "openai" in self.keys:
                self.keys["chatgpt"] = self.keys["openai"]

            # Ollama URL
            if ollama_url := llm_keys.get("ollama_base_url"):
                if isinstance(ollama_url, str) and ollama_url.strip():
                    self.keys["ollama_base_url"] = ollama_url.strip()
                    logger.debug(f"Loaded Ollama URL from {yaml_path.name}")

            # STT/TTSキー（将来実装用）
            stt_keys = data.get("stt", {})
            tts_keys = data.get("tts", {})
            # 現時点では読み込むが使用しない

        except yaml.YAMLError as e:
            logger.warning(f"Invalid YAML format in {yaml_path}: {e}")
        except PermissionError:
            logger.warning(f"Permission denied: {yaml_path}")
        except Exception as e:
            logger.warning(f"Failed to load {yaml_path}: {e}")

    def _load_from_environment(self) -> None:
        """環境変数からAPIキーを読み込み（最優先）

        開発規約書 v1.12準拠：DEBUGレベルでログ出力
        """
        for service, env_var in self.ENV_MAPPING.items():
            if env_value := os.getenv(env_var):
                if env_value.strip():
                    self.keys[service] = env_value.strip()
                    logger.debug(f"Loaded {service} from environment variable {env_var}")

    def get_api_key(self, service: str) -> Optional[str]:
        """APIキーを取得

        Args:
            service: サービス名（'claude', 'gemini', 'openai'等）

        Returns:
            APIキー（未設定の場合None）

        Note:
            IAPIKeyManagerインターフェースのget_api_key()簡易実装
        """
        return self.keys.get(service)

    def require_api_key(self, service: str) -> str:
        """APIキーを取得（必須）

        Args:
            service: サービス名

        Returns:
            APIキー

        Raises:
            ConfigurationError: APIキーが未設定（E2003）
        """
        api_key = self.get_api_key(service)
        if not api_key:
            env_var = self.ENV_MAPPING.get(service, f"{service.upper()}_API_KEY")
            error_msg = (
                f"APIキーが設定されていません: {service}。"
                f"環境変数 {env_var} "
                "または user_settings/api_keys.yaml で設定してください。"
            )
            logger.error(f"[E2003] {error_msg}")
            raise ConfigurationError(error_msg, error_code="E2003")
        return api_key

    def validate_api_keys(self) -> Dict[str, bool]:
        """設定されたAPIキーの有効性をチェック

        Returns:
            サービス名と設定状態のマッピング

        Raises:
            ConfigurationError: 最低1つのLLMキーが必要（E2003）

        Note:
            IAPIKeyManagerインターフェースのvalidate_keys()簡易実装
        """
        results = {}

        # LLMキーチェック
        for llm in ["claude", "gemini", "openai", "chatgpt"]:
            results[llm] = llm in self.keys

        # Ollamaもチェック
        results["ollama"] = "ollama_base_url" in self.keys

        # 最低1つのLLMキーが必要
        llm_available = any(
            [
                results.get("claude", False),
                results.get("gemini", False),
                results.get("openai", False),
                results.get("chatgpt", False),
                results.get("ollama", False),
            ]
        )

        if not llm_available:
            error_msg = (
                "LLM APIキーが設定されていません。"
                "以下のいずれかの方法でAPIキーを設定してください：\n"
                "1. 環境変数（例：CLAUDE_API_KEY）\n"
                "2. ~/.vioratalk/api_keys.yaml\n"
                "3. user_settings/api_keys.yaml"
            )
            logger.error(f"[E2003] {error_msg}")
            raise ConfigurationError(error_msg, error_code="E2003")

        logger.info(f"API keys validation passed: {sum(results.values())} keys available")
        return results

    def mask_api_key(self, api_key: str) -> str:
        """APIキーをマスキング（ログ出力用）

        Args:
            api_key: マスキングするAPIキー

        Returns:
            マスキングされたAPIキー

        Examples:
            >>> manager = APIKeyManager()
            >>> manager.mask_api_key("sk-1234567890abcdef")
            'sk-12...def'
            >>> manager.mask_api_key("short")
            '***'
        """
        if not api_key:
            return "***"

        if len(api_key) < 8:
            return "***"

        # 先頭と末尾の文字を残してマスキング
        if len(api_key) < 20:
            visible_chars = 2
        else:
            visible_chars = 4

        return f"{api_key[:visible_chars]}...{api_key[-visible_chars:]}"

    def get_summary(self) -> Dict[str, str]:
        """設定されているAPIキーのサマリーを取得（デバッグ用）

        Returns:
            マスキングされたAPIキーのサマリー

        開発規約書 v1.12準拠：センシティブ情報をマスキング
        """
        summary = {}
        for service in self.keys:
            if service == "ollama_base_url":
                # URLはマスキングしない
                summary[service] = self.keys[service]
            else:
                summary[service] = self.mask_api_key(self.keys[service])
        return summary

    def refresh(self) -> None:
        """APIキー情報を再読み込み

        Note:
            環境変数が変更された場合などに使用
        """
        logger.info("Refreshing API keys...")
        self.keys.clear()
        self._load_api_keys()
        logger.info(f"API keys refreshed: {len(self.keys)} keys loaded")

    def has_service_key(self, service: str) -> bool:
        """特定サービスのAPIキーが設定されているか確認

        Args:
            service: サービス名

        Returns:
            APIキーが設定されている場合True
        """
        return service in self.keys

    def get_available_services(self) -> list[str]:
        """APIキーが設定されているサービスのリストを取得

        Returns:
            利用可能なサービス名のリスト
        """
        return list(self.keys.keys())


# シングルトンインスタンス管理
_api_key_manager: Optional[APIKeyManager] = None


def get_api_key_manager() -> APIKeyManager:
    """APIKeyManagerのシングルトンインスタンスを取得

    Returns:
        APIKeyManagerインスタンス

    Note:
        アプリケーション全体で単一のインスタンスを共有
    """
    global _api_key_manager
    if _api_key_manager is None:
        _api_key_manager = APIKeyManager()
    return _api_key_manager


# 公開API（開発規約書 v1.12準拠）
__all__ = [
    "APIKeyManager",
    "get_api_key_manager",
]
