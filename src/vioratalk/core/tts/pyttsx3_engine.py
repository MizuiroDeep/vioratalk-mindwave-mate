"""
pyttsx3を使用したTTSエンジン実装

このモジュールは、pyttsx3ライブラリを使用した音声合成エンジンを提供します。
Windows環境でのフォールバック用途として設計されています。

Part 29調査結果反映：エンジン再利用不可問題への対応
インターフェース定義書 v1.34準拠
エラーハンドリング指針 v1.20準拠
開発規約書 v1.12準拠
"""

import asyncio
import os
import tempfile
from typing import Any, Dict, List, Optional

try:
    import pyttsx3
except ImportError:
    pyttsx3 = None

from vioratalk.core.base import ComponentState
from vioratalk.core.exceptions import InvalidVoiceError, TTSError
from vioratalk.core.tts.base import BaseTTSEngine, SynthesisResult, TTSConfig, VoiceInfo
from vioratalk.utils.logger_manager import LoggerManager


class Pyttsx3Engine(BaseTTSEngine):
    """
    pyttsx3を使用したTTSエンジン（フォールバック用）

    Windows環境でのエンジン再利用不可問題への対応:
    - synthesize()メソッドで毎回新しいエンジンを作成
    - 使用後は必ず破棄してリソースリークを防ぐ
    - イテレーティブ方式で非ブロッキング実行

    Part 29調査報告書の知見を反映した実装

    Attributes:
        config: TTSエンジン設定
        logger: ロガーインスタンス
        _voice_map: 音声IDと内部IDのマッピング
        _state: コンポーネント状態（VioraTalkComponent準拠）
    """

    # サポートする言語と音声のマッピング
    VOICE_MAP = {
        "ja": {
            "name": "Japanese - Haruka",
            "patterns": ["haruka", "japanese", "ja-jp", "日本"],
            "fallback": "com.apple.speech.synthesis.voice.kyoko",  # macOS用
        },
        "en": {
            "name": "English - Zira",
            "patterns": ["zira", "english", "david", "en-us", "us"],
            "fallback": "com.apple.speech.synthesis.voice.samantha",  # macOS用
        },
    }

    def __init__(self, config: Optional[TTSConfig] = None):
        """初期化

        Args:
            config: TTSエンジン設定

        Raises:
            TTSError: pyttsx3がインストールされていない場合（E3004）
        """
        if pyttsx3 is None:
            raise TTSError(
                "pyttsx3 is not installed. Please install it with: pip install pyttsx3",
                error_code="E3004",
            )

        super().__init__(config)
        self.logger = LoggerManager().get_logger(self.__class__.__name__)
        self._state = ComponentState.NOT_INITIALIZED

        # 初回のエンジン作成で利用可能な音声を取得
        self._initialize_voices()

        self.logger.info(
            "Pyttsx3Engine initialized",
            extra={
                "save_audio_data": self.config.save_audio_data,
                "default_language": self.config.language,
                "available_voices": len(self.available_voices),
            },
        )

    # ============================================================================
    # VioraTalkComponent抽象メソッドの実装
    # ============================================================================

    async def initialize(self) -> None:
        """エンジンの初期化

        VioraTalkComponent準拠の初期化処理。
        状態遷移: NOT_INITIALIZED → INITIALIZING → READY

        Raises:
            TTSError: 初期化に失敗した場合（E3004）
        """
        if self._state == ComponentState.READY:
            return

        self._state = ComponentState.INITIALIZING
        self.logger.debug("Starting Pyttsx3Engine initialization")

        try:
            # 初期化処理は基本的にコンストラクタで完了済み
            # 追加の初期化が必要な場合はここに記述

            # 音声の利用可能性をテスト
            engine = None
            try:
                engine = pyttsx3.init()
                # テスト用の短い合成を試みる
                engine.setProperty("volume", 0)  # 無音でテスト
                engine.say("test")
                # 実際には実行しない（runAndWaitは呼ばない）
            except Exception as e:
                self.logger.warning(f"Voice test during initialization failed: {e}")
                # 初期化失敗とはしない（フォールバック用途のため）
            finally:
                if engine:
                    try:
                        engine.stop()
                    except:
                        pass
                    del engine

            self._state = ComponentState.READY
            self.logger.info("Pyttsx3Engine initialization completed")

        except Exception as e:
            self._state = ComponentState.ERROR
            self.logger.error(f"[E3004] Initialization failed: {e}")
            raise TTSError(f"Failed to initialize Pyttsx3Engine: {str(e)}", error_code="E3004")

    async def cleanup(self) -> None:
        """リソースのクリーンアップ

        VioraTalkComponent準拠のクリーンアップ処理。
        状態遷移: READY/ERROR → TERMINATING → TERMINATED
        """
        if self._state == ComponentState.TERMINATED:
            return

        self._state = ComponentState.TERMINATING
        self.logger.debug("Starting Pyttsx3Engine cleanup")

        try:
            # 特にクリーンアップするリソースはない
            # エンジンは都度作成・破棄しているため
            pass

        except Exception as e:
            self.logger.error(f"Cleanup error: {e}")
            # クリーンアップは失敗してもTERMINATEDにする
        finally:
            self._state = ComponentState.TERMINATED
            self.logger.info("Pyttsx3Engine cleanup completed")

    def is_available(self) -> bool:
        """利用可能状態の確認

        Returns:
            bool: READYまたはRUNNING状態の場合True
        """
        return self._state in [ComponentState.READY, ComponentState.RUNNING]

    def get_status(self) -> Dict[str, Any]:
        """ステータス情報の取得

        Returns:
            Dict[str, Any]: ステータス情報
        """
        return {
            "state": self._state.value,
            "is_available": self.is_available(),
            "error": None,
            "current_voice": self.current_voice_id,
            "available_voices": len(self.available_voices),
            "save_audio_mode": self.config.save_audio_data,
            "language": self.config.language,
        }

    # ============================================================================
    # Pyttsx3Engine固有のメソッド
    # ============================================================================

    def _initialize_voices(self) -> None:
        """利用可能な音声情報を初期化

        Note:
            エンジンは使用後破棄するが、音声情報は保持する
            Part 29調査結果：音声検出が失敗する可能性があるため、
            確実なフォールバック処理を実装
        """
        engine = None
        try:
            engine = pyttsx3.init()
            voices = engine.getProperty("voices")

            self.available_voices = []
            self._voice_id_map = {}  # pyttsx3内部IDとのマッピング

            # voices が None または空の場合の対処
            if not voices:
                self.logger.warning("No voices detected, using fallback configuration")
                self._setup_fallback_voices()
                return

            for voice in voices:
                # 安全な属性アクセス
                voice_name = getattr(voice, "name", "").lower() if hasattr(voice, "name") else ""
                voice_id = getattr(voice, "id", "") if hasattr(voice, "id") else ""

                if not voice_name:
                    continue

                # 言語コードの判定
                lang_code = self._detect_language_from_voice_name(voice_name)
                if lang_code:
                    voice_info = VoiceInfo(
                        id=lang_code,  # シンプルな言語コードをIDとする
                        name=voice.name if hasattr(voice, "name") else f"Voice {lang_code}",
                        language=lang_code,
                        gender=self._detect_gender_from_voice_name(voice_name),
                        metadata={"pyttsx3_id": voice_id},
                    )
                    self.available_voices.append(voice_info)
                    self._voice_id_map[lang_code] = voice_id

            # 同じ言語で複数の音声がある場合の処理
            # 英語音声が複数ある場合（Zira, David）最初の1つだけを"en"として登録
            seen_languages = set()
            filtered_voices = []
            for voice in self.available_voices:
                if voice.language not in seen_languages:
                    filtered_voices.append(voice)
                    seen_languages.add(voice.language)
            self.available_voices = filtered_voices

            # 音声が1つも検出されなかった場合
            if not self.available_voices:
                self.logger.warning("No compatible voices found, using fallback configuration")
                self._setup_fallback_voices()

        except Exception as e:
            self.logger.error(f"Failed to initialize voices: {e}")
            self._setup_fallback_voices()
        finally:
            if engine:
                try:
                    engine.stop()
                except:
                    pass
                del engine

            # デフォルト音声の設定
            self._set_default_voice()

    def _setup_fallback_voices(self) -> None:
        """フォールバック音声情報を設定

        Part 29調査結果を反映：音声検出失敗時でも動作継続可能にする
        """
        self.available_voices = [
            VoiceInfo(id="ja", name="Japanese (Fallback)", language="ja", gender="female"),
            VoiceInfo(id="en", name="English (Fallback)", language="en", gender="female"),
        ]
        self._voice_id_map = {}
        self.logger.info("Fallback voices configured")

    def _set_default_voice(self) -> None:
        """デフォルト音声を設定"""
        if not self.current_voice_id and self.available_voices:
            # 日本語優先、なければ最初の音声
            for voice in self.available_voices:
                if voice.language == "ja":
                    self.current_voice_id = voice.id
                    break
            if not self.current_voice_id:
                self.current_voice_id = self.available_voices[0].id

    def _detect_language_from_voice_name(self, voice_name: str) -> Optional[str]:
        """音声名から言語コードを検出（改善版）

        Args:
            voice_name: 音声名（小文字）

        Returns:
            Optional[str]: 言語コード（"ja"または"en"）、不明な場合None
        """
        # 日本語音声の検出
        for pattern in self.VOICE_MAP["ja"]["patterns"]:
            if pattern in voice_name:
                return "ja"

        # 英語音声の検出
        for pattern in self.VOICE_MAP["en"]["patterns"]:
            if pattern in voice_name:
                return "en"

        # その他は英語として扱う（デフォルト）
        if any(lang in voice_name for lang in ["english", "us", "uk"]):
            return "en"

        return None

    def _detect_gender_from_voice_name(self, voice_name: str) -> str:
        """音声名から性別を検出

        Args:
            voice_name: 音声名（小文字）

        Returns:
            str: "male", "female", または "neutral"
        """
        # 既知の音声名から判定
        if any(name in voice_name for name in ["haruka", "zira", "kyoko", "female"]):
            return "female"
        elif any(name in voice_name for name in ["david", "mark", "male"]):
            return "male"

        # デフォルト
        return "neutral"

    async def synthesize(
        self,
        text: str,
        voice_id: Optional[str] = None,
        style: Optional[str] = None,
        save_audio: Optional[bool] = None,
        **kwargs,
    ) -> SynthesisResult:
        """テキストを音声に変換

        Part 29調査結果に基づき、エンジンは毎回新規作成する。

        Args:
            text: 変換するテキスト
            voice_id: 音声ID（"ja"または"en"）
            style: スタイル（未使用）
            save_audio: 音声データ保存フラグ（設定値をオーバーライド）
            **kwargs: 追加パラメータ

        Returns:
            SynthesisResult: 音声合成結果

        Raises:
            TTSError: 音声合成失敗（E3000）
            InvalidVoiceError: 無効な音声ID（E3001）
        """
        # 状態チェック
        if self._state != ComponentState.READY:
            raise TTSError(
                f"TTS engine is not ready (state: {self._state.value})", error_code="E3000"
            )

        # 空文字チェック
        if not text or text.isspace():
            return self._create_empty_result()

        # 音声データ保存モードの決定
        should_save = save_audio if save_audio is not None else self.config.save_audio_data

        # 音声IDの決定
        voice_id = voice_id or self.current_voice_id or "ja"

        # 音声IDの検証
        if voice_id not in [v.id for v in self.available_voices]:
            raise InvalidVoiceError(
                f"Voice ID '{voice_id}' is not available", error_code="E3001", voice_id=voice_id
            )

        self.logger.debug(
            "Synthesizing text",
            extra={"text_length": len(text), "voice_id": voice_id, "save_audio": should_save},
        )

        # 実際の音声合成
        if should_save:
            return await self._synthesize_with_data(text, voice_id)
        else:
            return await self._synthesize_direct(text, voice_id)

    async def _synthesize_direct(self, text: str, voice_id: str) -> SynthesisResult:
        """直接音声出力（デフォルトモード）- イテレーティブ方式

        Args:
            text: 変換するテキスト
            voice_id: 音声ID

        Returns:
            SynthesisResult: 音声合成結果（audio_dataは空）
        """
        engine = None

        try:
            # エンジンを新規作成（Part 29の教訓：再利用不可）
            engine = pyttsx3.init()

            # 音声設定
            self._configure_engine(engine, voice_id)

            # 音声合成を設定（まだ実行はしない）
            engine.say(text)

            # イテレーティブ方式で実行
            try:
                # 非ブロッキングモードで開始
                engine.startLoop(False)

                # 最大待機時間の設定（無限ループ防止）
                max_iterations = 10000  # 100秒相当（10ms × 10000）
                iteration_count = 0

                # 音声再生が完了するまでイテレーション
                while iteration_count < max_iterations:
                    engine.iterate()

                    # エンジンがビジーでなければ終了
                    if not engine.isBusy():
                        break

                    # 少し待機（CPUを占有しないように）
                    await asyncio.sleep(0.01)  # 10ms
                    iteration_count += 1

                if iteration_count >= max_iterations:
                    self.logger.warning(f"Synthesis timeout after {iteration_count} iterations")

                # ループを終了
                engine.endLoop()

            except AttributeError:
                # startLoop/endLoopがサポートされていない場合（一部の環境）
                self.logger.warning(
                    "Iterative mode not supported, falling back to synchronous mode"
                )

                # 同期モードにフォールバック（ブロッキング）
                engine.runAndWait()

            # 推定時間（1文字0.1秒として計算）
            duration = self._estimate_duration(text)

            return SynthesisResult(
                audio_data=b"",  # 空のbytes
                sample_rate=22050,
                duration=duration,
                format="direct_output",
                metadata={"mode": "speaker", "voice_id": voice_id, "text_length": len(text)},
            )

        except Exception as e:
            self.logger.error(f"[E3000] Direct synthesis failed: {e}")
            raise TTSError(f"Failed to synthesize speech: {str(e)}", error_code="E3000")
        finally:
            # エンジンのクリーンアップ（必須）
            if engine:
                try:
                    engine.stop()
                except Exception as stop_error:
                    self.logger.debug(f"Error stopping engine: {stop_error}")
                    pass

                # エンジンオブジェクトを削除
                try:
                    del engine
                except:
                    pass

    async def _synthesize_with_data(self, text: str, voice_id: str) -> SynthesisResult:
        """WAVデータ取得モード（オプション）- イテレーティブ方式

        Args:
            text: 変換するテキスト
            voice_id: 音声ID

        Returns:
            SynthesisResult: 音声合成結果（audio_data含む）
        """
        engine = None
        temp_file = None

        try:
            # 一時ファイルの作成
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                temp_file = tmp.name

            # エンジンを新規作成
            engine = pyttsx3.init()

            # 音声設定
            self._configure_engine(engine, voice_id)

            # ファイルに保存を設定
            engine.save_to_file(text, temp_file)

            # イテレーティブ方式で実行
            try:
                # 非ブロッキングモードで開始
                engine.startLoop(False)

                # 最大待機時間の設定
                max_iterations = 10000
                iteration_count = 0

                # ファイル生成が完了するまでイテレーション
                while iteration_count < max_iterations:
                    engine.iterate()

                    if not engine.isBusy():
                        break

                    await asyncio.sleep(0.01)
                    iteration_count += 1

                if iteration_count >= max_iterations:
                    self.logger.warning(
                        f"File synthesis timeout after {iteration_count} iterations"
                    )

                engine.endLoop()

            except AttributeError:
                # フォールバック
                self.logger.warning("Iterative mode not supported, using synchronous mode")
                engine.runAndWait()

            # ファイルからデータを読み込み
            audio_data = b""
            if os.path.exists(temp_file):
                with open(temp_file, "rb") as f:
                    audio_data = f.read()
            else:
                self.logger.warning(f"Temp file not created: {temp_file}")

            # 推定時間
            duration = self._estimate_duration(text)

            return SynthesisResult(
                audio_data=audio_data if audio_data else self._create_dummy_wav(),
                sample_rate=22050,
                duration=duration,
                format="wav",
                metadata={
                    "mode": "file",
                    "voice_id": voice_id,
                    "text_length": len(text),
                    "file_size": len(audio_data),
                },
            )

        except Exception as e:
            self.logger.error(f"[E3000] File synthesis failed: {e}")
            raise TTSError(f"Failed to synthesize speech to file: {str(e)}", error_code="E3000")
        finally:
            # クリーンアップ
            if engine:
                try:
                    engine.stop()
                except:
                    pass

                try:
                    del engine
                except:
                    pass

            # 一時ファイルの削除
            if temp_file and os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except Exception as e:
                    self.logger.debug(f"Could not delete temp file: {e}")

    def _configure_engine(self, engine: Any, voice_id: str) -> None:
        """エンジンの音声設定を行う

        Args:
            engine: pyttsx3エンジンインスタンス
            voice_id: 音声ID
        """
        # 基本パラメータの設定
        engine.setProperty("rate", int(150 * self.config.speed))  # 話速
        engine.setProperty("volume", self.config.volume)  # 音量

        # 音声の設定
        if voice_id in self._voice_id_map:
            pyttsx3_voice_id = self._voice_id_map[voice_id]
            engine.setProperty("voice", pyttsx3_voice_id)
        else:
            # 音声IDマッピングがない場合は、言語に基づいて検索
            try:
                voices = engine.getProperty("voices")
                if voices:
                    for voice in voices:
                        voice_name = (
                            getattr(voice, "name", "").lower() if hasattr(voice, "name") else ""
                        )
                        if voice_id == "ja" and any(
                            p in voice_name for p in ["haruka", "japanese"]
                        ):
                            engine.setProperty("voice", voice.id)
                            break
                        elif voice_id == "en" and any(p in voice_name for p in ["zira", "english"]):
                            engine.setProperty("voice", voice.id)
                            break
            except Exception as e:
                self.logger.warning(f"Could not set voice {voice_id}: {e}")

    def _estimate_duration(self, text: str) -> float:
        """音声の推定時間を計算

        Args:
            text: テキスト

        Returns:
            float: 推定時間（秒）
        """
        # 基本：1文字0.1秒
        base_duration = len(text) * 0.1

        # 話速による調整
        adjusted_duration = base_duration / self.config.speed

        return max(0.1, adjusted_duration)  # 最小0.1秒

    def _create_empty_result(self) -> SynthesisResult:
        """空の結果を作成（修正版）

        Returns:
            SynthesisResult: 空の音声合成結果
        """
        # MockTTSEngineと同様にWAVヘッダー付きダミーデータを生成
        return SynthesisResult(
            audio_data=self._create_dummy_wav(),
            sample_rate=22050,
            duration=0.0,
            format="direct_output",
            metadata={"reason": "empty_text"},
        )

    def _create_dummy_wav(self) -> bytes:
        """ダミーのWAVヘッダーを作成

        Returns:
            bytes: 最小限のWAVヘッダー
        """
        # 44バイトの最小WAVヘッダー
        return b'RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00"V\x00\x00D\xac\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00'

    def get_available_voices(self) -> List[VoiceInfo]:
        """利用可能な音声のリストを取得

        Returns:
            List[VoiceInfo]: 音声情報のリスト
        """
        return self.available_voices.copy()

    def set_voice(self, voice_id: str) -> None:
        """使用する音声を設定

        Args:
            voice_id: 音声ID（"ja"または"en"）

        Raises:
            InvalidVoiceError: 無効な音声ID（E3001）
        """
        if voice_id not in [v.id for v in self.available_voices]:
            raise InvalidVoiceError(
                f"Voice ID '{voice_id}' is not available", error_code="E3001", voice_id=voice_id
            )

        self.current_voice_id = voice_id
        self.logger.info(f"Voice changed to: {voice_id}")

    async def test_availability(self) -> bool:
        """エンジンが利用可能かテスト

        Returns:
            bool: 利用可能な場合True
        """
        try:
            # 短いテキストで合成を試みる
            result = await self.synthesize("Test", save_audio=False)  # 直接出力モードでテスト
            return result is not None
        except Exception as e:
            self.logger.warning(f"Availability test failed: {e}")
            return False
