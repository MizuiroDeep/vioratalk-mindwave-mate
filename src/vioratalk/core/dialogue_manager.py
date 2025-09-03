"""対話システム管理モジュール

DialogueManagerクラスを定義し、対話フローを制御する。
Phase 2の中核コンポーネントとして、会話の管理と処理を担当。
Phase 4拡張: LLMManager、TTSManager、STT、AudioCapture、VAD統合。

インターフェース定義書 v1.34準拠
DialogueManager統合ガイド v1.2準拠
データフォーマット仕様書 v1.5準拠
開発規約書 v1.12準拠
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from vioratalk.core.base import ComponentState, VioraTalkComponent
from vioratalk.core.dialogue_config import DialogueConfig
from vioratalk.core.dialogue_state import DialogueTurn
from vioratalk.core.exceptions import AudioError, LLMError, STTError, TTSError

logger = logging.getLogger(__name__)


class ConversationState(Enum):
    """会話の状態を表すEnum

    Phase 2の最小実装として基本的な状態のみ定義。
    """

    IDLE = "idle"  # アイドル状態
    PROCESSING = "processing"  # 処理中
    WAITING = "waiting"  # ユーザー入力待ち
    ERROR = "error"  # エラー状態


@dataclass
class ConversationContext:
    """会話のコンテキストを管理する内部クラス

    DialogueManagerで使用される会話状態を保持。
    Phase 2では最小限の実装。

    Attributes:
        conversation_id: 会話セッションID
        character_id: 使用中のキャラクターID
        turns: 対話ターンのリスト
        current_turn_number: 現在のターン番号
        state: 会話の状態
        started_at: 会話開始時刻
        last_activity: 最後の活動時刻
        metadata: 追加メタデータ
    """

    conversation_id: str
    character_id: str
    turns: List[DialogueTurn] = field(default_factory=list)
    current_turn_number: int = 0
    state: ConversationState = ConversationState.IDLE
    started_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_turn(self, turn: DialogueTurn) -> None:
        """対話ターンを追加

        Args:
            turn: 追加する対話ターン
        """
        self.turns.append(turn)
        self.current_turn_number += 1
        self.last_activity = datetime.now()

    def get_recent_turns(self, limit: int = 10) -> List[DialogueTurn]:
        """最近の対話ターンを取得

        Args:
            limit: 取得する最大ターン数

        Returns:
            List[DialogueTurn]: 最近の対話ターンのリスト
        """
        if limit <= 0:
            return []
        return self.turns[-limit:] if self.turns else []

    def clear(self) -> None:
        """会話履歴をクリア"""
        self.turns.clear()
        self.current_turn_number = 0
        self.state = ConversationState.IDLE
        self.last_activity = datetime.now()

    def is_active(self) -> bool:
        """会話がアクティブかチェック

        Returns:
            bool: アクティブな場合True
        """
        return self.state in [ConversationState.PROCESSING, ConversationState.WAITING]


class DialogueManager(VioraTalkComponent):
    """対話フロー管理クラス

    Phase 2の中核コンポーネント。
    ユーザー入力を処理し、応答を生成する。
    Phase 4拡張: LLMManager、TTSManager、STT、AudioCapture、VAD統合。

    インターフェース定義書 v1.34 セクション3.9準拠
    DialogueManager統合ガイド v1.2準拠

    Attributes:
        config: 対話設定
        character_manager: キャラクター管理（Phase 2ではMock）
        llm_manager: LLM管理（Phase 4追加）
        stt_engine: STTエンジン（Phase 4統合）
        tts_manager: TTSマネージャー（Phase 4統合）
        audio_capture: 音声入力管理（Phase 4統合）
        vad: 音声区間検出（Phase 4統合）
        _context: 現在の会話コンテキスト
        _initialized: 初期化済みフラグ
        _conversation_count: 生成した会話ID用カウンタ

    Example:
        >>> config = DialogueConfig()
        >>> llm_manager = LLMManager()
        >>> tts_manager = TTSManager()
        >>> manager = DialogueManager(config, llm_manager=llm_manager, tts_manager=tts_manager)
        >>> await manager.initialize()
        >>> turn = await manager.process_text_input("こんにちは")
        >>> print(turn.assistant_response)
    """

    def __init__(
        self,
        config: DialogueConfig,
        character_manager: Optional[Any] = None,
        llm_manager: Optional[Any] = None,
        stt_engine: Optional[Any] = None,
        tts_manager: Optional[Any] = None,  # TTSManagerを使用
        audio_capture: Optional[Any] = None,  # AudioCapture追加
        vad: Optional[Any] = None,  # VAD追加
    ) -> None:
        """コンストラクタ

        Args:
            config: 対話設定
            character_manager: キャラクター管理（Phase 2ではオプション）
            llm_manager: LLM管理（Phase 4追加）
            stt_engine: 音声認識エンジン（Phase 4統合）
            tts_manager: TTS統合管理（Phase 4統合）
            audio_capture: 音声入力管理（Phase 4統合）
            vad: 音声区間検出（Phase 4統合）
        """
        super().__init__()
        self.config = config
        self.character_manager = character_manager
        self.llm_manager = llm_manager
        self.stt_engine = stt_engine
        self.tts_manager = tts_manager  # TTSManagerを使用
        self.audio_capture = audio_capture
        self.vad = vad
        self._context: Optional[ConversationContext] = None
        self._initialized: bool = False
        self._conversation_count: int = 0

    async def initialize(self) -> None:
        """非同期初期化

        コンポーネントを初期化し、使用可能な状態にする。

        Raises:
            RuntimeError: 既に初期化済みの場合
        """
        if self._state == ComponentState.READY:
            raise RuntimeError("DialogueManager is already initialized")

        self._state = ComponentState.INITIALIZING
        logger.info("Initializing DialogueManager")

        # キャラクターマネージャーの初期化（存在する場合）
        if self.character_manager and hasattr(self.character_manager, "initialize"):
            if self.character_manager._state != ComponentState.READY:
                await self.character_manager.initialize()
                logger.info("Character manager initialized")

        # LLMManagerの初期化（Phase 4追加）
        if self.llm_manager and hasattr(self.llm_manager, "initialize"):
            if self.llm_manager._state != ComponentState.READY:
                await self.llm_manager.initialize()
                logger.info("LLM manager initialized")

        # STTエンジンの初期化（Phase 4統合）
        if self.stt_engine and hasattr(self.stt_engine, "initialize"):
            if self.stt_engine._state != ComponentState.READY:
                await self.stt_engine.initialize()
                logger.info("STT engine initialized")

        # TTSManagerの初期化（Phase 4統合）
        if self.tts_manager and hasattr(self.tts_manager, "initialize"):
            if self.tts_manager._state != ComponentState.READY:
                await self.tts_manager.initialize()
                logger.info("TTS manager initialized")

        # AudioCaptureの初期化（Phase 4統合）
        if self.audio_capture and hasattr(self.audio_capture, "safe_initialize"):
            if self.audio_capture._state != ComponentState.READY:
                await self.audio_capture.safe_initialize()
                logger.info("Audio capture initialized")

        # VADの初期化（Phase 4統合）
        if self.vad and hasattr(self.vad, "initialize"):
            if self.vad._state != ComponentState.READY:
                await self.vad.initialize()
                logger.info("VAD initialized")

        # 新しい会話コンテキストを作成
        self._create_new_context()

        self._initialized = True
        self._state = ComponentState.READY
        logger.info("DialogueManager initialization completed")

    async def cleanup(self) -> None:
        """リソースのクリーンアップ

        使用したリソースを解放する。
        """
        if self._state == ComponentState.TERMINATED:
            return

        self._state = ComponentState.TERMINATING
        logger.info("Cleaning up DialogueManager")

        # 各コンポーネントのクリーンアップ
        if self.llm_manager and hasattr(self.llm_manager, "cleanup"):
            await self.llm_manager.cleanup()

        if self.tts_manager and hasattr(self.tts_manager, "cleanup"):
            await self.tts_manager.cleanup()

        if self.stt_engine and hasattr(self.stt_engine, "cleanup"):
            await self.stt_engine.cleanup()

        if self.audio_capture and hasattr(self.audio_capture, "safe_cleanup"):
            await self.audio_capture.safe_cleanup()

        if self.vad and hasattr(self.vad, "cleanup"):
            await self.vad.cleanup()

        if self.character_manager and hasattr(self.character_manager, "cleanup"):
            await self.character_manager.cleanup()

        # コンテキストのクリア
        if self._context:
            self._context.clear()
            self._context = None

        self._initialized = False
        self._state = ComponentState.TERMINATED
        logger.info("DialogueManager cleanup completed")

    def _create_new_context(self) -> None:
        """新しい会話コンテキストを作成"""
        self._conversation_count += 1
        conversation_id = (
            f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self._conversation_count:04d}"
        )

        # 現在のキャラクターIDを取得
        character_id = self.config.default_character_id
        if self.character_manager and hasattr(self.character_manager, "get_current_character"):
            current_char = self.character_manager.get_current_character()
            if current_char:
                character_id = current_char.character_id

        self._context = ConversationContext(
            conversation_id=conversation_id, character_id=character_id
        )

    async def process_text_input(self, text: str) -> DialogueTurn:
        """テキスト入力を処理して応答を生成

        DialogueManager統合ガイド v1.2準拠の実装。
        テキスト入力 → LLM → TTS（オプション）の流れを実装。

        Args:
            text: ユーザーからの入力テキスト

        Returns:
            DialogueTurn: ユーザー入力とアシスタント応答のペア

        Raises:
            RuntimeError: 未初期化の場合
            ValueError: 入力テキストが空の場合
            LLMError: LLM応答生成に失敗した場合
            TTSError: 音声合成に失敗した場合
        """
        if not self._initialized:
            raise RuntimeError("DialogueManager is not initialized")

        if not text or not text.strip():
            raise ValueError("Input text cannot be empty")

        # 最大ターン数チェック
        if self._context and len(self._context.turns) >= self.config.max_turns:
            # 最古のターンを削除
            self._context.turns.pop(0)

        # 会話状態を処理中に変更
        if self._context:
            self._context.state = ConversationState.PROCESSING

        try:
            start_time = datetime.now()

            # 共通のLLM処理（_process_with_llm）
            assistant_response = await self._process_with_llm(text)

            # TTS処理（設定により音声合成するか決定）
            audio_response = None
            if self.tts_manager and getattr(self.config, "tts_enabled_for_text", True):
                try:
                    logger.debug("Starting TTS synthesis for text response")
                    # TTSManagerのsynthesize()を使用
                    synthesis_result = await self.tts_manager.synthesize(
                        text=assistant_response,
                        voice_id=getattr(self.config, "voice_id", None),
                        style=getattr(self.config, "voice_style", None),
                    )

                    # SynthesisResultからaudio_dataを取得
                    if hasattr(synthesis_result, "audio_data"):
                        audio_response = synthesis_result.audio_data
                    else:
                        audio_response = synthesis_result

                    logger.debug("TTS synthesis completed")

                except TTSError as e:
                    logger.warning(f"TTS synthesis failed, continuing without audio: {e}")
                    # TTSエラーは警告のみ、処理は継続
                except Exception as e:
                    logger.warning(f"Unexpected TTS error: {e}")

            # 処理時間を計算
            processing_time = (datetime.now() - start_time).total_seconds()

            # DialogueTurnを作成
            turn = DialogueTurn(
                user_input=text,
                assistant_response=assistant_response,
                audio_response=audio_response,  # 音声データを追加
                timestamp=datetime.now(),
                turn_number=self._context.current_turn_number + 1 if self._context else 1,
                character_id=self._context.character_id
                if self._context
                else self.config.default_character_id,
                emotion="neutral",  # Phase 7で感情分析実装
                confidence=0.95 if self.llm_manager else 0.5,
                processing_time=processing_time,
            )

            # コンテキストに追加
            if self._context:
                self._context.add_turn(turn)
                self._context.state = ConversationState.WAITING

            return turn

        except Exception as e:
            # エラー時の処理
            if self._context:
                self._context.state = ConversationState.ERROR
            logger.error(f"Failed to process text input: {e}")
            raise RuntimeError(f"Failed to process text input: {str(e)}")

    async def process_audio_input(self, audio_data: bytes = None) -> DialogueTurn:
        """音声入力を処理して応答を生成

        DialogueManager統合ガイド v1.2準拠の実装。
        音声入力 → VAD → STT → LLM → TTS → 音声出力の完全フローを実装。

        Args:
            audio_data: 音声データ（バイト列）。Noneの場合はマイクから録音

        Returns:
            DialogueTurn: 会話ターンの完全な情報

        Raises:
            AudioError: 音声入力エラー（E1001, E1002）
            STTError: 音声認識に失敗した場合（E2000）
            LLMError: 応答生成に失敗した場合
            TTSError: 音声合成に失敗した場合（E3000）
        """
        if not self._initialized:
            raise RuntimeError("DialogueManager is not initialized")

        # 会話状態を処理中に変更
        if self._context:
            self._context.state = ConversationState.PROCESSING

        try:
            start_time = datetime.now()

            # 1. 音声入力取得（audio_dataがない場合はマイクから録音）
            if audio_data is None:
                if not self.audio_capture:
                    raise AudioError("Audio capture not available", error_code="E1002")

                logger.info("Recording from microphone")
                # デフォルト5秒録音（設定可能にする場合は config から取得）
                duration = getattr(self.config, "recording_duration", 5.0)
                audio_data_obj = await self.audio_capture.record_from_microphone(duration=duration)
                audio_array = audio_data_obj.raw_data
            else:
                # バイト列をnumpy配列に変換（必要に応じて）
                import numpy as np

                if isinstance(audio_data, bytes):
                    audio_array = np.frombuffer(audio_data, dtype=np.float32)
                else:
                    audio_array = audio_data

            # 2. VADで音声区間検出（オプション）
            if self.vad:
                logger.info("Detecting speech segments with VAD")
                segments = self.vad.detect_segments(audio_array)

                if not segments:
                    logger.warning("No speech detected in audio")
                    raise AudioError("No speech detected in audio input", error_code="E1003")

                # 最初のセグメントを使用（将来的には複数セグメント対応）
                segment = segments[0]
                start_sample = segment.start_sample
                end_sample = segment.end_sample
                audio_for_stt = audio_array[start_sample:end_sample]
                logger.info(
                    f"Using speech segment: {segment.start_time:.2f}s - {segment.end_time:.2f}s"
                )
            else:
                # VADなしの場合は全音声を使用
                audio_for_stt = audio_array

            # 3. STTで音声認識
            if not self.stt_engine:
                raise STTError("STT engine not available", error_code="E2000")

            logger.info("Starting speech recognition")

            # STTエンジンに音声データを渡す
            # AudioDataオブジェクトを作成
            if hasattr(self.stt_engine, "transcribe"):
                from vioratalk.core.stt.base import AudioData, AudioMetadata

                metadata = AudioMetadata(
                    sample_rate=getattr(self.config, "sample_rate", 16000),
                    channels=1,
                    bit_depth=16,
                    duration=len(audio_for_stt) / getattr(self.config, "sample_rate", 16000),
                    format="pcm_float32",
                    filename=None,
                )

                stt_audio_data = AudioData(
                    raw_data=audio_for_stt, encoded_data=None, metadata=metadata
                )

                # 音声認識実行
                transcription_result = await self.stt_engine.transcribe(stt_audio_data)

                # TranscriptionResultから文字列を取得
                if hasattr(transcription_result, "text"):
                    transcribed_text = transcription_result.text
                else:
                    transcribed_text = str(transcription_result)

                logger.info(f"Transcribed: {transcribed_text}")
            else:
                raise STTError("STT engine does not support transcribe method", error_code="E2000")

            # 4. 共通のLLM処理（_process_with_llm）
            assistant_response = await self._process_with_llm(transcribed_text)

            # 5. TTSで音声合成
            audio_response = None
            if self.tts_manager:
                try:
                    logger.info("Starting speech synthesis")
                    synthesis_result = await self.tts_manager.synthesize(
                        text=assistant_response,
                        voice_id=getattr(self.config, "voice_id", None),
                        style=getattr(self.config, "voice_style", None),
                    )

                    # SynthesisResultからaudio_dataを取得
                    if hasattr(synthesis_result, "audio_data"):
                        audio_response = synthesis_result.audio_data
                    else:
                        audio_response = synthesis_result

                    logger.info("Speech synthesis completed")

                except TTSError as e:
                    logger.warning(f"TTS synthesis failed: {e}")
                    # TTSエラーは警告のみ、処理は継続

            # 処理時間を計算
            processing_time = (datetime.now() - start_time).total_seconds()

            # 6. ConversationTurnを作成
            turn = DialogueTurn(
                user_input=transcribed_text,
                assistant_response=assistant_response,
                audio_response=audio_response,
                timestamp=datetime.now(),
                turn_number=self._context.current_turn_number + 1 if self._context else 1,
                character_id=self._context.character_id
                if self._context
                else self.config.default_character_id,
                emotion="neutral",  # Phase 7で感情分析実装
                confidence=getattr(transcription_result, "confidence", 0.9)
                if "transcription_result" in locals()
                else 0.5,
                processing_time=processing_time,
            )

            # コンテキストに追加
            if self._context:
                self._context.add_turn(turn)
                self._context.state = ConversationState.WAITING

            return turn

        except (AudioError, STTError, LLMError, TTSError) as e:
            # 既知のエラーはそのまま再スロー
            if self._context:
                self._context.state = ConversationState.ERROR
            logger.error(f"Error in audio processing pipeline: {e}")
            raise

        except Exception as e:
            # 予期しないエラー
            if self._context:
                self._context.state = ConversationState.ERROR
            logger.error(f"Unexpected error in process_audio_input: {e}", exc_info=True)
            raise RuntimeError(f"Failed to process audio input: {str(e)}")

    async def _process_with_llm(self, input_text: str) -> str:
        """LLMでの処理（共通ロジック）

        DialogueManager統合ガイド v1.2準拠。
        テキスト入力と音声入力の両方で使用される共通処理。

        Args:
            input_text: 入力テキスト

        Returns:
            str: LLM応答テキスト

        Raises:
            LLMError: LLM処理失敗
        """
        # LLMManagerを使用した応答生成
        if self.llm_manager:
            logger.debug(f"Processing with LLMManager: {input_text[:50]}...")

            # 会話履歴からコンテキストを構築
            context = self._build_llm_context(input_text)

            # LLMManagerで応答生成
            try:
                response = await self.llm_manager.generate(
                    prompt=context,
                    temperature=getattr(self.config, "temperature", 0.7),
                    max_tokens=getattr(self.config, "max_tokens", 500),
                )

                # LLMResponseから文字列を取得（修正箇所: text → content）
                if hasattr(response, "content"):
                    assistant_response = response.content
                else:
                    assistant_response = str(response)

                logger.debug(f"LLM response received: {assistant_response[:50]}...")
                return assistant_response

            except Exception as e:
                logger.warning(f"LLMManager failed, falling back to mock: {e}")
                # フォールバック: モック応答を使用
                return await self._generate_mock_response(input_text)
        else:
            # LLMManagerがない場合はモック応答
            logger.debug("No LLMManager available, using mock response")
            return await self._generate_mock_response(input_text)

    def _build_llm_context(self, user_input: str) -> str:
        """LLM用のコンテキストを構築

        会話履歴を含むプロンプトを生成する。

        Args:
            user_input: ユーザー入力

        Returns:
            str: LLM用のプロンプト
        """
        # 基本的なシステムプロンプト
        system_prompt = "あなたは親切で丁寧な日本語アシスタントです。ユーザーの質問に対して適切に応答してください。"

        # キャラクター設定がある場合は追加
        if self.character_manager and hasattr(self.character_manager, "get_current_character"):
            character = self.character_manager.get_current_character()
            if character and hasattr(character, "personality"):
                system_prompt = f"あなたは{character.name}という名前のアシスタントです。{character.personality}"

        # 会話履歴を含める（最近の5ターンまで）
        conversation_history = ""
        if self._context:
            recent_turns = self._context.get_recent_turns(5)
            for turn in recent_turns:
                conversation_history += f"ユーザー: {turn.user_input}\n"
                conversation_history += f"アシスタント: {turn.assistant_response}\n"

        # プロンプトを構築
        if conversation_history:
            prompt = f"{system_prompt}\n\n以下は過去の会話履歴です:\n{conversation_history}\n\n現在のユーザーの入力: {user_input}\n\n応答:"
        else:
            prompt = f"{system_prompt}\n\nユーザー: {user_input}\n\nアシスタント:"

        return prompt

    async def _generate_mock_response(self, user_input: str) -> str:
        """モック応答を生成（フォールバック）

        LLMManagerが利用できない場合のフォールバック応答。

        Args:
            user_input: ユーザー入力

        Returns:
            str: 生成された応答
        """
        # キャラクターに応じた応答パターン
        responses = {
            "こんにちは": "こんにちは！今日はどんなお話をしましょうか？",
            "お元気ですか": "はい、元気です！あなたはいかがですか？",
            "ありがとう": "どういたしまして！お役に立てて嬉しいです。",
            "さようなら": "さようなら！またお話ししましょうね。",
            "おはよう": "おはようございます！今日も良い一日になりますように。",
            "おやすみ": "おやすみなさい。良い夢を見てくださいね。",
        }

        # 簡単なパターンマッチング
        for pattern, response in responses.items():
            if pattern in user_input:
                return response

        # デフォルト応答
        if self.character_manager and hasattr(self.character_manager, "get_current_character"):
            character = self.character_manager.get_current_character()
            if character and character.name == "あおい":
                return "はい、承知いたしました。他に何かお手伝いできることはありますか？"

        return "なるほど、そうですね。もう少し詳しく教えていただけますか？"

    def get_conversation_history(self, limit: int = 10) -> List[DialogueTurn]:
        """会話履歴を取得

        Args:
            limit: 取得する最大ターン数（デフォルト: 10）

        Returns:
            List[DialogueTurn]: 会話履歴のリスト

        Raises:
            RuntimeError: 未初期化の場合
        """
        if not self._initialized:
            raise RuntimeError("DialogueManager is not initialized")

        if not self._context:
            return []

        return self._context.get_recent_turns(limit)

    async def reset_conversation(self) -> None:
        """会話をリセット

        現在の会話をクリアし、新しい会話を開始する。

        Raises:
            RuntimeError: 未初期化の場合
        """
        if not self._initialized:
            raise RuntimeError("DialogueManager is not initialized")

        # 現在のコンテキストをクリア
        if self._context:
            self._context.clear()

        # 新しいコンテキストを作成
        self._create_new_context()
        logger.info("Conversation reset")

    def clear_conversation(self) -> None:
        """会話履歴をクリア

        現在の会話履歴を削除する。
        reset_conversationとは異なり、同じセッション内で履歴のみクリア。

        Raises:
            RuntimeError: 未初期化の場合
        """
        if not self._initialized:
            raise RuntimeError("DialogueManager is not initialized")

        if self._context:
            self._context.clear()
        logger.info("Conversation history cleared")

    def can_switch_character(self) -> bool:
        """キャラクター切り替え可能かチェック

        Returns:
            bool: 切り替え可能な場合True

        Note:
            インターフェース定義書 v1.34準拠
            会話がアクティブでない場合のみ切り替え可能
        """
        if not self._initialized:
            return False

        if not self._context:
            return True

        # 会話がアクティブでない場合のみ切り替え可能
        return not self._context.is_active()

    def is_conversation_active(self) -> bool:
        """会話がアクティブか確認

        Returns:
            bool: アクティブな場合True
        """
        if not self._initialized:
            return False

        if not self._context:
            return False

        return self._context.is_active()

    def get_current_context(self) -> Optional[ConversationContext]:
        """現在の会話コンテキストを取得（デバッグ用）

        Returns:
            Optional[ConversationContext]: 現在のコンテキスト

        Note:
            主にテストやデバッグで使用
        """
        return self._context

    def get_conversation_stats(self) -> Dict[str, Any]:
        """会話統計を取得

        Returns:
            Dict[str, Any]: 統計情報の辞書

        Example:
            >>> stats = manager.get_conversation_stats()
            >>> print(f"Total turns: {stats['total_turns']}")
        """
        if not self._context:
            return {
                "conversation_id": None,
                "total_turns": 0,
                "character_id": None,
                "state": "no_context",
                "duration_seconds": 0,
                "llm_available": self.llm_manager is not None,
                "tts_available": self.tts_manager is not None,
                "stt_available": self.stt_engine is not None,
                "audio_capture_available": self.audio_capture is not None,
                "vad_available": self.vad is not None,
            }

        duration = (datetime.now() - self._context.started_at).total_seconds()

        return {
            "conversation_id": self._context.conversation_id,
            "total_turns": len(self._context.turns),
            "current_turn_number": self._context.current_turn_number,
            "character_id": self._context.character_id,
            "state": self._context.state.value,
            "duration_seconds": duration,
            "started_at": self._context.started_at.isoformat(),
            "last_activity": self._context.last_activity.isoformat(),
            "llm_available": self.llm_manager is not None,
            "tts_available": self.tts_manager is not None,
            "stt_available": self.stt_engine is not None,
            "audio_capture_available": self.audio_capture is not None,
            "vad_available": self.vad is not None,
        }

    def __repr__(self) -> str:
        """開発用の詳細文字列表現

        Returns:
            str: オブジェクトの詳細情報
        """
        context_info = "no_context"
        if self._context:
            context_info = f"id={self._context.conversation_id}, turns={len(self._context.turns)}"

        components = []
        if self.llm_manager:
            components.append("LLM")
        if self.tts_manager:
            components.append("TTS")
        if self.stt_engine:
            components.append("STT")
        if self.audio_capture:
            components.append("Audio")
        if self.vad:
            components.append("VAD")

        return (
            f"DialogueManager("
            f"state={self._state.value}, "
            f"initialized={self._initialized}, "
            f"context=[{context_info}], "
            f"components=[{', '.join(components)}], "
            f"config=use_mock={self.config.use_mock_engines})"
        )
