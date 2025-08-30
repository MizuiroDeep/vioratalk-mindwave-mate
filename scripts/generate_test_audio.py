#!/usr/bin/env python3
"""
テスト用音声ファイル生成スクリプト。

FasterWhisperEngineのテスト用に必要な音声ファイルを生成する。
Windows SAPI（Speech API）を使用した音声合成と、
numpy/scipyを使用した無音・ノイズ生成を行う。

開発規約書 v1.12準拠
テストデータ・モック完全仕様書 v1.1準拠
"""

import sys
import wave
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


def create_directory(path: Path) -> None:
    """ディレクトリを作成する。
    
    Args:
        path: 作成するディレクトリのパス
        
    Raises:
        OSError: ディレクトリ作成に失敗した場合
    """
    try:
        path.mkdir(parents=True, exist_ok=True)
        print(f"✓ ディレクトリ作成: {path}")
    except OSError as e:
        print(f"✗ ディレクトリ作成失敗: {path}")
        raise


def generate_silence_audio(
    output_path: Path,
    duration: float = 2.0,
    sample_rate: int = 16000
) -> None:
    """無音の音声ファイルを生成する。
    
    Args:
        output_path: 出力ファイルパス
        duration: 無音の長さ（秒）
        sample_rate: サンプリングレート
        
    Raises:
        IOError: ファイル書き込みに失敗した場合
    """
    try:
        # 無音データ生成（全て0）
        num_samples = int(sample_rate * duration)
        silence_data = np.zeros(num_samples, dtype=np.int16)
        
        # WAVファイルとして保存
        with wave.open(str(output_path), 'wb') as wav_file:
            wav_file.setnchannels(1)  # モノラル
            wav_file.setsampwidth(2)  # 16bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(silence_data.tobytes())
        
        print(f"✓ 無音ファイル生成: {output_path} ({duration}秒)")
        
    except Exception as e:
        print(f"✗ 無音ファイル生成失敗: {e}")
        raise


def generate_noisy_audio(
    output_path: Path,
    text: str = "ノイズテスト",
    duration: float = 3.0,
    sample_rate: int = 16000,
    noise_level: float = 0.1
) -> None:
    """ノイズ付き音声ファイルを生成する。
    
    Windows SAPIで音声を生成し、ホワイトノイズを追加する。
    
    Args:
        output_path: 出力ファイルパス
        text: 読み上げるテキスト
        duration: 音声の長さ（秒）
        sample_rate: サンプリングレート
        noise_level: ノイズレベル（0.0-1.0）
        
    Raises:
        ImportError: pywin32がインストールされていない場合
        RuntimeError: 音声合成に失敗した場合
    """
    try:
        import win32com.client
    except ImportError:
        print("✗ pywin32がインストールされていません")
        print("  実行: poetry add --group dev pywin32")
        raise
    
    try:
        # まず通常の音声を生成
        temp_path = output_path.with_suffix('.tmp.wav')
        generate_speech_with_sapi(temp_path, text, lang='ja')
        
        # 音声データを読み込み
        with wave.open(str(temp_path), 'rb') as wav_file:
            frames = wav_file.readframes(wav_file.getnframes())
            audio_data = np.frombuffer(frames, dtype=np.int16)
            actual_rate = wav_file.getframerate()
        
        # ホワイトノイズを生成して追加
        noise = np.random.normal(0, noise_level * 32767, len(audio_data))
        noisy_data = audio_data + noise.astype(np.int16)
        
        # クリッピング防止
        noisy_data = np.clip(noisy_data, -32768, 32767).astype(np.int16)
        
        # 保存
        with wave.open(str(output_path), 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(actual_rate)
            wav_file.writeframes(noisy_data.tobytes())
        
        # 一時ファイル削除
        temp_path.unlink(missing_ok=True)
        
        print(f"✓ ノイズ付き音声生成: {output_path} (ノイズレベル: {noise_level})")
        
    except Exception as e:
        print(f"✗ ノイズ付き音声生成失敗: {e}")
        raise


def generate_speech_with_sapi(
    output_path: Path,
    text: str,
    lang: str = 'ja',
    rate: int = 0,
    volume: int = 100
) -> None:
    """Windows SAPIを使用して音声を生成する。
    
    Args:
        output_path: 出力ファイルパス
        text: 読み上げるテキスト
        lang: 言語（'ja'または'en'）
        rate: 話速（-10〜10、0が標準）
        volume: 音量（0〜100）
        
    Raises:
        ImportError: pywin32がインストールされていない場合
        RuntimeError: 音声合成に失敗した場合
    """
    try:
        import win32com.client
    except ImportError:
        print("✗ pywin32がインストールされていません")
        print("  実行: poetry add --group dev pywin32")
        raise
    
    try:
        # SAPI.SpVoiceオブジェクトの作成
        voice = win32com.client.Dispatch("SAPI.SpVoice")
        
        # 利用可能な音声を取得
        voices = voice.GetVoices()
        selected_voice = None
        
        # 言語に応じた音声を選択
        for i in range(voices.Count):
            v = voices.Item(i)
            desc = v.GetDescription()
            
            if lang == 'ja' and ('Japanese' in desc or '日本' in desc):
                selected_voice = v
                break
            elif lang == 'en' and 'English' in desc:
                selected_voice = v
                break
        
        if selected_voice:
            voice.Voice = selected_voice
            print(f"  使用音声: {selected_voice.GetDescription()}")
        else:
            print(f"  警告: {lang}音声が見つかりません。デフォルト音声を使用します。")
        
        # 音声パラメータ設定
        voice.Rate = rate
        voice.Volume = volume
        
        # ファイルストリームの作成
        stream = win32com.client.Dispatch("SAPI.SpFileStream")
        stream.Open(str(output_path), 3)  # 3 = SSFMCreateForWrite
        
        # 出力先をファイルに設定
        voice.AudioOutputStream = stream
        
        # 音声合成実行
        voice.Speak(text)
        
        # ストリームを閉じる
        stream.Close()
        
        print(f"✓ 音声ファイル生成: {output_path}")
        print(f"  テキスト: {text}")
        
    except Exception as e:
        print(f"✗ 音声生成失敗: {e}")
        raise


def generate_simple_tone(
    output_path: Path,
    text: str,
    duration: float = 3.0,
    sample_rate: int = 16000
) -> None:
    """簡易的なトーン音声を生成する（SAPI使用不可時のフォールバック）。
    
    音声認識テスト用の簡単な音声パターンを生成する。
    実際の音声ではないが、音声認識エンジンのテストには使用可能。
    
    Args:
        output_path: 出力ファイルパス
        text: ファイル名用のテキスト（実際には使用しない）
        duration: 音声の長さ（秒）
        sample_rate: サンプリングレート
    """
    print(f"  警告: SAPIが使用できないため、代替音声を生成します")
    
    # 複数の周波数を組み合わせた音声パターンを生成
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # 音声っぽいパターンを作る（複数の正弦波の組み合わせ）
    signal = np.zeros_like(t)
    frequencies = [200, 300, 400, 600, 800]  # 人の声の周波数帯
    
    for i, freq in enumerate(frequencies):
        amplitude = 0.2 * (1.0 - i * 0.15)  # 高周波ほど振幅を小さく
        signal += amplitude * np.sin(2 * np.pi * freq * t)
    
    # 音量調整とクリッピング
    signal = signal * 10000
    signal = np.clip(signal, -32768, 32767).astype(np.int16)
    
    # エンベロープ（フェードイン・フェードアウト）
    fade_samples = int(0.1 * sample_rate)
    signal[:fade_samples] *= np.linspace(0, 1, fade_samples)
    signal[-fade_samples:] *= np.linspace(1, 0, fade_samples)
    
    # WAVファイルとして保存
    with wave.open(str(output_path), 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(signal.tobytes())
    
    print(f"✓ 代替音声生成: {output_path} ({duration}秒)")


def main() -> None:
    """メイン処理。
    
    tests/fixtures/audio/ディレクトリに4つのテスト用音声ファイルを生成する。
    """
    print("=" * 60)
    print("テスト用音声ファイル生成スクリプト")
    print("=" * 60)
    
    # プロジェクトルートからの相対パス
    project_root = Path(__file__).parent.parent
    audio_dir = project_root / "tests" / "fixtures" / "audio"
    
    print(f"\n出力ディレクトリ: {audio_dir}")
    
    # ディレクトリ作成
    create_directory(audio_dir)
    
    # 生成する音声ファイルのリスト
    files_to_generate = [
        {
            "filename": "hello_japanese.wav",
            "text": "こんにちは、音声認識のテストです",
            "lang": "ja",
            "type": "speech"
        },
        {
            "filename": "hello_english.wav",
            "text": "Hello, this is a speech recognition test",
            "lang": "en",
            "type": "speech"
        },
        {
            "filename": "silence.wav",
            "text": "",
            "lang": "",
            "type": "silence"
        },
        {
            "filename": "noisy.wav",
            "text": "これはノイズ付きの音声テストです",
            "lang": "ja",
            "type": "noisy"
        }
    ]
    
    print("\n生成開始:")
    print("-" * 40)
    
    success_count = 0
    failed_files = []
    
    for file_info in files_to_generate:
        output_path = audio_dir / file_info["filename"]
        print(f"\n[{file_info['filename']}]")
        
        try:
            if file_info["type"] == "silence":
                # 無音ファイル生成
                generate_silence_audio(output_path)
                
            elif file_info["type"] == "noisy":
                # ノイズ付き音声生成
                generate_noisy_audio(output_path, file_info["text"])
                
            elif file_info["type"] == "speech":
                # 通常の音声生成
                try:
                    generate_speech_with_sapi(
                        output_path,
                        file_info["text"],
                        file_info["lang"]
                    )
                except Exception as sapi_error:
                    print(f"  SAPIエラー: {sapi_error}")
                    print("  代替方法で生成を試みます...")
                    generate_simple_tone(
                        output_path,
                        file_info["text"]
                    )
            
            # ファイルサイズ確認
            if output_path.exists():
                size = output_path.stat().st_size
                print(f"  サイズ: {size:,} bytes")
                success_count += 1
            else:
                raise FileNotFoundError(f"ファイルが作成されませんでした")
                
        except Exception as e:
            print(f"✗ エラー: {e}")
            failed_files.append(file_info["filename"])
    
    # 結果サマリー
    print("\n" + "=" * 60)
    print("生成結果:")
    print(f"  成功: {success_count}/{len(files_to_generate)}")
    
    if failed_files:
        print(f"  失敗: {', '.join(failed_files)}")
        print("\n注意: 失敗したファイルは手動で作成するか、")
        print("      別の方法で生成してください。")
        sys.exit(1)
    else:
        print("\n✓ すべての音声ファイルの生成が完了しました！")
        print(f"\n生成場所: {audio_dir}")
        print("\n次のステップ:")
        print("  1. 生成された音声ファイルを確認")
        print("  2. test_manual_faster_whisper.pyでテスト実行")
        
    return


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n中断されました。")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n予期しないエラー: {e}")
        sys.exit(1)
