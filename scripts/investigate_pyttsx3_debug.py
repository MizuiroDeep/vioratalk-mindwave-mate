#!/usr/bin/env python3
"""
pyttsx3問題の詳細調査スクリプト

エンジンを毎回再初期化して、どの文字が問題を引き起こすか特定する。
"""

import sys
import time
import traceback

try:
    import pyttsx3
except ImportError:
    print("ERROR: pyttsx3がインストールされていません。")
    sys.exit(1)


def test_single_text(text: str, description: str) -> bool:
    """単一のテキストをテスト（エンジンを毎回新規作成）"""
    print(f"\n{'='*60}")
    print(f"テスト: {description}")
    print(f"テキスト: '{text}'")
    print('-'*60)
    
    engine = None
    success = False
    
    try:
        # エンジンを新規作成
        engine = pyttsx3.init()
        
        # 日本語音声を設定
        voices = engine.getProperty('voices')
        for voice in voices:
            if 'haruka' in voice.name.lower():
                engine.setProperty('voice', voice.id)
                break
        
        engine.setProperty('rate', 120)
        engine.setProperty('volume', 1.0)
        
        print("エンジン初期化: OK")
        
        # 音声合成実行
        start_time = time.time()
        engine.say(text)
        engine.runAndWait()
        duration = time.time() - start_time
        
        print(f"実行時間: {duration:.2f}秒")
        
        # 音声が出たか確認
        if duration > 0.5:  # 0.5秒以上なら音声が出た可能性が高い
            print("✅ 音声出力: あり（推定）")
            success = True
        else:
            print("❌ 音声出力: なし（0.07秒は無音の典型）")
            success = False
            
    except Exception as e:
        print(f"❌ エラー: {e}")
        traceback.print_exc()
        success = False
        
    finally:
        # エンジンを確実に破棄
        if engine:
            try:
                engine.stop()
                del engine
            except:
                pass
    
    return success


def test_problem_characters():
    """問題のある文字を個別にテスト"""
    
    test_cases = [
        # 基本テスト
        ("こんにちは", "基本日本語"),
        ("コンニチハ", "カタカナ"),
        ("Hello", "英語"),
        
        # 問題の可能性がある文字を個別にテスト
        ("すごい", "感嘆符なし"),
        ("すごい！", "感嘆符あり"),
        ("すごい!", "半角感嘆符"),
        
        ("今日は良い天気です", "句点なし"),
        ("今日は良い天気です。", "句点あり"),
        ("今日は良い天気です.", "半角ピリオド"),
        
        ("今日は良い天気です", "読点なし"),
        ("今日は、良い天気です", "読点あり"),
        ("今日は，良い天気です", "全角カンマ"),
        
        ("こんにちはと言った", "かぎ括弧なし"),
        ("「こんにちは」と言った", "かぎ括弧あり"),
        ('"こんにちは"と言った', "ダブルクォート"),
        
        ("重要なお知らせです", "二重かぎ括弧なし"),
        ("『重要なお知らせ』です", "二重かぎ括弧あり"),
        
        # 組み合わせテスト
        ("良い天気。素晴らしい！", "句点と感嘆符"),
        ("「こんにちは」。「さようなら」", "かぎ括弧と句点"),
    ]
    
    print("="*60)
    print("pyttsx3 問題文字の特定テスト")
    print("="*60)
    print("\n各テストでエンジンを新規作成します。")
    print("音声が出るか自動判定します（0.5秒以上なら音声あり）\n")
    
    input("Enterキーを押してテスト開始...")
    
    results = []
    for text, description in test_cases:
        success = test_single_text(text, description)
        results.append((text, description, success))
        time.sleep(0.5)  # 次のテストまで待機
    
    # 結果サマリー
    print("\n" + "="*60)
    print("テスト結果サマリー")
    print("="*60)
    
    success_count = sum(1 for _, _, s in results if s)
    total_count = len(results)
    
    print(f"\n成功: {success_count}/{total_count}")
    
    print("\n【失敗したケース】")
    for text, desc, success in results:
        if not success:
            print(f"❌ {desc}: '{text}'")
    
    print("\n【成功したケース】")
    for text, desc, success in results:
        if success:
            print(f"✅ {desc}: '{text}'")
    
    # 問題の分析
    print("\n" + "="*60)
    print("問題の分析")
    print("="*60)
    
    failed_chars = set()
    for text, _, success in results:
        if not success:
            # 成功したケースとの差分を見つける
            for char in text:
                if char not in "あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをんアイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲンabcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 　":
                    failed_chars.add(char)
    
    if failed_chars:
        print("\n問題を引き起こす可能性がある文字:")
        for char in sorted(failed_chars):
            print(f"  '{char}' (Unicode: U+{ord(char):04X})")
    
    # 推奨事項
    print("\n【Pyttsx3Engine実装への推奨事項】")
    print("-"*40)
    
    if failed_chars:
        print("以下の文字を前処理で除去または置換することを推奨：")
        print(f"問題文字: {', '.join(f'「{c}」' for c in sorted(failed_chars))}")
        
        print("\n推奨される前処理コード：")
        print("""
def preprocess_for_pyttsx3(text: str) -> str:
    # 問題のある文字を置換
    replacements = {
        '。': ' ',  # 句点をスペースに
        '、': ' ',  # 読点をスペースに
        '！': '',   # 感嘆符を削除
        '？': '',   # 疑問符を削除
        '「': ' ',  # かぎ括弧を削除
        '」': ' ',
        '『': ' ',
        '』': ' ',
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # 連続スペースを単一スペースに
    text = ' '.join(text.split())
    
    return text.strip()
        """)
    else:
        print("✅ 特定の文字での問題は検出されませんでした。")


def test_engine_persistence():
    """エンジンの永続性テスト（同じエンジンを使い続けた場合）"""
    print("\n" + "="*60)
    print("エンジン永続性テスト")
    print("="*60)
    print("同じエンジンで連続してテキストを処理します。")
    
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    for voice in voices:
        if 'haruka' in voice.name.lower():
            engine.setProperty('voice', voice.id)
            break
    
    test_texts = [
        "こんにちは",
        "すごい！",
        "良い天気です。",
        "さようなら",
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n{i}. テスト: '{text}'")
        try:
            start = time.time()
            engine.say(text)
            engine.runAndWait()
            duration = time.time() - start
            
            if duration > 0.5:
                print(f"  ✅ 音声出力あり ({duration:.2f}秒)")
            else:
                print(f"  ❌ 音声出力なし ({duration:.2f}秒)")
                print("  → エンジンが壊れた可能性")
                break
        except Exception as e:
            print(f"  ❌ エラー: {e}")
            break
    
    engine.stop()
    del engine


if __name__ == "__main__":
    print("調査モードを選択してください：")
    print("1. 問題文字の特定（推奨）")
    print("2. エンジン永続性テスト")
    print("3. 両方実行")
    
    choice = input("\n選択 (1/2/3): ").strip()
    
    if choice == "1":
        test_problem_characters()
    elif choice == "2":
        test_engine_persistence()
    elif choice == "3":
        test_problem_characters()
        input("\nEnterキーを押して永続性テストへ...")
        test_engine_persistence()
    else:
        print("無効な選択です")
