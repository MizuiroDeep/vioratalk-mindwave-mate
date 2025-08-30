#!/usr/bin/env python3
"""
pyttsx3がどのタイミングで壊れるか詳細調査
"""

import pyttsx3
import time


def test_timing():
    """エンジンがどのタイミングで壊れるか調査"""
    
    print("="*60)
    print("pyttsx3 タイミング調査")
    print("="*60)
    
    # テスト1: say()だけで壊れるか？
    print("\n【テスト1: say()のみ実行】")
    engine = pyttsx3.init()
    engine.say("テスト1")
    print("say()実行: OK")
    engine.say("テスト2")
    print("say()実行: OK（2回目）")
    # runAndWait()を実行せずに破棄
    del engine
    print("結果: say()だけでは壊れない")
    
    # テスト2: runAndWait()後に壊れるか？
    print("\n【テスト2: runAndWait()後の状態】")
    engine = pyttsx3.init()
    
    print("1回目:")
    start = time.time()
    engine.say("こんにちは")
    engine.runAndWait()
    duration = time.time() - start
    print(f"  実行時間: {duration:.2f}秒")
    
    print("2回目（同じエンジン）:")
    start = time.time()
    engine.say("さようなら")
    engine.runAndWait()
    duration = time.time() - start
    print(f"  実行時間: {duration:.2f}秒")
    
    if duration < 0.5:
        print("  → エンジンが壊れた！")
    
    del engine
    
    # テスト3: stop()で復活するか？
    print("\n【テスト3: stop()で復活するか】")
    engine = pyttsx3.init()
    
    print("1回目:")
    engine.say("テスト")
    engine.runAndWait()
    
    print("stop()実行...")
    engine.stop()
    
    print("2回目（stop後）:")
    start = time.time()
    engine.say("復活した？")
    engine.runAndWait()
    duration = time.time() - start
    print(f"  実行時間: {duration:.2f}秒")
    
    if duration > 0.5:
        print("  → stop()で復活！")
    else:
        print("  → stop()では復活しない")
    
    del engine
    
    # テスト4: プロパティ再設定で復活するか？
    print("\n【テスト4: プロパティ再設定】")
    engine = pyttsx3.init()
    
    print("1回目:")
    engine.say("初回")
    engine.runAndWait()
    
    print("プロパティ再設定...")
    voices = engine.getProperty('voices')
    for voice in voices:
        if 'haruka' in voice.name.lower():
            engine.setProperty('voice', voice.id)
            break
    engine.setProperty('rate', 120)
    
    print("2回目（再設定後）:")
    start = time.time()
    engine.say("復活？")
    engine.runAndWait()
    duration = time.time() - start
    print(f"  実行時間: {duration:.2f}秒")
    
    if duration > 0.5:
        print("  → プロパティ再設定で復活！")
    else:
        print("  → プロパティ再設定では復活しない")
    
    del engine
    
    # テスト5: エンジンのリセット方法を探す
    print("\n【テスト5: エンジン内部メソッド調査】")
    engine = pyttsx3.init()
    
    # 利用可能なメソッドを表示
    methods = [m for m in dir(engine) if not m.startswith('_')]
    print("利用可能なメソッド:", methods)
    
    del engine


if __name__ == "__main__":
    test_timing()
