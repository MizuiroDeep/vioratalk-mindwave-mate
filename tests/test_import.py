"""最小限のインポートテスト"""


def test_import_vioratalk():
    """vioratalkパッケージがインポートできることを確認"""
    import vioratalk

    assert vioratalk.__version__ == "0.3.0"


def test_author_info():
    """作者情報が正しく設定されているか確認"""
    import vioratalk

    assert vioratalk.__author__ == "MizuiroDeep"  # ← ここを修正
    assert vioratalk.__license__ == "MIT"
