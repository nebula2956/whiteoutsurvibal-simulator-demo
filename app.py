"""Streamlit Community Cloud エントリーポイント。"""
import runpy, sys, os

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.dirname(__file__))

# ui/app.py を実行
runpy.run_path(os.path.join(os.path.dirname(__file__), "ui", "app.py"), run_name="__main__")
