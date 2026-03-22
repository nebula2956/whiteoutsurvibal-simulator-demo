"""永続キャッシュストア。expected modeの結果をSQLiteに保存。

expected modeは決定論的なので、一度計算した結果は永久にキャッシュ可能。
2回目以降のPhase 1リーダー探索はほぼゼロコストになる。
"""
from __future__ import annotations

import hashlib
import sqlite3
import threading
from pathlib import Path
from typing import Dict, Optional, Tuple


_DEFAULT_DB_PATH = Path(__file__).parent.parent / "cache" / "expected_cache.sqlite"

# expected modeの計算ロジック変更時にインクリメント → 古いキャッシュを自動無効化
CACHE_VERSION = 3


class ExpectedCacheStore:
    """SQLite-backed persistent cache for expected-mode evaluation scores."""

    def __init__(self, db_path: Path | str | None = None):
        self._db_path = Path(db_path) if db_path else _DEFAULT_DB_PATH
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        self._local = threading.local()
        # L1: in-memory cache for fast repeated lookups
        self._l1: Dict[str, float] = {}

        # Initialize DB schema
        conn = self._get_conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS expected_scores (
                key_hash TEXT PRIMARY KEY,
                score REAL NOT NULL
            )
        """)
        conn.commit()

        # バージョンチェック → 不一致なら全削除
        stored_version = self._get_version(conn)
        if stored_version != CACHE_VERSION:
            self.clear()
            self._set_version(conn, CACHE_VERSION)

    def _get_conn(self) -> sqlite3.Connection:
        """Thread-local connection (SQLite connections are not thread-safe)."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(str(self._db_path), timeout=10)
        return self._local.conn

    @staticmethod
    def _hash_key(key: Tuple) -> str:
        return hashlib.sha256(str(key).encode()).hexdigest()

    def get(self, key: Tuple) -> Optional[float]:
        """キャッシュからスコアを取得。L1→L2の順で探す。"""
        h = self._hash_key(key)

        # L1
        if h in self._l1:
            return self._l1[h]

        # L2 (SQLite)
        try:
            conn = self._get_conn()
            row = conn.execute(
                "SELECT score FROM expected_scores WHERE key_hash = ?", (h,)
            ).fetchone()
        except sqlite3.Error:
            return None

        if row is not None:
            self._l1[h] = row[0]
            return row[0]

        return None

    def put(self, key: Tuple, score: float) -> None:
        """スコアをキャッシュに保存（L1 + L2）。"""
        h = self._hash_key(key)
        self._l1[h] = score

        try:
            conn = self._get_conn()
            conn.execute(
                "INSERT OR REPLACE INTO expected_scores (key_hash, score) VALUES (?, ?)",
                (h, score),
            )
            conn.commit()
        except sqlite3.Error:
            pass  # DB書き込み失敗は致命的ではない

    def put_batch(self, items: list[Tuple[Tuple, float]]) -> None:
        """複数エントリを一括保存。"""
        if not items:
            return
        rows = []
        for key, score in items:
            h = self._hash_key(key)
            self._l1[h] = score
            rows.append((h, score))

        try:
            conn = self._get_conn()
            conn.executemany(
                "INSERT OR REPLACE INTO expected_scores (key_hash, score) VALUES (?, ?)",
                rows,
            )
            conn.commit()
        except sqlite3.Error:
            pass

    def size(self) -> int:
        """DB内のエントリ数。"""
        try:
            conn = self._get_conn()
            row = conn.execute("SELECT COUNT(*) FROM expected_scores").fetchone()
            return row[0] if row else 0
        except sqlite3.Error:
            return len(self._l1)

    def clear(self) -> None:
        """キャッシュを全削除。"""
        self._l1.clear()
        try:
            conn = self._get_conn()
            conn.execute("DELETE FROM expected_scores")
            conn.commit()
        except sqlite3.Error:
            pass

    def _get_version(self, conn: sqlite3.Connection) -> int:
        """格納済みのキャッシュバージョンを取得。"""
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_meta (
                    key TEXT PRIMARY KEY,
                    value INTEGER NOT NULL
                )
            """)
            conn.commit()
            row = conn.execute(
                "SELECT value FROM cache_meta WHERE key = 'version'"
            ).fetchone()
            return row[0] if row else 0
        except sqlite3.Error:
            return 0

    def _set_version(self, conn: sqlite3.Connection, version: int) -> None:
        """キャッシュバージョンを設定。"""
        try:
            conn.execute(
                "INSERT OR REPLACE INTO cache_meta (key, value) VALUES ('version', ?)",
                (version,),
            )
            conn.commit()
        except sqlite3.Error:
            pass
