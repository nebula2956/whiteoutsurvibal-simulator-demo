"""Streamlit rerun耐性のある最適化セッション管理。

Streamlitはウィジェット操作のたびにスクリプト全体を再実行するため、
最適化を別スレッドで実行し、session_stateで状態を管理する。
"""
from __future__ import annotations

import queue
import threading
import traceback
from dataclasses import dataclass
from typing import Optional

from simulator.models import ArmyConfig
from .genome import HeroPool
from .fitness import FitnessEvaluator, FitnessWeights
from .beam_runner import BeamSearchOptimizer, BeamSearchConfig, BeamSearchResult, PhaseInfo
from .ga_runner import GAOptimizer, GAConfig, GAResult


@dataclass
class SessionStatus:
    running: bool = False
    finished: bool = False
    error: Optional[str] = None


class OptimizerSession:
    """最適化の非同期実行を管理するセッションオブジェクト。

    Usage in Streamlit:
        if "opt_session" not in st.session_state:
            st.session_state["opt_session"] = None

        session = st.session_state["opt_session"]
        if session and session.status.running:
            # Poll for updates
            info = session.get_latest_info()
            if info: display(info)
            if session.status.finished:
                result = session.result
        else:
            if st.button("実行"):
                session = OptimizerSession()
                session.start(evaluator, pool, config)
                st.session_state["opt_session"] = session
    """

    def __init__(self):
        self.status = SessionStatus()
        self.result: Optional[BeamSearchResult] = None
        self._info_queue: queue.Queue[PhaseInfo] = queue.Queue()
        self._latest_info: Optional[PhaseInfo] = None
        self._thread: Optional[threading.Thread] = None
        self.score_history: list = []  # List of (overall_progress: float, best_score: float)
        self.info_history: list = []  # List of all PhaseInfo received

    # Phase weight mapping for overall progress calculation (must match optimizer_panel.py)
    _PHASE_WEIGHTS = {
        "leader_screen": (0.0, 0.3),
        "beam_search": (0.3, 0.7),
        "ga_search": (0.2, 0.7),
        "cma_refine": (0.7, 0.9),
        "reeval": (0.9, 1.0),
    }

    def start(self, evaluator: FitnessEvaluator, pool: HeroPool,
              config: BeamSearchConfig | GAConfig) -> None:
        """最適化を別スレッドで開始。"""
        if self.status.running:
            return

        self.status = SessionStatus(running=True)
        self.result = None
        self.score_history = []
        self.info_history = []

        self._thread = threading.Thread(
            target=self._run,
            args=(evaluator, pool, config),
            daemon=True,
        )
        self._thread.start()

    def _run(self, evaluator: FitnessEvaluator, pool: HeroPool,
             config: BeamSearchConfig | GAConfig) -> None:
        try:
            if isinstance(config, GAConfig):
                optimizer = GAOptimizer(evaluator, pool, config)
            else:
                optimizer = BeamSearchOptimizer(evaluator, pool, config)
            self.result = optimizer.run(
                callback=lambda info: self._info_queue.put(info)
            )
            self.status.finished = True
        except Exception as e:
            self.status.error = f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}"
            self.status.finished = True
        finally:
            self.status.running = False

    def get_latest_info(self) -> Optional[PhaseInfo]:
        """キューから最新のPhaseInfoを取得（非ブロッキング）。スコア履歴も更新。"""
        info = None
        while not self._info_queue.empty():
            try:
                info = self._info_queue.get_nowait()
                self.info_history.append(info)
                # Update score history for each received info
                lo, hi = self._PHASE_WEIGHTS.get(info.phase, (0.0, 1.0))
                overall = lo + (hi - lo) * info.progress
                if info.best_score > 0:
                    self.score_history.append((round(overall, 4), round(info.best_score, 4)))
            except queue.Empty:
                break
        if info is not None:
            self._latest_info = info
        return self._latest_info

    def is_running(self) -> bool:
        return self.status.running

    def is_finished(self) -> bool:
        return self.status.finished
