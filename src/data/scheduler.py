"""
Background scheduler for AURA.
Runs the Data Agent at a configurable interval.

Usage:
    python -m src.data.scheduler
or imported and started by app.py on startup.
"""
import time
import threading
import logging
from src.utils.config import REFRESH_INTERVAL_MINUTES
from src.database.connection import init_db

logger = logging.getLogger(__name__)

_scheduler_thread: threading.Thread | None = None
_stop_event = threading.Event()


def _run_loop():
    """Worker loop — init DB then fetch data every N minutes."""
    # Import here to avoid circular imports at module load time
    from src.agents.data_agent import DataAgent

    init_db()
    agent = DataAgent()

    logger.info(
        f"[Scheduler] Started. Polling every {REFRESH_INTERVAL_MINUTES} minute(s)."
    )

    while not _stop_event.is_set():
        logger.info("[Scheduler] Triggering DataAgent for all cities...")
        try:
            agent.run_all_cities()
        except Exception as e:
            logger.error(f"[Scheduler] DataAgent error: {e}")

        # Wait for the interval or until stop is requested
        _stop_event.wait(timeout=REFRESH_INTERVAL_MINUTES * 60)

    logger.info("[Scheduler] Stopped.")


def start_scheduler():
    """Start the background scheduler thread (non-blocking)."""
    global _scheduler_thread, _stop_event
    if _scheduler_thread and _scheduler_thread.is_alive():
        logger.warning("[Scheduler] Already running.")
        return

    _stop_event.clear()
    _scheduler_thread = threading.Thread(
        target=_run_loop, name="aura-scheduler", daemon=True
    )
    _scheduler_thread.start()
    logger.info("[Scheduler] Thread launched.")


def stop_scheduler():
    """Signal the scheduler to stop after the current cycle."""
    _stop_event.set()
    logger.info("[Scheduler] Stop signal sent.")


def trigger_now():
    """Manually trigger one fetch cycle (blocking, for UI refresh buttons)."""
    from src.agents.data_agent import DataAgent
    init_db()
    agent = DataAgent()
    agent.run_all_cities()


if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    logger.info("Running scheduler in foreground (Ctrl+C to stop)...")
    init_db()
    _run_loop()
