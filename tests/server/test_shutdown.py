import threading

import pytest

from lightwork.server.hub import _Dispatcher


def test_close_drains_accepted_actions_and_rejects_late_actions():
    dispatcher = _Dispatcher('test-drain')
    delivered = []
    for i in range(20):
        dispatcher.submit(delivered.append, i)
    dispatcher.close()
    assert delivered == list(range(20))
    with pytest.raises(RuntimeError, match='closed'):
        dispatcher.submit(delivered.append, 99)
    dispatcher.close()
    assert delivered == list(range(20))


def test_close_reports_blocked_callback_and_can_finish_after_release():
    dispatcher = _Dispatcher('test-blocked')
    entered, release = threading.Event(), threading.Event()
    def callback():
        entered.set()
        release.wait()
    dispatcher.submit(callback)
    assert entered.wait(1)
    try:
        with pytest.raises(TimeoutError):
            dispatcher.close(timeout=0)
    finally:
        release.set()
        dispatcher.close()
