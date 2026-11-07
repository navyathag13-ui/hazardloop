import pytest

from app.video import SmoothedAlerts


def run(flags, **kw):
    s = SmoothedAlerts(**kw)
    return [s.update(f) for f in flags]


def test_single_frame_blip_is_ignored():
    assert run([False, True, False, False]) == [False, False, False, False]


def test_two_of_three_raises_an_alert():
    assert run([True, False, True, True]) == [False, False, True, True]


def test_alert_clears_when_violation_stops():
    assert run([True, True, False, False]) == [False, True, True, False]


def test_need_one_is_per_frame():
    assert run([False, True], window=1, need=1) == [False, True]


def test_bad_settings_rejected():
    with pytest.raises(ValueError):
        SmoothedAlerts(window=2, need=3)
