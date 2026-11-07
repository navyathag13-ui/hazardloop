import numpy as np

from app.detector import has_hard_hat, run_inspection


def test_blank_frame_has_no_people():
    assert run_inspection(np.zeros((320, 320, 3), dtype=np.uint8)) == []


def test_yellow_above_head_counts_as_hard_hat():
    frame = np.full((200, 200, 3), 90, dtype=np.uint8)
    frame[40:80, 60:140] = (0, 220, 255)  # BGR yellow where the hat sits
    assert has_hard_hat(frame, (70, 70, 60, 60))


def test_grey_above_head_is_not_a_hard_hat():
    frame = np.full((200, 200, 3), 90, dtype=np.uint8)
    assert not has_hard_hat(frame, (70, 70, 60, 60))
