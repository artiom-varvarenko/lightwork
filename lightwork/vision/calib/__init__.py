"""Camera-pixel to projector-page calibration (C7; D5).

Binding API: ``pattern_for_page``, ``solve``, ``calibrate``. The page must ACK its device-snapped marker rectangles;
never solve against the requested rectangles. ``calibrate`` writes ``.lightwork/calib/calib.json`` by default.

Rig workflow (two terminals):
  ``uv run lightwork calibrate`` and ``uv run lightwork tank up``.
Print ``lightwork calibrate card --output /tmp/lightwork-card.svg`` at 100 %, check its 100 mm ruler, then run
``lightwork calibrate check`` for C1. C2: ``lightwork calibrate --repeat 10 --mm nominal`` with partial occlusion.
C3: ``lightwork calibrate watch --check-after-bump``; nudge camera then phone, three times each, hands out.
C4: put a white board at plate height, ``lightwork calibrate plate``, then the card on that plane and
``lightwork calibrate check --plane plate``. Each command starts its own local server; stop live/replay first.

Flat fields are optional: remove all objects and printed cards, then ``lightwork calibrate flat --work-area X Y W H``.
Only this explicitly selected crop is saved. Pass its C7 origin to MeasureE's ``cell_stats(flat_field_origin=...)``;
do not pad the crop (zero padding biases its gain/Lab normalisation).
A physical bump invalidates the flat field and plate-plane H; re-capture them, and update/recheck the layout if
``recalibrate`` reports camera motion or an unobservable cause. BumpMonitor requires settled, hand-free frames.
"""

from .flow import calibrate
from .pattern import pattern_for_page
from .solve import CalibrationError, solve

__all__ = ["CalibrationError", "calibrate", "pattern_for_page", "solve"]
