"""Colour and optical-density readings from decoded keyframes (pure; PLAN.md §6.1, ``briefs/rig.md`` §6).

Pipeline per cell:

1. Pixels of the ROI disc (:mod:`lightwork.measure.grid`) are taken from the *decoded keyframe JPEG*; live and
   replay measure the same bytes, so the numbers are identical (PLAN.md §4.2).
2. Pixels with any channel ``>= reject_ge`` (default 250: specular highlights, clipping) are dropped.
3. sRGB is linearised per pixel (IEC 61966-2-1 transfer curve, exact 256-entry table).
4. Optional flat field: the linear image is divided by a linearised frame of the empty white board taken in the same
   measurement mode, normalised per channel by its mean, which cancels projector/lamp non-uniformity.
5. Outliers on the measured channel (bubbles, debris) beyond ``median +- max(4 x 1.4826 MAD, 2 % of median)`` are
   dropped; the survivors are averaged in linear light.

``OD = -log10(I / I_ref)`` on the dye's complementary channel: ``"r"`` for blue dye, ``"b"`` for tea or coffee.
``I_ref`` is the same cell before dye arrived, or a blank cell in the same frame (the reducer picks; see H).

Lab is CIE L*a*b* of the mean *linear* BGR colour via OpenCV ``COLOR_LBGR2Lab`` on float32 input, which uses the
sRGB primaries and the **D65** white point (Xn = 0.950456, Zn = 1.088754); linear white (1, 1, 1) is L* = 100.
Colour differences use CIEDE2000.

All values that reach events are quantized with the contract helpers (``q_od``, ``q_lab``).

Run as a script for the home colour test ``rig:D1`` (separable 1:2 steps in a photo series)::

    uv run python -m lightwork.measure.colour --help
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ..contracts.canonical import q_lab, q_od
from ..contracts.stepgraph import Grid
from ..geometry import GridGeom
from .grid import cell_rois, disc_indices

CHANNELS = {"b": 0, "g": 1, "r": 2}

# Exact sRGB -> linear table for 8-bit values.
_V = np.arange(256, dtype=np.float64) / 255.0
SRGB_TO_LINEAR = np.where(_V <= 0.04045, _V / 12.92, ((_V + 0.055) / 1.055) ** 2.4)
del _V

# Linear intensity floor for OD: one tenth of the darkest non-zero 8-bit level, caps OD near 4.5 on a white reference.
I_FLOOR = float(SRGB_TO_LINEAR[1]) / 10.0
# Robust clipping band on the measured channel (see module docstring, step 5).
CLIP_K = 4.0
CLIP_REL = 0.02
# Linear intensities are rounded before use so tiny summation-order differences cannot reach quantized outputs.
_I_DIGITS = 7


def decode(jpeg: bytes) -> np.ndarray:
    """Decode JPEG (or PNG) bytes to a BGR uint8 image; raises ``ValueError`` if the bytes are not an image."""
    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("bytes are not a decodable image")
    return img


@dataclass(frozen=True)
class CellStats:
    """One cell's reading.

    ``i_lin``: mean linear intensity of the chosen channel (flat-field corrected when a flat field was given).
    ``lab``: CIE Lab (D65) of the mean linear colour, quantized to 0.01.
    ``n``: pixels used; ``n == 0`` means nothing usable (disc outside the image or every pixel rejected) and the
    other fields are zero; callers must treat the cell as unreadable (check ``inconclusive``).
    ``sd``: per-pixel standard deviation of the linear channel intensity (same units as ``i_lin``).
    """

    i_lin: float
    lab: tuple[float, float, float]
    n: int
    sd: float


_EMPTY = CellStats(i_lin=0.0, lab=(0.0, 0.0, 0.0), n=0, sd=0.0)


def _flat_gain(flat_field: np.ndarray) -> np.ndarray:
    """Per-channel mean of the linearised flat field (normaliser so corrected values stay in intensity units)."""
    return SRGB_TO_LINEAR[flat_field].reshape(-1, 3).mean(axis=0)


def cell_stats(
    img: np.ndarray,
    centres: Mapping[str, tuple[float, float]],
    radius_px: float,
    channel: str,
    *,
    origin: tuple[float, float] = (0, 0),
    flat_field: np.ndarray | None = None,
    flat_field_origin: tuple[float, float] = (0, 0),
    reject_ge: int = 250,
) -> dict[str, CellStats]:
    """Read every cell's ROI disc from ``img`` (BGR uint8, usually :func:`decode` of a keyframe crop).

    ``centres`` are camera px; ``origin`` is the camera-px position of ``img``'s top-left pixel (the keyframe's
    ``crop_xywh[:2]``). ``radius_px`` is the ROI radius (:func:`lightwork.measure.grid.roi_radius`).
    ``flat_field`` is a BGR uint8 work-area crop; ``flat_field_origin`` locates its top-left in camera px.
    Pixels where it is clipped (``>= reject_ge``) or black are rejected.
    """
    if channel not in CHANNELS:
        raise ValueError(f"channel must be one of {sorted(CHANNELS)}, got {channel!r}")
    if img.ndim != 3 or img.shape[2] != 3 or img.dtype != np.uint8:
        raise ValueError("img must be a BGR uint8 image")
    ci = CHANNELS[channel]
    ox, oy = int(round(origin[0])), int(round(origin[1]))
    if flat_field is not None:
        if flat_field.ndim != 3 or flat_field.shape[2] != 3 or flat_field.dtype != np.uint8:
            raise ValueError("flat_field must be a BGR uint8 image")
        fh, fw = flat_field.shape[:2]
        fx = ox - int(round(flat_field_origin[0]))
        fy = oy - int(round(flat_field_origin[1]))
        if fy < 0 or fx < 0 or fy + img.shape[0] > fh or fx + img.shape[1] > fw:
            raise ValueError("flat_field does not cover the image at origin")
        gain = _flat_gain(flat_field)
        if not (gain > 0).all():
            raise ValueError("flat_field has a black channel")
    out: dict[str, CellStats] = {}
    for name in sorted(centres):
        cx, cy = centres[name]
        ys, xs = disc_indices(img.shape, (float(cx) - ox, float(cy) - oy), radius_px)
        px = img[ys, xs]
        keep = (px < reject_ge).all(axis=1)
        if flat_field is not None:
            fpx = flat_field[ys + fy, xs + fx]
            keep &= (fpx < reject_ge).all(axis=1) & (fpx > 0).all(axis=1)
            lin = SRGB_TO_LINEAR[px[keep]] / (SRGB_TO_LINEAR[fpx[keep]] / gain)
        else:
            lin = SRGB_TO_LINEAR[px[keep]]
        if len(lin) == 0:
            out[name] = _EMPTY
            continue
        ch = lin[:, ci]
        med = float(np.median(ch))
        band = max(CLIP_K * 1.4826 * float(np.median(np.abs(ch - med))), CLIP_REL * med)
        lin = lin[np.abs(ch - med) <= band]
        ch = lin[:, ci]
        mean_bgr = lin.mean(axis=0)
        lab = cv2.cvtColor(mean_bgr.astype(np.float32).reshape(1, 1, 3), cv2.COLOR_LBGR2Lab)[0, 0]
        out[name] = CellStats(
            i_lin=round(float(ch.mean()), _I_DIGITS),
            lab=(q_lab(lab[0]), q_lab(lab[1]), q_lab(lab[2])),
            n=int(len(ch)),
            sd=round(float(ch.std()), _I_DIGITS),
        )
    return out


def od(i_lin: float, i_ref: float) -> float:
    """``-log10(i_lin / i_ref)``, quantized to 1e-4 (``q_od``).

    Signed on purpose: a value below 0 means the cell is brighter than its reference (noise, lighting drift), which
    the checks need to see. ``i_lin`` below :data:`I_FLOOR` (black) is clamped to it. Raises ``ValueError`` for a
    non-positive reference or a negative intensity (e.g. from an unreadable ``n == 0`` cell, which must not reach
    here).
    """
    if not (i_ref > 0 and math.isfinite(i_ref)):
        raise ValueError(f"reference intensity must be positive, got {i_ref!r}")
    if not (i_lin >= 0 and math.isfinite(i_lin)):
        raise ValueError(f"intensity must be >= 0, got {i_lin!r}")
    return q_od(-math.log10(max(i_lin, I_FLOOR) / i_ref))


def od_sd(stats: CellStats) -> float:
    """First-order per-pixel SD of the cell's OD (``sd / (i_lin ln 10)``), quantized; 0 for an unreadable cell."""
    if stats.n == 0 or stats.i_lin <= 0:
        return 0.0
    return q_od(stats.sd / (max(stats.i_lin, I_FLOOR) * math.log(10)))


def delta_e(lab1: Sequence[float], lab2: Sequence[float]) -> float:
    """CIEDE2000 colour difference (kL = kC = kH = 1), quantized to 0.01 (``q_lab``)."""
    L1, a1, b1 = (float(v) for v in lab1)
    L2, a2, b2 = (float(v) for v in lab2)
    c_bar = (math.hypot(a1, b1) + math.hypot(a2, b2)) / 2
    g = 0.5 * (1 - math.sqrt(c_bar**7 / (c_bar**7 + 25.0**7)))
    a1p, a2p = (1 + g) * a1, (1 + g) * a2
    c1p, c2p = math.hypot(a1p, b1), math.hypot(a2p, b2)
    h1p = math.degrees(math.atan2(b1, a1p)) % 360 if c1p else 0.0
    h2p = math.degrees(math.atan2(b2, a2p)) % 360 if c2p else 0.0
    dLp, dCp = L2 - L1, c2p - c1p
    if c1p * c2p == 0:
        dhp = 0.0
    elif abs(h2p - h1p) <= 180:
        dhp = h2p - h1p
    elif h2p - h1p > 180:
        dhp = h2p - h1p - 360
    else:
        dhp = h2p - h1p + 360
    dHp = 2 * math.sqrt(c1p * c2p) * math.sin(math.radians(dhp) / 2)
    l_bar, cp_bar = (L1 + L2) / 2, (c1p + c2p) / 2
    if c1p * c2p == 0:
        hp_bar = h1p + h2p
    elif abs(h1p - h2p) <= 180:
        hp_bar = (h1p + h2p) / 2
    elif h1p + h2p < 360:
        hp_bar = (h1p + h2p + 360) / 2
    else:
        hp_bar = (h1p + h2p - 360) / 2
    t = (
        1
        - 0.17 * math.cos(math.radians(hp_bar - 30))
        + 0.24 * math.cos(math.radians(2 * hp_bar))
        + 0.32 * math.cos(math.radians(3 * hp_bar + 6))
        - 0.20 * math.cos(math.radians(4 * hp_bar - 63))
    )
    d_theta = 30 * math.exp(-(((hp_bar - 275) / 25) ** 2))
    r_c = 2 * math.sqrt(cp_bar**7 / (cp_bar**7 + 25.0**7))
    s_l = 1 + 0.015 * (l_bar - 50) ** 2 / math.sqrt(20 + (l_bar - 50) ** 2)
    s_c = 1 + 0.045 * cp_bar
    s_h = 1 + 0.015 * cp_bar * t
    r_t = -math.sin(math.radians(2 * d_theta)) * r_c
    de = math.sqrt(
        (dLp / s_l) ** 2 + (dCp / s_c) ** 2 + (dHp / s_h) ** 2 + r_t * (dCp / s_c) * (dHp / s_h)
    )
    return q_lab(de)


def _labels(n: int, names: Sequence[str] | None) -> list[str]:
    if names is None:
        return [f"#{i + 1}" for i in range(n)]
    if len(names) != n:
        raise ValueError("names must match the values")
    return list(names)


def series_ok(
    ods: Sequence[float],
    ratio: float,
    tol: float,
    floor: float = 0.03,
    *,
    slack: float = 0.01,
    names: Sequence[str] | None = None,
) -> tuple[bool, str]:
    """Dilution-series test over ODs in series order (PLAN.md §6.1 ``series``).

    For each adjacent pair (a, b) where b > ``floor``: ``log(b / a) / log(ratio)`` (the number of dilution steps
    between them; ``log2(a / b)`` for ratio 0.5) must lie in ``1 +- tol``. Where b <= ``floor`` the dye is below the
    camera's noise floor and only monotonic non-increase is required (b <= a + ``slack``). At least one pair must be
    ratio-checked, so a blank row never passes. ``names`` label the cells in the reason string.
    """
    if not 0 < ratio < 1:
        raise ValueError(f"ratio must be in (0, 1), got {ratio!r}")
    labels = _labels(len(ods), names)
    if len(ods) < 2:
        return False, "series needs at least 2 cells"
    step = math.log(ratio)
    checked = 0
    for i in range(len(ods) - 1):
        a, b = float(ods[i]), float(ods[i + 1])
        pair = f"{labels[i]}->{labels[i + 1]}"
        if b > floor:
            if a <= floor:
                return False, f"{pair}: OD rises from {a:.4f} to {b:.4f}"
            steps = math.log(b / a) / step
            if abs(steps - 1) > tol:
                return False, f"{pair}: {steps:.2f} dilution steps (OD {a:.4f} -> {b:.4f}), expected 1+-{tol:g}"
            checked += 1
        elif b > a + slack:
            return False, f"{pair}: OD rises from {a:.4f} to {b:.4f}"
    if checked == 0:
        return False, f"no cell above OD {floor:g}"
    return True, f"{checked} dilution steps within 1+-{tol:g}; {len(ods) - 1 - checked} pairs below OD {floor:g} monotonic"


def transfer_ok(
    od_src_after: float,
    od_tgt_after: float,
    od_src_before: float,
    od_tgt_before: float = 0.0,
    *,
    min_delta: float = 0.02,
    drop_frac: float = 0.25,
    ratio_lo: float = 0.75,
    ratio_hi: float = 1.25,
    floor: float = 0.03,
    names: tuple[str, str] = ("source", "target"),
) -> tuple[str, str]:
    """Check one equal-volume transfer of a 1:2 dilution (``briefs/procedure.md`` §1 physics).

    After transferring half of the source into an equal volume of diluent, source and target hold the same amount of
    dye: the source loses half its OD and the target gains it. Returns ``(status, reason)`` with status ``pass``,
    ``fail`` or ``inconclusive`` (C1 check vocabulary), testing in this order:

    1. ``od_src_before < max(floor, 3 * min_delta)``: ``inconclusive``. Below that the expected change
       (``od_src_before / 2``) is under 1.5 x ``min_delta``, so a correct transfer could not be told from noise.
    2. Target darker: ``od_tgt_after - od_tgt_before >= max(min_delta, drop_frac * od_src_before)``.
    3. Source lighter: ``od_src_before - od_src_after >=`` the same threshold (expected gain/loss is half).
    4. Equal OD: ``od_tgt_after / od_src_after`` in ``[ratio_lo, ratio_hi]`` when both are above ``floor``
       (catches a double dispense or a wrong pipette volume); below the floor the ratio is not tested.

    ODs are relative to each cell's pre-dye reference. ``names`` label (source, target) in the reason.
    """
    src, tgt = names
    readable = max(floor, 3 * min_delta)
    if od_src_before < readable:
        return "inconclusive", f"{src} OD {od_src_before:.4f} is below the readable {readable:g}"
    need = max(min_delta, drop_frac * od_src_before)
    gain = od_tgt_after - od_tgt_before
    if gain < need:
        return "fail", f"{tgt} did not darken (OD change {gain:+.4f}, need +{need:.4f})"
    drop = od_src_before - od_src_after
    if drop < need:
        return "fail", f"{src} did not get lighter (OD change {-drop:+.4f}, need -{need:.4f})"
    if od_src_after > floor and od_tgt_after > floor:
        r = od_tgt_after / od_src_after
        if not ratio_lo <= r <= ratio_hi:
            return "fail", f"{tgt}/{src} OD ratio {r:.2f} outside {ratio_lo:g}-{ratio_hi:g}"
        return "pass", f"{src} {od_src_before:.4f} -> {od_src_after:.4f}, {tgt} {od_tgt_after:.4f} (ratio {r:.2f})"
    return "pass", f"{src} {od_src_before:.4f} -> {od_src_after:.4f}, {tgt} {od_tgt_after:.4f} (ratio below floor)"


def wrong_cell(
    expected: str,
    deltas: Mapping[str, float],
    *,
    exclude: Sequence[str] = (),
    min_delta: float = 0.02,
) -> str | None:
    """The cell that darkened instead of ``expected`` (PLAN.md §6.2 hero recovery: "liquid went into A7, expected A6").

    ``deltas`` maps cell names to their OD change over the step (after - before); pass the candidate cells only
    (e.g. :func:`lightwork.measure.grid.neighbours` of the target). ``exclude`` holds cells that legitimately change
    (the source). Returns ``None`` when ``expected`` darkened by ``min_delta`` or no other cell did; otherwise the
    candidate with the largest darkening (ties: first in name order).
    """
    if deltas.get(expected, 0.0) >= min_delta:
        return None
    skip = {expected, *exclude}
    cands = [(name, d) for name, d in sorted(deltas.items()) if name not in skip and d >= min_delta]
    if not cands:
        return None
    return max(cands, key=lambda nd: nd[1])[0]


# ---------------------------------------------------------------------------------------------------- rig:D1 tool


def _parse_xy(s: str) -> tuple[float, float]:
    x, y = s.split(",")
    return float(x), float(y)


def _read_image(path: Path) -> np.ndarray:
    return decode(path.read_bytes())


def separable_steps(means: Sequence[float], sds: Sequence[float], *, k_sd: float = 3.0, floor: float = 0.03,
                    lo: float = 0.35, hi: float = 0.65) -> int:
    """Leading adjacent steps that are told apart (``briefs/procedure.md`` §2 pass rule).

    Step i -> i+1 counts when ``means[i] - means[i+1] > k_sd x pooled SD`` and, where ``means[i+1] > floor``, the
    ratio ``means[i+1] / means[i]`` lies in ``[lo, hi]``. Counting stops at the first step that fails.
    """
    n = 0
    for i in range(len(means) - 1):
        a, b = means[i], means[i + 1]
        pooled = math.sqrt((sds[i] ** 2 + sds[i + 1] ** 2) / 2)
        if not a - b > k_sd * pooled:
            break
        if b > floor and not (a > 0 and lo <= b / a <= hi):
            break
        n += 1
    return n


def _rois(args: argparse.Namespace) -> tuple[list[str], dict[str, tuple[float, float]], float]:
    corners = np.array([_parse_xy(c) for c in args.corners], dtype=np.float64)
    geom = GridGeom(corners, Grid(rows=args.rows, cols=args.cols, naming="A1"))
    cells = list(args.cells) if args.cells else [c for c in geom.names() if c != args.blank]
    centres, radius = cell_rois(geom, cells + ([args.blank] if args.blank else []))
    return cells, centres, args.radius or radius


def _series_report(args: argparse.Namespace) -> dict:
    cells, centres, radius = _rois(args)
    flat = _read_image(args.flat) if args.flat else None

    def read(path: Path) -> dict[str, CellStats]:
        st = cell_stats(_read_image(path), centres, radius, args.channel, flat_field=flat)
        bad = [c for c in sorted(centres) if st[c].n == 0]
        if bad:
            raise ValueError(f"{path}: no usable pixels in {bad}")
        return st

    ref_i = {}
    if args.ref:
        ref_reads = [read(p) for p in args.ref]
        ref_i = {c: float(np.mean([r[c].i_lin for r in ref_reads])) for c in cells}
    reads = [read(p) for p in args.images]
    per_frame = [{c: od(st[c].i_lin, st[args.blank].i_lin if args.blank else ref_i[c]) for c in cells} for st in reads]

    means = [float(np.mean([f[c] for f in per_frame])) for c in cells]
    if len(per_frame) >= 2:
        sds = [float(np.std([f[c] for f in per_frame], ddof=1)) for c in cells]
        sd_kind = "frame-to-frame SD"
    else:
        st = reads[0]
        sds = [st[c].sd / (max(st[c].i_lin, I_FLOOR) * math.log(10)) / math.sqrt(st[c].n) for c in cells]
        sd_kind = "pixel standard error (single frame: no drift included)"
    ok, reason = series_ok([q_od(m) for m in means], args.ratio, args.tol, names=cells)
    return {
        "channel": args.channel,
        "frames": len(per_frame),
        "roi_radius_px": round(radius, 2),
        "reference": f"blank cell {args.blank}" if args.blank else f"{len(args.ref)} pre-dye image(s)",
        "sd_kind": sd_kind,
        "cells": {c: {"od": round(m, 4), "sd": round(s, 4)} for c, m, s in zip(cells, means, sds, strict=True)},
        "separable_steps": separable_steps(means, sds),
        "series_ok": ok,
        "series_reason": reason,
    }


def _annotate(args: argparse.Namespace) -> None:
    _, centres, radius = _rois(args)
    img = _read_image(args.images[0]).copy()
    for name, (x, y) in centres.items():
        colour = (0, 200, 255) if name == args.blank else (0, 0, 255)
        cv2.circle(img, (round(x), round(y)), max(round(radius), 1), colour, 1, cv2.LINE_AA)
        cv2.putText(img, name, (round(x + radius + 2), round(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1, cv2.LINE_AA)
    if not cv2.imwrite(str(args.annotate), img):
        raise ValueError(f"cannot write {args.annotate}")


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m lightwork.measure.colour",
        description="rig:D1 colour test: read a 1:2 dilution series from photos and count separable steps.",
    )
    p.add_argument("images", nargs="+", type=Path, help="photos/keyframes of the filled series (same framing)")
    p.add_argument("--corners", nargs=4, required=True, metavar="X,Y",
                   help="image px centres of the corner cells TL (A1), TR, BR, BL")
    p.add_argument("--rows", type=int, default=1)
    p.add_argument("--cols", type=int, required=True)
    p.add_argument("--cells", nargs="+", help="cells of the series in dilution order (default: all, row-major)")
    ref = p.add_mutually_exclusive_group(required=True)
    ref.add_argument("--blank", help="blank (diluent-only) cell used as I_ref in each frame")
    ref.add_argument("--ref", nargs="+", type=Path, help="pre-dye photo(s) of the same cells used as I_ref")
    p.add_argument("--channel", choices=sorted(CHANNELS), default="b", help="b for tea/coffee, r for blue dye")
    p.add_argument("--flat", type=Path, help="flat-field photo of the empty white board, same framing")
    p.add_argument("--radius", type=float, help="ROI radius px (default 0.19 x cell pitch)")
    p.add_argument("--ratio", type=float, default=0.5)
    p.add_argument("--tol", type=float, default=0.15)
    p.add_argument("--min-steps", type=int, default=6, help="exit 1 if fewer separable steps (default 6)")
    p.add_argument("--annotate", type=Path, help="write the first image with the ROI discs drawn, to check placement")
    p.add_argument("--json", action="store_true", help="print the report as JSON")
    args = p.parse_args(argv)
    if args.cells and args.blank and args.blank in args.cells:
        p.error("--blank must not be one of --cells")
    try:
        if args.annotate:
            _annotate(args)
        rep = _series_report(args)
    except (OSError, ValueError, KeyError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(rep, indent=2, sort_keys=True))
    else:
        print(f"channel {rep['channel']}, {rep['frames']} frame(s), ROI r={rep['roi_radius_px']} px, "
              f"I_ref: {rep['reference']}; SD = {rep['sd_kind']}")
        prev = None
        for c, v in rep["cells"].items():
            extra = ""
            if prev is not None and prev > 0:
                extra = f"  ratio {v['od'] / prev:.2f}"
            print(f"  {c:>4}  OD {v['od']:+.4f} +- {v['sd']:.4f}{extra}")
            prev = v["od"]
        print(f"separable steps: {rep['separable_steps']} (target >= {args.min_steps})")
        print(f"series({args.ratio:g}, {args.tol:g}): {'ok' if rep['series_ok'] else 'FAIL'} - {rep['series_reason']}")
    return 0 if rep["separable_steps"] >= args.min_steps else 1


if __name__ == "__main__":
    raise SystemExit(main())
