#!/usr/bin/env python3
"""Generates the physical/on-screen calibration targets: a ChArUco board
(PNG at exactly 1920x1200 for the laptop panel, plus a print-ready PDF), a
~5-degree slanted-edge target, and a gray ramp -- all at 1920x1200 PNG.
Board geometry (squares, square size in mm for each output) is recorded in
``board_geometry.json`` next to the images, per the Wave 2B task prompt.

Run with ``.venv/bin/python targets/generate_targets.py [--out targets/]``.
Uses ``calsuite.lens``'s board definition (``lens/charuco.py``,
``lens/constants.py``) so the generated target and the detector agree on
squares/marker ratio/dictionary -- see lens/constants.py's geometry
constants for why 10x6 squares at these particular sizes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from calsuite.lens import charuco
from calsuite.lens.constants import (
    BOARD_DICT_NAME,
    BOARD_MARKER_RATIO,
    BOARD_SQUARES_X,
    BOARD_SQUARES_Y,
    DISPLAY_PIXEL_PITCH_MM,
    DISPLAY_SQUARE_MM,
    DISPLAY_SQUARE_PX,
    PRINT_DPI,
    PRINT_SQUARE_MM,
)

DISPLAY_SIZE_PX = (1920, 1200)  # (width, height)


def _paste_centered(board_img: np.ndarray, canvas_size: tuple, background: int = 255) -> np.ndarray:
    width, height = canvas_size
    canvas = np.full((height, width), background, dtype=np.uint8)
    bh, bw = board_img.shape
    y0, x0 = (height - bh) // 2, (width - bw) // 2
    canvas[y0 : y0 + bh, x0 : x0 + bw] = board_img
    return canvas


def generate_charuco(out_dir: Path) -> dict:
    board = charuco.build_board(square_length=1.0)

    board_px_w = BOARD_SQUARES_X * DISPLAY_SQUARE_PX
    board_px_h = BOARD_SQUARES_Y * DISPLAY_SQUARE_PX
    display_board = charuco.render_board_image(board, (board_px_w, board_px_h))
    display_canvas = _paste_centered(display_board, DISPLAY_SIZE_PX)
    Image.fromarray(display_canvas).save(out_dir / "charuco_display.png")

    print_square_px = round(PRINT_SQUARE_MM / 25.4 * PRINT_DPI)
    print_board = charuco.render_board_image(
        board, (BOARD_SQUARES_X * print_square_px, BOARD_SQUARES_Y * print_square_px), margin_px=print_square_px // 2
    )
    Image.fromarray(print_board).convert("L").save(
        out_dir / "charuco_print.pdf", "PDF", resolution=float(PRINT_DPI)
    )

    return {
        "dictionary": BOARD_DICT_NAME,
        "squares_x": BOARD_SQUARES_X,
        "squares_y": BOARD_SQUARES_Y,
        "marker_ratio": BOARD_MARKER_RATIO,
        "display": {
            "file": "charuco_display.png",
            "size_px": list(DISPLAY_SIZE_PX),
            "square_size_px": DISPLAY_SQUARE_PX,
            "square_size_mm": DISPLAY_SQUARE_MM,
            "pixel_pitch_mm": DISPLAY_PIXEL_PITCH_MM,
        },
        "print": {
            "file": "charuco_print.pdf",
            "square_size_mm": PRINT_SQUARE_MM,
            "dpi": PRINT_DPI,
        },
    }


def generate_slanted_edge(out_dir: Path, angle_deg: float = 5.0) -> dict:
    width, height = DISPLAY_SIZE_PX
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
    cx = width / 2.0
    theta = np.radians(angle_deg)
    edge_x = cx + (yy - height / 2.0) * np.tan(theta)
    img = np.where(xx >= edge_x, 255, 0).astype(np.uint8)
    Image.fromarray(img).save(out_dir / "slanted_edge.png")
    return {"file": "slanted_edge.png", "size_px": list(DISPLAY_SIZE_PX), "angle_deg": angle_deg}


def generate_gray_ramp(out_dir: Path, n_steps: int = 25) -> dict:
    width, height = DISPLAY_SIZE_PX
    band_w = width // n_steps
    img = np.zeros((height, width), dtype=np.uint8)
    for i in range(n_steps):
        level = round(i / (n_steps - 1) * 255)
        img[:, i * band_w : (i + 1) * band_w if i < n_steps - 1 else width] = level
    Image.fromarray(img).save(out_dir / "gray_ramp.png")
    return {"file": "gray_ramp.png", "size_px": list(DISPLAY_SIZE_PX), "n_steps": n_steps}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    geometry = {
        "charuco": generate_charuco(args.out),
        "slanted_edge": generate_slanted_edge(args.out),
        "gray_ramp": generate_gray_ramp(args.out),
    }
    (args.out / "board_geometry.json").write_text(json.dumps(geometry, indent=2))
    print(f"wrote targets to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
