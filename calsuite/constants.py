"""Every numeric threshold in the foundation layer, with the reason for its
value -- same convention as hydrationTracker's ``model/constants.py``. A
constant with no comment explaining *why* that number and not another is a
bug waiting to be filed. Wave 2 areas (camera/, lens/, display/) add their
own analysis-specific thresholds in their own modules, next to the checks
that use them, for the same reason store.py keeps SHELF_LIFE_DAYS' reasons
next to the table rather than here.
"""

from __future__ import annotations

# -- EDID (VESA E-EDID 1.4) ------------------------------------------------

EDID_BASE_BLOCK_LENGTH = 128
# The EDID 1.x base block is fixed at 128 bytes, ending in a checksum byte
# that must make the whole block sum to 0 mod 256. Extension blocks (CTA-861
# etc.) are separate 128-byte blocks this parser does not read, because
# nothing we need (chromaticity, physical size, gamma, name/serial
# descriptors) lives outside the base block.

EDID_HEADER_MAGIC = bytes([0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x00])
# The fixed 8-byte pattern every EDID block starts with. Checked before
# attempting to parse a blob as EDID at all, so a garbage read (e.g. an
# unplugged sysfs EDID node, which reads back as 0 bytes) fails with a clear
# message instead of a confusing IndexError deep in the chromaticity math.

EDID_CHROMATICITY_STEPS = 1024
# Each chromaticity coordinate is a 10-bit fixed-point fraction of 1.0
# (VESA E-EDID 1.4 §3.10.2), i.e. steps of 1/1024 ≈ 0.000977. That is where
# the "±0.001" precision the devices.py EDID tests check against comes from
# -- it's the format's actual resolution, not a chosen tolerance.

EDID_YEAR_OFFSET = 1990
# EDID stores manufacture year as (year - 1990) in a single byte (bytes
# 1990-2245 fit in 8 bits), per §3.4.

# -- record / device identity ----------------------------------------------

DEVICE_ID_SERIAL_HASH_LEN = 8
# 8 hex chars of sha256(serial) = 32 bits of a one-way hash. Collisions
# between two serials of the *same* model landing in the same device id are
# astronomically unlikely for a personal collection of a handful of bodies,
# and 8 chars keeps directory names short. The full serial is never stored
# (devices.py) -- this hash exists only to tell two units of one model
# apart, never to be reversed back into the serial.

# -- accuracy estimates (docs/design.md §10) -------------------------------
# Not measured yet -- every value here is a placeholder a real measurement
# will supersede. House rule 7: "every estimate in this plan gets replaced
# by a measured value once it exists." Reports must show these labeled as
# *estimates* to compare an achieved result against, never assert them as
# fact. Units and meaning match the design doc's table exactly.

ESTIMATED_ACCURACY = {
    "gain_pct": 2.5,  # ±2-3% with 20+ PTC pairs -- source stability, pair count
    "read_noise_pct": 5.0,  # ±5% -- limited by bias pair count
    "distortion_rms_px": 0.3,  # < 0.3 px RMS reprojection -- target coverage/flatness
    "vignetting_pct": 1.0,  # ±1% -- source non-uniformity (why the flat is self-calibrating)
    "mtf50_pct": 5.0,  # ±5% -- edge quality, focus repeatability
    "display_colorimeter_de00": 1.0,  # ΔE00 ≈ 1, instrument-limited
    "display_camera_de00": 3.5,  # ΔE00 2-5 (midpoint kept here); camera metamerism vs. narrow primaries
}
