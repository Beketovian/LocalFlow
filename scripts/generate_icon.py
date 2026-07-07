"""Draw the LocalFlow app icon (the recording pill's look, as a squircle).

Usage: venv/bin/python scripts/generate_icon.py OUT.png [SIZE]
Pure AppKit so it needs no extra dependencies beyond pyobjc.
"""

from __future__ import annotations

import sys

from AppKit import (
    NSBezierPath,
    NSBitmapImageRep,
    NSColor,
    NSGraphicsContext,
    NSMakeRect,
    NSPNGFileType,
)
from Foundation import NSData  # noqa: F401  (linked by pyobjc)


def rgb(spec: str, alpha: float = 1.0) -> NSColor:
    r, g, b = (int(spec[i:i + 2], 16) / 255.0 for i in (1, 3, 5))
    return NSColor.colorWithCalibratedRed_green_blue_alpha_(r, g, b, alpha)


def draw(size: int) -> NSBitmapImageRep:
    rep = NSBitmapImageRep.alloc().initWithBitmapDataPlanes_pixelsWide_pixelsHigh_bitsPerSample_samplesPerPixel_hasAlpha_isPlanar_colorSpaceName_bytesPerRow_bitsPerPixel_(
        None, size, size, 8, 4, True, False, "NSCalibratedRGBColorSpace", 0, 0
    )
    ctx = NSGraphicsContext.graphicsContextWithBitmapImageRep_(rep)
    NSGraphicsContext.saveGraphicsState()
    NSGraphicsContext.setCurrentContext_(ctx)

    s = float(size)
    # macOS icon grid: content in the middle ~80% as a rounded squircle
    inset = s * 0.10
    body = NSMakeRect(inset, inset, s - 2 * inset, s - 2 * inset)
    rgb("#16161A").setFill()
    NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
        body, s * 0.18, s * 0.18
    ).fill()

    # coral recording dot on the left
    dot_r = s * 0.045
    dot_cx, cy = s * 0.30, s * 0.5
    rgb("#FF6A3D").setFill()
    NSBezierPath.bezierPathWithOvalInRect_(
        NSMakeRect(dot_cx - dot_r, cy - dot_r, dot_r * 2, dot_r * 2)
    ).fill()

    # waveform bars
    heights = [0.10, 0.22, 0.34, 0.20, 0.28, 0.14]
    bar_w = s * 0.030
    gap = s * 0.055
    x = s * 0.385
    rgb("#FFFFFF").setFill()
    for h in heights:
        bh = s * h * 2
        NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
            NSMakeRect(x, cy - bh / 2, bar_w, bh), bar_w / 2, bar_w / 2
        ).fill()
        x += gap

    NSGraphicsContext.restoreGraphicsState()
    return rep


def main() -> int:
    out = sys.argv[1]
    size = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
    rep = draw(size)
    rep.representationUsingType_properties_(NSPNGFileType, None).writeToFile_atomically_(
        out, True
    )
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
