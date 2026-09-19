"""Step 0 environment check.

Records the facts the plan asks for and proves the PyObjC bindings import and
run on this machine.

Deliberately does NOT capture the screen and does NOT request permission:
`CGPreflightScreenCaptureAccess` reports status without triggering a TCC
prompt, and the Vision check runs on an image this module draws in memory.
Pass --request to explicitly ask for Screen Recording permission.
"""

from __future__ import annotations

import argparse
import platform
import subprocess
import sys
from pathlib import Path

OK = "ok"
FAIL = "FAIL"

# Text drawn into the synthetic OCR fixture. Synthetic only -- never real data.
FIXTURE_TEXT = "Safe Share OCR check 12345"


def _run(*cmd: str) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def report_versions() -> None:
    print("== environment ==")
    print(f"  macOS          : {platform.mac_ver()[0]} (build {_run('sw_vers', '-buildVersion')})")
    print(f"  hardware       : {_run('sysctl', '-n', 'hw.model')} / {_run('sysctl', '-n', 'machdep.cpu.brand_string')}")
    print(f"  python         : {platform.python_version()} ({sys.executable})")

    import sysconfig

    framework = sysconfig.get_config_var("PYTHONFRAMEWORK") or "(not a framework build)"
    print(f"  python build   : PYTHONFRAMEWORK={framework}")

    teams = Path("/Applications/Microsoft Teams.app/Contents/Info.plist")
    if teams.exists():
        version = _run("defaults", "read", str(teams), "CFBundleShortVersionString")
        print(f"  teams          : {version}")
    else:
        print("  teams          : not installed")


def check_imports() -> bool:
    print("\n== bindings import ==")
    ok = True
    for name in ("objc", "Quartz", "ScreenCaptureKit", "Vision"):
        try:
            module = __import__(name)
            version = getattr(module, "__version__", "")
            print(f"  {name:<18} {OK} {version}")
        except Exception as exc:  # noqa: BLE001 - report any import failure
            print(f"  {name:<18} {FAIL} {type(exc).__name__}: {exc}")
            ok = False
    return ok


def check_screencapturekit_symbols() -> bool:
    """Confirm the classes we depend on are reachable, without capturing."""
    print("\n== ScreenCaptureKit symbols ==")
    import ScreenCaptureKit

    ok = True
    for name in ("SCShareableContent", "SCContentFilter", "SCStream", "SCStreamConfiguration"):
        present = hasattr(ScreenCaptureKit, name)
        print(f"  {name:<24} {OK if present else FAIL}")
        ok = ok and present
    return ok


def check_permission() -> bool:
    """Report Screen Recording status. Does not prompt."""
    print("\n== screen recording permission ==")
    import Quartz

    granted = bool(Quartz.CGPreflightScreenCaptureAccess())
    print(f"  granted        : {granted}")
    if not granted:
        print("  (run with --request to ask for it, or grant it in")
        print("   System Settings > Privacy & Security > Screen & System Audio Recording)")
    return granted


def request_permission() -> None:
    print("\n== requesting screen recording permission ==")
    import Quartz

    print(f"  CGRequestScreenCaptureAccess -> {bool(Quartz.CGRequestScreenCaptureAccess())}")
    print("  macOS may require restarting this process for a new grant to take effect.")


def _synthetic_image_data():
    """Render FIXTURE_TEXT to PNG bytes in memory. Touches no screen content.

    Drawn with AppKit rather than CoreText: the CoreText bindings are not part
    of pyobjc-framework-Quartz, and NSBitmapImageRep.CGImage() comes back as an
    unbridged pointer, so Vision is handed PNG data instead of a CGImage.
    """
    from AppKit import (
        NSAttributedString,
        NSBitmapImageRep,
        NSCalibratedRGBColorSpace,
        NSColor,
        NSFont,
        NSGraphicsContext,
        NSMakePoint,
        NSPNGFileType,
        NSRectFill,
    )
    from Foundation import NSMakeRect

    width, height = 700, 120
    rep = NSBitmapImageRep.alloc().initWithBitmapDataPlanes_pixelsWide_pixelsHigh_bitsPerSample_samplesPerPixel_hasAlpha_isPlanar_colorSpaceName_bytesPerRow_bitsPerPixel_(
        None, width, height, 8, 4, True, False, NSCalibratedRGBColorSpace, 0, 0
    )
    if rep is None:
        raise RuntimeError("NSBitmapImageRep allocation returned None")

    context = NSGraphicsContext.graphicsContextWithBitmapImageRep_(rep)
    NSGraphicsContext.saveGraphicsState()
    try:
        NSGraphicsContext.setCurrentContext_(context)
        NSColor.whiteColor().setFill()
        NSRectFill(NSMakeRect(0, 0, width, height))
        attributes = {
            "NSFont": NSFont.systemFontOfSize_(48.0),
            "NSColor": NSColor.blackColor(),
        }
        NSAttributedString.alloc().initWithString_attributes_(
            FIXTURE_TEXT, attributes
        ).drawAtPoint_(NSMakePoint(20, 40))
    finally:
        NSGraphicsContext.restoreGraphicsState()

    data = rep.representationUsingType_properties_(NSPNGFileType, None)
    if data is None:
        raise RuntimeError("PNG representation returned None")
    return data


def check_vision() -> bool:
    """Run Vision OCR on the synthetic image and confirm it reads the text back."""
    print("\n== vision OCR (synthetic image, no screen capture) ==")
    import time

    import Vision

    try:
        image_data = _synthetic_image_data()
    except Exception as exc:  # noqa: BLE001
        print(f"  fixture        : {FAIL} {type(exc).__name__}: {exc}")
        return False
    print(f"  fixture        : {OK} drawn in memory ({len(image_data)} bytes)")

    handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(image_data, None)
    request = Vision.VNRecognizeTextRequest.alloc().init()
    request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)

    try:
        started = time.perf_counter()
        success, error = handler.performRequests_error_([request], None)
        elapsed = time.perf_counter() - started

        if not success:
            print(f"  performRequests: {FAIL} {error}")
            return False

        # A candidate array can be empty; skip observations without one rather
        # than indexing blindly (an OCR error must FAIL cleanly, not traceback).
        recognized = []
        for obs in request.results() or []:
            candidates = obs.topCandidates_(1)
            if candidates:
                recognized.append(candidates[0].string())
    except Exception as exc:  # noqa: BLE001 - any OCR failure is a clean FAIL
        print(f"  performRequests: {FAIL} {type(exc).__name__}: {exc}")
        return False

    print(f"  performRequests: {OK} in {elapsed * 1000:.0f} ms")
    print(f"  recognized     : {recognized}")

    # Join before matching: Vision may split the line into several observations,
    # so no single candidate need contain the whole fixture string.
    matched = FIXTURE_TEXT in " ".join(recognized)
    print(f"  round-trip     : {OK if matched else FAIL}")
    return matched


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--request",
        action="store_true",
        help="explicitly request Screen Recording permission (triggers a system prompt)",
    )
    args = parser.parse_args()

    report_versions()

    checks = [check_imports()]
    if checks[0]:
        checks.append(check_screencapturekit_symbols())
        checks.append(check_vision())

    check_permission()
    if args.request:
        request_permission()

    passed = all(checks)
    print(f"\n== step 0: {'PASS' if passed else 'FAIL'} ==")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
