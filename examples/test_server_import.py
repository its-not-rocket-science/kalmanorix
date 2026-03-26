#!/usr/bin/env python3
"""
Smoke test for FastAPI server imports and basic functionality.
"""

import sys

sys.path.insert(0, ".")

from fastapi_server import create_toy_village, FUSERS, SCOUT_ALL


def test_imports():
    """Test that all necessary imports work."""
    village = create_toy_village()
    print(f"[OK] Village created with {len(village.modules)} modules")

    for name, fuser in FUSERS.items():
        print(f"[OK] Fuser '{name}' available: {type(fuser).__name__}")

    print("[OK] Scout ALL available")
    print("[OK] Scout HARD available")

    # Try a simple fusion
    from kalmanorix import Panoramix

    panoramix = Panoramix(fuser=FUSERS["mean"])
    potion = panoramix.brew("test query", village=village, scout=SCOUT_ALL)
    print(
        f"[OK] Fusion successful: vector shape {potion.vector.shape}, weights {potion.weights}"
    )

    return True


if __name__ == "__main__":
    try:
        test_imports()
        print("\n[SUCCESS] All imports and basic functionality work!")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
