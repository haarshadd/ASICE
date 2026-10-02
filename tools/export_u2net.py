"""One-time export: U^2-Net-small (u2netp) PyTorch weights -> ONNX.

Run this yourself, once, on a machine with internet + PyTorch. The output
.onnx file then gets committed into asice/models/ and shipped with the repo,
so end users of `asice` never need PyTorch or a download.

Usage:
    pip install torch torchvision onnx
    # get u2netp.pth from the official U^2-Net repo release, place it next
    # to this script or pass --weights
    python tools/export_u2net.py --weights u2netp.pth --output ../asice/asice/models/u2netp.onnx

This script is NOT part of the asice package and is not imported by it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", default="u2netp.pth", help="Path to the official u2netp.pth checkpoint.")
    p.add_argument("--output", default="u2netp.onnx", help="Where to write the ONNX file.")
    p.add_argument("--opset", type=int, default=12)
    args = p.parse_args()

    try:
        import torch
    except ImportError:
        print(
            "This script needs PyTorch (developer-side only; not an asice dependency).\n"
            "  pip install torch torchvision",
            file=sys.stderr,
        )
        return 1

    weights_path = Path(args.weights)
    if not weights_path.is_file():
        print(
            f"Checkpoint not found: {weights_path}\n"
            "Download u2netp.pth from the official U^2-Net repository release page:\n"
            "  https://github.com/xuebinqin/U-2-Net\n"
            "and pass its path with --weights.",
            file=sys.stderr,
        )
        return 1

    # Imported lazily: the model definition file (u2net.py from the official
    # repo) must be importable. Clone it next to this script, or adjust the
    # import below to wherever you placed it.
    try:
        from u2net import U2NETP  # from the official U^2-Net repo's model/ folder
    except ImportError:
        print(
            "Could not import U2NETP. Clone https://github.com/xuebinqin/U-2-Net "
            "and put its model/u2net.py on your PYTHONPATH, or copy it next to "
            "this script.",
            file=sys.stderr,
        )
        return 1

    model = U2NETP(3, 1)
    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    dummy = torch.randn(1, 3, 320, 320)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=["input"],
        output_names=["output"],
        opset_version=args.opset,
        dynamic_axes=None,  # fixed 320x320 input; asice resizes to match before calling
    )
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"Wrote {out_path} ({size_mb:.1f} MB)")
    print("Remember: commit this file and its Apache-2.0 notice, see asice/models/README.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())