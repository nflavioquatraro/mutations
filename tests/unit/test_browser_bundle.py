"""Transport corruption rejection and Python/JavaScript byte parity."""

import json
import shutil
import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest

from anchored_moe.branching import (
    GenerativeRouterConfig,
    RoutingMessage,
    apply_cact_patch,
    create_cact_patch,
)


def test_patch_roundtrip_shares_bytes_and_rejects_corruption():
    base = np.random.default_rng(14).integers(0, 256, 65536, dtype=np.uint8).tobytes()
    target = base[:9000] + b"updated suffix" + base[9014:]
    patch = create_cact_patch(base, target)
    assert len(patch) < len(target) / 8
    assert apply_cact_patch(base, patch) == target
    assert apply_cact_patch(base, create_cact_patch(base, base)) == base
    # Also handle a changed layout/file size, without assuming fixed offsets.
    longer = base + b"new head"
    assert apply_cact_patch(base, create_cact_patch(base, longer)) == longer
    for wrong_base, wrong_patch in [
        (base[1:], patch),
        (base, patch[:-1]),
        (base, b"bad" + patch[3:]),
        (base, patch[:-1] + bytes([patch[-1] ^ 1])),
    ]:
        with pytest.raises(ValueError):
            apply_cact_patch(wrong_base, wrong_patch)
    header_size = struct.unpack_from("<I", patch, 8)[0]
    header = json.loads(patch[12 : 12 + header_size])
    header["operations"][0][1] = -1
    raw = json.dumps(header).encode()
    with pytest.raises(ValueError, match="range"):
        apply_cact_patch(
            base,
            patch[:8] + struct.pack("<I", len(raw)) + raw + patch[12 + header_size :],
        )


def test_javascript_transport_context_and_runtime(tmp_path):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is required for JavaScript contract tests")
    base = np.random.default_rng(5).integers(0, 256, 16384, dtype=np.uint8).tobytes()
    target = base[:8192] + b"x" * 100 + base[8292:]
    (tmp_path / "base.bin").write_bytes(base)
    (tmp_path / "patch.bin").write_bytes(create_cact_patch(base, target))
    (tmp_path / "target.bin").write_bytes(target)
    config = GenerativeRouterConfig(("base", "specialist"), max_input_bytes=150)
    history = [
        RoutingMessage("user", "ancient" * 100),
        RoutingMessage("assistant", "Tomorrow 🌍?"),
        RoutingMessage("tool", '{"time":"noon"}'),
    ]
    (tmp_path / "contract.json").write_text(
        json.dumps(
            {
                "config": {**config.to_dict(), "model_id": "router"},
                "request": "sí, tomorrow",
                "history": [vars(m) for m in history],
                "expected": config.render_request("sí, tomorrow", history),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    script = Path(__file__).parents[1] / "browser" / "test_needle_browser.mjs"
    result = subprocess.run(
        [node, str(script), str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=45,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
