"""Single-command runner for the LongProbe RAG Regression Starter Kit.

Executes the three canonical stages:
  1. Known-Good: Run check (100% recall) and save trusted baseline 'launch-demo'
  2. Degraded:   Switch fixture to degraded mode, run check + diff, verify regression exit code
  3. Corrected:  Switch fixture to corrected mode, re-run check + diff, verify recovery

Saves raw terminal outputs and exit codes to sample_outputs/.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen

PORT = int(os.environ.get("LONGPROBE_FIXTURE_PORT", 8000))
BASE_URL = f"http://127.0.0.1:{PORT}"
HEALTH_URL = f"{BASE_URL}/health"
MODE_URL = f"{BASE_URL}/mode"

STARTER_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = STARTER_DIR / "sample_outputs"


def wait_for_server(timeout: float = 6.0) -> bool:
    """Poll health endpoint until server responds."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            with urlopen(HEALTH_URL, timeout=0.5) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            time.sleep(0.15)
    return False


def set_fixture_mode(mode: str) -> None:
    """Dynamically set fixture server state."""
    payload = json.dumps({"mode": mode}).encode("utf-8")
    req = Request(MODE_URL, data=payload, headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=3.0) as resp:
        if resp.status != 200:
            raise RuntimeError(f"Failed to switch fixture mode to {mode}")


def run_cli_cmd(args: list[str], output_filename: str) -> tuple[int, str, float]:
    """Execute a longprobe CLI command from STARTER_DIR and capture output."""
    cmd = [sys.executable, "-m", "longprobe.cli.main", *args]
    env = {**os.environ, "COLUMNS": "140", "LINES": "60"}
    start_time = time.time()
    proc = subprocess.run(
        cmd,
        cwd=STARTER_DIR,
        env=env,
        capture_output=True,
        text=True,
    )
    elapsed = time.time() - start_time
    combined_output = proc.stdout
    if proc.stderr:
        combined_output += ("\n[STDERR]\n" if combined_output else "") + proc.stderr

    out_file = OUTPUTS_DIR / output_filename
    out_file.write_text(combined_output, encoding="utf-8")
    return proc.returncode, combined_output, elapsed


def main() -> None:
    print("=" * 72)
    print("LongProbe RAG Regression Starter Kit — Pass -> Fail -> Fix Demo")
    print("=" * 72)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Reset baseline store for clean reproducible demo
    db_dir = STARTER_DIR / ".longprobe"
    if db_dir.exists():
        shutil.rmtree(db_dir)
    print(f"[*] Cleaned database directory: {db_dir}")

    # 2. Launch fixture server in the background
    print(f"[*] Starting local retrieval fixture on {BASE_URL}...")
    server_proc = subprocess.Popen(
        [sys.executable, str(STARTER_DIR / "fixture_server.py"), "--port", str(PORT), "--mode", "good"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        if not wait_for_server():
            print("[!] Error: Fixture server failed to start.")
            sys.exit(3)
        print("[✓] Fixture server ready.\n")

        results = []

        # ---------------------------------------------------------------------
        # STAGE 1: Known-Good
        # ---------------------------------------------------------------------
        print("[1/3] STAGE 1: Known-Good Run & Trusted Baseline Save")
        print("  Running: longprobe check")
        rc, _out, dur = run_cli_cmd(["check"], "01_known_good_check.txt")
        print(f"    Exit code: {rc} (expected: 0) | Duration: {dur:.3f}s")
        assert rc == 0, f"Stage 1 check failed with code {rc}"
        results.append(("Stage 1 Check", rc, dur, "PASS"))

        print("  Running: longprobe baseline save --label launch-demo")
        rc, _out, dur = run_cli_cmd(["baseline", "save", "--label", "launch-demo"], "01_baseline_save.txt")
        print(f"    Exit code: {rc} (expected: 0) | Duration: {dur:.3f}s")
        assert rc == 0, f"Stage 1 baseline save failed with code {rc}"
        results.append(("Stage 1 Baseline Save", rc, dur, "PASS"))
        print("  -> Baseline 'launch-demo' saved successfully.\n")

        # ---------------------------------------------------------------------
        # STAGE 2: Degraded
        # ---------------------------------------------------------------------
        print("[2/3] STAGE 2: Degraded Run (Simulating Retrieval Regression)")
        set_fixture_mode("degraded")
        print("  Fixture switched to 'degraded' (3 chunks missing, recall = 85.0%).")

        print("  Running: longprobe check --baseline launch-demo --output github")
        rc, _out, dur = run_cli_cmd(
            ["check", "--baseline", "launch-demo", "--output", "github"],
            "02_degraded_check_github.txt",
        )
        print(f"    Exit code: {rc} (expected: 1) | Duration: {dur:.3f}s")
        assert rc == 1, f"Expected nonzero exit code 1 for degraded check, got {rc}"
        results.append(("Stage 2 Check (GitHub Output)", rc, dur, "EXPECTED REGRESSION (FAIL)"))

        print("  Running: longprobe diff --baseline launch-demo")
        rc, _out, dur = run_cli_cmd(["diff", "--baseline", "launch-demo"], "02_degraded_diff.txt")
        print(f"    Exit code: {rc} (expected: 1) | Duration: {dur:.3f}s")
        assert rc == 1, f"Expected nonzero exit code 1 for degraded diff, got {rc}"
        results.append(("Stage 2 Diff", rc, dur, "EXPECTED REGRESSION (FAIL)"))
        print("  -> Regression caught with nonzero exit code.\n")

        # ---------------------------------------------------------------------
        # STAGE 3: Corrected
        # ---------------------------------------------------------------------
        print("[3/3] STAGE 3: Corrected Run (Restoring Missing Chunks)")
        set_fixture_mode("corrected")
        print("  Fixture switched to 'corrected' (100.0% recall restored).")

        print("  Running: longprobe check --baseline launch-demo")
        rc, _out, dur = run_cli_cmd(["check", "--baseline", "launch-demo"], "03_corrected_check.txt")
        print(f"    Exit code: {rc} (expected: 0) | Duration: {dur:.3f}s")
        assert rc == 0, f"Expected code 0 for corrected check, got {rc}"
        results.append(("Stage 3 Check", rc, dur, "PASS"))

        print("  Running: longprobe diff --baseline launch-demo")
        rc, _out, dur = run_cli_cmd(["diff", "--baseline", "launch-demo"], "03_corrected_diff.txt")
        print(f"    Exit code: {rc} (expected: 0) | Duration: {dur:.3f}s")
        assert rc == 0, f"Expected code 0 for corrected diff, got {rc}"
        results.append(("Stage 3 Diff", rc, dur, "PASS"))
        print("  -> Full retrieval quality restored. Zero regressions detected.\n")

        # ---------------------------------------------------------------------
        # Summary
        # ---------------------------------------------------------------------
        print("=" * 72)
        print("Demo Summary Results:")
        print(f"{'Step':<32} {'Exit Code':<12} {'Duration':<12} {'Outcome'}")
        print("-" * 72)
        for step, code, dur, outcome in results:
            print(f"{step:<32} {code:<12} {dur:>6.3f}s     {outcome}")
        print("=" * 72)
        print(f"[✓] All stage outputs successfully saved to: {OUTPUTS_DIR}")

    finally:
        server_proc.terminate()
        server_proc.wait(timeout=3)
        print("[*] Fixture server stopped.")


if __name__ == "__main__":
    main()
