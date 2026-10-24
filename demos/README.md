# LongProbe terminal demonstrations

The recording scripts demonstrate the implemented v0.2.0 Python interface with fixed sample documents and an in-memory retriever. They exercise real selection, scoring, comparison, and policy-gated SQLite promotion. They do not benchmark a vector database or call an embedding model.

## Run the scenarios

From the repository root, with Python 3.11 or newer:

```bash
make setup-demos
make check-demos
```

The dedicated `demos/.recording_venv` directory is generated and rebuilt by setup. The legacy `demos/.demo_venv` directory is not used: it contains machine-specific files and may have broken Python links.

Setup installs this checkout into `demos/.recording_venv`; it does not install an older published package or download embedding models. Package installation may need network access. Once installed, these scenarios run without external services. Each run uses temporary golden/configuration/baseline files and cleans them up afterward.

Run an individual scenario with recording pauses:

```bash
demos/.recording_venv/bin/python demo_run/complete_workflow.py
demos/.recording_venv/bin/python demo_run/monitor_rag_quality.py
demos/.recording_venv/bin/python demo_run/detect_regressions.py
```

Add `--no-delay` for immediate output.

## Presentation

The demonstrations retain the original terminal dashboard style: a single updating progress panel, per-question result tables, baseline comparison, and a clear final verdict. They display no Python source code or assertion snippets. The recording theme, dimensions, and logo are unchanged. GIF playback runs twice as fast as the earlier recordings; tape playback settings preserve that speed when regenerating. Each result displays recall percentage, evidence found / required, and PASS/FAIL together. For example, `0.0% | 0 / 1 | FAIL` means the one required evidence item was not found; `100.0% | 1 / 1 | PASS` means it was found. Synthetic similarity scores are not displayed.

## What the recordings show

| Recording source | Scenario |
|---|---|
| `01-complete-workflow.tape` | Draft candidates, source-review explanation, approved cases, eligible promotion, and missing-evidence regression |
| `01-quick-check.tape` | One evaluation: two passing questions, one failing question, and a failed quality verdict |
| `02-python-api.tape` | An 85% threshold, two passing questions, one failing question, and missing-evidence details |
| `03-baseline-tracking.tape` | Explicit `production-v1` selection, compatibility, regression/improvement counts, and quality failure |

The workflow uses fixed sample evidence supplied with the script. Its scripted status transition illustrates a review decision; it does not perform human approval. Real generated or captured evidence must be checked against source documents before approval.

The regression scenarios deliberately remove refund evidence after promotion. The second evaluation reports one regression and failed quality. The script checks this expected result and exits successfully while showing the failed quality verdict. The saved reference is preserved.

## Regenerate README and documentation GIFs

Install [VHS and its recording prerequisites](https://github.com/charmbracelet/vhs#installation), including ttyd, Chromium, and ffmpeg. Then run from the repository root:

```bash
make demos
```

This records the four existing tapes into `demos/` and copies the resulting GIFs into both `assets/` and `docs/assets/`. Commit the tapes, scripts, and regenerated assets together when recording tools are available. Changing a script alone does not update an existing GIF.

After changing the presentation scripts, regenerate the GIFs to refresh the images shown in the README and documentation. VHS requires permission to start a local terminal server and browser.

To record individually:

```bash
vhs demos/01-complete-workflow.tape
vhs demos/01-quick-check.tape
vhs demos/02-python-api.tape
vhs demos/03-baseline-tracking.tape
```

Individual recording does not perform the asset-copy steps. Use `make demos` to synchronize the files displayed in the README and documentation.

## Editing presentation

Keep the existing theme, font, dimensions, and README layout. Adjust tape pauses if a scenario needs more reading time. Display the quality verdict and comparison compatibility alongside recall; a good average or zero regression count alone does not establish deployment readiness.

## Public README and PyPI images

The root README uses public GIF URLs from `Abdullahbinmushtaq/LongProbe/main`. Push updated assets to that repository to refresh the published recordings. Open the local files directly to preview changes before pushing.

Package builds pin these images to the package version's Git tag, keeping older PyPI descriptions stable. The checkout README stays unchanged during builds. See the [release process](../docs/contributing.md#release-process) for tag requirements and verification commands.
