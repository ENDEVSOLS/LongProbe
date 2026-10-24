# CLI reference

Use `longprobe COMMAND --help` for implemented options. Default paths are `goldens.yaml` and `longprobe.yaml`.

| Command | Purpose |
|---|---|
| `init` | Create example files, preserving existing files unless `--force` |
| `check` | Evaluate selected questions with shared scoring/comparison rules |
| `capture` | Query supplied questions and select evidence; `--auto` creates drafts |
| `generate PATH` | Generate questions from documents; auto-capture creates draft cases |
| `edit` | Review status, question text/mode, approve drafts, and save |
| `baseline save` | Fresh approved-only evaluation and explicit policy-gated promotion |
| `baseline list` | List snapshots at `--db-path` |
| `baseline delete --label LABEL` | Confirm deletion of a snapshot at `--db-path` |
| `diff --baseline LABEL` | Compare a live run with an explicit reference |
| `diff LABEL_A LABEL_B` | Compare stored target B against reference A |
| `report REPORT_PATH` | Render saved JSON without configuration or retrieval |
| `explain QUESTION_ID` | Rerun evaluation and explain missing evidence |
| `watch` | Poll the golden file and rerun after changes |

## Evaluation and output

```bash
longprobe check --goldens tests/goldens.yaml --config longprobe.yaml
longprobe check --threshold 0.9 --top-k 10
longprobe check --tag critical --tag billing
longprobe check --include-drafts
longprobe check --baseline release-a
longprobe check --no-compare
longprobe check --output json > result.json
longprobe report result.json --output github
longprobe baseline save --label latest --output json
```

Multiple tags require all specified tags. Empty suites and empty selections fail. Drafts are excluded by default and cannot be promoted. Global threshold overrides do not override explicit per-question settings.

`check --baseline LABEL` requires the label to exist. Optional automatic `latest` absence is visible in the report; no historical safety is implied. `--no-compare` deliberately disables comparison and cannot be combined with `--baseline`.

## JSON and exit codes

`check`, `diff`, `baseline save`, and `report` support `table`, `json`, and `github` output. Parsed JSON-mode invocations emit one complete JSON object on stdout, including handled command errors. Notices and diagnostics go to stderr.

Success/evaluation payloads include existing aggregate/result fields, `schema_version`, recorded contracts/policy, `quality`, `baseline_identity`, `comparison`, and a `status`. `diff` remains an alias for the comparison object for automation compatibility. Error envelopes contain `schema_version`, `status: error`, and `error.code`/`error.message`.

| Code | Meaning |
|---|---|
| `0` | Successful operation, or an explicitly nonblocking completed quality failure |
| `1` | Completed evaluation/comparison or baseline quality failure |
| `2` | Invalid input/setup, empty selection, missing explicit reference, or incompatible comparison |
| `3` | Operational or unexpected execution failure |

A failed quality decision stays recorded as failed even when `scoring.fail_on_regression: false` allows a check to exit 0. That setting does not relax setup/execution failures or incompatible comparisons. Explicit `diff` gates quality; baseline promotion always applies its own policy.

Parser-level usage errors that prevent establishing a valid invocation use normal CLI stderr and code 2, without promising a JSON object. Watch and interactive authoring remain human-oriented. `report` renders completed data and does not re-enforce the saved quality verdict; malformed input still fails.

## Authoring

```bash
longprobe capture --question "What is the refund policy?" --auto
longprobe capture --questions-file questions.txt --match-mode id
longprobe generate ./documents --num-questions 20 --auto-capture
longprobe edit
```

Install generation dependencies and configure the provider/model before generation. Review required evidence in YAML; the editor does not expose every field. Automatic capture is candidate evidence, not independently verified truth. Generating questions does not guarantee complete document coverage.

Question-list export is separate:

```bash
longprobe generate ./documents --output questions.yaml
longprobe generate ./documents --output questions.yaml --overwrite
```

`--overwrite` requires `--output`, and output mode cannot be combined with `--auto-capture`. Invalid existing golden files fail preflight and are never silently reset.

## Baselines and watch

Saving another label does not update literal `latest`. New reports enforce contract compatibility; see [baseline management](baseline-management.md) for old reference migration and intentional rebaselining.

Watch runs after golden-file changes, not on startup or configuration/remote index changes. Failed iterations are reported and watching continues. Explain lists missing evidence and retrieved IDs; it does not calculate rank or distance deltas.
