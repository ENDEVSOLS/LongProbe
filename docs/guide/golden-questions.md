# Golden questions

A golden suite is a reviewed set of example questions and the evidence your retriever must return. It tests retrieval after documents have been indexed and before answer generation. It does not judge the final LLM answer.

## File structure

The default filename is `goldens.yaml`; `golden.yaml` also works when selected with `--goldens golden.yaml`.

```yaml
name: policy-retrieval
version: "1.0"
questions:
  - id: refund-policy
    question: "What is the refund policy?"
    status: draft
    match_mode: text
    required_chunks:
      - "30-day money-back guarantee"
      - "full refund within 30 days"
    top_k: 5
    tags: [policy, critical]
```

`required_chunks` contains stable chunk IDs in `id` mode or evidence text in `text`/`semantic` modes. `top_k` sets the requested result limit. Tags help select cases. Use explicit `status: draft` for unreviewed cases: omitted status defaults to `approved` for backward compatibility.

## Create and review the suite

1. Prepare documents, chunk them, and build your application's retrieval index.
2. Write representative questions manually, or generate candidates from documents using a configured generation provider.
3. Write required evidence from verified sources, or capture candidate results from the configured retriever.
4. Review each question and every requirement against its source document. Current retrieval output can itself be incomplete or wrong.
5. Change verified cases from `draft` to `approved`, using `longprobe edit` or a reviewed YAML edit. The editor does not expose every evidence field; inspect YAML when necessary.
6. Run checks and explicitly save an eligible baseline after review.

```bash
longprobe generate ./documents --num-questions 20 --auto-capture
longprobe edit
longprobe check --no-compare
longprobe baseline save --label latest
```

Generation needs its optional dependencies and configured provider. Ordinary `generate` produces questions; `--auto-capture` additionally retrieves candidate evidence and saves draft cases. Alternatively, `capture --question "What is the refund policy?" --auto` creates a draft from a supplied question. `init` copies examples; it does not read your documents or establish correct evidence.

## Human review and approval

| Stage | Human action |
|---|---|
| Generate questions or auto-capture evidence | Review candidates afterward; automatic output is not trusted ground truth |
| Promote `draft` to `approved` | Verify relevance, source correctness, and required evidence before changing status |
| Run approved questions | Runs automatically in CLI, pytest, or CI; no interactive approval is required |
| Save or replace a trusted baseline | Explicitly promote an eligible report after reviewing changes |
| Change documents, questions, evidence, or settings | Review the intended change and establish a new reference when contracts become incompatible |

These are explicit editing and promotion actions, not a separate enforced manager sign-off system. Passing checks never automatically rewrite approved evidence or replace a baseline. Drafts are excluded by default; `check --include-drafts` can inspect them, but draft-containing reports cannot become trusted baselines.

## Coverage and maintenance

A suite covers selected cases, not every possible document question. Start with critical facts, representative topics, and known failure cases. Add cases when documents or real query patterns expose a gap.

A live user question absent from the suite has no golden expected evidence, so LongProbe cannot automatically certify that question's retrieval correctness. Review it, add expected evidence, and approve the case to include it in future checks.

The file remains editable. Make deliberate, reviewable changes in version control; do not copy current retrieval output into approved requirements just to make a failed test pass. Changed coverage or evaluation contracts make strict baseline comparison incompatible. Inspect intentional changes with `check --no-compare`, then explicitly save a reviewed reference when eligible. See [baseline management](baseline-management.md).

## Scoring example

If two required evidence items are specified and only one is found, recall is `1 / 2 = 50%`. Passing depends on that question's effective threshold. An unchanged average does not hide lost required evidence when comparing compatible reports.

See [match modes](match-modes.md), [configuration](../getting-started/configuration.md), and the [CLI reference](cli-reference.md).
