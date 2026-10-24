"""Execute workflow JavaScript with local API stubs; no GitHub writes occur."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        shutil.which("node") is None, reason="Node is required for GitHub workflow-script checks"
    ),
]


def script(name):
    path = Path(__file__).parents[2] / ".github/workflows/longprobe.yml"
    steps = yaml.safe_load(path.read_text())["jobs"]["longprobe-check"]["steps"]
    return next(step["with"]["script"] for step in steps if step.get("name") == name)


def execute(source, prefix, tmp_path):
    code = (
        prefix
        + "\nasync function execute() {\n"
        + source
        + "\n}\nexecute().then(() => console.log(JSON.stringify(state)));"
    )
    result = subprocess.run(
        ["node", "-e", code], cwd=tmp_path, capture_output=True, text=True, timeout=10
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


@pytest.mark.parametrize(
    "change,denied",
    [
        ({}, False),
        ({"conclusion": "failure"}, True),
        ({"head_branch": "feature"}, True),
        ({"event": "pull_request"}, True),
        ({"head_repository": {"full_name": "outsider/fork"}}, True),
    ],
)
def test_only_successful_default_branch_source_runs_are_trusted(tmp_path, change, denied):
    run = {
        "conclusion": "success",
        "head_branch": "main",
        "event": "workflow_dispatch",
        "head_repository": {"full_name": "example/app"},
        **change,
    }
    prefix = """
const state = {failed: []};
const context = {repo: {owner: 'example', repo: 'app'}};
const core = {setFailed: message => state.failed.push(message)};
const github = {rest: {
  actions: {getWorkflowRun: async () => ({data: RUN})},
  repos: {get: async () => ({data: {default_branch: 'main', full_name: 'example/app'}})}
}};
""".replace("RUN", json.dumps(run))
    result = execute(script("Verify trusted baseline source run"), prefix, tmp_path)
    assert bool(result["failed"]) == denied


@pytest.mark.parametrize("existing,denied", [(False, False), (True, False), (False, True)])
def test_comment_create_update_and_permission_failure_are_separate_from_verdict(
    tmp_path, existing, denied
):
    (tmp_path / "longprobe-comment.md").write_text("<!-- longprobe-report -->\nQuality: PASS")
    comments = (
        [{"id": 7, "user": {"type": "Bot"}, "body": "<!-- longprobe-report -->"}]
        if existing
        else []
    )
    prefix = """
const state = {created: 0, updated: 0, warnings: []};
const context = {repo: {owner: 'example', repo: 'app'}, issue: {number: 42}};
const core = {warning: message => state.warnings.push(message)};
const github = {
  paginate: async () => COMMENTS,
  rest: {issues: {
    listComments: () => {},
    createComment: async () => { if (DENIED) throw new Error('permission denied'); state.created += 1; },
    updateComment: async () => { state.updated += 1; }
  }}
};
""".replace("COMMENTS", json.dumps(comments)).replace("DENIED", str(denied).lower())
    result = execute(script("Publish PR report when permitted"), prefix, tmp_path)
    assert result["created"] == int(not existing and not denied)
    assert result["updated"] == int(existing)
    assert bool(result["warnings"]) == denied
