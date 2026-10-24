"""Verify built v0.2.0 artifacts from a clean base-only wheel environment.

Run with that environment's Python and ``-I``. A controlled localhost fixture
is the only service used. This script does not publish, push, or promote a
production reference.
"""

import hashlib
import importlib.metadata as metadata
import importlib.util
import json
import os
import socket
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

import yaml


def main():
    repo = Path(__file__).resolve().parents[1]
    cli = str(Path(sys.executable).parent / "longprobe")
    wheel = repo / "dist/longprobe-0.2.0-py3-none-any.whl"
    sdist = repo / "dist/longprobe-0.2.0.tar.gz"
    with tempfile.TemporaryDirectory(prefix="longprobe-v020-smoke-") as temporary:
        os.chdir(temporary)
        connect = socket.socket.connect

        def denied(*args, **kwargs):
            raise AssertionError("startup attempted network access")

        socket.socket.connect = denied
        from typer.testing import CliRunner

        import longprobe
        from longprobe.cli.main import app
        from longprobe.core.baseline import BaselineStore

        assert CliRunner().invoke(app, ["--help"]).exit_code == 0
        assert not list(Path(temporary).iterdir())
        assert metadata.version("longprobe") == longprobe.__version__ == "0.2.0"
        assert metadata.metadata("longprobe")["Requires-Python"] == ">=3.11"
        assert not Path(longprobe.__file__).is_relative_to(repo)
        for module in (
            "chromadb",
            "pinecone",
            "qdrant_client",
            "langchain_core",
            "llama_index",
            "litellm",
            "pytest",
        ):
            assert importlib.util.find_spec(module) is None, module
        socket.socket.connect = connect

        process = subprocess.Popen(
            [sys.executable, "-u", str(repo / "examples/http_retriever.py"), "--port", "0"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            line = process.stdout.readline().strip()
            assert line.startswith("Development retriever:"), line
            endpoint = line.split(" ", 2)[-1]
            config = {
                "retriever": {
                    "type": "http",
                    "http": {
                        "url": endpoint,
                        "response_mapping": {
                            "results_path": "data.chunks",
                            "id_field": "chunk_id",
                            "text_field": "content",
                            "score_field": "similarity",
                        },
                    },
                },
                "baseline": {"db_path": ".longprobe/baselines.db"},
            }
            Path("longprobe.yaml").write_text(yaml.safe_dump(config))
            golden = {
                "name": "artifact-smoke",
                "version": "1",
                "questions": [
                    {
                        "id": "refund",
                        "question": "What is the refund policy?",
                        "required_chunks": ["refund-policy"],
                        "status": "approved",
                    }
                ],
            }
            Path("goldens.yaml").write_text(yaml.safe_dump(golden))

            def run(arguments, expected=0):
                result = subprocess.run(
                    [cli, *arguments], capture_output=True, text=True, timeout=30
                )
                assert result.returncode == expected, result.stdout + result.stderr
                return json.loads(result.stdout)

            report = run(["check", "--output", "json"])
            assert report["schema_version"] == 2
            assert report["quality"]["passed"]
            assert report["baseline_identity"]["status"] == "not_found"
            Path("result.json").write_text(json.dumps(report))
            Path("longprobe.yaml").rename("config-backup.yaml")
            assert run(["report", "result.json", "--output", "json"])["overall_recall"] == 1
            Path("config-backup.yaml").rename("longprobe.yaml")

            run(["baseline", "save", "--label", "latest", "--output", "json"])
            reference = BaselineStore().load()
            assert run(["diff", "--baseline", "latest", "--output", "json"])["comparison"][
                "compatible"
            ]
            assert (
                run(["check", "--baseline", "latest", "--output", "json"])["baseline_identity"][
                    "resolved_label"
                ]
                == "latest"
            )

            weak = {
                **golden,
                "questions": [{**golden["questions"][0], "id": f"refund-{i}"} for i in range(4)]
                + [
                    {
                        "id": "cancellation",
                        "question": "Cancellation?",
                        "required_chunks": ["not-found"],
                        "status": "approved",
                    }
                ],
            }
            Path("goldens.yaml").write_text(yaml.safe_dump(weak))
            rejected = run(["baseline", "save", "--output", "json"], expected=1)
            assert rejected["overall_recall"] == 0.8
            assert rejected["quality"]["failed_questions"] == ["cancellation"]
            assert BaselineStore().load() == reference
            Path("goldens.yaml").write_text(yaml.safe_dump(golden))
            assert (
                run(["check", "--tag", "missing", "--output", "json"], expected=2)["status"]
                == "error"
            )

            ci_environment = dict(os.environ)
            ci_environment.pop("GITHUB_OUTPUT", None)
            ci = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "longprobe.ci",
                    "run",
                    "--config",
                    "longprobe.yaml",
                    "--goldens",
                    "goldens.yaml",
                    "--baseline",
                    "latest",
                    "--directory",
                    "ci-results",
                ],
                capture_output=True,
                text=True,
                timeout=30,
                env=ci_environment,
            )
            assert ci.returncode == 0, ci.stdout + ci.stderr
            assert json.loads(Path("ci-results/longprobe-result.json").read_text())["comparison"][
                "compatible"
            ]
            assert "PASS" in Path("ci-results/longprobe-comment.md").read_text()
        finally:
            process.terminate()
            process.communicate(timeout=10)

    with tarfile.open(sdist) as archive:
        info = archive.extractfile("longprobe-0.2.0/PKG-INFO").read().decode()
        assert "Version: 0.2.0" in info
        assert "Requires-Python: >=3.11" in info
        assert "longprobe-0.2.0/.github/workflows/longprobe.yml" in archive.getnames()
    print(
        json.dumps(
            {
                "python": sys.version.split()[0],
                "runtime_version": longprobe.__version__,
                "import_path": longprobe.__file__,
                "base_only": True,
                "startup_no_network_or_project_writes": True,
                "installed_cli_http_workflow": "passed",
                "default_gate_rejects_high_average_failure": True,
                "saved_report_without_config": "passed",
                "local_ci_from_installed_wheel": "passed",
                "sdist_metadata": "0.2.0",
                "artifacts": {
                    path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in (wheel, sdist)
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
