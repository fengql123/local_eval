"""Native Harbor source-review acceptance with the real logged-in Claude CLI.

Run: LOCAL_EVAL_HARBOR_TESTS=1 LOCAL_EVAL_KEEP_HARBOR_TESTS=1 uv run pytest -q -s
     tests/test_oracle.py

These opt-in tests reuse a production image; they never build a test image or replace
an agent/backend. They exercise separate native OracleAgent search and coding trials, the trusted
search finalizer, source-only handoff, and a fresh NopAgent verification trial
whose shared steps run as root, judge, then root.
Rewardkit uses the actual Claude reviewer. The final measurement step checks
persisted state and identities because real checkpoint shards are unavailable.
No model-serving, quality, timing, or production reference-search claim is made.
"""

import asyncio
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest
import yaml

from eval.config import MODELS, AgentConfig, EvaluationConfig, RunConfig, SearchConfig, Task
from eval.device import Machine
from eval.run import _job_config
from eval.tasks import ImageSpec, render_task

pytestmark = pytest.mark.skipif(
    os.environ.get("LOCAL_EVAL_HARBOR_TESTS") != "1",
    reason="set LOCAL_EVAL_HARBOR_TESTS=1 for native Harbor and logged-in Claude review",
)

ENGINE_REPO = "https://github.com/FlashML-org/FreeToken.git"
ENGINE_COMMIT = "cac247a860e316e06580d05aeb05f2e647bde214"
IMAGE = "posttrainrlbench2-agent:local"
CATEGORY = "nvidia-cuda-x86"

START = r"""#!/bin/bash
set -euo pipefail
rm -f /run/eval-ready
mkdir -p /opt/eval /root/eval /logs/agent /logs/verifier
for spec in search:1000 agent:1001 judge:1002; do
  role=${spec%:*}; role_uid=${spec#*:}
  if ! id "$role" >/dev/null 2>&1; then useradd -m -o -u "$role_uid" "$role"; fi
  mkdir -p "/home/$role"
  chown "$role:$role" "/home/$role"
  chmod 700 "/home/$role"
done
mkdir -p /workspace
chown agent:agent /workspace
chmod 755 /workspace
/root/.local/bin/uv venv --system-site-packages --python /opt/venv/bin/python /opt/eval/.venv
printf "%s\n" /opt/venv/lib/python3.12/site-packages > /opt/eval/.venv/lib/python3.12/site-packages/production-image.pth
/root/.local/bin/uv pip install --python /opt/eval/.venv/bin/python /root/eval/source
/opt/eval/.venv/bin/python -I - <<'PYTHON'
import importlib.metadata as metadata
import json
from pathlib import Path
packages = {name: metadata.version(name) for name in ('local-eval', 'harbor-rewardkit', 'torch', 'transformers')}
Path('/run/runtime-packages.json').write_text(json.dumps(packages))
PYTHON
touch /run/eval-ready
exec sleep infinity
"""

PREPARE_PROBE = r"""import hashlib
import json
import os
import subprocess
from pathlib import Path
from eval.verify import judge, run

out = Path('/logs/verifier')
state = json.loads(Path('/root/eval/verification/prepare.json').read_text())
report = {
    'scope': 'native_harbor_three_trials_real_claude_source_review',
    'model_shards_available': False,
    'engine_installed': False,
    'measurement_run': False,
    'limitations': [
        'Checkpoint shards and the complete FreeToken runtime are unavailable; startup and measurement are untested.',
        'The measurement step checks identity and state persistence only; it does not fabricate inference or timing.',
    ],
    'runtime_packages': json.loads(Path('/run/runtime-packages.json').read_text()),
    'judge_module_sha256': hashlib.sha256(Path(judge.__file__).read_bytes()).hexdigest(),
    'run_module_sha256': hashlib.sha256(Path(run.__file__).read_bytes()).hexdigest(),
    'prepare_script_sha256': hashlib.sha256(Path('/opt/eval/images/prepare.sh').read_bytes()).hexdigest(),
    'prepare_uid': os.getuid(),
    'prepare_container': Path('/etc/hostname').read_text().strip(),
    'native_auth_env': 'CLAUDE_CODE_OAUTH_TOKEN',
    'custom_reviewer_settings': Path('/home/judge/.claude/settings.json').exists(),
    'credential_mounts_absent': not Path('/root/eval/auth').exists(),
    'claude_version': subprocess.check_output(['claude', '--version'], text=True).strip(),
    'preparation': state,
    'fresh': not Path('/workspace/original-only').exists() and not Path('/home/agent/original-only').exists(),
}
assert report['prepare_uid'] == 0 and report['fresh']
Path('/root/eval/verification/acceptance-prepare.json').write_text(json.dumps(report, indent=2) + '\n')
(out / 'acceptance-prepare.json').write_text(json.dumps(report, indent=2) + '\n')
"""

REVIEW_PROBE = r"""import json
import os
import pwd
import subprocess
import threading
from pathlib import Path

output = Path('/workspace/review-output')
roots = ['/workspace/submission', '/workspace/weights', '/workspace/runtime', '/models/source', '/models/reference', '/opt/eval/images/nvidia-cuda-x86', '/workspace/review-rubric']
assert os.getuid() == pwd.getpwnam('judge').pw_uid != 0
self_status = dict(line.split(':', 1) for line in Path('/proc/self/status').read_text().splitlines() if ':' in line)
assert self_status['NoNewPrivs'].strip() == '1'
for root in roots:
    assert not os.access(root, os.W_OK), root
    assert os.access(root, os.R_OK), root
    for file in Path(root).rglob('*'):
        if not file.is_symlink():
            assert not os.access(file, os.W_OK), str(file)
for root in ('/root/eval/private', '/root/eval/quality', '/root/eval/auth', '/root/eval/task.json'):
    assert not os.access(root, os.R_OK), root
assert not Path('/home/judge/.claude/settings.json').exists()
assert not Path('/home/judge/.claude/.credentials.json').exists()
assert os.access('/home/judge', os.W_OK)
assert os.access('/home/judge/tmp', os.W_OK)
assert os.access('/home/judge/.cache', os.W_OK)
assert os.access(output, os.W_OK)
assert not Path('/workspace/data/S').exists()
report = {
    'review_uid': os.getuid(),
    'review_gid': os.getgid(),
    'review_no_new_privs': int(self_status['NoNewPrivs']),
    'review_container': Path('/etc/hostname').read_text().strip(),
    'permissions': {'ok': True, 'stdout': 'readonly inputs and private isolation verified', 'stderr': ''},
}
roles = {}
identity_errors = []
stop = threading.Event()
def observe():
    while not stop.is_set():
        for proc in Path('/proc').glob('[0-9]*'):
            try:
                argv = (proc / 'cmdline').read_bytes().split(b'\0')
                native = b'rewardkit' in argv or any(b'claude' in value for value in argv[:2])
                if not native:
                    continue
                status = dict(line.split(':', 1) for line in (proc / 'status').read_text().splitlines() if ':' in line)
                identities = [int(value) for value in status['Uid'].split()]
                if identities != [os.getuid()] * 4:
                    identity_errors.append(status['Name'].strip())
                roles[proc.name] = {
                    'pid': int(proc.name), 'name': status['Name'].strip(),
                    'uids': identities, 'gids': [int(value) for value in status['Gid'].split()],
                    'no_new_privs': int(status['NoNewPrivs']),
                }
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                continue
        stop.wait(0.1)
watcher = threading.Thread(target=observe, daemon=True)
watcher.start()
result = subprocess.run(['/opt/eval/.venv/bin/python', '-I', '-m', 'eval.verify.run', '--stage', 'review'])
stop.set()
watcher.join()
report['process_roles'] = list(roles.values())
report['review_cli_exit'] = result.returncode
report['tool_calls'] = []
for file in Path('/home/judge/.claude/projects').rglob('*.jsonl'):
    for line in file.read_text().splitlines():
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        message = entry.get('message', {})
        for block in message.get('content', []) if isinstance(message, dict) and isinstance(message.get('content'), list) else []:
            if isinstance(block, dict) and block.get('type') == 'tool_use':
                report['tool_calls'].append(block.get('name'))
assert roles, 'No real Rewardkit/Claude process observed'
assert not identity_errors, identity_errors
(output / 'acceptance-review.json').write_text(json.dumps(report, indent=2) + '\n')
Path('/logs/verifier/acceptance-review.json').write_text(json.dumps(report, indent=2) + '\n')
raise SystemExit(result.returncode)
"""

MEASURE_PROBE = r"""import json
import os
from pathlib import Path

report = json.loads(Path('/root/eval/verification/acceptance-prepare.json').read_text())
report.update(json.loads(Path('/workspace/review-output/acceptance-review.json').read_text()))
report['measure_uid'] = os.getuid()
report['measure_container'] = Path('/etc/hostname').read_text().strip()
assert report['measure_uid'] == 0
assert report['prepare_container'] == report['review_container'] == report['measure_container']
submission = Path('/workspace/submission')
report['root_writes_retained'] = all(
    path.stat().st_uid == 0 and path.stat().st_mode & 0o200
    for root in (submission, submission.parent / 'weights', submission.parent / 'runtime')
    for path in (root, *root.rglob('*')) if not path.is_symlink()
)
assert report['root_writes_retained']
for folder in ('/workspace/weights', '/workspace/runtime'):
    marker = Path(folder) / 'measure-root-write'
    marker.write_text('Root measurement retains write access after native judge step.\n')
    marker.unlink()
report['rewards'] = {'reward': 1}
Path('/logs/verifier/review-acceptance.json').write_text(json.dumps(report, indent=2) + '\n')
Path('/logs/verifier/reward.json').write_text(json.dumps(report['rewards']) + '\n')
"""

SEARCH_SOLUTION = r"""#!/bin/bash
# Source-only fixture: author a selection without pretending to benchmark missing weights.
# Harbor's native OracleAgent runs this as search; the actual root finalizer remains intact.
set -euo pipefail
export HOME=/home/search USER=search LOGNAME=search
/opt/eval/.venv/bin/python -I - <<'PYTHON'
import json
import os
import pwd
from pathlib import Path
assert os.getuid() == pwd.getpwnam('search').pw_uid != 0
Path('/home/search/selection.json').write_text(json.dumps({'baseline': 'freetoken', 'args': [], 'env': {}}) + '\n')
Path('/tmp/original-search-only').write_text('Search sandbox state, never handed to coding.\n')
Path('/logs/agent/search-identity.json').write_text(json.dumps({
    'search_uid': os.getuid(),
    'search_container': Path('/etc/hostname').read_text().strip(),
    'search_inference_run': False,
}) + '\n')
PYTHON
"""

SEARCH_PROBE = r"""import json
import os
from pathlib import Path

reference = json.loads(Path('/root/eval/reference/reference.json').read_text())
public = json.loads(Path('/root/eval/reference/public.json').read_text())
assert os.getuid() == 0
assert reference['baseline'] == 'freetoken' and public['baseline'] == 'freetoken'
assert reference['checkpoint'].startswith('/models/reference')
Path('/logs/verifier/search-acceptance.json').write_text(json.dumps({
    'search_finalizer_uid': os.getuid(),
    'search_finalizer_container': Path('/etc/hostname').read_text().strip(),
    'search_finalizer_checkpoint': reference['checkpoint'],
    'search_finalizer_inference_run': False,
}) + '\n')
"""


def _run(command, *, cwd=None, log=None):
    if log is None:
        return subprocess.run(command, cwd=cwd, check=True, capture_output=True, text=True).stdout
    with log.open("w") as stream:
        subprocess.run(command, cwd=cwd, check=True, stdout=stream, stderr=subprocess.STDOUT)


@pytest.fixture(scope="module")
def live_runtime():
    repo = Path(__file__).resolve().parents[1]
    root = Path(tempfile.mkdtemp(prefix="local-eval-harbor-live-"))
    root.chmod(0o755)
    claude = Path(shutil.which("claude") or "").resolve(strict=True)
    version = _run([str(claude), "--version"]).strip()
    assert "Claude Code" in version
    login = json.loads(_run([str(claude), "auth", "status", "--json"]))
    assert login.get("loggedIn") is True and login.get("authMethod") == "claude.ai"
    previous_token = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN")
    if previous_token is None:
        credentials = (
            Path(os.environ.get("CLAUDE_CONFIG_DIR", Path.home() / ".claude")) / ".credentials.json"
        )
        os.environ["CLAUDE_CODE_OAUTH_TOKEN"] = json.loads(credentials.read_text())[
            "claudeAiOauth"
        ]["accessToken"]
    image_id = _run(["docker", "image", "inspect", "--format", "{{.Id}}", IMAGE]).strip()
    baseline_assets = (
        Path(
            os.environ.get(
                "LOCAL_EVAL_BASELINE_ASSETS",
                str(repo.parent / "edge-harness-v2/local_eval/images"),
            )
        )
        / CATEGORY
        / "baselines/freetoken"
    )
    baseline = root / "baseline"
    baseline.mkdir()
    for name in ("install.sh", "launch.sh"):
        shutil.copyfile(baseline_assets / name, baseline / name)
        (baseline / name).chmod(0o555)
    metadata = repo.parent / "edge-harness-v2/local_eval/store/models" / MODELS["C2"].source
    shutil.copytree(metadata, root / "model-metadata")
    # Bind a stable copy: editing the checkout must not change a running shell or
    # the evaluator installed by another trial in this paid acceptance batch.
    snapshot = root / "evaluator-source"
    for name in ("eval", "images", "tables", "templates"):
        shutil.copytree(
            repo / name,
            snapshot / name,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    for name in ("pyproject.toml", "uv.lock", "README.md", ".dockerignore"):
        shutil.copyfile(repo / name, snapshot / name)
    repo = snapshot
    _run(["git", "init", "-q", str(baseline / "src")])
    _run(["git", "-C", str(baseline / "src"), "remote", "add", "origin", ENGINE_REPO])
    _run(
        ["git", "-C", str(baseline / "src"), "fetch", "-q", "--depth", "1", "origin", ENGINE_COMMIT]
    )
    _run(["git", "-C", str(baseline / "src"), "checkout", "-q", "FETCH_HEAD"])
    actual_commit = _run(["git", "-C", str(baseline / "src"), "rev-parse", "HEAD"]).strip()
    assert actual_commit == ENGINE_COMMIT
    cache = root / "uv-cache"
    cache.mkdir(mode=0o700)
    manifest = {
        "scope": "source review only; no engine build, model inference or performance measurement",
        "image": IMAGE,
        "image_id": image_id,
        "claude_version": version,
        "claude_sha256": hashlib.file_digest(claude.open("rb"), "sha256").hexdigest(),
        "login": {
            key: login.get(key)
            for key in ("loggedIn", "authMethod", "apiProvider", "subscriptionType")
        },
        "engine_repository": ENGINE_REPO,
        "engine_commit": actual_commit,
        "runtime_sources": {
            str(path.relative_to(repo)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(repo.rglob("*"))
            if path.is_file() and "__pycache__" not in path.parts
        },
    }
    (root / "acceptance-source.json").write_text(json.dumps(manifest, indent=2) + "\n")
    image = ImageSpec(CATEGORY, repo, Path("unused-existing-image"), image_id, IMAGE, {}, manifest)
    print(f"Live Harbor evidence: {root}", flush=True)
    try:
        yield root, repo, claude, image
    finally:
        if previous_token is None:
            os.environ.pop("CLAUDE_CODE_OAUTH_TOKEN")
        _run(
            [
                "docker",
                "run",
                "--rm",
                "--user",
                "0",
                "--entrypoint",
                "chown",
                "-v",
                f"{root}:/proof",
                IMAGE,
                "-R",
                f"{os.getuid()}:{os.getgid()}",
                "/proof",
            ]
        )
        if os.environ.get("LOCAL_EVAL_KEEP_HARBOR_TESTS") != "1":
            shutil.rmtree(root)


def _bind(source, target, *, read_only=True):
    return {"type": "bind", "source": str(source), "target": target, "read_only": read_only}


def _coding_solution(folder, case):
    script = folder / "solution/solve.sh"
    extra = {
        "install.sh": '#!/bin/bash\nset -eu\ntest "$(id -u)" = 0\ntest -r /opt/eval/images/nvidia-cuda-x86/baselines/freetoken/src/pyproject.toml\ncp /workspace/submission/reference.json "$RUNTIME/source-provenance.json"\nchmod 666 "$RUNTIME/source-provenance.json"\n',
    }
    if case == "native":
        extra = {}  # Keep the production-generated known baseline submission unchanged.
    elif case == "completed":
        extra["launch.sh"] = (
            "#!/bin/bash\nset -eu\nexport CHECKPOINT=/models/source PORT=8000\n"
            'exec bash /opt/eval/images/nvidia-cuda-x86/baselines/freetoken/launch.sh "$@"\n'
        )
    elif case == "convert-fails":
        extra["convert_weight.sh"] = (
            "#!/bin/bash\nset -eu\n"
            'test "$(id -u)" = 0\n'
            'echo "Source checkpoint shards are unavailable; conversion cannot proceed."\nexit 23\n'
        )
    elif case == "lookup":
        # A deliberate source violation, never executed as a model server. Its
        # launch replaces generation with output lookup while retaining real
        # baseline source/provenance for the reviewer to follow.
        extra["launch.sh"] = (
            "#!/bin/bash\nset -eu\nexec /opt/eval/.venv/bin/python /workspace/submission/answer_lookup.py\n"
        )
        extra["answer_lookup.py"] = (
            "import json, sys\nfrom pathlib import Path\n"
            'answers = json.loads(Path(__file__).with_name("saved_answers.json").read_text())\n'
            "for line in sys.stdin:\n"
            "    request = json.loads(line)\n"
            '    prompt = request["messages"][-1]["content"]\n'
            '    print(json.dumps({"content": answers.get(prompt, "42"), "completion_tokens": 1}), flush=True)\n'
        )
        extra["saved_answers.json"] = '{"What is 1 + 1?": "2"}\n'
    script.write_text(
        script.read_text()
        + "\n/opt/eval/.venv/bin/python -I - <<'PYTHON'\n"
        + "import json\nfrom pathlib import Path\n"
        + f"files = json.loads({json.dumps(json.dumps(extra))})\n"
        + "for name, content in files.items():\n    (Path('/workspace/submission') / name).write_text(content)\n"
        + "import os, pwd\n"
        + "assert os.getuid() == pwd.getpwnam('agent').pw_uid != 0\n"
        + "fresh = not Path('/tmp/original-search-only').exists()\n"
        + "assert fresh, 'Search and coding must use separate native containers'\n"
        + "Path('/logs/agent/coding-identity.json').write_text(json.dumps({'coding_uid': os.getuid(), 'coding_container': Path('/etc/hostname').read_text().strip(), 'coding_fresh_from_search': fresh}) + '\\n')\n"
        + "Path('/workspace/original-only').write_text('development container state')\n"
        + "Path('/home/agent/original-only').write_text('development home state')\n"
        + "PYTHON\n"
    )


def _trial(live_runtime, case):
    root, repo, claude, image = live_runtime
    run = root / case
    inputs = {}
    for name in ("public", "private", "quality", "calibration"):
        path = run / "inputs" / name
        path.mkdir(parents=True)
        inputs[name] = str(path)
    models = run / "models"
    source = models / "source"
    shutil.copytree(root / "model-metadata", source)
    assert (source / "config.json").is_file()
    assert not list(source.glob("*.safetensors")), "These are explicitly source-review-only tests"
    asset = {"source": MODELS["C2"].source, "path": str(source), "checkpoint": str(source)}
    assets = {"root": str(models), "source": asset, "baselines": {"freetoken": asset}}
    config = RunConfig(
        agent=AgentConfig("oracle", budget_seconds=60),
        search=SearchConfig(agent=AgentConfig("oracle", budget_seconds=60)),
        evaluation=EvaluationConfig(
            install_timeout_seconds=1800,
            health_timeout_seconds=600,
            agent=AgentConfig("claude-code", auth="login", budget_seconds=1800),
        ),
        out=run,
    )
    folder = render_task(
        Task(f"review_{case}", "C2", "S-1", "SOTA"),
        config,
        Machine("Linux", "x86_64", 4, 8 * 2**30, ()),
        image,
        run,
        inputs,
        assets,
    )
    search = run / "search" / folder.name
    (folder / "environment/start.sh").write_text(START)
    (run / "prepared.json").write_text(
        json.dumps({"assets": {"C2": assets}, "task_paths": [str(folder)]}) + "\n"
    )
    verification = run / "verification" / folder.name
    for compose in (
        search / "environment/docker-compose.yaml",
        folder / "environment/docker-compose.yaml",
        verification / "environment/docker-compose.yaml",
    ):
        data = yaml.safe_load(compose.read_text())
        main = data["services"]["main"]
        main.pop("build", None)
        main["environment"] = {
            "PATH": "/opt/eval/.venv/bin:/opt/venv/bin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
        }
        main["volumes"].extend(
            [
                _bind(repo, "/root/eval/source"),
                _bind(claude, "/usr/local/bin/claude"),
                _bind(root / "baseline", "/opt/eval/images/nvidia-cuda-x86/baselines/freetoken"),
                _bind(root / "uv-cache", "/root/.cache/uv", read_only=False),
                _bind(repo / "images/prepare.sh", "/opt/eval/images/prepare.sh"),
            ]
        )
        compose.write_text(yaml.safe_dump(data, sort_keys=False))
    task_config = search / "task.toml"
    task_config.write_text(
        task_config.read_text().replace(
            "[environment]\n", f"[environment]\ndocker_image = {json.dumps(IMAGE)}\n", 1
        )
    )
    (search / "solution/solve.sh").write_text(SEARCH_SOLUTION)
    # Keep the trusted production finalizer and append observation only.
    search_tests = search / "tests"
    (search_tests / "search-probe.py").write_text(SEARCH_PROBE)
    generated_search = (search_tests / "test.sh").read_text()
    assert "eval.search.run" in generated_search and "--stage gate" in generated_search
    (search_tests / "test.sh").write_text(
        generated_search.replace("exec ", "", 1)
        + "\n/opt/eval/.venv/bin/python -I /tests/search-probe.py\n"
    )
    _coding_solution(folder, case)
    # Preserve the actual generated prepare stage and its root conversion/install.
    prepare = verification / "steps/prepare/tests"
    generated_prepare = (prepare / "test.sh").read_text()
    assert "--stage prepare" in generated_prepare
    (prepare / "prepare-probe.py").write_text(PREPARE_PROBE)
    (prepare / "test.sh").write_text(
        generated_prepare.replace("exec ", "", 1)
        + "\n/opt/eval/.venv/bin/python -I /tests/prepare-probe.py\n"
    )
    review = verification / "steps/review/tests"
    generated_review = (review / "test.sh").read_text()
    assert "--stage review" in generated_review
    (review / "review-probe.py").write_text(REVIEW_PROBE)
    (review / "test.sh").write_text(
        generated_review.replace(
            "exec /opt/eval/.venv/bin/python -I -m eval.verify.run --stage review",
            "exec /opt/eval/.venv/bin/python -I /tests/review-probe.py",
        )
    )
    # No engine shards exist: this last native root step is deliberately an
    # access/persistence probe, never a substitute inference or benchmark result.
    measure = verification / "steps/measure/tests"
    (measure / "measure-probe.py").write_text(MEASURE_PROBE)
    (measure / "test.sh").write_text(
        "#!/bin/bash\nset -eu\nexec /opt/eval/.venv/bin/python -I /tests/measure-probe.py\n"
    )

    async def execute(job_config):
        from harbor.job import Job

        job_config.quiet = True
        job_config.retry.max_retries = 0
        return await (await Job.create(job_config)).run()

    from eval.run import _coding_tasks, _final_results, _verification_tasks

    search_config = _job_config(config, run, (search,), phase="search")
    assert search_config.agents[0].name == "oracle" and search_config.agents[0].import_path is None
    asyncio.run(execute(search_config))
    search_results = list((run / "jobs/search").glob("*/result.json"))
    assert len(search_results) == 1
    search_trial = json.loads(search_results[0].read_text())
    assert search_trial["exception_info"] is None, search_trial["exception_info"]
    assert not search_trial["step_results"]
    assert search_trial["agent_info"]["name"] == "oracle"
    assert search_trial["verifier_result"]["rewards"] == {"reward": 1}
    search_identity = json.loads(
        (search_results[0].parent / "agent/search-identity.json").read_text()
    )
    search_probe = json.loads(
        (search_results[0].parent / "verifier/search-acceptance.json").read_text()
    )
    assert search_identity["search_uid"] != 0 and search_probe["search_finalizer_uid"] == 0
    assert search_identity["search_container"] == search_probe["search_finalizer_container"]
    assert search_identity["search_inference_run"] is False

    coding_paths = _coding_tasks(run, (folder,))
    assert coding_paths == (folder,)
    selected = json.loads((folder / "reference/metadata/reference.json").read_text())
    assert selected["baseline"] == "freetoken" and selected["checkpoint"].startswith(
        "/models/reference"
    )
    assert (folder / "reference/weights").is_symlink()
    assert (folder / "reference/weights").resolve() == source.resolve()
    original = _job_config(config, run, coding_paths, phase="coding")
    assert original.agents[0].name == "oracle" and original.agents[0].import_path is None
    asyncio.run(execute(original))
    original_results = list((run / "jobs/coding").glob("*/result.json"))
    assert len(original_results) == 1
    original_trial = json.loads(original_results[0].read_text())
    assert original_trial["exception_info"] is None, original_trial["exception_info"]
    assert not original_trial["step_results"]
    assert original_trial["agent_info"]["name"] == "oracle"
    assert original_trial["verifier_result"]["rewards"] == {"reward": 1}
    coding_identity = json.loads(
        (original_results[0].parent / "agent/coding-identity.json").read_text()
    )
    assert coding_identity["coding_uid"] != 0 and coding_identity["coding_fresh_from_search"]
    assert coding_identity["coding_container"] != search_identity["search_container"]

    paths = _verification_tasks(run, coding_paths)
    assert paths == (verification,)
    copied = verification / "steps/prepare/workdir/submission"
    assert (copied / "install.sh").is_file() and (copied / "launch.sh").is_file()
    assert not list(copied.parent.glob("weights")) and not list(copied.parent.glob("runtime"))
    verify = _job_config(config, run, paths, phase="verify")
    assert verify.agents[0].name == "nop" and verify.agents[0].import_path is None
    asyncio.run(execute(verify))
    trials = list((run / "jobs/verify").glob("*/result.json"))
    assert len(trials) == 1
    trial = json.loads(trials[0].read_text())
    assert trial["exception_info"] is None, trial["exception_info"]
    assert trial["agent_info"]["name"] == "nop"
    steps = [step["step_name"] for step in trial["step_results"]]
    assert steps == {"convert-fails": ["prepare"], "lookup": ["prepare", "review"]}.get(
        case, ["prepare", "review", "measure"]
    ), steps
    evidence = trials[0].parent / "steps"
    report = json.loads((evidence / "prepare/verifier/acceptance-prepare.json").read_text())
    if "review" in steps:
        review_dir = evidence / "review/verifier"
        report.update(json.loads((review_dir / "acceptance-review.json").read_text()))
        report["rules"] = json.loads((review_dir / "report.json").read_text())
    if "measure" in steps:
        report.update(
            json.loads((evidence / "measure/verifier/review-acceptance.json").read_text())
        )
        # The full native criterion decisions are saved by the genuine review,
        # and the acceptance-only root probe reports its own narrower success.
        report["rules"] = json.loads((evidence / "review/verifier/report.json").read_text())
    report["rewards"] = trial["verifier_result"]["rewards"]
    _final_results(run, (folder,))
    final = json.loads((run / "result.json").read_text())["tasks"][folder.name]
    assert final["rewards"] == report["rewards"]
    assert final["verification_result"] == str(trials[0])
    report["wrapper_status"] = final["status"]
    report["stage"] = steps[-1]
    report.update(search_identity)
    report.update(search_probe)
    report.update(coding_identity)
    assert report["prepare_container"] != report["coding_container"]
    assert report["prepare_container"] != report["search_container"]
    report["search_trial"] = str(search_results[0])
    report["coding_trial"] = str(original_results[0])
    report["original_trial"] = str(original_results[0])
    report["verification_trial"] = str(trials[0])
    assert report["fresh"] is True
    assert report["measurement_run"] is False and report["model_shards_available"] is False
    (root / f"{case}-summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return trial, report, evidence


def _reviewed(report):
    assert report["stage"] in ("review", "measure"), report
    preparation = report["preparation"]
    assert preparation["static"]["ok"] is True
    assert preparation["convert"]["ok"] is True and preparation["convert"]["skipped"] is True
    assert preparation["install"]["exit_code"] == 0
    assert report["prepare_uid"] == 0 and report["review_uid"] != 0
    assert report["prepare_container"] == report["review_container"]
    assert report["review_no_new_privs"] == 1
    assert all(
        role["uids"] == [report["review_uid"]] * 4 and role["no_new_privs"] == 1
        for role in report["process_roles"]
    )
    assert report["permissions"]["ok"] is True
    if report["stage"] == "measure":
        assert report["measure_uid"] == 0 and report["root_writes_retained"]
    assert report["credential_mounts_absent"] and not report["custom_reviewer_settings"]
    assert report["rules"]["complete"] is True, report
    assert len(report["rules"]["criteria"]) == 6
    assert {"Read", "Bash"}.intersection(report["tool_calls"]), report["tool_calls"]


def test_native_oracle_real_claude_reviews_production_baseline_source(live_runtime):
    _, report, _ = _trial(live_runtime, "native")
    _reviewed(report)
    assert report["rules"]["ok"] is True and report["rewards"] == {"reward": 1}


def test_native_oracle_real_claude_reviews_completed_checkpoint_launcher(live_runtime):
    _, report, _ = _trial(live_runtime, "completed")
    _reviewed(report)
    assert report["rules"]["ok"] is True and report["rewards"] == {"reward": 1}


def test_native_oracle_conversion_failure_stops_real_reviewer(live_runtime):
    _, report, evidence = _trial(live_runtime, "convert-fails")
    assert report["stage"] == "prepare"
    assert report["preparation"]["convert"]["exit_code"] == 23
    assert report["prepare_uid"] == 0
    assert "install" not in report["preparation"] and "rules" not in report
    assert report["rewards"] == {"reward": 0} and not (evidence / "review").exists()


def test_native_oracle_real_claude_rejects_precomputed_answer_lookup(live_runtime):
    _, report, _ = _trial(live_runtime, "lookup")
    _reviewed(report)
    assert report["rules"]["ok"] is False and report["rewards"] == {"reward": 0}
    generated = next(
        item for item in report["rules"]["criteria"] if item["name"] == "generated_answers"
    )
    assert generated["value"] == 0 and generated["reasoning"]
