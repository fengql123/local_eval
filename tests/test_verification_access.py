"""Two working verification boundaries and two adversarial cases, without GPU/model use."""

import asyncio
import json
import os
import subprocess
import sysconfig
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from harbor.environments.base import ExecResult
from harbor.environments.docker.docker_unix import UnixOps
from harbor.models.task.config import TaskOS
from harbor.models.task.task import Task
from harbor.models.trial.paths import TrialPaths
from harbor.verifier.verifier import Verifier

ROOT = Path(__file__).resolve().parents[1]


def docker(*args, input=None):
    result = subprocess.run(["docker", *args], input=input, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr or result.stdout
    return result.stdout


@pytest.fixture(scope="module")
def sandbox():
    image = os.environ.get("LOCAL_EVAL_VERIFY_IMAGE")
    if image is None:
        pytest.skip("set LOCAL_EVAL_VERIFY_IMAGE to a production verification image")
    name = "local-eval-access-" + uuid4().hex[:12]
    docker(
        "create",
        "--name",
        name,
        "--user",
        "root",
        "--runtime",
        "runc",
        "--init",
        "--memory",
        "1g",
        "--cpus",
        "1",
        "--security-opt",
        "no-new-privileges:true",
        "--mount",
        f"type=bind,src={ROOT / 'eval'},dst=/opt/eval/eval,readonly",
        "--mount",
        f"type=bind,src={ROOT / 'tables'},dst=/opt/eval/tables,readonly",
        "--mount",
        f"type=bind,src={sysconfig.get_path('purelib')},dst=/packages,readonly",
        "--env",
        "PYTHONPATH=/opt/eval:/packages",
        image,
    )
    try:
        docker("start", name)
        probe(
            name,
            r"""
from pathlib import Path
from eval.verify.access import readonly, prepare_workspace
prepare_workspace('verification')
private = Path('/var/lib/local-eval/private')
private.mkdir()
(private / 'answer').write_text('private fixture answer')
readonly(private, private=True, account='root')
Path('/models/target/model').write_text('fixed target')
readonly(Path('/models/target'))
submission = Path('/workspace')
(submission / 'install.sh').write_text(
    '#!/bin/bash\nset -eu\n'
    'printf "install:%s\\n" "$(id -u)" > ./order\n'
    'printf "#!/bin/bash\\nexit 0\\n" > "./server.launch"\n'
    'chmod +x "./server.launch"\n'
    'echo acquired > ./installed-asset\n'
    'echo temporary > "$TMPDIR/local-eval-install-leftover"\n')
print('{}')
""",
        )
        yield name
    finally:
        docker("rm", "--force", name)


def probe(name, code):
    result = docker("exec", "-i", name, "python3", "-", input=code)
    return json.loads(result.strip().splitlines()[-1])


def test_preparation_and_runtime_identities_have_separate_writable_outputs(sandbox):
    result = probe(
        sandbox,
        r"""
import asyncio, json, os, subprocess
from pathlib import Path
from eval.bench.serve import process_command
from eval.verify.access import reset_runtime, seal_submission
from eval.verify import run
out = Path('/logs/verifier')
task = {'evaluation': {'install_timeout_seconds': 5}}
prepared = asyncio.run(run._prepare_script(task, Path('/workspace'), 'install', out/'install.log'))
seal_submission(Path('/workspace'))
assert Path('/workspace').stat().st_uid == 0
assert not Path('/workspace').stat().st_mode & 0o222
reset_runtime('verifier')
assert not Path('/workspace/runtime/candidate/tmp/local-eval-install-leftover').exists()
observed = []
for uid, account in ((1004, 'verifier'), (1005, 'reference')):
    command, env = process_command(['python3', '-c',
        'import json,os; from pathlib import Path; '
        'Path(os.environ["XDG_CACHE_HOME"], "working").write_text("ok"); '
        'print(json.dumps([os.getuid(),os.getgid(),os.getgroups(),'
        'next(x.split()[1] for x in Path("/proc/self/status").read_text().splitlines() '
        'if x.startswith("NoNewPrivs:"))]))'], run.runtime_environment(account), uid)
    observed.append(json.loads(subprocess.check_output(command, env=env, text=True)))
print(json.dumps({'controller': os.getuid(), 'prepared': prepared['ok'],
    'order': Path('/workspace/order').read_text().splitlines(), 'runtime': observed}))
""",
    )
    assert result == {
        "controller": 0,
        "prepared": True,
        "order": ["install:1004"],
        "runtime": [[1004, 1004, [], "1"], [1005, 1005, [], "1"]],
    }


def test_native_verifier_logs_stay_protected_and_download_to_host(sandbox, tmp_path):
    class Container:
        os = TaskOS.LINUX
        capabilities = SimpleNamespace(mounted=False)
        default_user = "root"

        async def exec(self, command, env=None, user=None):
            args = ["exec", "--user", user or self.default_user]
            for key, value in (env or {}).items():
                args.extend(["--env", f"{key}={value}"])
            return ExecResult(
                stdout=docker(*args, sandbox, "bash", "-c", command), stderr="", return_code=0
            )

        async def upload_dir(self, source_dir, target_dir):
            docker("cp", f"{source_dir}/.", f"{sandbox}:{target_dir}")

        async def _run_docker_compose_command(self, command, **kwargs):
            converted = [part.replace("main:", f"{sandbox}:") for part in command]
            return ExecResult(stdout=docker(*converted), stderr="", return_code=0)

        async def download_dir(self, source_dir, target_dir):
            await UnixOps(self).download_dir(source_dir, target_dir)

        @staticmethod
        def runtime():
            return SimpleNamespace(supports_compose_cp=True)

    environment = Container()
    for step, user in (
        ("reference", "root"),
        ("prepare", "root"),
        ("review", "judge"),
        ("measure", "root"),
    ):
        # Use the real reset protection with a stand-in for native empty_dirs.
        probe(
            sandbox,
            r"""
import asyncio, json, subprocess
from types import SimpleNamespace
from eval.providers.transfer import VerifierLogs
class Environment(VerifierLogs):
    manifest = {'phase': 'verification'}
    async def exec(self, command, user=None, env=None, timeout_sec=None):
        subprocess.run(command, shell=True, check=True)
        return SimpleNamespace(return_code=0)
async def empty(dirs, chmod):
    assert chmod is False
    subprocess.run(['find', dirs[0], '-mindepth', '1', '-delete'], check=True)
asyncio.run(Environment()._verifier_logs(empty, ['/logs/verifier'], True))
print('{}')
""",
        )
        folder = tmp_path / step
        (folder / "environment").mkdir(parents=True)
        (folder / "tests").mkdir()
        (folder / "instruction.md").write_text("Native verifier access check.")
        (folder / "task.toml").write_text('[environment]\ndocker_image="production"\n')
        (folder / "tests/test.sh").write_text(
            f"#!/bin/bash\nset -eu\ntest $(id -un) = {user}\n"
            "test -r /workspace/order\n"
            "echo '{\"reward\":1}' > /logs/verifier/reward.json\n"
        )
        environment.default_user = user
        paths = TrialPaths(trial_dir=tmp_path / f"trial-{step}")
        paths.verifier_dir.mkdir(parents=True)
        result = asyncio.run(Verifier(Task(folder), paths, environment).verify())
        assert result.rewards == {"reward": 1}
        assert paths.reward_json_path.stat().st_uid == os.getuid()
    modes = probe(
        sandbox,
        r"""
import json
from pathlib import Path
p = Path('/logs/verifier').stat()
print(json.dumps([p.st_uid,p.st_gid,p.st_mode & 0o777]))
""",
    )
    assert modes == [0, 1003, 0o770]


def test_submitted_process_cannot_read_answers_or_forge_trusted_outputs(sandbox):
    result = probe(
        sandbox,
        r"""
import json, subprocess
from eval.bench.serve import process_command
from eval.verify.run import runtime_environment
from pathlib import Path
code = '''import json, os
from pathlib import Path
denied = []
for path, operation in [
    ('/var/lib/local-eval/private/answer', 'read_text'),
    ('/var/lib/local-eval/verification/install.json', 'write_text'),
    ('/logs/verifier/reward.json', 'write_text'),
    ('/workspace/.harbor/review-rubric/prompt.txt', 'write_text'),
    ('/workspace/installed-asset', 'write_text'),
    ('/models/target/model', 'write_text'),
    ('/workspace/install.sh', 'write_text'),
    ('/workspace/order', 'write_text'),
    ('/root/.local-eval-cache/forged', 'write_text')]:
    try:
        getattr(Path(path), operation)(*(['forged'] if operation == 'write_text' else []))
    except PermissionError:
        denied.append(path)
Path(os.environ['XDG_CACHE_HOME'], 'allowed').write_text('runtime scratch')
assert Path('/workspace/installed-asset').read_text() == 'acquired' + chr(10)
assert Path('/models/target/model').read_text() == 'fixed target'
print(json.dumps({'denied': len(denied), 'readable_weights': True,
                  'credentials': 'ANTHROPIC_API_KEY' in os.environ}))'''
observed = []
for uid, account in ((1004, 'verifier'), (1003, 'judge')):
    from eval.verify.access import directory
    for p in (Path('/workspace/.harbor/scratch')/account, Path('/workspace/.harbor/scratch')/account/'cache', Path('/workspace/.harbor/scratch')/account/'tmp'):
        directory(p, account, 0o700)
    command, env = process_command(['python3','-c',code], runtime_environment(account), uid)
    observed.append(json.loads(subprocess.check_output(command,env=env,text=True)))
print(json.dumps(observed))
""",
    )
    assert result == [{"denied": 9, "readable_weights": True, "credentials": False}] * 2


def test_loader_environment_is_applied_after_demotion_and_detached_children_stop(sandbox):
    result = probe(
        sandbox,
        r"""
import asyncio, json, os, subprocess
from pathlib import Path
from eval.bench.serve import process_command
from eval.verify.run import runtime_environment, stop_runtime_users
folder = Path('/workspace')
(folder / 'loader.c').write_text(
    '#include <stdio.h>\n#include <unistd.h>\n'
    '__attribute__((constructor)) void record_uid(void) {'
    'FILE *f=fopen("/workspace/runtime/candidate/tmp/loader-uid","w");'
    'if(f){fprintf(f,"%d",getuid());fclose(f);}}')
subprocess.run(['gcc','-shared','-fPIC',str(folder/'loader.c'),'-o',str(folder/'loader.so')],check=True)
command, outer = process_command(['python3','-c','print("ready")'],
    {**runtime_environment(),'LD_PRELOAD':str(folder/'loader.so')},1004)
subprocess.run(command,env=outer,check=True,capture_output=True)
assert 'LD_PRELOAD' not in outer
children = []
for uid, account in ((1004,'verifier'),(1005,'reference')):
    code = ('import subprocess,sys; p=subprocess.Popen(["sleep","60"],start_new_session=True,'
            'stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL);print(p.pid)')
    command, env = process_command(['python3','-c',code],runtime_environment(account),uid)
    children.append(int(subprocess.check_output(command,env=env,text=True)))
asyncio.run(stop_runtime_users())
alive = [pid for pid in children if Path(f'/proc/{pid}/status').exists() and
    not next(x for x in Path(f'/proc/{pid}/status').read_text().splitlines() if x.startswith('State:')).split()[1] == 'Z']
print(json.dumps({'loader_uid':int(Path('/workspace/runtime/candidate/tmp/loader-uid').read_text()),'alive':alive}))
""",
    )
    assert result == {"loader_uid": 1004, "alive": []}
