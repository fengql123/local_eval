#!/usr/bin/env bash
# Root image preparation: prepare.sh SUBMISSION RUBRIC_DIR OUTPUT_DIR
# Harbor selects root/judge/root for the native verification steps.
set -euo pipefail

submission=$1
rubric=$2
output=$3

# Newly installed artifacts are readable by the reviewer, writable only by root.
# Walk without following links so this cannot chmod a model mount or system tree.
/opt/eval/.venv/bin/python -I - "$submission" "$rubric" "$output" <<'PYTHON'
import os
import pwd
import sys
from pathlib import Path

submission, rubric, output = map(Path, sys.argv[1:])
workspace = submission.parent
completed = (submission, workspace / 'weights', workspace / 'runtime')
trusted = tuple(Path(name) for name in (
    '/models/source', '/models/reference', '/usr', '/opt', '/lib', '/lib64',
    '/bin', '/sbin', '/etc',
))
allowed = tuple(path.resolve() for path in (*completed, *trusted))

def protect(path):
    if path.is_symlink():
        target = path.resolve()
        if not any(target.is_relative_to(root) for root in allowed):
            print(f'Submission link escapes readable artifacts: {path} -> {target}', file=sys.stderr)
            sys.exit(65)
        return
    mode = path.stat().st_mode & 0o777
    os.chown(path, 0, 0)
    path.chmod((mode & ~0o022) | (0o755 if path.is_dir() else 0o644))

protect(workspace)
for root in (*completed, rubric):
    if root.is_symlink():
        print(f'Artifact root must be a directory: {root}', file=sys.stderr)
        sys.exit(65)
    root.mkdir(parents=True, exist_ok=True)
    protect(root)
    for directory, folders, files in os.walk(root, followlinks=False):
        for name in (*folders, *files):
            protect(Path(directory) / name)

account = pwd.getpwnam('judge')
for parent in (output.parent,):
    parent.mkdir(parents=True, exist_ok=True)
    os.chown(parent, 0, 0)
    parent.chmod(0o755)
home = Path(account.pw_dir)
for path in (home, home / 'tmp', home / '.cache', output):
    path.mkdir(parents=True, exist_ok=True)
    os.chown(path, account.pw_uid, account.pw_gid)
    path.chmod(0o700)
PYTHON
