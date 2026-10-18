"""Launch native harnesses from the registry and collect their delegation evidence."""

import json
import os
import re
import shutil
import socket
import sys
from functools import lru_cache
from pathlib import Path
from uuid import uuid4

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from ..config import OUTPUT_ALLOWANCE, TABLES

TEMPLATES = (
    TABLES.parent / ("_templates" if TABLES.name == "_tables" else "templates") / "harnesses"
)


def registry():
    return json.loads((TABLES / "harnesses.json").read_text())


def profile(name, pool):
    """Apply a pool's declared settings while retaining the provider connection."""
    table = registry()
    base = table["harnesses"][name]
    override = base.get("pools", {}).get(pool, {})
    if override is None:
        raise ValueError(f"{name} has no {pool} recording mode")
    selected = {**table["pools"][pool], **base, **override}
    for key in ("env", "files"):
        selected[key] = {**base.get(key, {}), **override.get(key, {})}
    return selected


@lru_cache(maxsize=32)
def _host_runtimes(programs, search_path):
    """Read-only installations, including a venv's interpreter and editable Hermes source."""
    roots = {Path(sys.prefix), Path(sys.base_prefix), Path(sys._base_executable).parent.parent}
    files, aliases, bins = set(), [], [str(Path(sys.executable).parent)]
    for program in programs:
        found = shutil.which(program, path=search_path)
        if found is None:
            continue  # Rendering a profile does not require its CLI to be installed.
        executable = Path(found)
        resolved = executable.resolve()
        bins.extend((str(executable.parent), str(resolved.parent)))
        if "node_modules" in resolved.parts:
            roots.add(Path(*resolved.parts[: resolved.parts.index("node_modules") + 1]))
        elif Path(program).name == "node" and resolved.parent.name == "bin":
            roots.add(resolved.parent.parent)
        else:
            files.add(resolved)
            if resolved.name == "codex":
                companion = resolved.with_name("codex-code-mode-host")
                if companion.is_file():
                    files.add(companion)
        if (executable.parent.parent / "pyvenv.cfg").is_file():
            roots.add(executable.parent.parent)
            interpreter = executable.parent / "python3"
            roots.add(interpreter.resolve().parent.parent)
            while interpreter.is_symlink():
                target = Path(os.readlink(interpreter))
                interpreter = target if target.is_absolute() else interpreter.parent / target
                roots.add(interpreter.parent.parent)
            source = executable.parent.parent.parent / "src"
            if Path(program).name == "hermes" and (source / "hermes_cli").is_dir():
                roots.add(source)
        if executable != resolved:
            aliases.append((resolved, executable))
    return tuple(sorted(roots)), tuple(sorted(files)), tuple(aliases), tuple(dict.fromkeys(bins))


def _host_sandbox(command, environment, workspace, config, name=None, tool_socket=None):
    """Give a host harness its own filesystem and PID view, preserving proxy networking."""
    workspace, config = workspace.resolve(), config.resolve()
    home = config / "home"
    home.mkdir(parents=True, exist_ok=True)
    programs = tuple(dict.fromkeys([command[0], "node", *([name] if name else [])]))
    roots, files, aliases, bins = _host_runtimes(programs, environment["PATH"])
    prefix = [
        "bwrap",
        "--unshare-user",
        "--unshare-pid",
        "--unshare-ipc",
        "--unshare-uts",
        "--new-session",
        "--die-with-parent",
        "--cap-drop",
        "ALL",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",
        "--tmpfs",
        "/run",
        "--dir",
        "/var",
        "--symlink",
        "/run",
        "/var/run",
    ]
    for name in ("/usr", "/bin", "/sbin", "/lib", "/lib64"):
        path = Path(name)
        if path.is_symlink():
            prefix += ["--symlink", os.readlink(path), name]
        elif path.exists():
            prefix += ["--ro-bind", name, name]
    for name in (
        "/etc/ssl",
        "/etc/pki",
        "/etc/ld.so.cache",
        "/etc/resolv.conf",
        "/etc/hosts",
        "/etc/nsswitch.conf",
        "/etc/passwd",
        "/etc/group",
        "/etc/localtime",
    ):
        if Path(name).exists():
            prefix += ["--ro-bind", name, name]
    for path in roots:
        if not path.is_relative_to("/usr"):
            prefix += ["--ro-bind", str(path), str(path)]
    for path in files:
        if not any(path.is_relative_to(root) for root in (*roots, Path("/usr"))):
            prefix += ["--ro-bind", str(path), str(path)]
    for target, alias in aliases:
        system = (Path("/usr"), Path("/bin"), Path("/sbin"), Path("/lib"), Path("/lib64"))
        if not any(alias.is_relative_to(root) for root in (*roots, *system)):
            prefix += ["--symlink", str(target), str(alias)]
    # An explicitly supplied harness script is trusted executable input, not its parent folder.
    if len(command) > 1 and command[1].endswith(".py") and Path(command[1]).is_file():
        script = Path(command[1]).resolve()
        if not script.is_relative_to(workspace) and not script.is_relative_to(config):
            prefix += ["--ro-bind", str(script), str(script)]
    prefix += [
        "--bind",
        str(workspace),
        str(workspace),
        "--bind",
        str(config),
        str(config),
    ]
    if tool_socket is not None:
        prefix += ["--ro-bind", str(tool_socket), "/run/record-tools.sock"]
    environment.update(
        HOME=str(home),
        TMPDIR="/tmp",
        XDG_CACHE_HOME=str(config / "cache"),
        XDG_CONFIG_HOME=str(config / "user-config"),
        PYTHONDONTWRITEBYTECODE="1",
        PATH=":".join([*bins, "/usr/local/bin", "/usr/bin", "/bin"]),
    )
    return [*prefix, "--chdir", str(workspace), "--", *command], environment


def harness(item, workspace, config, url, *, context_window=131072):
    """Render one private native provider connection and pass the first prompt."""
    environment = {
        key: value
        for key, value in os.environ.items()
        if key in ("PATH", "HOME", "USER", "LANG", "TERM", "SHELL", "TMPDIR")
        or key.startswith("LC_")
    }
    environment.update(
        item.get("env", {}), RECORD_ENDPOINT=url + "/v1", RECORD_WORKSPACE=str(workspace)
    )
    if not item.get("container"):
        environment["GIT_CEILING_DIRECTORIES"] = str(workspace.resolve().parent.parent)
    if item.get("command"):
        if item.get("container"):
            return item["command"], environment
        return _host_sandbox(
            item["command"], environment, workspace, config, tool_socket=item.get("tool_socket")
        )
    selected = profile(item["harness"], item["pool"])
    port = None
    if selected.get("gateway"):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
    context = {
        **selected,
        "url": url,
        "workspace": str(workspace),
        "cwd": item.get("container_workspace", str(workspace)),
        "config": str(config),
        "container": bool(item.get("container")),
        "session": uuid4().hex,
        "port": port,
        "multi": item["pool"] == "A-MA",
        "allowance": OUTPUT_ALLOWANCE,
        "context_window": context_window,
        "prompt": item["prompt"] + item.get("prompt_append", "") + selected["prompt_append"],
    }
    renderer = Environment(
        loader=FileSystemLoader(TEMPLATES), undefined=StrictUndefined, keep_trailing_newline=True
    )

    def render(value):
        return renderer.from_string(value).render(context)

    config.mkdir(mode=0o700, parents=True, exist_ok=True)
    for filename, template in selected["files"].items():
        path = config / filename
        path.write_text(renderer.get_template(template).render(context))
        path.chmod(0o600)
    environment.update({key: render(value) for key, value in selected["env"].items()})
    command = []
    location = "container" if item.get("container") else "host"
    for argument in selected["argv"]:
        values = argument[location] if isinstance(argument, dict) else [argument]
        command.extend(render(value) for value in values)
    if item.get("container"):
        return command, environment
    return _host_sandbox(
        command, environment, workspace, config, item["harness"], item.get("tool_socket")
    )


def sub_agents(calls, name, log: Path | None = None):
    """Count successful native delegation results, never inferred graph starts."""
    if name == "codex":
        if log is None:
            return 0
        completed = set()
        for line in log.read_text().split("\n"):
            if line.startswith("{"):
                item = json.loads(line).get("item", {})
                completed.update(
                    child
                    for child, state in item.get("agents_states", {}).items()
                    if state["status"] == "completed"
                )
        for call in calls:
            if call.get("error") or call["abandoned"]:
                continue
            for message in call.get("request", {}).get("messages", []):
                content = message.get("content")
                if (
                    message["role"] == "user"
                    and isinstance(content, str)
                    and content.startswith("<subagent_notification>\n")
                    and content.endswith("\n</subagent_notification>")
                ):
                    notice = json.loads(
                        content.removeprefix("<subagent_notification>\n").removesuffix(
                            "\n</subagent_notification>"
                        )
                    )
                    if "completed" in notice["status"]:
                        completed.add(notice["agent_path"])
        recorded = {
            call.get("raw_request", {}).get("client_metadata", {}).get("thread_id")
            for call in calls
            if not call.get("error") and not call["abandoned"]
        }
        return len(completed & recorded)
    spec = registry()["harnesses"].get(name, {})
    if log is not None and spec.get("children_file"):
        path = log.parent / "config" / spec["children_file"]
        if path.is_file():
            return json.loads(path.read_text())["count"]
    delegation = spec.get("delegation", {})
    delegates, children = {}, set()
    messages = [
        message
        for call in calls
        if not call["abandoned"]
        for message in call["request"]["messages"]
    ]
    for message in messages:
        for tool in message.get("tool_calls", []):
            if tool["function"]["name"] in delegation.get("tools", []):
                delegates[tool["id"]] = tool["function"]["name"]
    for message in messages:
        call_id = message.get("tool_call_id")
        if message["role"] != "tool" or call_id not in delegates:
            continue
        content = message.get("content") or ""
        if not isinstance(content, str):
            content = "".join(part.get("text", "") for part in content)
        if delegation.get("pattern"):
            children.update(re.findall(delegation["pattern"], content))
        elif delegation.get("results"):
            # Native delegation can append a tool-loop warning after its JSON result.
            result, _ = json.JSONDecoder().raw_decode(content)
            children.update(
                (call_id, row["task_index"])
                for row in result.get("results", [])
                if row["status"] == "completed" and row.get("api_calls", 0) > 0
            )
    metadata = delegation.get("log_result")
    if metadata and log is not None:
        # Some native Agent results omit IDs from the model-facing text.
        for line in log.read_text().split("\n"):
            if line.startswith("{"):
                result = json.loads(line).get(metadata["field"])
                if (
                    isinstance(result, dict)
                    and result.get("status") == metadata["status"]
                    and result.get(metadata["id"])
                ):
                    children.add(result[metadata["id"]])
    return len(children)
