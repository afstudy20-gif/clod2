"""Git tools — local git operations (status, commit, push, pull, branch, diff, log, PR)."""
import os
import subprocess


def _git(args: list[str], cwd: str | None = None) -> tuple[str, int]:
    """Run a git command and return (output, returncode)."""
    from .implementations import _project_root
    work_dir = cwd or _project_root or os.getcwd()
    try:
        result = subprocess.run(
            ["git"] + args,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=work_dir,
        )
        output = (result.stdout + result.stderr).strip()
        return output or "(no output)", result.returncode
    except subprocess.TimeoutExpired:
        return "Error: git command timed out after 30s", 1
    except FileNotFoundError:
        return "Error: git is not installed", 1
    except Exception as e:
        return f"Error: {e}", 1


def git_status() -> str:
    """Show working tree status (staged, unstaged, untracked files)."""
    out, code = _git(["status", "--short", "--branch"])
    return out


def git_diff(staged: bool = False, path: str = "") -> str:
    """Show changes in working tree or staging area."""
    args = ["diff"]
    if staged:
        args.append("--staged")
    if path:
        args.extend(["--", path])
    out, code = _git(args)
    if not out.strip() or out == "(no output)":
        return "No changes" + (" in staging area" if staged else "")
    return out[:5000]


def git_log(count: int = 10, oneline: bool = True) -> str:
    """Show recent commit history."""
    count = min(count, 50)
    args = ["log", f"-{count}"]
    if oneline:
        args.append("--oneline")
    out, code = _git(args)
    return out


def git_add(paths: str = ".") -> str:
    """Stage files for commit. Use '.' for all changes."""
    file_list = paths.split()
    out, code = _git(["add"] + file_list)
    if code == 0:
        # Show what was staged
        status_out, _ = _git(["status", "--short"])
        return f"Staged: {paths}\n{status_out}"
    return out


def git_commit(message: str) -> str:
    """Create a commit with the given message."""
    if not message:
        return "Error: commit message is required"
    out, code = _git(["commit", "-m", message])
    return out


def git_push(remote: str = "origin", branch: str = "", force: bool = False) -> str:
    """Push commits to remote repository."""
    args = ["push", remote]
    if branch:
        args.append(branch)
    if force:
        args.append("--force-with-lease")
    out, code = _git(args)
    if code != 0:
        return f"Error: {out}"
    return out


def git_pull(remote: str = "origin", branch: str = "") -> str:
    """Pull changes from remote repository."""
    args = ["pull", remote]
    if branch:
        args.append(branch)
    out, code = _git(args)
    return out


def git_branch(name: str = "", delete: bool = False, list_all: bool = False) -> str:
    """Create, delete, or list branches."""
    if list_all or not name:
        out, _ = _git(["branch", "-a"])
        return out
    if delete:
        out, code = _git(["branch", "-d", name])
        return out
    out, code = _git(["checkout", "-b", name])
    return out


def git_checkout(ref: str) -> str:
    """Switch to a branch, tag, or commit."""
    out, code = _git(["checkout", ref])
    return out


def git_create_pr(
    title: str,
    body: str = "",
    base: str = "main",
    head: str = "",
) -> str:
    """Create a pull request using the GitHub CLI (gh).
    Requires gh CLI installed and authenticated.
    """
    args = ["pr", "create", "--title", title, "--base", base]
    if body:
        args.extend(["--body", body])
    if head:
        args.extend(["--head", head])

    from .implementations import _project_root
    work_dir = _project_root or os.getcwd()
    try:
        result = subprocess.run(
            ["gh"] + args,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=work_dir,
        )
        output = (result.stdout + result.stderr).strip()
        return output or "(no output)"
    except FileNotFoundError:
        return "Error: gh CLI is not installed. Install from https://cli.github.com/"
    except Exception as e:
        return f"Error: {e}"
