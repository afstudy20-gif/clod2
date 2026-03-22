"""GitHub file tools — read/write/list files in any GitHub repo via the REST API."""
import base64
import os
from typing import Optional

import requests


def _headers(token: str | None = None) -> dict:
    tok = token or os.environ.get("GITHUB_TOKEN", "")
    h = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    if tok:
        h["Authorization"] = f"Bearer {tok}"
    return h


def github_read_file(repo: str, path: str, ref: str = "HEAD", token: str | None = None) -> str:
    """
    Read a file from a GitHub repository.
    repo: 'owner/repo'  e.g. 'afstudy20-gif/Cclaude'
    path: file path inside the repo e.g. 'src/api.py'
    ref: branch, tag or commit SHA (default: HEAD)
    """
    url = f"https://api.github.com/repos/{repo}/contents/{path}"
    params = {"ref": ref}
    resp = requests.get(url, headers=_headers(token), params=params, timeout=15)
    if resp.status_code == 404:
        return f"Error: File not found: {repo}/{path} (ref={ref})"
    if not resp.ok:
        return f"Error {resp.status_code}: {resp.text[:500]}"
    data = resp.json()
    if isinstance(data, list):
        # It's a directory
        lines = [f"{'d' if i.get('type') == 'dir' else 'f'} {i['name']}" for i in data]
        return "\n".join(lines)
    content = base64.b64decode(data["content"]).decode("utf-8", errors="replace")
    lines = content.splitlines()
    numbered = [f"{i+1:4d}\t{line}" for i, line in enumerate(lines)]
    return "\n".join(numbered)


def github_write_file(
    repo: str,
    path: str,
    content: str,
    message: str,
    branch: str = "main",
    token: str | None = None,
) -> str:
    """
    Create or update a file in a GitHub repository.
    repo: 'owner/repo'
    path: file path inside the repo
    content: new file content (plain text)
    message: commit message
    branch: branch to commit to (default: main)
    """
    url = f"https://api.github.com/repos/{repo}/contents/{path}"
    tok = token or os.environ.get("GITHUB_TOKEN", "")

    # Get current SHA if file exists (needed for updates)
    sha: str | None = None
    resp = requests.get(url, headers=_headers(tok), params={"ref": branch}, timeout=15)
    if resp.ok and isinstance(resp.json(), dict):
        sha = resp.json().get("sha")

    encoded = base64.b64encode(content.encode()).decode()
    body: dict = {
        "message": message,
        "content": encoded,
        "branch": branch,
    }
    if sha:
        body["sha"] = sha

    resp = requests.put(url, headers=_headers(tok), json=body, timeout=15)
    if resp.ok:
        action = "Updated" if sha else "Created"
        commit = resp.json().get("commit", {}).get("sha", "")[:8]
        return f"{action} {repo}/{path} on branch '{branch}' (commit {commit})"
    return f"Error {resp.status_code}: {resp.text[:500]}"


def github_list_dir(repo: str, path: str = "", ref: str = "HEAD", token: str | None = None) -> str:
    """
    List files and directories in a GitHub repository path.
    repo: 'owner/repo'
    path: directory path (empty string = repo root)
    ref: branch, tag or commit SHA
    """
    url = f"https://api.github.com/repos/{repo}/contents/{path}"
    resp = requests.get(url, headers=_headers(token), params={"ref": ref}, timeout=15)
    if resp.status_code == 404:
        return f"Error: Path not found: {repo}/{path}"
    if not resp.ok:
        return f"Error {resp.status_code}: {resp.text[:300]}"
    data = resp.json()
    if isinstance(data, dict):
        return f"'{path}' is a file, not a directory. Use github_read_file to read it."
    lines = []
    for item in sorted(data, key=lambda x: (x["type"] != "dir", x["name"])):
        prefix = "d" if item["type"] == "dir" else "f"
        size = f" ({item.get('size', 0):,} bytes)" if item["type"] == "file" else ""
        lines.append(f"{prefix} {item['name']}{size}")
    return "\n".join(lines) or "(empty)"


def github_delete_file(
    repo: str,
    path: str,
    message: str,
    branch: str = "main",
    token: str | None = None,
) -> str:
    """
    Delete a file from a GitHub repository.
    repo: 'owner/repo'
    path: file path to delete
    message: commit message
    branch: branch to commit to
    """
    url = f"https://api.github.com/repos/{repo}/contents/{path}"
    tok = token or os.environ.get("GITHUB_TOKEN", "")

    resp = requests.get(url, headers=_headers(tok), params={"ref": branch}, timeout=15)
    if resp.status_code == 404:
        return f"Error: File not found: {repo}/{path}"
    if not resp.ok:
        return f"Error {resp.status_code}: {resp.text[:300]}"
    sha = resp.json().get("sha")

    resp = requests.delete(
        url,
        headers=_headers(tok),
        json={"message": message, "sha": sha, "branch": branch},
        timeout=15,
    )
    if resp.ok:
        return f"Deleted {repo}/{path} from branch '{branch}'"
    return f"Error {resp.status_code}: {resp.text[:300]}"


def github_search_code(repo: str, query: str, token: str | None = None) -> str:
    """
    Search for code inside a GitHub repository.
    repo: 'owner/repo'
    query: search string (GitHub code search syntax supported)
    """
    url = "https://api.github.com/search/code"
    params = {"q": f"{query} repo:{repo}", "per_page": 20}
    resp = requests.get(url, headers=_headers(token), params=params, timeout=15)
    if not resp.ok:
        return f"Error {resp.status_code}: {resp.text[:300]}"
    items = resp.json().get("items", [])
    if not items:
        return f"No results for '{query}' in {repo}"
    lines = [f"{i['path']} (score: {i.get('score', '?')})" for i in items]
    return "\n".join(lines)
