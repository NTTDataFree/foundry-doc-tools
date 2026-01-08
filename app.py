import base64
import io
import os
import re
import tempfile
import zipfile
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from uuid import uuid4

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from azure.identity import DefaultAzureCredential
from azure.storage.blob import (
    BlobServiceClient,
    ContentSettings,
    generate_blob_sas,
    BlobSasPermissions,
)

app = FastAPI(
    title="Foundry Documentation Tools",
    version="1.0.0",
    description="Tools for cloning repos, storing generated docs/diagrams in Blob, zipping artifacts, and creating PRs."
)

# -----------------------------
# Config (App Settings)
# -----------------------------
STORAGE_ACCOUNT_NAME = os.getenv("STORAGE_ACCOUNT_NAME", "")
STORAGE_CONTAINER = os.getenv("STORAGE_CONTAINER", "documentation-data")

# OPTION A (recommended): Managed Identity to read/write blobs
# OPTION B (simple): Connection string (if you want)
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")

# For SAS generation (simple path):
STORAGE_ACCOUNT_KEY = os.getenv("STORAGE_ACCOUNT_KEY", "")

# GitHub
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
GITHUB_API = os.getenv("GITHUB_API", "https://api.github.com")
DEFAULT_GITHUB_OWNER = os.getenv("DEFAULT_GITHUB_OWNER", "")

# Safety limits
MAX_ZIP_MB = int(os.getenv("MAX_ZIP_MB", "80"))  # avoid huge repos in this demo
MAX_FILE_CHARS_DEFAULT = int(os.getenv("MAX_FILE_CHARS_DEFAULT", "12000"))


def _blob_service_client() -> BlobServiceClient:
    if AZURE_STORAGE_CONNECTION_STRING:
        return BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
    if not STORAGE_ACCOUNT_NAME:
        raise RuntimeError("STORAGE_ACCOUNT_NAME is required if no AZURE_STORAGE_CONNECTION_STRING is set.")
    account_url = f"https://{STORAGE_ACCOUNT_NAME}.blob.core.windows.net"
    cred = DefaultAzureCredential()
    return BlobServiceClient(account_url=account_url, credential=cred)


def _container_client():
    return _blob_service_client().get_container_client(STORAGE_CONTAINER)


def _safe_repo_name(repo_full_name: str) -> str:
    # "org/repo" -> "repo"
    if "/" in repo_full_name:
        return repo_full_name.split("/", 1)[1].strip()
    return repo_full_name.strip()


def _safe_path(rel_path: str) -> str:
    rel_path = rel_path.replace("\\", "/").strip()
    if rel_path.startswith("/") or ".." in rel_path.split("/"):
        raise HTTPException(status_code=400, detail="Invalid relative path")
    return rel_path


def _github_headers() -> Dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
    }
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    return headers


def _download_github_zip(repo_full_name: str, ref: str) -> bytes:
    # Zipball: GET /repos/{owner}/{repo}/zipball/{ref}
    url = f"{GITHUB_API}/repos/{repo_full_name}/zipball/{ref}"
    r = requests.get(url, headers=_github_headers(), timeout=120)
    if r.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"GitHub zip download failed: {r.status_code} {r.text}")
    content = r.content
    if len(content) > MAX_ZIP_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"Repo zip too large (> {MAX_ZIP_MB} MB).")
    return content


def _put_blob(blob_name: str, data: bytes, content_type: str = "application/octet-stream") -> str:
    cc = _container_client()
    blob = cc.get_blob_client(blob_name)
    blob.upload_blob(
        data,
        overwrite=True,
        content_settings=ContentSettings(content_type=content_type),
    )
    return blob.url


def _get_blob_bytes(blob_name: str) -> bytes:
    cc = _container_client()
    blob = cc.get_blob_client(blob_name)
    stream = blob.download_blob()
    return stream.readall()


def _list_blobs(prefix: str) -> List[str]:
    cc = _container_client()
    return [b.name for b in cc.list_blobs(name_starts_with=prefix)]


def _make_sas_url(blob_name: str, minutes: int = 120) -> Optional[str]:
    # Simplest: SAS with account key.
    if not STORAGE_ACCOUNT_NAME or not STORAGE_ACCOUNT_KEY:
        return None
    sas = generate_blob_sas(
        account_name=STORAGE_ACCOUNT_NAME,
        container_name=STORAGE_CONTAINER,
        blob_name=blob_name,
        account_key=STORAGE_ACCOUNT_KEY,
        permission=BlobSasPermissions(read=True),
        expiry=datetime.utcnow() + timedelta(minutes=minutes),
    )
    return f"https://{STORAGE_ACCOUNT_NAME}.blob.core.windows.net/{STORAGE_CONTAINER}/{blob_name}?{sas}"


def _extract_zip_to_temp(zip_bytes: bytes) -> str:
    tmpdir = tempfile.mkdtemp(prefix="repo_")
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        zf.extractall(tmpdir)
    return tmpdir


def _iter_text_files(root_dir: str):
    # GitHub zipball usually has a single top folder, e.g. org-repo-sha/
    # We'll walk everything but skip obvious binaries.
    skip_dirs = {"node_modules", ".git", "dist", "build", ".venv", "venv", "__pycache__"}
    text_ext_allow = {
        ".md", ".txt", ".py", ".js", ".ts", ".java", ".cs", ".go", ".rb", ".php",
        ".yml", ".yaml", ".json", ".toml", ".ini", ".cfg",
        ".sql", ".sh", ".ps1",
        ".xml", ".html", ".css",
        ".gradle", ".properties",
        ".dockerfile", "dockerfile",
    }

    for dirpath, dirnames, filenames in os.walk(root_dir):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs and not d.startswith(".git")]
        for fn in filenames:
            full = os.path.join(dirpath, fn)
            lower = fn.lower()
            _, ext = os.path.splitext(lower)
            if lower == "dockerfile":
                yield full
                continue
            if ext in text_ext_allow:
                yield full


def _search_in_repo(zip_bytes: bytes, query: str, max_results: int) -> List[Dict[str, Any]]:
    root = _extract_zip_to_temp(zip_bytes)
    # simple case-insensitive substring search
    q = query.strip()
    if not q:
        return []
    pattern = re.compile(re.escape(q), re.IGNORECASE)

    results = []
    for file_path in _iter_text_files(root):
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
        except Exception:
            continue

        for i, line in enumerate(lines):
            if pattern.search(line):
                # Build a small context snippet
                start = max(0, i - 2)
                end = min(len(lines), i + 3)
                snippet = "".join(lines[start:end])
                rel = os.path.relpath(file_path, root).replace("\\", "/")
                results.append({
                    "file": rel,
                    "line": i + 1,
                    "snippet": snippet[:1500],
                })
                if len(results) >= max_results:
                    return results
    return results


def _read_file_from_repo(zip_bytes: bytes, path: str, max_chars: int) -> str:
    root = _extract_zip_to_temp(zip_bytes)
    # find file by suffix match (because zipball has top folder)
    # Example: "org-repo-sha/README.md" endswith "README.md"
    normalized = path.replace("\\", "/").lstrip("/")
    candidates = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, root).replace("\\", "/")
            if rel.endswith(normalized):
                candidates.append(full)
    if not candidates:
        raise HTTPException(status_code=404, detail=f"File not found in repo zip: {path}")
    target = candidates[0]
    with open(target, "r", encoding="utf-8", errors="ignore") as f:
        return f.read(max_chars)


# -----------------------------
# OpenAPI Models (schemas)
# -----------------------------
class RepoInitIn(BaseModel):
    repo_full_name: str = Field(..., description="GitHub repo in 'org/repo' format, or just 'repo' if DEFAULT_GITHUB_OWNER is set.")
    ref: str = Field("main", description="Branch, tag, or SHA.")


class RepoInitOut(BaseModel):
    repo_full_name: str
    repo_name: str
    repo_root_prefix: str
    source_zip_blob: str
    source_zip_url: str


class RepoSearchIn(BaseModel):
    repo_root_prefix: str
    query: str
    max_results: int = 8


class RepoSearchOut(BaseModel):
    results: List[Dict[str, Any]]


class RepoReadFileIn(BaseModel):
    repo_root_prefix: str
    path: str
    max_chars: int = MAX_FILE_CHARS_DEFAULT


class RepoReadFileOut(BaseModel):
    content: str


class SaveFileIn(BaseModel):
    repo_root_prefix: str
    relative_path: str = Field(..., description="Path inside the repo where the file should be committed, e.g. 'docs/ai/technical.md'")
    content: str
    content_type: str = "text/plain; charset=utf-8"


class SaveFileOut(BaseModel):
    blob_name: str
    blob_url: str


class ZipDiagramsIn(BaseModel):
    repo_root_prefix: str
    diagrams_prefix: str = Field(..., description="Prefix inside repo to include, e.g. 'docs/ai/diagrams/'")
    zip_blob_name: str = Field("artifacts/diagrams.zip", description="Blob path relative to repo_root_prefix")


class ZipDiagramsOut(BaseModel):
    zip_blob: str
    zip_url: str
    zip_sas_url: Optional[str] = None


class CreatePrIn(BaseModel):
    repo_full_name: str
    base_branch: str = "main"
    repo_root_prefix: str
    pr_title: str = "docs: AI-generated documentation update"
    pr_body: str = "This PR was generated by an Azure AI Foundry multi-agent workflow."
    commit_message: str = "docs: update generated documentation"


class CreatePrOut(BaseModel):
    pr_url: str
    branch_name: str


# -----------------------------
# Endpoints (operationId must be simple)
# -----------------------------
@app.get("/health", operation_id="health_check")
def health_check():
    return {"ok": True}


@app.post("/repo/init", response_model=RepoInitOut, operation_id="repo_init")
def repo_init(body: RepoInitIn):
    repo_full_name = body.repo_full_name.strip()
    if "/" not in repo_full_name:
        if not DEFAULT_GITHUB_OWNER:
            raise HTTPException(status_code=400, detail="repo_full_name must be 'org/repo' unless DEFAULT_GITHUB_OWNER is set.")
        repo_full_name = f"{DEFAULT_GITHUB_OWNER}/{repo_full_name}"

    repo_name = _safe_repo_name(repo_full_name)
    repo_root_prefix = f"{repo_name}/"

    # create a "folder" marker
    _put_blob(f"{repo_root_prefix}.keep", b"", "text/plain")

    # download zip from GitHub and upload
    zip_bytes = _download_github_zip(repo_full_name, body.ref)
    source_zip_blob = f"{repo_root_prefix}source/source.zip"
    source_zip_url = _put_blob(source_zip_blob, zip_bytes, "application/zip")

    return RepoInitOut(
        repo_full_name=repo_full_name,
        repo_name=repo_name,
        repo_root_prefix=repo_root_prefix,
        source_zip_blob=source_zip_blob,
        source_zip_url=source_zip_url,
    )


@app.post("/repo/search", response_model=RepoSearchOut, operation_id="repo_search")
def repo_search(body: RepoSearchIn):
    repo_root_prefix = body.repo_root_prefix.strip()
    source_zip_blob = f"{repo_root_prefix}source/source.zip"
    zip_bytes = _get_blob_bytes(source_zip_blob)
    results = _search_in_repo(zip_bytes, body.query, body.max_results)
    return RepoSearchOut(results=results)


@app.post("/repo/read-file", response_model=RepoReadFileOut, operation_id="repo_read_file")
def repo_read_file(body: RepoReadFileIn):
    repo_root_prefix = body.repo_root_prefix.strip()
    source_zip_blob = f"{repo_root_prefix}source/source.zip"
    zip_bytes = _get_blob_bytes(source_zip_blob)
    content = _read_file_from_repo(zip_bytes, body.path, body.max_chars)
    return RepoReadFileOut(content=content)


@app.post("/storage/save-generated-file", response_model=SaveFileOut, operation_id="save_generated_file")
def save_generated_file(body: SaveFileIn):
    repo_root_prefix = body.repo_root_prefix.strip()
    rel = _safe_path(body.relative_path)

    blob_name = f"{repo_root_prefix}pr_files/{rel}"
    url = _put_blob(blob_name, body.content.encode("utf-8"), body.content_type)
    return SaveFileOut(blob_name=blob_name, blob_url=url)


@app.post("/storage/zip-diagrams", response_model=ZipDiagramsOut, operation_id="zip_diagrams")
def zip_diagrams(body: ZipDiagramsIn):
    repo_root_prefix = body.repo_root_prefix.strip()
    diagrams_prefix = _safe_path(body.diagrams_prefix)
    zip_rel = _safe_path(body.zip_blob_name)

    prefix = f"{repo_root_prefix}pr_files/{diagrams_prefix}"
    blobs = _list_blobs(prefix)
    if not blobs:
        raise HTTPException(status_code=404, detail=f"No diagram files found under prefix: {prefix}")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for bname in blobs:
            data = _get_blob_bytes(bname)
            arcname = bname[len(f"{repo_root_prefix}pr_files/"):]  # keep repo-relative paths
            zf.writestr(arcname, data)

    zip_blob = f"{repo_root_prefix}{zip_rel}"
    zip_url = _put_blob(zip_blob, buf.getvalue(), "application/zip")
    sas_url = _make_sas_url(zip_blob, minutes=240)

    return ZipDiagramsOut(zip_blob=zip_blob, zip_url=zip_url, zip_sas_url=sas_url)


def _github_request(method: str, url: str, json_body: Optional[dict] = None) -> dict:
    r = requests.request(method, url, headers=_github_headers(), json=json_body, timeout=120)
    if r.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"GitHub API failed: {r.status_code} {r.text}")
    return r.json() if r.text else {}


@app.post("/github/create-pr-from-blob", response_model=CreatePrOut, operation_id="create_pr_from_blob")
def create_pr_from_blob(body: CreatePrIn):
    if not GITHUB_TOKEN:
        raise HTTPException(status_code=500, detail="GITHUB_TOKEN is not configured in App Settings.")

    repo_full_name = body.repo_full_name.strip()
    repo_root_prefix = body.repo_root_prefix.strip()

    # 1) gather generated files from blob
    prefix = f"{repo_root_prefix}pr_files/"
    blob_names = _list_blobs(prefix)
    if not blob_names:
        raise HTTPException(status_code=404, detail=f"No generated files found under {prefix}")

    files: List[Dict[str, Any]] = []
    for bname in blob_names:
        data = _get_blob_bytes(bname)
        rel_path = bname[len(prefix):]  # path inside repo
        files.append({"path": rel_path, "bytes": data})

    # 2) get base branch SHA
    base_ref_url = f"{GITHUB_API}/repos/{repo_full_name}/git/ref/heads/{body.base_branch}"
    base_ref = _github_request("GET", base_ref_url)
    base_sha = base_ref["object"]["sha"]

    # 3) create new branch
    suffix = uuid4().hex[:8]
    branch_name = f"docs/ai-foundry/{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}-{suffix}"
    create_ref_url = f"{GITHUB_API}/repos/{repo_full_name}/git/refs"
    _github_request("POST", create_ref_url, {"ref": f"refs/heads/{branch_name}", "sha": base_sha})

    # 4) upsert files
    for f in files:
        path = f["path"]
        content_b64 = base64.b64encode(f["bytes"]).decode("utf-8")

        # check if file exists to get sha
        get_url = f"{GITHUB_API}/repos/{repo_full_name}/contents/{path}?ref={branch_name}"
        r = requests.get(get_url, headers=_github_headers(), timeout=60)
        sha = None
        if r.status_code == 200:
            sha = r.json().get("sha")
        elif r.status_code not in (404,):
            raise HTTPException(status_code=502, detail=f"GitHub content check failed: {r.status_code} {r.text}")

        put_url = f"{GITHUB_API}/repos/{repo_full_name}/contents/{path}"
        payload = {
            "message": body.commit_message,
            "content": content_b64,
            "branch": branch_name,
        }
        if sha:
            payload["sha"] = sha

        _github_request("PUT", put_url, payload)

    # 5) create PR
    pr_url = f"{GITHUB_API}/repos/{repo_full_name}/pulls"
    pr = _github_request("POST", pr_url, {
        "title": body.pr_title,
        "head": branch_name,
        "base": body.base_branch,
        "body": body.pr_body,
    })

    return CreatePrOut(pr_url=pr["html_url"], branch_name=branch_name)
