import configparser
import hashlib
import json
import os
import queue
import re
import threading
import time
import urllib.parse
import uuid

import requests
from aiohttp import web

import folder_paths
import server


class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False


ANY = AnyType("*")
DEFAULT_ALLOWED_HOSTS = ("huggingface.co", "hf-mirror.com", "modelscope.cn", "github.com", "githubusercontent.com")
ALWAYS_ALLOWED_HOSTS = ("civitai.com",)
DEFAULT_MODELS_JSON = json.dumps(
    [
        {
            "name": "Qwen Image VAE",
            "url": "https://modelscope.cn/models/Comfy-Org/Qwen-Image_ComfyUI/resolve/master/split_files/vae/qwen_image_vae.safetensors",
            "download_directory": "vae",
            "file_name": "qwen_image_vae.safetensors",
            "overwrite_existing": False,
            "size": "",
            "sha256": "a70580f0213e67967ee9c95f05bb400e8fb08307e017a924bf3441223e023d1f",
            "description": ""
        }
    ],
    indent=2,
)

_TASKS = {}
_TASK_LOCK = threading.Lock()
_DOWNLOAD_QUEUE = queue.Queue()
_TERMINAL_STATES = {"success", "error", "partial", "skipped"}
_CONFIG_CACHE = None


def _now():
    return time.time()


def _set_task(task_key, **updates):
    with _TASK_LOCK:
        task = _TASKS.setdefault(task_key, {})
        task.update(updates)
        task["updated_at"] = _now()
        return dict(task)


def _get_task(task_id):
    with _TASK_LOCK:
        task = _TASKS.get(task_id)
        if not task:
            return None
        return dict(task)


def _cleanup_tasks(max_age_seconds=3600):
    cutoff = _now() - max_age_seconds
    with _TASK_LOCK:
        stale_ids = [
            task_id
            for task_id, task in _TASKS.items()
            if task.get("status") in _TERMINAL_STATES and task.get("updated_at", 0) < cutoff
        ]
        for task_id in stale_ids:
            _TASKS.pop(task_id, None)


def _parse_urls(raw_urls):
    if isinstance(raw_urls, (list, tuple)):
        candidates = raw_urls
    else:
        candidates = str(raw_urls or "").splitlines()

    urls = []
    for candidate in candidates:
        url = str(candidate).strip()
        if not url or url.startswith("#"):
            continue
        urls.append(url)
    return urls


def _normalize_host(value):
    text = str(value or "").strip().lower()
    if not text:
        return ""
    text = text.removeprefix("*.").strip()
    parsed = urllib.parse.urlparse(text if "://" in text else f"//{text}")
    host = parsed.hostname or text.split("/", 1)[0].split(":", 1)[0]
    return host.strip().strip(".")


def _host_matches(host, allowed_host):
    host = _normalize_host(host)
    allowed_host = _normalize_host(allowed_host)
    return bool(host and allowed_host and (host == allowed_host or host.endswith(f".{allowed_host}")))


def _is_allowed_download_host(host):
    return any(_host_matches(host, allowed) for allowed in [*DEFAULT_ALLOWED_HOSTS, *ALWAYS_ALLOWED_HOSTS])


def _as_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("true", "1", "yes", "y", "on"):
            return True
        if normalized in ("false", "0", "no", "n", "off"):
            return False
    return default


def _parse_expected_size(value):
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        raise ValueError("Invalid size value: use an exact byte count, for example 123456789.")
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, float):
        if value >= 0 and value.is_integer():
            return int(value)
        raise ValueError("Invalid size value: use an exact byte count, for example 123456789.")

    text = str(value).strip()
    if not text:
        return None

    if not re.fullmatch(r"\d+", text):
        raise ValueError("Invalid size value: use an exact byte count, for example 123456789.")

    return int(text)


def _clean_hash(value):
    if value in (None, ""):
        return ""
    return re.sub(r"[^a-fA-F0-9]", "", str(value)).lower()


def _hashes_from_entry(entry):
    hash_data = entry.get("hashes") if isinstance(entry.get("hashes"), dict) else {}

    sha256 = _clean_hash(
        _first_present(entry, ("sha256", "hash_sha256"), _first_present(hash_data, ("sha256",), ""))
    )
    sha1 = _clean_hash(_first_present(entry, ("sha1", "hash_sha1"), _first_present(hash_data, ("sha1",), "")))
    md5 = _clean_hash(_first_present(entry, ("md5", "hash_md5"), _first_present(hash_data, ("md5",), "")))

    generic = _clean_hash(_first_present(entry, ("hash", "checksum"), ""))
    if generic:
        if len(generic) == 64 and not sha256:
            sha256 = generic
        elif len(generic) == 40 and not sha1:
            sha1 = generic
        elif len(generic) == 32 and not md5:
            md5 = generic

    for label, expected, length in (("sha256", sha256, 64), ("sha1", sha1, 40), ("md5", md5, 32)):
        if expected and len(expected) != length:
            raise ValueError(f"Invalid {label} hash length.")

    return {"sha256": sha256, "sha1": sha1, "md5": md5}


def _first_present(data, names, default=None):
    for name in names:
        if isinstance(data, dict) and name in data and data[name] not in (None, ""):
            return data[name]
    return default


def _infer_model_name(entry, urls, index):
    explicit = _first_present(entry, ("name", "title", "label", "id"))
    if explicit:
        return str(explicit)

    filename = _first_present(entry, ("file_name", "filename", "file"))
    if filename:
        return os.path.basename(str(filename))

    if urls:
        return _filename_from_url(urls[0])

    return f"Model {index}"


def _normalize_model_entry(entry, defaults, index):
    if isinstance(entry, str):
        entry = {"url": entry}
    if not isinstance(entry, dict):
        raise ValueError(f"Model #{index} must be an object or URL string.")

    urls = _parse_urls(
        _first_present(entry, ("urls", "model_urls", "url", "model_url", "download_url"), "")
    )
    if not urls:
        raise ValueError(f"Model #{index} has no URL.")

    download_directory = _first_present(
        entry,
        ("download_directory", "directory", "folder", "model_folder", "target_dir", "save_to"),
        _first_present(defaults, ("download_directory", "directory", "folder", "model_folder"), "checkpoints"),
    )
    file_name = _first_present(
        entry,
        ("file_name", "filename", "file"),
        _first_present(defaults, ("file_name", "filename"), ""),
    )
    overwrite_existing = _as_bool(
        _first_present(entry, ("overwrite_existing", "overwrite"), None),
        _as_bool(_first_present(defaults, ("overwrite_existing", "overwrite"), None), False),
    )
    expected_size = _parse_expected_size(
        _first_present(
            entry,
            ("size", "file_size", "size_bytes", "bytes", "expected_size"),
            _first_present(defaults, ("size", "file_size", "size_bytes", "bytes", "expected_size"), None),
        )
    )
    hashes = _hashes_from_entry(entry)

    name = _infer_model_name(entry, urls, index)
    description = str(_first_present(entry, ("description", "desc", "notes"), ""))

    return {
        "name": name,
        "description": description,
        "urls": urls,
        "download_directory": str(download_directory or "checkpoints"),
        "file_name": str(file_name or ""),
        "overwrite_existing": overwrite_existing,
        "expected_size": expected_size,
        "sha256": hashes["sha256"],
        "sha1": hashes["sha1"],
        "md5": hashes["md5"],
    }


def _model_items_from_config(models_json):
    raw = str(models_json or "").strip()
    if not raw:
        return []

    data = json.loads(raw)
    defaults = {}

    if isinstance(data, list):
        entries = data
    elif isinstance(data, dict):
        defaults = data.get("defaults") if isinstance(data.get("defaults"), dict) else {}
        if isinstance(data.get("models"), list):
            entries = data["models"]
        elif isinstance(data.get("models"), dict):
            entries = [
                {"name": name, **value} if isinstance(value, dict) else {"name": name, "url": value}
                for name, value in data["models"].items()
            ]
        elif any(key in data for key in ("url", "urls", "model_url", "model_urls", "download_url")):
            entries = [data]
        else:
            entries = [
                {"name": name, **value} if isinstance(value, dict) else {"name": name, "url": value}
                for name, value in data.items()
                if name != "defaults"
            ]
    else:
        raise ValueError("JSON config must be an object or array.")

    return [_normalize_model_entry(entry, defaults, index) for index, entry in enumerate(entries, start=1)]


def _normalize_download_directory(download_directory):
    raw = str(download_directory or "checkpoints").strip().strip("\"'")
    if not raw:
        raw = "checkpoints"

    expanded = os.path.expandvars(os.path.expanduser(raw))
    if os.path.isabs(expanded):
        return os.path.abspath(expanded), None

    normalized = expanded.replace("\\", "/").lstrip("/")
    if normalized == "models":
        normalized = ""
    elif normalized.startswith("models/"):
        normalized = normalized[len("models/") :]

    return normalized, normalized


def _get_registered_folder_paths(folder_name):
    names = []
    for name in (folder_name, str(folder_name).lower()):
        if name and name not in names:
            names.append(name)

    map_legacy = getattr(folder_paths, "map_legacy", None)
    if callable(map_legacy):
        for name in list(names):
            mapped = map_legacy(name)
            if mapped and mapped not in names:
                names.append(mapped)

    for name in names:
        try:
            paths = folder_paths.get_folder_paths(name)
        except Exception:
            continue
        if paths:
            return name, [os.path.abspath(path) for path in paths]
    return None, []


def _registered_download_roots(download_directory):
    normalized_or_abs, normalized = _normalize_download_directory(download_directory)
    if normalized is None:
        return None, "", [normalized_or_abs]

    if not normalized:
        return None, "", [os.path.abspath(folder_paths.models_dir)]

    parts = normalized.split("/", 1)
    candidates = [(normalized, ""), (parts[0], parts[1] if len(parts) > 1 else "")]
    for folder_name, subdir in candidates:
        registered_name, roots = _get_registered_folder_paths(folder_name)
        if roots:
            return registered_name, subdir, roots

    return None, normalized, [os.path.abspath(folder_paths.models_dir)]


def _download_dirs_for_directory(download_directory):
    _folder_name, subdir, roots = _registered_download_roots(download_directory)
    return [os.path.abspath(os.path.join(root, subdir)) for root in roots]


def _resolve_download_dir(download_directory):
    normalized_or_abs, normalized = _normalize_download_directory(download_directory)
    if normalized is None:
        return normalized_or_abs
    if not normalized:
        return os.path.abspath(folder_paths.models_dir)
    return os.path.abspath(os.path.join(folder_paths.models_dir, normalized))


def _filename_from_url(url):
    parsed = urllib.parse.urlparse(url)
    filename = urllib.parse.unquote(os.path.basename(parsed.path or ""))
    return filename or "download.bin"


def _filename_from_content_disposition(header):
    if not header:
        return ""

    match = re.search(r"filename\*=UTF-8''([^;]+)", header, flags=re.IGNORECASE)
    if match:
        return urllib.parse.unquote(match.group(1).strip().strip("\"'"))

    match = re.search(r'filename="([^"]+)"', header, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()

    match = re.search(r"filename=([^;]+)", header, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip().strip("\"'")

    return ""


def _safe_filename(filename, fallback="download.bin"):
    name = os.path.basename(str(filename or "").strip())
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name)
    name = name.strip().strip(".")
    return (name or fallback)[:240]


def _format_filename(template, source_filename, index, total):
    source_filename = _safe_filename(source_filename)
    if not template:
        return source_filename

    template = str(template).strip()
    source_stem, source_ext = os.path.splitext(source_filename)
    source_ext_clean = source_ext[1:] if source_ext.startswith(".") else source_ext

    if any(token in template for token in ("{index}", "{name}", "{ext}", "{original}")):
        try:
            rendered = template.format(
                index=index,
                name=source_stem,
                ext=source_ext_clean,
                original=source_filename,
            )
        except Exception:
            rendered = template
    elif total > 1:
        template_stem, template_ext = os.path.splitext(template)
        ext = template_ext or source_ext
        rendered = f"{template_stem}_{index:02d}{ext}"
    else:
        rendered = template
        if not os.path.splitext(rendered)[1] and source_ext:
            rendered += source_ext

    return _safe_filename(rendered, source_filename)


def _validate_url(url, task=None):
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError("Only http and https URLs are supported.")
    if not parsed.netloc:
        raise ValueError("URL is missing a host.")

    task = task or {}
    if _as_bool(task.get("safe_mode"), True):
        host = _normalize_host(parsed.netloc)
        if not _is_allowed_download_host(host):
            allowed_text = ", ".join([*DEFAULT_ALLOWED_HOSTS, *ALWAYS_ALLOWED_HOSTS])
            raise ValueError(
                f"Safe mode blocked host '{host}'. Disable Safe mode only if you trust this URL. "
                f"Allowed hosts: {allowed_text}."
            )


def _validate_task_urls(task):
    for url in task.get("urls", []):
        _validate_url(url, task)


def _normalize_download_url(url):
    parsed = urllib.parse.urlparse(url)
    host = parsed.netloc.lower()
    if host == "civitai.com" or host.endswith(".civitai.com"):
        if "/models/" in parsed.path and "/api/download/" not in parsed.path:
            query = urllib.parse.parse_qs(parsed.query)
            version_ids = query.get("modelVersionId") or query.get("modelversionid")
            if version_ids and version_ids[0]:
                return f"https://civitai.com/api/download/models/{version_ids[0]}"
    return url


def _load_config_tokens():
    global _CONFIG_CACHE
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.ini")
    config_mtime = os.path.getmtime(config_path) if os.path.exists(config_path) else None
    if _CONFIG_CACHE is not None and _CONFIG_CACHE.get("mtime") == config_mtime:
        return _CONFIG_CACHE.get("tokens", {})

    tokens = {}
    if os.path.exists(config_path):
        config = configparser.ConfigParser()
        config.read(config_path, encoding="utf-8")
        tokens["civitai"] = config.get("civitai", "api_key", fallback="").strip()
        tokens["huggingface"] = config.get("huggingface", "token", fallback="").strip()

    _CONFIG_CACHE = {"mtime": config_mtime, "tokens": tokens}
    return tokens


def _is_real_token(token):
    return bool(token and not token.upper().startswith("YOUR_"))


def _request_headers(url):
    headers = {"User-Agent": "ComfyUI-Model-Downloader-plus/modern"}
    tokens = _load_config_tokens()
    host = urllib.parse.urlparse(url).netloc.lower()

    if (host == "civitai.com" or host.endswith(".civitai.com")) and _is_real_token(tokens.get("civitai")):
        headers["Authorization"] = f"Bearer {tokens['civitai']}"
    elif (host == "huggingface.co" or host.endswith(".huggingface.co")) and _is_real_token(
        tokens.get("huggingface")
    ):
        headers["Authorization"] = f"Bearer {tokens['huggingface']}"

    return headers


def _human_size(num_bytes):
    if num_bytes is None:
        return "unknown"
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def _hashers_for_task(task):
    hashers = {}
    if task.get("sha256"):
        hashers["sha256"] = hashlib.sha256()
    if task.get("sha1"):
        hashers["sha1"] = hashlib.sha1()
    if task.get("md5"):
        hashers["md5"] = hashlib.md5()
    return hashers


def _verify_file(path, task, hash_values=None):
    expected_size = task.get("expected_size")
    actual_size = os.path.getsize(path)
    if expected_size is not None and actual_size != expected_size:
        raise RuntimeError(
            f"Size mismatch for {os.path.basename(path)}: expected {_human_size(expected_size)}, got {_human_size(actual_size)}"
        )

    expected_hashes = {
        name: task.get(name)
        for name in ("sha256", "sha1", "md5")
        if task.get(name)
    }
    if not expected_hashes:
        return {}

    if hash_values is None:
        hashers = {name: getattr(hashlib, name)() for name in expected_hashes}
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                for hasher in hashers.values():
                    hasher.update(chunk)
        hash_values = {name: hasher.hexdigest().lower() for name, hasher in hashers.items()}

    for name, expected in expected_hashes.items():
        actual = (hash_values.get(name) or "").lower()
        if actual != expected.lower():
            raise RuntimeError(
                f"{name.upper()} mismatch for {os.path.basename(path)}: expected {expected}, got {actual}"
            )

    return hash_values


def _download_one(task_id, task, url, index, total):
    _validate_url(url, task)
    url = _normalize_download_url(url)

    destination_dir = _resolve_download_dir(task["download_directory"])
    os.makedirs(destination_dir, exist_ok=True)

    timeout = (15, 60)
    headers = _request_headers(url)

    base_progress = ((index - 1) / max(total, 1)) * 100
    _set_task(
        task_id,
        status="downloading",
        message=f"Connecting to file {index}/{total}...",
        progress=base_progress,
        current_url=url,
        file_index=index,
        total_files=total,
    )

    with requests.get(url, stream=True, allow_redirects=True, timeout=timeout, headers=headers) as response:
        response.raise_for_status()

        header_filename = _filename_from_content_disposition(response.headers.get("content-disposition"))
        source_filename = header_filename or _filename_from_url(response.url or url)
        filename = _format_filename(task["file_name"], source_filename, index, total)
        file_path = os.path.join(destination_dir, filename)
        partial_path = file_path + ".partial"

        existing_path = None
        if not task["overwrite_existing"]:
            for candidate_path in _candidate_paths_for_filename(task, filename):
                if os.path.exists(candidate_path):
                    existing_path = candidate_path
                    break

        if existing_path:
            verified_hashes = _verify_file(existing_path, task)
            size = os.path.getsize(existing_path)
            _set_task(
                task_id,
                message=f"Skipped existing file: {filename}",
                current_file=filename,
                bytes_downloaded=size,
                bytes_total=size,
                hashes=verified_hashes,
            )
            return {"status": "skipped", "path": existing_path, "filename": filename, "message": "File already exists"}

        total_size = int(response.headers.get("content-length", 0) or 0)
        downloaded = 0
        last_update = 0
        start_time = time.time()
        hashers = _hashers_for_task(task)

        if os.path.exists(partial_path):
            try:
                os.remove(partial_path)
            except OSError:
                pass

        try:
            with open(partial_path, "wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue

                    handle.write(chunk)
                    downloaded += len(chunk)
                    for hasher in hashers.values():
                        hasher.update(chunk)
                    current_time = time.time()

                    if current_time - last_update >= 0.25:
                        if total_size > 0:
                            file_progress = downloaded / total_size
                            overall = base_progress + (file_progress * (100 / max(total, 1)))
                            size_message = f"{_human_size(downloaded)} / {_human_size(total_size)}"
                        else:
                            overall = base_progress
                            size_message = f"{_human_size(downloaded)} downloaded"

                        elapsed = max(current_time - start_time, 0.001)
                        speed_mbps = downloaded / elapsed / 1024 / 1024
                        _set_task(
                            task_id,
                            status="downloading",
                            message=f"{filename}: {size_message} at {speed_mbps:.2f} MB/s",
                            progress=min(max(overall, 1), 99),
                            current_file=filename,
                            current_url=url,
                            bytes_downloaded=downloaded,
                            bytes_total=total_size,
                            speed_mbps=round(speed_mbps, 2),
                        )
                        last_update = current_time
        except Exception:
            if os.path.exists(partial_path):
                try:
                    os.remove(partial_path)
                except OSError:
                    pass
            raise

        actual_size = os.path.getsize(partial_path)
        if total_size > 0 and actual_size != total_size:
            try:
                os.remove(partial_path)
            except OSError:
                pass
            raise RuntimeError(
                f"Downloaded size mismatch for {filename}: expected {total_size}, got {actual_size}"
            )
        hash_values = {name: hasher.hexdigest().lower() for name, hasher in hashers.items()}

        try:
            _verify_file(partial_path, task, hash_values=hash_values)
        except Exception:
            try:
                os.remove(partial_path)
            except OSError:
                pass
            raise

        if os.path.exists(file_path) and task["overwrite_existing"]:
            os.remove(file_path)

        os.replace(partial_path, file_path)
        _set_task(
            task_id,
            message=f"Installed {filename}",
            current_file=filename,
            bytes_downloaded=actual_size,
            bytes_total=actual_size,
            hashes=hash_values,
        )

        return {"status": "downloaded", "path": file_path, "filename": filename, "message": "Downloaded"}


def _run_task(task):
    task_id = task["task_id"]
    urls = task["urls"]
    total = len(urls)
    downloaded_files = []
    skipped_files = []
    errors = []

    _set_task(
        task_id,
        model_name=task.get("model_name", ""),
        status="queued",
        message="Waiting for downloader worker...",
        progress=0,
        urls=urls,
        download_directory=task["download_directory"],
        file_name=task["file_name"],
        overwrite_existing=task["overwrite_existing"],
        safe_mode=_as_bool(task.get("safe_mode"), True),
        expected_size=task.get("expected_size"),
        sha256=task.get("sha256", ""),
        sha1=task.get("sha1", ""),
        md5=task.get("md5", ""),
        total_files=total,
        downloaded_files=[],
        skipped_files=[],
        errors=[],
    )

    for index, url in enumerate(urls, start=1):
        try:
            result = _download_one(task_id, task, url, index, total)
            if result["status"] == "skipped":
                skipped_files.append(result)
            else:
                downloaded_files.append(result)

            _set_task(
                task_id,
                downloaded_files=downloaded_files,
                skipped_files=skipped_files,
                errors=errors,
                progress=(index / max(total, 1)) * 100,
            )
        except Exception as exc:
            error = {"url": url, "message": str(exc)}
            errors.append(error)
            _set_task(
                task_id,
                status="error",
                message=str(exc),
                errors=errors,
                progress=(index / max(total, 1)) * 100,
            )

    if errors and (downloaded_files or skipped_files):
        status = "partial"
        message = f"Finished with {len(errors)} error(s)."
    elif errors:
        status = "error"
        message = errors[-1]["message"] if errors else "Download failed."
    elif downloaded_files:
        status = "success"
        message = f"Downloaded {len(downloaded_files)} file(s)."
    else:
        status = "skipped"
        message = f"All {len(skipped_files)} file(s) already exist."

    _set_task(
        task_id,
        status=status,
        message=message,
        progress=100,
        downloaded_files=downloaded_files,
        skipped_files=skipped_files,
        errors=errors,
        current_file="",
        current_url="",
    )


def _worker_loop():
    while True:
        task = _DOWNLOAD_QUEUE.get()
        try:
            if task is None:
                return
            _run_task(task)
        finally:
            _DOWNLOAD_QUEUE.task_done()


threading.Thread(target=_worker_loop, daemon=True).start()


def _build_task_from_payload(payload):
    urls = _parse_urls(payload.get("urls") or payload.get("model_urls") or payload.get("model_url"))
    if not urls:
        raise ValueError("No URL provided.")

    task_id = str(payload.get("task_id") or uuid.uuid4())
    task = {
        "task_id": task_id,
        "model_name": payload.get("name") or payload.get("model_name") or "",
        "urls": urls,
        "download_directory": payload.get("download_directory") or payload.get("model_folder") or "checkpoints",
        "file_name": payload.get("file_name") or payload.get("filename") or "",
        "overwrite_existing": _as_bool(payload.get("overwrite_existing"), False),
        "safe_mode": _as_bool(payload.get("safe_mode"), True),
        "expected_size": _parse_expected_size(
            payload.get("expected_size", payload.get("size", payload.get("file_size", None)))
        ),
        "sha256": _clean_hash(payload.get("sha256", "")),
        "sha1": _clean_hash(payload.get("sha1", "")),
        "md5": _clean_hash(payload.get("md5", "")),
    }
    for label, expected, length in (
        ("sha256", task["sha256"], 64),
        ("sha1", task["sha1"], 40),
        ("md5", task["md5"], 32),
    ):
        if expected and len(expected) != length:
            raise ValueError(f"Invalid {label} hash length.")
    return task


def _candidate_paths_for_filename(task, filename):
    preferred = os.path.join(_resolve_download_dir(task["download_directory"]), filename)
    paths = [preferred]
    for directory in _download_dirs_for_directory(task["download_directory"]):
        candidate = os.path.join(directory, filename)
        if os.path.abspath(candidate) not in {os.path.abspath(path) for path in paths}:
            paths.append(candidate)
    return paths


def _candidate_paths_for_task(task):
    urls = task["urls"]
    total = len(urls)
    paths = []

    for index, url in enumerate(urls, start=1):
        normalized_url = _normalize_download_url(url)
        source_filename = _filename_from_url(normalized_url or url)
        filename = _format_filename(task.get("file_name", ""), source_filename, index, total)
        candidates = _candidate_paths_for_filename(task, filename)
        paths.append(
            {
                "url": url,
                "filename": filename,
                "path": candidates[0],
                "candidate_paths": candidates,
                "index": index,
            }
        )

    return paths


def _check_task_files(task):
    files = []
    installed_count = 0
    missing_count = 0
    error_count = 0

    for candidate in _candidate_paths_for_task(task):
        info = dict(candidate)
        path = next((candidate_path for candidate_path in candidate["candidate_paths"] if os.path.exists(candidate_path)), None)
        if not path:
            missing_count += 1
            info.update({"status": "missing", "message": "File not found"})
            files.append(info)
            continue

        try:
            hashes = _verify_file(path, task)
            installed_count += 1
            info.update(
                {
                    "status": "installed",
                    "message": "Installed",
                    "path": path,
                    "size": os.path.getsize(path),
                    "hashes": hashes,
                }
            )
        except Exception as exc:
            error_count += 1
            info.update(
                {
                    "status": "error",
                    "message": str(exc),
                    "path": path,
                    "size": os.path.getsize(path) if os.path.exists(path) else 0,
                }
            )
        files.append(info)

    total = len(files)
    if error_count:
        status = "error" if error_count == total else "partial"
        message = f"{error_count}/{total} file(s) failed verification."
    elif missing_count:
        status = "missing" if missing_count == total else "partial"
        message = f"{installed_count}/{total} file(s) installed."
    else:
        status = "installed"
        message = f"Installed: {installed_count}/{total} file(s)."

    return {
        "status": status,
        "message": message,
        "progress": 100 if status == "installed" else 0,
        "model_name": task.get("model_name", ""),
        "files": files,
        "installed_count": installed_count,
        "missing_count": missing_count,
        "error_count": error_count,
        "total_files": total,
    }


@server.PromptServer.instance.routes.post("/model_downloader_plus/general/check")
async def general_check(request):
    try:
        payload = await request.json()
        task = _build_task_from_payload(payload)
        result = _check_task_files(task)
    except json.JSONDecodeError:
        return web.json_response({"error": "Invalid JSON body"}, status=400)
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=400)

    return web.json_response(result, headers={"Cache-Control": "no-store"})


@server.PromptServer.instance.routes.post("/model_downloader_plus/general/download")
async def general_download(request):
    _cleanup_tasks()
    try:
        payload = await request.json()
        task = _build_task_from_payload(payload)
        _validate_task_urls(task)
    except json.JSONDecodeError:
        return web.json_response({"error": "Invalid JSON body"}, status=400)
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=400)

    _set_task(
        task["task_id"],
        task_id=task["task_id"],
        model_name=task.get("model_name", ""),
        status="queued",
        message="Queued in backend...",
        progress=0,
        urls=task["urls"],
        download_directory=task["download_directory"],
        file_name=task["file_name"],
        overwrite_existing=task["overwrite_existing"],
        safe_mode=task.get("safe_mode", True),
        expected_size=task.get("expected_size"),
        sha256=task.get("sha256", ""),
        sha1=task.get("sha1", ""),
        md5=task.get("md5", ""),
        total_files=len(task["urls"]),
        downloaded_files=[],
        skipped_files=[],
        errors=[],
    )
    _DOWNLOAD_QUEUE.put(task)
    return web.json_response({"status": "queued", "task_id": task["task_id"]})


@server.PromptServer.instance.routes.get("/model_downloader_plus/general/status/{task_id}")
async def general_status(request):
    _cleanup_tasks()
    task_id = request.match_info.get("task_id")
    task = _get_task(task_id)
    if not task:
        return web.json_response(
            {"task_id": task_id, "status": "unknown", "message": "No task found.", "progress": 0},
            headers={"Cache-Control": "no-store"},
        )
    return web.json_response(task, headers={"Cache-Control": "no-store"})


class GeneralModelDownloader:
    NAME = "GeneralModelDownloader"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "models_json": (
                    "STRING",
                    {
                        "default": DEFAULT_MODELS_JSON,
                        "multiline": True,
                        "tooltip": "JSON model list. Use a top-level array for multiple models; legacy {\"models\": [...]} is still supported.",
                    },
                ),
                "config_locked": ("BOOLEAN", {"default": False}),
                "download_on_execute": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "When false, use the node panel buttons to download without running the workflow.",
                    },
                ),
                "safe_mode": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "When enabled, downloads are limited to built-in trusted hosts plus Civitai.",
                    },
                ),
            },
            "optional": {
                "anything": (ANY, {}),
            },
        }

    RETURN_TYPES = (ANY, "STRING", "STRING")
    RETURN_NAMES = ("output", "downloaded_files", "status_report")
    FUNCTION = "run"
    CATEGORY = "utils/download"

    def run(
        self,
        models_json,
        config_locked=False,
        download_on_execute=False,
        safe_mode=True,
        anything=None,
    ):
        try:
            models = _model_items_from_config(models_json)
        except Exception as exc:
            return (anything, "", f"Invalid JSON model list: {exc}")

        if not models:
            return (anything, "", "No models in JSON model list.")

        if not download_on_execute:
            return (
                anything,
                "",
                f"Ready: {len(models)} model(s). Use the node panel buttons to download.",
            )

        downloaded_all = []
        skipped_all = []
        errors_all = []
        report_lines = []

        for model in models:
            task = {
                "task_id": f"execute-{uuid.uuid4()}",
                "model_name": model["name"],
                "urls": model["urls"],
                "download_directory": model["download_directory"],
                "file_name": model["file_name"],
                "overwrite_existing": bool(model["overwrite_existing"]),
                "safe_mode": _as_bool(safe_mode, True),
                "expected_size": model.get("expected_size"),
                "sha256": model.get("sha256", ""),
                "sha1": model.get("sha1", ""),
                "md5": model.get("md5", ""),
            }
            _run_task(task)
            final_state = _get_task(task["task_id"]) or {}
            downloaded = final_state.get("downloaded_files", [])
            skipped = final_state.get("skipped_files", [])
            errors = final_state.get("errors", [])

            downloaded_all.extend(downloaded)
            skipped_all.extend(skipped)
            errors_all.extend(errors)
            report_lines.append(f"{model['name']}: {final_state.get('message', 'Done.')}")

        paths = [item.get("path", "") for item in downloaded_all + skipped_all if item.get("path")]
        report_lines.extend(item.get("path", "") for item in downloaded_all if item.get("path"))
        report_lines.extend(f"Skipped: {item.get('path', '')}" for item in skipped_all if item.get("path"))
        report_lines.extend(f"Error: {item.get('message', '')}" for item in errors_all)

        return (anything, "\n".join(paths), "\n".join(report_lines))
