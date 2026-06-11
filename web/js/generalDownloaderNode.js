import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const ACTIVE_STATES = new Set(["queued", "downloading"]);
const DONE_STATES = new Set(["success", "error", "partial", "skipped", "installed", "missing"]);
const INTERNAL_WIDGET_NAMES = ["models_json", "config_locked", "download_on_execute", "safe_mode"];
const DEFAULT_ALLOWED_HOSTS = ["huggingface.co", "hf-mirror.com", "modelscope.cn", "github.com", "githubusercontent.com"];
const ALWAYS_ALLOWED_HOSTS = ["civitai.com"];
const DEFAULT_NODE_WIDTH = 520;
const DEFAULT_NODE_HEIGHT = 500;
const MIN_NODE_WIDTH = 460;

function getWidget(node, name) {
    return node?.widgets?.find((widget) => widget.name === name);
}

function setWidgetValue(node, name, value) {
    const widget = getWidget(node, name);
    if (!widget) return;
    widget.value = value;
}

function getWidgetValue(node, name, fallback = "") {
    const widget = getWidget(node, name);
    return widget ? widget.value : fallback;
}

function hideWidget(widget) {
    if (!widget) return;
    if (!widget._gmdOriginal) {
        widget._gmdOriginal = {
            type: widget.type,
            computeSize: widget.computeSize,
            serializeValue: widget.serializeValue,
            draw: widget.draw,
        };
    }
    widget.hidden = true;
    widget.computeSize = () => [0, -4];
    widget.serializeValue = () => widget.value;
    widget.type = "converted-widget:gmd";
    widget.draw = () => {};
    widget._gmdHidden = true;
    for (const child of widget.linkedWidgets || []) {
        hideWidget(child);
    }
    for (const key of ["element", "inputEl", "textElement", "textarea", "domElement"]) {
        if (widget[key]?.style) {
            widget[key].style.display = "none";
        }
    }
}

function hideInternalWidgets(node) {
    INTERNAL_WIDGET_NAMES.forEach((name) => hideWidget(getWidget(node, name)));
}

function scheduleHideInternalWidgets(node) {
    hideInternalWidgets(node);
    if (typeof requestAnimationFrame === "function") {
        requestAnimationFrame(() => hideInternalWidgets(node));
    }
    setTimeout(() => hideInternalWidgets(node), 100);
    setTimeout(() => hideInternalWidgets(node), 500);
}

function parseUrls(raw) {
    if (Array.isArray(raw)) {
        return raw.map((url) => String(url || "").trim()).filter(Boolean);
    }
    return String(raw || "")
        .split(/\r?\n/)
        .map((line) => line.trim())
        .filter((line) => line && !line.startsWith("#"));
}

function firstPresent(source, names, fallback = "") {
    if (!source || typeof source !== "object") return fallback;
    for (const name of names) {
        if (source[name] !== undefined && source[name] !== null && source[name] !== "") {
            return source[name];
        }
    }
    return fallback;
}

function asBool(value, fallback = false) {
    if (value === undefined || value === null || value === "") return fallback;
    if (typeof value === "boolean") return value;
    if (typeof value === "number") return Boolean(value);
    if (typeof value === "string") {
        const normalized = value.trim().toLowerCase();
        if (["true", "1", "yes", "y", "on"].includes(normalized)) return true;
        if (["false", "0", "no", "n", "off"].includes(normalized)) return false;
    }
    return fallback;
}

function parseExpectedSize(value) {
    if (value === undefined || value === null || value === "") return null;
    if (typeof value === "boolean") {
        throw new Error("Invalid size value: use an exact byte count, for example 123456789.");
    }
    if (typeof value === "number") {
        if (value >= 0 && Number.isInteger(value)) return value;
        throw new Error("Invalid size value: use an exact byte count, for example 123456789.");
    }

    const text = String(value).trim();
    if (!text) return null;
    if (!/^\d+$/.test(text)) throw new Error("Invalid size value: use an exact byte count, for example 123456789.");
    return Number(text);
}

function cleanHash(value) {
    if (value === undefined || value === null || value === "") return "";
    return String(value).replace(/[^a-fA-F0-9]/g, "").toLowerCase();
}

function hashesFromEntry(entry) {
    const hashData = entry.hashes && typeof entry.hashes === "object" ? entry.hashes : {};
    let sha256 = cleanHash(firstPresent(entry, ["sha256", "hash_sha256"], firstPresent(hashData, ["sha256"], "")));
    let sha1 = cleanHash(firstPresent(entry, ["sha1", "hash_sha1"], firstPresent(hashData, ["sha1"], "")));
    let md5 = cleanHash(firstPresent(entry, ["md5", "hash_md5"], firstPresent(hashData, ["md5"], "")));
    const generic = cleanHash(firstPresent(entry, ["hash", "checksum"], ""));

    if (generic) {
        if (generic.length === 64 && !sha256) sha256 = generic;
        if (generic.length === 40 && !sha1) sha1 = generic;
        if (generic.length === 32 && !md5) md5 = generic;
    }

    const checks = [
        ["sha256", sha256, 64],
        ["sha1", sha1, 40],
        ["md5", md5, 32],
    ];
    for (const [name, value, length] of checks) {
        if (value && value.length !== length) throw new Error(`Invalid ${name} hash length.`);
    }

    return { sha256, sha1, md5 };
}

function fileNameFromUrl(url) {
    try {
        const parsed = new URL(url);
        const parts = decodeURIComponent(parsed.pathname || "").split("/");
        return parts.filter(Boolean).pop() || "download.bin";
    } catch (_error) {
        const parts = String(url || "").split("?")[0].split("/");
        return parts.filter(Boolean).pop() || "download.bin";
    }
}

function modelKey(model, index) {
    return `${index}:${model.name}:${model.download_directory}:${model.urls.join("|")}`;
}

function formatPercent(value) {
    const percent = Math.max(0, Math.min(100, Number(value) || 0));
    return `${Math.round(percent)}%`;
}

function formatBytes(value) {
    const bytes = Number(value);
    if (!Number.isFinite(bytes) || bytes < 0) return "";

    const units = ["B", "KB", "MB", "GB", "TB"];
    let size = bytes;
    for (const unit of units) {
        if (size < 1024 || unit === units[units.length - 1]) {
            if (unit === "B") return `${Math.round(size)} ${unit}`;
            return `${size.toFixed(1)} ${unit}`;
        }
        size /= 1024;
    }
    return `${size.toFixed(1)} TB`;
}

function normalizeModelEntry(entry, defaults, index) {
    if (typeof entry === "string") {
        entry = { url: entry };
    }
    if (!entry || typeof entry !== "object") {
        throw new Error(`Model ${index} is not an object or URL string.`);
    }

    const urls = parseUrls(firstPresent(entry, ["urls", "model_urls", "url", "model_url", "download_url"], ""));
    if (!urls.length) {
        throw new Error(`Model ${index} has no URL.`);
    }

    const downloadDirectory = firstPresent(
        entry,
        ["download_directory", "directory", "folder", "model_folder", "target_dir", "save_to"],
        firstPresent(defaults, ["download_directory", "directory", "folder", "model_folder"], "checkpoints")
    );
    const fileName = firstPresent(
        entry,
        ["file_name", "filename", "file"],
        firstPresent(defaults, ["file_name", "filename"], "")
    );
    const overwriteExisting = asBool(
        firstPresent(entry, ["overwrite_existing", "overwrite"], null),
        asBool(firstPresent(defaults, ["overwrite_existing", "overwrite"], null), false)
    );
    const expectedSize = parseExpectedSize(
        firstPresent(
            entry,
            ["size", "file_size", "size_bytes", "bytes", "expected_size"],
            firstPresent(defaults, ["size", "file_size", "size_bytes", "bytes", "expected_size"], null)
        )
    );
    const hashes = hashesFromEntry(entry);

    const name =
        firstPresent(entry, ["name", "title", "label", "id"], "") ||
        fileName ||
        fileNameFromUrl(urls[0]) ||
        `Model ${index}`;

    return {
        name: String(name),
        description: String(firstPresent(entry, ["description", "desc", "notes"], "")),
        urls,
        download_directory: String(downloadDirectory || "checkpoints"),
        file_name: String(fileName || ""),
        overwrite_existing: overwriteExisting,
        expected_size: expectedSize,
        sha256: hashes.sha256,
        sha1: hashes.sha1,
        md5: hashes.md5,
        key_hint: String(firstPresent(entry, ["key", "id"], "")),
    };
}

function parseModelsJson(rawJson) {
    const raw = String(rawJson || "").trim();
    if (!raw) return { models: [], error: "" };

    let data;
    try {
        data = JSON.parse(raw);
    } catch (error) {
        return { models: [], error: error.message || String(error) };
    }

    const defaults = data && !Array.isArray(data) && typeof data === "object" && data.defaults
        ? data.defaults
        : {};

    let entries;
    if (Array.isArray(data)) {
        entries = data;
    } else if (data && typeof data === "object") {
        if (Array.isArray(data.models)) {
            entries = data.models;
        } else if (data.models && typeof data.models === "object") {
            entries = Object.entries(data.models).map(([name, value]) =>
                value && typeof value === "object" ? { name, ...value } : { name, url: value }
            );
        } else if (["url", "urls", "model_url", "model_urls", "download_url"].some((field) => field in data)) {
            entries = [data];
        } else {
            entries = Object.entries(data)
                .filter(([name]) => name !== "defaults")
                .map(([name, value]) =>
                    value && typeof value === "object" ? { name, ...value } : { name, url: value }
                );
        }
    } else {
        return { models: [], error: "JSON must be an object or array." };
    }

    try {
        const models = entries.map((entry, index) => normalizeModelEntry(entry, defaults, index + 1));
        return { models, error: "" };
    } catch (error) {
        return { models: [], error: error.message || String(error) };
    }
}

function defaultConfigText() {
    return JSON.stringify(
        [
            {
                "name": "Qwen Image VAE",
                "url": "https://modelscope.cn/models/Comfy-Org/Qwen-Image_ComfyUI/resolve/master/split_files/vae/qwen_image_vae.safetensors",
                "download_directory": "vae",
                "file_name": "qwen_image_vae.safetensors",
                "overwrite_existing": false,
                "size": "",
                "sha256": "a70580f0213e67967ee9c95f05bb400e8fb08307e017a924bf3441223e023d1f",
                "description": ""
            },
        ],
        null,
        2
    );
}

class GeneralDownloaderPanel {
    constructor(node, container) {
        this.node = node;
        this.container = container;
        this.modelStatuses = new Map();
        this.modelTasks = new Map();
        this.pollTimers = new Map();
        this.precheckSignature = "";
        this.precheckTimer = null;

        this.build();
        this.render();
    }

    ensureInitialNodeSize() {
        const width = Math.max(Number(this.node?.size?.[0]) || DEFAULT_NODE_WIDTH, MIN_NODE_WIDTH);
        const height = Math.max(Number(this.node?.size?.[1]) || DEFAULT_NODE_HEIGHT, DEFAULT_NODE_HEIGHT);
        if (this.node?.setSize && (this.node.size?.[0] !== width || this.node.size?.[1] !== height)) {
            this.node.setSize([width, height]);
            this.node.setDirtyCanvas?.(true, true);
            app.canvas?.setDirty?.(true, true);
        }
    }

    applyLayout() {
        this.container.style.height = "100%";
        this.container.style.minHeight = "0";
        this.container.style.maxHeight = "none";
        if (this.panel) {
            this.panel.style.height = "100%";
            this.panel.style.minHeight = "0";
            this.panel.style.maxHeight = "none";
        }
    }

    build() {
        this.container.innerHTML = "";
        this.container.style.width = "100%";
        this.container.style.height = "100%";
        this.container.style.minHeight = "0";
        this.container.style.boxSizing = "border-box";
        this.container.style.padding = "0";
        this.container.style.overflow = "hidden";
        this.container.style.fontFamily = "Inter, system-ui, -apple-system, BlinkMacSystemFont, sans-serif";

        const style = document.createElement("style");
        style.textContent = `
            .gmd-panel {
                display: flex;
                flex-direction: column;
                height: 100%;
                min-height: 0;
                background: #222;
                border: 1px solid #3b3b3b;
                border-radius: 8px;
                overflow: hidden;
                color: #ddd;
                box-sizing: border-box;
            }
            .gmd-config {
                border-bottom: 1px solid #343434;
                background: #1f1f1f;
                padding: 8px;
                display: flex;
                flex-direction: column;
                gap: 7px;
                flex: 0 0 auto;
                min-width: 0;
            }
            .gmd-config.locked {
                flex-direction: row;
                align-items: center;
                justify-content: space-between;
                gap: 8px;
            }
            .gmd-config-head {
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 8px;
                min-width: 0;
            }
            .gmd-title {
                color: #f0f0f0;
                font-size: 12px;
                font-weight: 700;
                min-width: 0;
                overflow: hidden;
                text-overflow: ellipsis;
                white-space: nowrap;
            }
            .gmd-summary {
                color: #aaa;
                font-size: 11px;
                min-width: 0;
                overflow: hidden;
                text-overflow: ellipsis;
                white-space: nowrap;
                text-align: right;
            }
            .gmd-hints {
                display: flex;
                flex-wrap: wrap;
                gap: 5px;
            }
            .gmd-hint {
                border: 1px solid #454545;
                border-radius: 999px;
                color: #b8b8b8;
                background: #252525;
                padding: 2px 7px;
                font-size: 10px;
                font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
            }
            .gmd-config-actions {
                display: flex;
                flex-wrap: wrap;
                justify-content: flex-end;
                gap: 6px;
                flex: 0 0 auto;
            }
            .gmd-json {
                width: 100%;
                height: 142px;
                box-sizing: border-box;
                resize: none;
                min-height: 88px;
                max-height: 220px;
                overflow: auto;
                background: #151515;
                color: #e8e8e8;
                border: 1px solid #454545;
                border-radius: 6px;
                padding: 8px;
                outline: none;
                font: 11px/1.35 ui-monospace, SFMono-Regular, Consolas, monospace;
            }
            .gmd-json:focus {
                border-color: #596bc4;
            }
            .gmd-list-head {
                padding: 9px 10px;
                border-bottom: 1px solid #333;
                background: #202020;
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 8px;
                flex: 0 0 auto;
            }
            .gmd-list-title {
                font-size: 12px;
                font-weight: 700;
                color: #eee;
                overflow: hidden;
                white-space: nowrap;
                text-overflow: ellipsis;
            }
            .gmd-list-actions {
                display: flex;
                align-items: center;
                justify-content: flex-end;
                gap: 8px;
                flex: 0 0 auto;
                min-width: 0;
            }
            .gmd-safe-mode {
                border: 1px solid #454545;
                border-radius: 999px;
                background: #252525;
                color: #ddd;
                display: flex;
                align-items: center;
                gap: 5px;
                max-width: 210px;
                min-height: 24px;
                padding: 2px 8px;
                font-size: 10px;
                box-sizing: border-box;
                cursor: pointer;
            }
            .gmd-safe-mode input {
                width: 12px;
                height: 12px;
                margin: 0;
                flex: 0 0 auto;
            }
            .gmd-safe-title {
                font-weight: 700;
                white-space: nowrap;
            }
            .gmd-safe-hint {
                color: #a8a8a8;
                overflow: hidden;
                text-overflow: ellipsis;
                white-space: nowrap;
                min-width: 0;
            }
            .gmd-list {
                flex: 1;
                min-height: 0;
                max-height: none;
                overflow-y: auto;
                padding: 8px;
                display: flex;
                flex-direction: column;
                gap: 8px;
                background: #222;
            }
            .gmd-empty, .gmd-error {
                color: #8f8f8f;
                text-align: center;
                padding: 18px 12px;
                font-size: 12px;
                line-height: 1.4;
            }
            .gmd-error {
                color: #ff9c9c;
            }
            .gmd-card {
                position: relative;
                flex: 0 0 auto;
                overflow: hidden;
                border: 1px solid #444;
                border-radius: 8px;
                background: #313131;
            }
            .gmd-card.success {
                border-color: #4f8c5d;
                background: #253027;
            }
            .gmd-card.installed, .gmd-card.skipped {
                border-color: #4f8c5d;
                background: #253027;
            }
            .gmd-card.missing {
                border-color: #666;
                background: #303030;
            }
            .gmd-card.checking {
                border-color: #596bc4;
                background: #292d42;
            }
            .gmd-card.error, .gmd-card.partial {
                border-color: #9b5656;
                background: #362727;
            }
            .gmd-card.queued {
                border-color: #8a7b3c;
                background: #33301f;
            }
            .gmd-card.downloading {
                border-color: #596bc4;
                background: #292d42;
            }
            .gmd-progress-bg {
                position: absolute;
                inset: 0 auto 0 0;
                width: 0%;
                background: rgba(78, 120, 85, 0.45);
                transition: width 0.2s linear;
                pointer-events: none;
            }
            .gmd-card-body {
                position: relative;
                z-index: 1;
                padding: 9px;
                display: grid;
                grid-template-columns: minmax(0, 1fr) auto;
                grid-template-areas:
                    "top controls"
                    "desc controls"
                    "meta controls"
                    "status controls";
                column-gap: 10px;
                row-gap: 4px;
                align-items: center;
            }
            .gmd-card-top {
                grid-area: top;
                display: flex;
                align-items: flex-start;
                justify-content: space-between;
                gap: 8px;
                min-width: 0;
            }
            .gmd-model-name {
                color: #f2f2f2;
                font-size: 13px;
                font-weight: 700;
                line-height: 1.3;
                min-width: 0;
                word-break: break-word;
            }
            .gmd-pill {
                border: 1px solid #555;
                border-radius: 999px;
                color: #bbb;
                background: #2c2c2c;
                padding: 2px 7px;
                font-size: 10px;
                flex: 0 0 auto;
            }
            .gmd-pill.downloading, .gmd-pill.queued {
                border-color: #6573c5;
                color: #dbe0ff;
                background: #30344f;
            }
            .gmd-pill.success, .gmd-pill.skipped {
                border-color: #4f8c5d;
                color: #cdf2d4;
                background: #213624;
            }
            .gmd-pill.installed {
                border-color: #4f8c5d;
                color: #cdf2d4;
                background: #213624;
            }
            .gmd-pill.missing {
                border-color: #666;
                color: #d0d0d0;
                background: #303030;
            }
            .gmd-pill.checking {
                border-color: #6573c5;
                color: #dbe0ff;
                background: #30344f;
            }
            .gmd-pill.error, .gmd-pill.partial {
                border-color: #a85d5d;
                color: #ffd8d8;
                background: #402828;
            }
            .gmd-desc {
                grid-area: desc;
                color: #b9b9b9;
                font-size: 10px;
                line-height: 1.35;
                overflow: hidden;
                display: -webkit-box;
                -webkit-line-clamp: 2;
                -webkit-box-orient: vertical;
            }
            .gmd-meta {
                grid-area: meta;
                color: #9d9d9d;
                font-size: 10px;
                font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
                overflow: hidden;
                text-overflow: ellipsis;
                white-space: nowrap;
            }
            .gmd-status-line {
                grid-area: status;
                color: #cfcfcf;
                font-size: 10px;
                min-height: 14px;
                overflow: hidden;
                text-overflow: ellipsis;
                white-space: nowrap;
            }
            .gmd-card-controls {
                grid-area: controls;
                display: flex;
                align-items: center;
                justify-content: flex-end;
                gap: 8px;
                align-self: end;
            }
            .gmd-btn {
                border: 1px solid #555;
                border-radius: 6px;
                min-height: 24px;
                padding: 4px 9px;
                background: #333;
                color: #eee;
                cursor: pointer;
                font-size: 11px;
                font-weight: 700;
                line-height: 1.2;
                white-space: nowrap;
                box-sizing: border-box;
                flex: 0 0 auto;
                max-width: 100%;
            }
            .gmd-btn.primary {
                background: #4557a6;
                border-color: #596bc4;
                color: white;
            }
            .gmd-btn.success {
                background: #357147;
                border-color: #4f8c5d;
            }
            .gmd-btn.warning {
                background: #6b5b24;
                border-color: #8a7b3c;
            }
            .gmd-btn:hover:not(:disabled) {
                filter: brightness(1.12);
            }
            .gmd-btn:disabled {
                opacity: 0.55;
                cursor: default;
            }
            .gmd-footer {
                padding: 8px 10px;
                border-top: 1px solid #333;
                background: #1f1f1f;
                flex: 0 0 auto;
            }
            .gmd-footer .gmd-btn {
                width: 100%;
            }
        `;
        this.container.appendChild(style);

        this.panel = document.createElement("div");
        this.panel.className = "gmd-panel";

        this.configSection = document.createElement("div");
        this.configSection.className = "gmd-config";

        this.configHead = document.createElement("div");
        this.configHead.className = "gmd-config-head";

        this.configTitle = document.createElement("div");
        this.configTitle.className = "gmd-title";
        this.configTitle.textContent = "JSON model list";

        this.configSummary = document.createElement("div");
        this.configSummary.className = "gmd-summary";

        this.hints = document.createElement("div");
        this.hints.className = "gmd-hints";
        ["array []", "name", "url", "download_directory", "file_name", "size?/hash?"].forEach((label) => {
            const hint = document.createElement("span");
            hint.className = "gmd-hint";
            hint.textContent = label;
            this.hints.appendChild(hint);
        });

        this.configActions = document.createElement("div");
        this.configActions.className = "gmd-config-actions";

        this.formatBtn = document.createElement("button");
        this.formatBtn.className = "gmd-btn";
        this.formatBtn.type = "button";
        this.formatBtn.textContent = "Format";
        this.formatBtn.onclick = () => this.formatJson();

        this.exampleBtn = document.createElement("button");
        this.exampleBtn.className = "gmd-btn";
        this.exampleBtn.type = "button";
        this.exampleBtn.textContent = "Example";
        this.exampleBtn.onclick = () => this.insertExample();

        this.lockBtn = document.createElement("button");
        this.lockBtn.className = "gmd-btn primary";
        this.lockBtn.type = "button";
        this.lockBtn.onclick = () => this.toggleLock();

        this.configActions.appendChild(this.exampleBtn);
        this.configActions.appendChild(this.formatBtn);
        this.configActions.appendChild(this.lockBtn);
        this.configHead.appendChild(this.configTitle);
        this.configHead.appendChild(this.configActions);

        this.jsonInput = document.createElement("textarea");
        this.jsonInput.className = "gmd-json";
        this.jsonInput.spellcheck = false;
        this.jsonInput.value = this.getJsonText();
        this.jsonInput.oninput = () => {
            setWidgetValue(this.node, "models_json", this.jsonInput.value);
            this.render();
        };

        this.configSection.appendChild(this.configHead);
        this.configSection.appendChild(this.configSummary);
        this.configSection.appendChild(this.hints);
        this.configSection.appendChild(this.jsonInput);

        this.listHead = document.createElement("div");
        this.listHead.className = "gmd-list-head";
        this.listTitle = document.createElement("div");
        this.listTitle.className = "gmd-list-title";

        this.listActions = document.createElement("div");
        this.listActions.className = "gmd-list-actions";

        this.safeModeLabel = document.createElement("label");
        this.safeModeLabel.className = "gmd-safe-mode";
        this.safeModeLabel.title = `Safe mode blocks downloads outside built-in trusted hosts plus ${ALWAYS_ALLOWED_HOSTS.join(", ")}.`;

        this.safeModeInput = document.createElement("input");
        this.safeModeInput.type = "checkbox";
        this.safeModeInput.onchange = () => {
            this.setSafeMode(this.safeModeInput.checked);
            this.render();
        };

        this.safeModeTitle = document.createElement("span");
        this.safeModeTitle.className = "gmd-safe-title";
        this.safeModeTitle.textContent = "Safe";

        this.safeModeHint = document.createElement("span");
        this.safeModeHint.className = "gmd-safe-hint";
        this.safeModeHint.textContent = "trusted hosts";

        this.safeModeLabel.appendChild(this.safeModeInput);
        this.safeModeLabel.appendChild(this.safeModeTitle);
        this.safeModeLabel.appendChild(this.safeModeHint);

        this.checkBtn = document.createElement("button");
        this.checkBtn.className = "gmd-btn";
        this.checkBtn.type = "button";
        this.checkBtn.textContent = "Check";
        this.checkBtn.onclick = () => this.checkAll(true);
        this.listHead.appendChild(this.listTitle);
        this.listActions.appendChild(this.safeModeLabel);
        this.listActions.appendChild(this.checkBtn);
        this.listHead.appendChild(this.listActions);

        this.listArea = document.createElement("div");
        this.listArea.className = "gmd-list";

        this.footer = document.createElement("div");
        this.footer.className = "gmd-footer";
        this.downloadAllBtn = document.createElement("button");
        this.downloadAllBtn.className = "gmd-btn primary";
        this.downloadAllBtn.type = "button";
        this.downloadAllBtn.textContent = "Download All";
        this.downloadAllBtn.onclick = () => this.downloadAll();
        this.footer.appendChild(this.downloadAllBtn);

        this.panel.appendChild(this.configSection);
        this.panel.appendChild(this.listHead);
        this.panel.appendChild(this.listArea);
        this.panel.appendChild(this.footer);
        this.container.appendChild(this.panel);
    }

    getJsonText() {
        let value = String(getWidgetValue(this.node, "models_json", "") || "");
        if (!value.trim()) {
            value = defaultConfigText();
            setWidgetValue(this.node, "models_json", value);
        }
        return value;
    }

    isLocked() {
        return asBool(getWidgetValue(this.node, "config_locked", false), false);
    }

    setLocked(value) {
        setWidgetValue(this.node, "config_locked", Boolean(value));
    }

    isSafeMode() {
        return asBool(getWidgetValue(this.node, "safe_mode", true), true);
    }

    setSafeMode(value) {
        setWidgetValue(this.node, "safe_mode", Boolean(value));
    }

    getParsed() {
        return parseModelsJson(this.getJsonText());
    }

    formatJson() {
        const raw = this.getJsonText();
        try {
            const formatted = JSON.stringify(JSON.parse(raw), null, 2);
            this.jsonInput.value = formatted;
            setWidgetValue(this.node, "models_json", formatted);
            this.render();
        } catch (error) {
            this.flashConfigError(error.message || String(error));
        }
    }

    insertExample() {
        const value = defaultConfigText();
        this.jsonInput.value = value;
        setWidgetValue(this.node, "models_json", value);
        this.render();
        app.canvas?.setDirty?.(true, true);
    }

    flashConfigError(message) {
        this.configSummary.textContent = message;
        this.configSummary.style.color = "#ff9c9c";
    }

    toggleLock() {
        if (!this.isLocked()) {
            const parsed = this.getParsed();
            if (parsed.error) {
                this.flashConfigError(parsed.error);
                return;
            }
            this.setLocked(true);
        } else {
            this.setLocked(false);
        }
        this.render();
        app.canvas?.setDirty?.(true, true);
    }

    payloadForModel(model) {
        return {
            name: model.name,
            urls: model.urls,
            download_directory: model.download_directory,
            file_name: model.file_name,
            overwrite_existing: model.overwrite_existing,
            safe_mode: this.isSafeMode(),
            expected_size: model.expected_size,
            sha256: model.sha256,
            sha1: model.sha1,
            md5: model.md5,
        };
    }

    async checkModel(model, key, showChecking = false) {
        const current = this.modelStatuses.get(key);
        if (current && ACTIVE_STATES.has(current.status)) return;

        if (showChecking) {
            this.modelStatuses.set(key, {
                status: "checking",
                message: "Checking local file...",
                progress: 0,
            });
            this.render();
        }

        try {
            const response = await api.fetchApi("/model_downloader_plus/general/check", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(this.payloadForModel(model)),
            });
            const data = await response.json().catch(() => ({}));
            if (!response.ok) {
                throw new Error(data.error || `HTTP ${response.status}`);
            }
            this.modelStatuses.set(key, data);
            this.render();
        } catch (error) {
            this.modelStatuses.set(key, {
                status: "error",
                message: error.message || String(error),
                progress: 100,
            });
            this.render();
        }
    }

    schedulePrecheck(models) {
        const signature = JSON.stringify(models.map((model, index) => ({
            key: model.key_hint || modelKey(model, index),
            urls: model.urls,
            download_directory: model.download_directory,
            file_name: model.file_name,
            safe_mode: this.isSafeMode(),
            expected_size: model.expected_size,
            sha256: model.sha256,
            sha1: model.sha1,
            md5: model.md5,
        })));
        if (signature === this.precheckSignature) return;

        this.precheckSignature = signature;
        if (this.precheckTimer) clearTimeout(this.precheckTimer);
        this.precheckTimer = setTimeout(() => this.checkAll(false), 150);
    }

    async checkAll(showChecking = false) {
        const parsed = this.getParsed();
        if (parsed.error || !parsed.models.length) return;
        await Promise.all(parsed.models.map((model, index) => {
            const key = model.key_hint || modelKey(model, index);
            return this.checkModel(model, key, showChecking);
        }));
    }

    async startDownload(model, key) {
        const payload = this.payloadForModel(model);

        this.modelStatuses.set(key, {
            status: "queued",
            message: "Sending download request...",
            progress: 1,
        });
        this.render();

        try {
            const response = await api.fetchApi("/model_downloader_plus/general/download", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });
            const data = await response.json().catch(() => ({}));
            if (!response.ok) {
                throw new Error(data.error || `HTTP ${response.status}`);
            }

            this.modelTasks.set(key, data.task_id);
            this.modelStatuses.set(key, {
                status: data.status || "queued",
                message: "Queued in backend...",
                progress: 1,
            });
            this.render();
            this.startPolling(key);
            await this.pollStatus(key);
        } catch (error) {
            this.modelStatuses.set(key, {
                status: "error",
                message: error.message || String(error),
                progress: 100,
            });
            this.render();
        }
    }

    async downloadAll() {
        const parsed = this.getParsed();
        if (parsed.error || !parsed.models.length) return;

        for (let index = 0; index < parsed.models.length; index += 1) {
            const model = parsed.models[index];
            const key = model.key_hint || modelKey(model, index);
            const status = this.modelStatuses.get(key);
            if (status && ACTIVE_STATES.has(status.status)) continue;
            if (status && ["installed", "success", "skipped"].includes(status.status)) continue;
            await this.startDownload(model, key);
        }
    }

    startPolling(key) {
        if (this.pollTimers.has(key)) {
            clearInterval(this.pollTimers.get(key));
        }
        this.pollTimers.set(
            key,
            setInterval(() => {
                this.pollStatus(key);
            }, 1000)
        );
    }

    stopPolling(key) {
        const timer = this.pollTimers.get(key);
        if (timer) clearInterval(timer);
        this.pollTimers.delete(key);
    }

    async pollStatus(key) {
        const taskId = this.modelTasks.get(key);
        if (!taskId) return;

        try {
            const response = await api.fetchApi(
                `/model_downloader_plus/general/status/${encodeURIComponent(taskId)}?t=${Date.now()}`,
                {
                    cache: "no-store",
                    headers: { "Cache-Control": "no-cache" },
                }
            );
            const data = await response.json();
            this.modelStatuses.set(key, data);
            if (DONE_STATES.has(data.status)) {
                this.stopPolling(key);
            }
            this.render();
        } catch (error) {
            this.modelStatuses.set(key, {
                status: "error",
                message: error.message || String(error),
                progress: 100,
            });
            this.stopPolling(key);
            this.render();
        }
    }

    stateText(status) {
        const state = String(status?.status || "ready");
        if (state === "idle") return "Ready";
        if (state === "installed") return "Installed";
        if (state === "missing") return "Missing";
        if (state === "checking") return "Checking";
        return state.charAt(0).toUpperCase() + state.slice(1);
    }

    renderCard(model, index) {
        const key = model.key_hint || modelKey(model, index);
        const status = this.modelStatuses.get(key) || { status: "ready", message: "Ready.", progress: 0 };
        const active = ACTIVE_STATES.has(status.status);
        const progress = Number(status.progress) || 0;

        const card = document.createElement("div");
        card.className = `gmd-card ${status.status || "ready"}`;
        card.dataset.modelKey = key;

        const bg = document.createElement("div");
        bg.className = "gmd-progress-bg";
        const isGreen = ["success", "skipped", "installed"].includes(status.status);
        bg.style.width = active || (DONE_STATES.has(status.status) && status.status !== "missing") ? formatPercent(progress || 100) : "0%";
        if (status.status === "error" || status.status === "partial") {
            bg.style.background = "rgba(120, 54, 54, 0.35)";
        } else if (status.status === "queued") {
            bg.style.background = "rgba(120, 105, 54, 0.35)";
        }
        card.appendChild(bg);

        const body = document.createElement("div");
        body.className = "gmd-card-body";

        const top = document.createElement("div");
        top.className = "gmd-card-top";

        const name = document.createElement("div");
        name.className = "gmd-model-name";
        name.textContent = model.name;

        const pill = document.createElement("div");
        pill.className = `gmd-pill ${status.status || "ready"}`;
        pill.textContent = this.stateText(status);

        top.appendChild(name);
        top.appendChild(pill);
        body.appendChild(top);

        if (model.description) {
            const desc = document.createElement("div");
            desc.className = "gmd-desc";
            desc.textContent = model.description;
            body.appendChild(desc);
        }

        const meta = document.createElement("div");
        meta.className = "gmd-meta";
        const fileText = model.file_name ? ` | ${model.file_name}` : "";
        const expectedSizeText = model.expected_size !== null && model.expected_size !== undefined
            ? ` | size: ${formatBytes(model.expected_size)}`
            : "";
        const checks = [];
        if (model.expected_size !== null && model.expected_size !== undefined) checks.push("size");
        if (model.sha256) checks.push("sha256");
        if (model.sha1) checks.push("sha1");
        if (model.md5) checks.push("md5");
        const checkText = checks.length ? ` | verify: ${checks.join("+")}` : "";
        meta.textContent = `${model.download_directory}${fileText}${expectedSizeText} | ${model.urls.length} URL${model.urls.length === 1 ? "" : "s"}${checkText}`;
        meta.title = [
            model.expected_size !== null && model.expected_size !== undefined ? `Expected size: ${model.expected_size} bytes` : "",
            ...model.urls,
        ].filter(Boolean).join("\n");
        body.appendChild(meta);

        const statusLine = document.createElement("div");
        statusLine.className = "gmd-status-line";
        const parts = [];
        if (status.message) parts.push(status.message);
        if (status.current_file) parts.push(status.current_file);
        if (status.file_index && status.total_files) parts.push(`file ${status.file_index}/${status.total_files}`);
        statusLine.textContent = parts.join(" | ") || "Ready.";
        statusLine.title = statusLine.textContent;
        body.appendChild(statusLine);

        const controls = document.createElement("div");
        controls.className = "gmd-card-controls";

        const downloadBtn = document.createElement("button");
        downloadBtn.className = `gmd-btn ${isGreen ? "success" : "primary"}`;
        if (status.status === "error" || status.status === "partial") downloadBtn.className = "gmd-btn warning";
        downloadBtn.type = "button";
        downloadBtn.disabled = active || isGreen;
        downloadBtn.textContent = active
            ? "Working..."
            : (isGreen ? "Installed" : (status.status === "error" || status.status === "partial" ? "Retry" : "Download"));
        downloadBtn.onclick = () => this.startDownload(model, key);
        controls.appendChild(downloadBtn);

        body.appendChild(controls);
        card.appendChild(body);
        return card;
    }

    render() {
        const parsed = this.getParsed();
        const locked = this.isLocked();
        const safeMode = this.isSafeMode();
        const modelCount = parsed.models.length;

        this.applyLayout();
        this.jsonInput.value = this.getJsonText();
        this.safeModeInput.checked = safeMode;
        this.safeModeTitle.textContent = safeMode ? "Safe On" : "Safe Off";
        this.safeModeHint.textContent = safeMode ? "trusted hosts + Civitai" : "all hosts";
        this.configSection.classList.toggle("locked", locked);
        this.jsonInput.style.display = locked ? "none" : "block";
        this.hints.style.display = locked ? "none" : "flex";
        this.exampleBtn.style.display = locked ? "none" : "inline-block";
        this.formatBtn.style.display = locked ? "none" : "inline-block";
        this.lockBtn.textContent = locked ? "Edit JSON" : "Lock";
        this.lockBtn.className = locked ? "gmd-btn" : "gmd-btn primary";

        this.configSummary.style.color = parsed.error ? "#ff9c9c" : "#aaa";
        if (locked) {
            this.configTitle.textContent = `${modelCount} model${modelCount === 1 ? "" : "s"} locked`;
            this.configSummary.textContent = parsed.error ? parsed.error : "Config hidden";
            if (!this.configSection.contains(this.configSummary)) {
                this.configSection.insertBefore(this.configSummary, this.configActions);
            }
        } else {
            this.configTitle.textContent = "JSON model list";
            this.configSummary.textContent = parsed.error || (
                modelCount
                    ? `${modelCount} model${modelCount === 1 ? "" : "s"} | use [...] for lists`
                    : "0 models - Example uses [...]"
            );
        }

        this.listTitle.textContent = parsed.error
            ? "Config error"
            : `${modelCount} model${modelCount === 1 ? "" : "s"}`;

        this.listArea.innerHTML = "";
        this.downloadAllBtn.disabled = Boolean(parsed.error) || !modelCount;
        this.checkBtn.disabled = Boolean(parsed.error) || !modelCount;

        if (parsed.error) {
            const error = document.createElement("div");
            error.className = "gmd-error";
            error.textContent = parsed.error;
            this.listArea.appendChild(error);
            return;
        }

        if (!modelCount) {
            const empty = document.createElement("div");
            empty.className = "gmd-empty";
            empty.textContent = "No models.";
            this.listArea.appendChild(empty);
            return;
        }

        parsed.models.forEach((model, index) => {
            this.listArea.appendChild(this.renderCard(model, index));
        });
        this.schedulePrecheck(parsed.models);
    }
}

app.registerExtension({
    name: "ModelDownloaderPlus.GeneralDownloader",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "GeneralModelDownloader") {
            return;
        }

        const syncDOMWidgetWidth = (node, widgetName) => {
            const widget = node?.widgets?.find((item) => item.name === widgetName);
            const nodeWidth = Number(node?.size?.[0]);
            if (widget && Number.isFinite(nodeWidth) && nodeWidth > 0) {
                if (!widget._gmdWidthBound) {
                    Object.defineProperty(widget, "width", {
                        configurable: true,
                        get() {
                            const width = Number(this._node?.size?.[0]);
                            return Number.isFinite(width) && width > 0 ? width : undefined;
                        },
                        set(_value) {},
                    });
                    widget._gmdWidthBound = true;
                }
                if (typeof widget.triggerDraw === "function") widget.triggerDraw();
            }
        };

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            if (onNodeCreated) onNodeCreated.apply(this, arguments);

            scheduleHideInternalWidgets(this);

            const container = document.createElement("div");
            Object.assign(container.style, {
                width: "100%",
                height: "100%",
                display: "flex",
                flexDirection: "column",
                boxSizing: "border-box",
            });

            const domWidget = this.addDOMWidget("general_downloader_panel", "div", container, {
                serialize: false,
                hideOnZoom: false,
            });
            this.generalDownloaderPanelWidget = domWidget;
            this.generalDownloaderPanel = new GeneralDownloaderPanel(this, container);

            this.generalDownloaderPanel.ensureInitialNodeSize();
            this.generalDownloaderPanel.applyLayout();
            syncDOMWidgetWidth(this, "general_downloader_panel");
            requestAnimationFrame(() => syncDOMWidgetWidth(this, "general_downloader_panel"));
        };

        const onResize = nodeType.prototype.onResize;
        nodeType.prototype.onResize = function (size) {
            if (onResize) onResize.apply(this, arguments);
            scheduleHideInternalWidgets(this);
            this.generalDownloaderPanel?.applyLayout();
            syncDOMWidgetWidth(this, "general_downloader_panel");
            requestAnimationFrame(() => syncDOMWidgetWidth(this, "general_downloader_panel"));
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            if (onConfigure) onConfigure.apply(this, arguments);
            scheduleHideInternalWidgets(this);
            setTimeout(() => {
                this.generalDownloaderPanel?.render();
                this.generalDownloaderPanel?.applyLayout();
                syncDOMWidgetWidth(this, "general_downloader_panel");
            }, 100);
        };
    },
});
