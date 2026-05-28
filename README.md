# ComfyUI Model Downloader Plus (ComfyUI 模型下载器Plus)

这是一个用于 ComfyUI 的模型下载器插件Plus增强版本。原始项目主要提供基于 Civitai / Hugging Face `model_id` 的分类下载节点；此版本在保留原有节点的基础上，额外加入了直连 URL 下载能力：

- `Simple Model Downloader`
- `Simple Batch Downloader`
- `General Model Downloader`

> 注意：上面三个下载器是此版本新增功能，原始源节点仓库中没有这些节点。如果只安装原始项目，不会看到 Simple / Batch / General 下载器。

## 节点列表

### 原始兼容节点

这些节点保留原项目的使用方式，适合通过 Civitai 或 Hugging Face 的模型 ID 下载并关联模型信息。

- `(Down)load Checkpoint`
- `(Down)load LoRA`
- `(Down)load VAE`
- `(Down)load UNET`
- `(Down)load ControlNet`

使用方式：

- `source = civitai` 时，`model_id` 填 Civitai 模型 ID，例如 `https://civitai.com/models/123456` 中的 `123456`。
- `source = huggingface` 时，`model_id` 填 Hugging Face 仓库名，例如 `runwayml/stable-diffusion-v1-5`。
- `version_id` 仅 Civitai 生效，留空时默认下载最新版本。
- `file_names` 仅 Hugging Face 生效，支持多行文件名；为空时下载仓库内可用模型文件。

### Simple Model Downloader

单 URL 下载节点，适合临时下载一个模型文件。

- 输入：`model_url`、`model_folder`、`run_download`、`overwrite_existing`
- 输出：模型文件名和下载状态
- 保存位置：`ComfyUI/models/<model_folder>/`
- 会先检查同组模型目录中是否已有同名文件，存在时默认跳过
- 直连 URL 仅允许可信模型站点：`huggingface.co`、`hf-mirror.com`、`modelscope.cn`、`github.com`、`githubusercontent.com`

### Simple Batch Downloader

最多 5 个 URL 的轻量批量下载节点。

- 输入：`url1` 到 `url5`、`model_folder`、`run_download`、`overwrite_existing`
- 输出：批量下载状态文本
- 保存位置：`ComfyUI/models/<model_folder>/`
- 直连 URL 仅允许可信模型站点：`huggingface.co`、`hf-mirror.com`、`modelscope.cn`、`github.com`、`githubusercontent.com`

### General Model Downloader

推荐的新下载器。它使用 JSON model list 管理多个模型，前端界面支持锁定配置、逐个下载、批量下载、状态刷新、本地存在性预检、文件大小和 hash 校验。

默认示例使用顶层数组：

```json
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
  }
]
```

兼容格式：

- 顶层数组：`[{...}, {...}]`
- 单模型对象：`{"name": "...", "url": "..."}`
- 旧格式：`{"models": [{...}]}`
- 命名对象映射：`{"Model A": {"url": "..."}}`

常用字段：

- `name`：界面显示名
- `url` 或 `urls`：下载地址；`urls` 可写数组或多行文本
- `download_directory`：模型目录名，例如 `checkpoints`、`diffusion_models`、`loras`
- `file_name`：保存文件名；为空时从 URL 或响应头推断
- `overwrite_existing`：是否覆盖已有文件
- `size` / `expected_size`：可选文件大小校验，只能填写精确字节数，例如 `123456789`
- `sha256`、`sha1`、`md5` 或 `hash`：可选 hash 校验
- `description`：卡片说明文本

General 节点会使用 ComfyUI 已注册的模型目录做存在性检测，因此可以识别 `extra_model_paths.yaml` 中配置的额外目录。若文件已存在于注册目录中，下载时默认跳过。

## Safe Mode

`General Model Downloader` 带有界面 Safe mode 开关，默认开启。

Safe mode 开启时，只允许从以下 host 下载：

- `huggingface.co`
- `hf-mirror.com`
- `modelscope.cn`
- `github.com`
- `githubusercontent.com`
- `civitai.com`

这些可信 host 是代码内置的，不从 JSON 读取。这样可以避免工作流分享者在模型清单里加入恶意白名单。即使 JSON 中出现 `allowed_hosts`、`allowed_domains` 等字段，`General Model Downloader` 也会忽略。

如果关闭 Safe mode，则只保留基础的 `http/https` URL 校验。建议仅在你本人确认下载源可信时关闭。

## 安装

将此版本插件放入 ComfyUI 的 `custom_nodes` 目录，例如：

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Windecay/ComfyUI-Model-Downloader-plus.git
```

安装依赖：

```bash
pip install -r requirements.txt
```

重启 ComfyUI 后，在节点菜单中查找：

- `Model (Down)load`
- `utils/download`
- `General Model Downloader`

## Token 配置

对于需要认证的下载，例如 Civitai 的部分模型或 Hugging Face 私有模型：

1. 复制 `config.ini.example` 为 `config.ini`
2. 填入 API key 或 token
3. 重启 ComfyUI

```ini
[civitai]
api_key = YOUR_CIVITAI_API_KEY_HERE

[huggingface]
token = YOUR_HUGGINGFACE_TOKEN_HERE
```

请不要分享包含真实 token 的 `config.ini`。

## 与原项目的区别

此版本插件保留原始分类下载节点，并额外提供面向工作流分发和自定义 URL 的下载器：

- Simple：单 URL，快速下载
- Batch：最多 5 个 URL，轻量批量下载
- General：JSON 驱动、现代 UI、逐个下载、批量下载、预检、状态报告、安全白名单、大小/hash 校验

如果你需要的是原项目的 Civitai / Hugging Face `model_id` 工作流，继续使用原始兼容节点即可。如果你需要把模型下载清单直接写进工作流，推荐使用 `General Model Downloader`。

## License

MIT
