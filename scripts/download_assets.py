"""下载固定版本的公开模型或数据，保留文件哈希。"""
import argparse
import hashlib
import json
import os
import fnmatch
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import truststore

# requests 使用 Windows 已信任的证书链；保留 HTTPS 校验。
truststore.inject_into_ssl()
os.environ["HF_HUB_DISABLE_XET"] = "1"
import requests

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=["student", "data", "glaive"], required=True)
    args = parser.parse_args()
    if args.kind == "student":
        source = json.loads((ROOT / "configs/models.json").read_text(encoding="utf-8"))["student"]
        target = ROOT / ".local/models/Qwen3-1.7B"
        repo_type = "model"
        patterns = ["*.json", "*.safetensors", "*.jinja", "*.txt", "README.md", "LICENSE*"]
    elif args.kind == "data":
        source = {"repo": "NousResearch/hermes-function-calling-v1", "revision": "dae3e1d28cfbcf4b915c04ea1e072030529b4bda"}
        target = ROOT / ".local/data/raw/hermes"
        repo_type = "dataset"
        patterns = ["README.md", "func-calling*.json", "glaive-*.json"]
    else:
        source = json.loads((ROOT / "configs/data-sources.json").read_text(encoding="utf-8"))["glaive"]
        target = ROOT / ".local/data/raw/glaive"
        repo_type = "dataset"
        patterns = ["README.md", "glaive-function-calling-v2.json"]
    print(f"开始下载 {source['repo']} @ {source['revision']}", flush=True)
    # 本机代理的 HEAD 响应缺少 Hub 元数据；用固定 revision 的 GET 和源哈希校验。
    api_type = "models" if repo_type == "model" else "datasets"
    metadata = requests.get(f"https://huggingface.co/api/{api_type}/{source['repo']}/revision/{source['revision']}",
                            params={"blobs": "true"}, timeout=30)
    metadata.raise_for_status()
    metadata = metadata.json()
    assert metadata["sha"] == source["revision"]
    entries = [p for p in metadata["siblings"] if any(fnmatch.fnmatch(p["rfilename"], pattern) for pattern in patterns)]
    target.mkdir(parents=True, exist_ok=True)
    (target / "source-metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    def verified(path, entry):
        if not path.exists() or path.stat().st_size != entry["size"]:
            return False
        if entry.get("lfs"):
            return digest(path) == entry["lfs"]["sha256"]
        data = path.read_bytes()
        return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest() == entry["blobId"]

    def download(entry):
        path = target / entry["rfilename"]
        if verified(path, entry):
            print(f"已校验 {entry['rfilename']}", flush=True)
            return
        prefix = "datasets/" if repo_type == "dataset" else ""
        url = f"https://huggingface.co/{prefix}{source['repo']}/resolve/{source['revision']}/{entry['rfilename']}"
        partial = path.with_suffix(path.suffix + ".part")
        if entry["size"] > 128 * 1024 * 1024:
            # 大文件整包响应在本机网络中等待过久；逐段下载后仍验证完整源哈希。
            chunks = target / ".parts" / path.name
            chunks.mkdir(parents=True, exist_ok=True)
            chunk_size = 64 * 1024 * 1024
            count = (entry["size"] + chunk_size - 1) // chunk_size

            def segment(index):
                begin = index * chunk_size
                end = min(begin + chunk_size, entry["size"]) - 1
                chunk = chunks / f"{index:05d}.bin"
                if chunk.exists() and chunk.stat().st_size == end - begin + 1:
                    return chunk
                for attempt in range(1, 5):
                    try:
                        with requests.get(url, headers={"Range": f"bytes={begin}-{end}"}, stream=True,
                                          timeout=(30, 120)) as response:
                            response.raise_for_status()
                            expected = f"bytes {begin}-{end}/{entry['size']}"
                            if response.status_code != 206 or response.headers.get("Content-Range") != expected:
                                raise RuntimeError("服务器没有按请求范围返回数据")
                            with chunk.open("wb") as handle:
                                for block in response.iter_content(1024 * 1024):
                                    if block:
                                        handle.write(block)
                        if chunk.stat().st_size != end - begin + 1:
                            raise RuntimeError("分段长度不一致")
                        print(f"分段完成 {path.name} {index + 1}/{count}", flush=True)
                        return chunk
                    except Exception as error:
                        print(f"分段重试 {path.name} {index + 1} 第 {attempt} 次：{type(error).__name__}", flush=True)
                        if attempt == 4:
                            raise
                        time.sleep(attempt * 2)
            with ThreadPoolExecutor(max_workers=4) as segments:
                pieces = list(segments.map(segment, range(count)))
            with partial.open("wb") as handle:
                for piece in pieces:
                    with piece.open("rb") as source_file:
                        for block in iter(lambda: source_file.read(8 * 1024 * 1024), b""):
                            handle.write(block)
            if not verified(partial, entry):
                raise RuntimeError(f"完整文件源哈希不匹配：{path.name}")
            partial.replace(path)
            for piece in pieces:
                piece.unlink()
            print(f"已下载并校验 {path.name} ({entry['size']} bytes)", flush=True)
            return
        for attempt in range(1, 5):
            try:
                offset = partial.stat().st_size if partial.exists() else 0
                headers = {"Range": f"bytes={offset}-"} if offset else {}
                with requests.get(url, headers=headers, stream=True, timeout=(30, 120)) as response:
                    response.raise_for_status()
                    mode = "ab" if offset and response.status_code == 206 else "wb"
                    with partial.open(mode) as handle:
                        for block in response.iter_content(8 * 1024 * 1024):
                            if block:
                                handle.write(block)
                if not verified(partial, entry):
                    raise RuntimeError(f"源哈希不匹配：{entry['rfilename']}")
                partial.replace(path)
                print(f"已下载并校验 {entry['rfilename']} ({entry['size']} bytes)", flush=True)
                return
            except Exception as error:
                print(f"下载重试 {entry['rfilename']}，第 {attempt} 次：{type(error).__name__}", flush=True)
                if attempt == 4:
                    raise
                time.sleep(attempt * 2)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(download, entries))
    files = [{"file": str(p.relative_to(target)).replace("\\", "/"), "bytes": p.stat().st_size,
              "sha256": digest(p)} for p in sorted(target.rglob("*"))
             if p.is_file() and not any(x in p.parts for x in [".cache", ".parts"]) and not p.name.endswith(".part")
             and p.name not in ["download-manifest.json", "source-metadata.json"]]
    (target / "download-manifest.json").write_text(json.dumps({**source, "files": files}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"kind": args.kind, "files": len(files), "bytes": sum(p["bytes"] for p in files)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
