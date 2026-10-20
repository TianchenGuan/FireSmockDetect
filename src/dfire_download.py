#!/usr/bin/env python3
"""Track C: download D-Fire dataset from official OneDrive links (README of
https://github.com/gaia-solutions-on-demand/DFireDataset) using the anonymous
Badger token flow that the OneDrive web client uses for migrated-to-SPO shares.
"""
import base64
import json
import os
import sys
import time
import urllib.request

RAW = "/local/home/tianrgua/cvproject/data/raw/dfire"
LINK_DATASET = "https://1drv.ms/u/c/c0bd25b6b048b01d/EbLgD7bES4FDvUN37Grxn8QBF5gIBBc7YV2qklF08GCiBw"
LINK_SPLITS = "https://1drv.ms/f/c/c0bd25b6b048b01d/Ema8FFze8mFIlM1Hn81BUUgBE3vnnmK4SQxybS-nHRt2pA?e=6rk0aN"
UA = "Mozilla/5.0"


def enc(url: str) -> str:
    return "u!" + base64.urlsafe_b64encode(url.encode()).decode().rstrip("=")


def badger_token() -> str:
    req = urllib.request.Request(
        "https://api-badgerp.svc.ms/v1.0/token",
        data=json.dumps({"appId": "5cbed6ac-a083-4e14-b191-b4ba07653de2"}).encode(),
        headers={"Content-Type": "application/json", "User-Agent": UA},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)["token"]


def api_get(path: str, token: str):
    req = urllib.request.Request(
        f"https://my.microsoftpersonalcontent.com/_api/v2.0/{path}",
        headers={"User-Agent": UA, "Authorization": "Badger " + token, "Prefer": "autoredeem"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def download(url: str, dest: str, expected_size: int | None):
    tmp = dest + ".part"
    pos = os.path.getsize(tmp) if os.path.exists(tmp) else 0
    headers = {"User-Agent": UA}
    if pos:
        headers["Range"] = f"bytes={pos}-"
    req = urllib.request.Request(url, headers=headers)
    mode = "ab" if pos else "wb"
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, mode) as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            pos += len(chunk)
            if pos % (200 << 20) < (1 << 20):
                print(f"  {dest}: {pos/1e6:.0f} MB, {pos/1e6/max(time.time()-t0,1):.1f} MB/s", flush=True)
    if expected_size is not None and pos != expected_size:
        raise RuntimeError(f"size mismatch for {dest}: got {pos}, expected {expected_size}")
    os.replace(tmp, dest)
    print(f"DONE {dest} ({pos} bytes)", flush=True)


def fetch_item(share_url: str, token: str, dest_dir: str, name_override=None):
    d = api_get(f"shares/{enc(share_url)}/driveItem", token)
    name = name_override or d["name"]
    dest = os.path.join(dest_dir, name)
    if os.path.exists(dest) and os.path.getsize(dest) == d.get("size"):
        print(f"SKIP {dest} (already complete)", flush=True)
        return d
    print(f"downloading {name} ({d.get('size')} bytes)", flush=True)
    download(d["@content.downloadUrl"], dest, d.get("size"))
    return d


def fetch_folder(share_url: str, token: str, dest_dir: str):
    root = api_get(f"shares/{enc(share_url)}/driveItem?$expand=children", token)
    drive_id = root["parentReference"]["driveId"]
    os.makedirs(dest_dir, exist_ok=True)
    stack = [(c, dest_dir) for c in root.get("children", [])]
    while stack:
        item, ddir = stack.pop()
        if "folder" in item:
            sub = os.path.join(ddir, item["name"])
            os.makedirs(sub, exist_ok=True)
            full = api_get(f"drives/{drive_id}/items/{item['id']}?$expand=children", token)
            stack.extend((c, sub) for c in full.get("children", []))
        else:
            dest = os.path.join(ddir, item["name"])
            if os.path.exists(dest) and os.path.getsize(dest) == item.get("size"):
                print(f"SKIP {dest}", flush=True)
                continue
            full = api_get(f"drives/{drive_id}/items/{item['id']}", token)
            download(full["@content.downloadUrl"], dest, item.get("size"))


def main():
    os.makedirs(RAW, exist_ok=True)
    token = badger_token()
    print("== splits folder ==", flush=True)
    fetch_folder(LINK_SPLITS, token, os.path.join(RAW, "data_splitting"))
    print("== main dataset ==", flush=True)
    token = badger_token()  # refresh
    meta = fetch_item(LINK_DATASET, token, RAW)
    with open(os.path.join(RAW, "onedrive_item_meta.json"), "w") as f:
        json.dump({k: v for k, v in meta.items() if not k.startswith("@")}, f, indent=2)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    sys.exit(main())
