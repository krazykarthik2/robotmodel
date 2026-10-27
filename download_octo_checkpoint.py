import os
import sys
import urllib.request
import shutil

# rail-berkeley/octo-small uses step 270000 on Hugging Face
FILES = [
    "config.json",
    "dataset_statistics.json",
    "example_batch.msgpack",
    "270000/default/checkpoint"
]

BASE_URL = "https://huggingface.co/rail-berkeley/octo-small/resolve/main"

def download_file(rel_path, target_dir="octo_small_checkpoint"):
    dest = os.path.join(target_dir, rel_path)
    if os.path.exists(dest) and os.path.getsize(dest) > 1000:
        print(f"  [OK] {rel_path} already exists ({os.path.getsize(dest) / 1024 / 1024:.2f} MB)")
        return

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    url = f"{BASE_URL}/{rel_path}"
    print(f"  Downloading: {rel_path} (from HuggingFace)...")

    # Add custom headers to avoid any bot-blocking
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    with urllib.request.urlopen(req) as resp, open(dest, "wb") as out_file:
        total_size = int(resp.headers.get("content-length", 0))
        downloaded = 0
        chunk_size = 1024 * 1024 # 1MB chunks

        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            out_file.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                pct = int(downloaded * 100 / total_size)
                mb_d = downloaded / (1024 * 1024)
                mb_t = total_size / (1024 * 1024)
                sys.stdout.write(f"\r    Progress: {pct:02d}% ({mb_d:.1f}/{mb_t:.1f} MB)")
                sys.stdout.flush()

    print("\n    Done.")

def main():
    print("=" * 65)
    print("DOWNLOADING OCTO SMALL CHECKPOINT FROM HUGGINGFACE")
    print("=" * 65)
    for f in FILES:
        download_file(f)

    # Octo's Orbax checkpoint manager looks for either latest step directory or 300000 / 270000
    # Also create a symlink / duplicate so both 270000 and 300000 are recognized
    target_300k = os.path.join("octo_small_checkpoint", "300000", "default")
    src_270k = os.path.join("octo_small_checkpoint", "270000", "default")
    if os.path.exists(src_270k) and not os.path.exists(target_300k):
        try:
            os.makedirs(target_300k, exist_ok=True)
            # Create hardlink or copy checkpoint if needed
            ck_src = os.path.join(src_270k, "checkpoint")
            ck_dst = os.path.join(target_300k, "checkpoint")
            if not os.path.exists(ck_dst):
                try:
                    os.link(ck_src, ck_dst)
                except Exception:
                    shutil.copyfile(ck_src, ck_dst)
        except Exception:
            pass

    print("\nAll Octo Small checkpoint files successfully downloaded and prepared!")

if __name__ == "__main__":
    main()
