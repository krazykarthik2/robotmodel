import os
import sys
import urllib.request

FILES = [
    "config.json",
    "dataset_statistics.json",
    "example_batch.msgpack",
    "300000/default/checkpoint"
]

BASE_URL = "https://huggingface.co/rail-berkeley/octo-small/resolve/main"

def download_file(rel_path, target_dir="octo_small_checkpoint"):
    dest = os.path.join(target_dir, rel_path)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"  [OK] {rel_path} already exists ({os.path.getsize(dest) / 1024 / 1024:.2f} MB)")
        return

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    url = f"{BASE_URL}/{rel_path}"
    print(f"  Downloading: {rel_path}...")

    def reporthook(count, block_size, total_size):
        if total_size > 0:
            percent = int(count * block_size * 100 / total_size)
            mb_downloaded = (count * block_size) / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            sys.stdout.write(f"\r    Progress: {percent:02d}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)")
            sys.stdout.flush()

    urllib.request.urlretrieve(url, dest, reporthook=reporthook)
    print("\n    Done.")

def main():
    print("=" * 65)
    print("DOWNLOADING OCTO SMALL CHECKPOINT FROM HUGGINGFACE")
    print("Direct Streamlined Downloader (Bypasses httpx2/brotli bug)")
    print("=" * 65)
    for f in FILES:
        download_file(f)
    print("\nAll Octo Small checkpoint files successfully downloaded!")

if __name__ == "__main__":
    main()
