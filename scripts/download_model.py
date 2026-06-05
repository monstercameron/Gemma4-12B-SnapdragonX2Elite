#!/usr/bin/env python
"""Download the Gemma 4 12B weights into models/<name>.

The engine loads the fp16 safetensors and quantizes to int4 on startup, so the full fp16 model is
needed (~24 GB). The download is resumable -- re-run if it's interrupted.

  python scripts/download_model.py                 # default repo -> models/gemma-4-12B-it
  python scripts/download_model.py --repo <id>     # if the HF repo id differs
  HF_TOKEN=... python scripts/download_model.py    # if the repo needs auth

Gemma weights are governed by Google's Gemma Terms of Use (not this repo's MIT license).
"""
import os
import sys
import argparse


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=os.environ.get("GEMMA4_REPO", "google/gemma-4-12B-it"),
                    help="Hugging Face repo id (default: google/gemma-4-12B-it)")
    ap.add_argument("--dest", default=os.environ.get("GEMMA4_DEST", "models/gemma-4-12B-it"),
                    help="local destination dir (must match MODEL in src/vk_engine.py)")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"), help="HF token (if the repo is gated)")
    args = ap.parse_args()

    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError
    except ImportError:
        sys.exit("huggingface_hub is not installed.\n  pip install huggingface_hub   (or run scripts/setup)")

    dest = os.path.abspath(args.dest)
    print(f"Downloading '{args.repo}' -> {dest}")
    print("  fp16 safetensors (~24 GB) + tokenizer/config. Resumable -- safe to re-run.\n")
    try:
        snapshot_download(
            repo_id=args.repo,
            local_dir=dest,
            token=args.token,
            allow_patterns=["*.safetensors", "*.json", "*.model", "tokenizer*", "*.txt"],
        )
    except RepositoryNotFoundError:
        sys.exit(f"\nRepo '{args.repo}' not found. If the id differs, pass --repo <id>, "
                 "or download manually and place the files in the --dest dir.")
    except GatedRepoError:
        sys.exit(f"\nRepo '{args.repo}' is gated. Accept the license on its HF page and pass a token:\n"
                 "  HF_TOKEN=hf_... python scripts/download_model.py")
    print(f"\nDone. Weights in {dest}")
    print("Verify with:  python scripts/check_env.py")


if __name__ == "__main__":
    main()
