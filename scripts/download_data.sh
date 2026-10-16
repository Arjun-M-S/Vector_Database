#!/usr/bin/env bash
# Downloads TEXMEX datasets into data/ (git-ignored).
#   scripts/download_data.sh            # siftsmall: 10k vectors, ~5 MB
#   scripts/download_data.sh sift       # SIFT1M: 1M vectors, ~170 MB download
set -euo pipefail

name="${1:-siftsmall}"
case "$name" in
  siftsmall | sift) ;;
  *) echo "usage: $0 [siftsmall|sift]" >&2; exit 1 ;;
esac

data_dir="$(cd "$(dirname "$0")/.." && pwd)/data"
mkdir -p "$data_dir"
curl -fL --retry 3 -o "$data_dir/$name.tar.gz" "ftp://ftp.irisa.fr/local/texmex/corpus/$name.tar.gz"
tar -xzf "$data_dir/$name.tar.gz" -C "$data_dir"
rm "$data_dir/$name.tar.gz"
ls -lh "$data_dir/$name"
