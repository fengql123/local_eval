#!/usr/bin/env bash
# CHECKPOINT is a GGUF file or a directory containing one mapped quantization.
set -euo pipefail
: "${CHECKPOINT:?GGUF checkpoint required}"
cd "$(dirname "${BASH_SOURCE[0]}")"
model=$CHECKPOINT
if [ -d "$model" ]; then
    shopt -s globstar nullglob
    files=()
    for file in "$model"/**/*.gguf; do
        name=${file##*/}
        case "${name,,}" in
            *mmproj*|*mtp-*|*dflash-*|*dspark*) continue ;;
            *-of-*) [[ "$file" == *-00001-of-* ]] || continue ;;
        esac
        files+=("$file")
    done
    if [ "${#files[@]}" != 1 ]; then
        printf 'Expected one GGUF model (or first split shard), found %s in %s\n' "${#files[@]}" "$model" >&2
        exit 65
    fi
    model=${files[0]}
fi
export CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH
exec build/bin/llama-server --model "$model" --alias served --host 127.0.0.1 --port 8000 "$@"
