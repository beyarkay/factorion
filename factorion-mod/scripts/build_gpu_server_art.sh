#!/usr/bin/env bash
# Package the generated GPU-server masters into aligned, 2x Factorio sprites.
set -euo pipefail

if ! command -v magick >/dev/null 2>&1; then
  printf 'ImageMagick (magick) is required to package GPU-server art.\n' >&2
  exit 1
fi

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
mod_dir=$(cd "$script_dir/../mod" && pwd)
masters="$script_dir/../art/gpu-server"
art_dir="$mod_dir/graphics/entity/gpu-server"

for direction in north east south west; do
  source_file="$masters/source/$direction.png"
  bbox=$(magick "$source_file" -alpha extract -threshold 5% -format '%@' info:)
  magick "$source_file" -crop "$bbox" +repage \
    -filter Lanczos -resize 320x320 \
    -gravity center -background none -extent 384x384 \
    -strip -define png:color-type=6 "$art_dir/$direction.png"
done

magick "$art_dir/north.png" "$art_dir/east.png" \
  "$art_dir/south.png" "$art_dir/west.png" \
  +append -strip -define png:color-type=6 "$masters/directions.png"

icon_source="$masters/source/icon.png"
icon_bbox=$(magick "$icon_source" -alpha extract -threshold 5% -format '%@' info:)
magick "$icon_source" -crop "$icon_bbox" +repage \
  -filter Lanczos -resize 116x116 \
  -gravity center -background none -extent 128x128 \
  -strip -define png:color-type=6 "$mod_dir/graphics/icons/gpu-server.png"
