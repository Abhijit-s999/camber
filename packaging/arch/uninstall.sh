#!/usr/bin/env bash
# Remove what install.sh added. Your projects in ~/Camber are left alone.
set -euo pipefail
DATA="${XDG_DATA_HOME:-$HOME/.local/share}"
rm -rf "$DATA/camber/venv"
rm -f "$HOME/.local/bin/camber-app" "$HOME/.local/bin/camber" \
      "$DATA/applications/camber.desktop" "$DATA/icons/hicolor/scalable/apps/camber.svg"
command -v update-desktop-database >/dev/null && update-desktop-database -q "$DATA/applications" || true
echo "Camber removed. Projects in ~/Camber were not touched."
