#!/usr/bin/env bash
# Install Camber as a desktop app on Arch (and derivatives: CachyOS, EndeavourOS, Manjaro).
#
# What it does, all inside your home directory -- nothing needs root except the
# pacman line it may ask you to run:
#   ~/.local/share/camber/venv          Camber's own Python environment
#   ~/.local/bin/camber, camber-app     launchers
#   ~/.local/share/applications/camber.desktop, and an icon
#
# The environment is created with --system-site-packages so it can see Qt
# WebEngine from pacman (python-pyqt6-webengine). Camber's numerical stack
# (MuJoCo, trimesh, pyamg, ...) is installed into the environment itself, where
# a system update cannot break it. Camber is installed *editable* from this
# checkout: keep the folder where it is, and `git pull` updates the app.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA="${XDG_DATA_HOME:-$HOME/.local/share}"
PREFIX="$DATA/camber"
VENV="$PREFIX/venv"
BIN="$HOME/.local/bin"
say() { printf '\033[1m%s\033[0m\n' "$*"; }
warn() { printf '\033[33m%s\033[0m\n' "$*"; }

say "Camber: installing from $REPO"

# ---- system packages -------------------------------------------------------------
missing=()
for pkg in python python-pyqt6 python-pyqt6-webengine ffmpeg; do
  pacman -Q "$pkg" >/dev/null 2>&1 || missing+=("$pkg")
done
if ((${#missing[@]})); then
  warn "Missing system packages: ${missing[*]}"
  echo "Install them, then run this again:"
  echo "    sudo pacman -S --needed ${missing[*]}"
  exit 1
fi

# ---- environment ---------------------------------------------------------------------
mkdir -p "$PREFIX" "$BIN"
if command -v uv >/dev/null 2>&1; then
  uv venv -q --python /usr/bin/python3 --system-site-packages --allow-existing "$VENV"
  say "Installing Camber and its numerical stack (this takes a minute)"
  uv pip install -q --python "$VENV/bin/python" -e "$REPO[app]"
else
  /usr/bin/python3 -m venv --system-site-packages "$VENV"
  say "Installing Camber and its numerical stack (this takes a few minutes)"
  "$VENV/bin/python" -m pip install --quiet --upgrade pip
  "$VENV/bin/python" -m pip install --quiet -e "$REPO[app]"
fi
"$VENV/bin/python" -c "import camber.app.server, mujoco, trimesh, PyQt6.QtWebEngineWidgets" \
  || { warn "The environment is incomplete; see the errors above."; exit 1; }

# ---- launchers -------------------------------------------------------------------------
cat > "$BIN/camber-app" <<EOF
#!/bin/sh
# Camber desktop app. Installed by $REPO/packaging/arch/install.sh
exec "$VENV/bin/python" -m camber.cli app "\$@"
EOF
cat > "$BIN/camber" <<EOF
#!/bin/sh
exec "$VENV/bin/python" -m camber.cli "\$@"
EOF
chmod +x "$BIN/camber-app" "$BIN/camber"

# ---- desktop entry -----------------------------------------------------------------------
ICONS="$DATA/icons/hicolor/scalable/apps"
mkdir -p "$ICONS" "$DATA/applications"
cp "$REPO/camber/app/static/icon.svg" "$ICONS/camber.svg"
sed "s|@BIN@|$BIN|g" "$REPO/packaging/arch/camber.desktop" > "$DATA/applications/camber.desktop"
command -v update-desktop-database >/dev/null && update-desktop-database -q "$DATA/applications" || true
command -v gtk-update-icon-cache >/dev/null && gtk-update-icon-cache -q "$DATA/icons/hicolor" 2>/dev/null || true

# ---- optional tools ------------------------------------------------------------------------
if ! ls "$HOME"/.local/opt/OpenFOAM-* >/dev/null 2>&1 && ! command -v simpleFoam >/dev/null; then
  warn "OpenFOAM not found: the Aerodynamics step will not run until it is installed."
fi
case ":$PATH:" in *":$BIN:"*) ;; *) warn "$BIN is not on your PATH; the app menu entry still works." ;; esac

say "Done. Open Camber from your app launcher, or run: camber-app"
