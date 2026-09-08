#!/usr/bin/env bash
# Single entry point to the OpenFOAM environment for this project.
#
# VENDORED from an earlier CFD study on 2026-09-07, and
# deliberately a copy rather than a shared import. That study's results
# must stay reproducible, so
# nothing here may reach back into it (DECISIONS.md ENV-01). Duplication is the
# correct trade.
#
# The install itself is READ ONLY. Camber must never rebuild, patch or upgrade
# ~/.local/opt/OpenFOAM-v2606: it is shared with that study, and changing it
# would silently alter the tool that produced its results, with no error
# to warn anyone. If Camber ever needs a different version it gets its own
# prefix via FOAM_INSTALL.
#
# Why this exists (see DECISIONS.md ENV-02):
#   * OpenFOAM lives at a non-standard path because the AUR package build
#     failed on two components we do not use; the solvers were built by hand
#     against the libraries that did build.
#   * Your login shell is fish, which cannot source OpenFOAM's etc/bashrc.
#     Nothing in this project may therefore depend on your interactive
#     environment -- every OpenFOAM call goes through bash via this file.
#
# Usage:
#   source scripts/foam-env.sh          # inside a bash script
#   scripts/foam-env.sh <cmd> [args]    # run one command in the environment
#
# From fish:
#   bash scripts/foam-env.sh checkMesh -case cases/Ac0.000_a08

FOAM_INSTALL="${FOAM_INSTALL:-$HOME/.local/opt/OpenFOAM-v2606}"

if [ ! -f "$FOAM_INSTALL/etc/bashrc" ]; then
    echo "ERROR: OpenFOAM not found at $FOAM_INSTALL" >&2
    echo "       Set FOAM_INSTALL to the installation directory." >&2
    return 1 2>/dev/null || exit 1
fi

# OpenFOAM's bashrc refuses to load if a version is already sourced, and
# returns a spurious non-zero status under `set -e`.
unset WM_PROJECT_DIR
export FOAM_CONFIG_MODE=o
unset FOAM_SETTINGS

set +e
# shellcheck disable=SC1090,SC1091
source "$FOAM_INSTALL/etc/bashrc" >/dev/null 2>&1
set -e

# PETSc (GPU linear solvers) lives in a userland build. Adding it here rather
# than to the OpenFOAM installation keeps ENV-01 intact: the install shared
# with that study is never modified, and a shell without this wrapper sees no
# change at all.
PETSC_DIR="${PETSC_DIR:-$HOME/.local/opt/src/petsc}"
PETSC_ARCH="${PETSC_ARCH:-arch-cuda-opt}"
if [ -d "$PETSC_DIR/$PETSC_ARCH/lib" ]; then
    export PETSC_DIR PETSC_ARCH
    export LD_LIBRARY_PATH="$PETSC_DIR/$PETSC_ARCH/lib:${LD_LIBRARY_PATH}"
fi

if ! command -v simpleFoam >/dev/null 2>&1; then
    echo "ERROR: sourced $FOAM_INSTALL but simpleFoam is not on PATH" >&2
    return 1 2>/dev/null || exit 1
fi

# If arguments were given, run them in this environment and exit with their
# status. If sourced with no arguments, just leave the environment set up.
if [ "${BASH_SOURCE[0]}" = "${0}" ] && [ "$#" -gt 0 ]; then
    exec "$@"
fi
