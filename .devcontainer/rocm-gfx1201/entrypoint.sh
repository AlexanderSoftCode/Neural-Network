#!/usr/bin/env bash
set -euo pipefail

# --------------------------------------------------------------------------
# Single, explicit startup path. No silent fallback.
#
# GPU device group membership (/dev/kfd, /dev/dri/render*) can only be
# fixed by root, and the privilege drop to `ubuntu` must go through gosu
# so the final process gets a *freshly computed* supplementary group list
# (gosu does setuid+setgid+setgroups+exec - it re-reads /etc/group at
# that moment). A plain `exec "$@"` here would keep whatever group list
# this process already had, silently shipping a container that starts
# "fine" but can't open /dev/kfd.
# --------------------------------------------------------------------------

if [ "$(id -u)" -ne 0 ]; then
    echo "[entrypoint] FATAL: container must start as root (got uid=$(id -u)/$(id -un))." >&2
    echo "[entrypoint] GPU group membership is fixed here, then handed off via" >&2
    echo "[entrypoint] gosu so the final process gets fresh supplementary groups." >&2
    echo "[entrypoint] Don't pass --user/-u to 'docker run' for this image - let" >&2
    echo "[entrypoint] ENTRYPOINT do the fix-then-drop sequence itself." >&2
    exit 1
fi

if [ ! -x /usr/local/bin/fix-gpu-groups.sh ]; then
    echo "[entrypoint] FATAL: /usr/local/bin/fix-gpu-groups.sh missing or not executable." >&2
    echo "[entrypoint] Refusing to start - GPU device group membership cannot be verified." >&2
    exit 1
fi

/usr/local/bin/fix-gpu-groups.sh

# The one and only path to the final process. gosu re-reads /etc/group
# fresh as part of the privilege drop - never a plain `exec "$@"`.
exec gosu ubuntu "$@"