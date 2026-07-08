#!/bin/bash
# Build LocalFlow.app - a double-clickable, menu-bar macOS app around the
# dictation daemon. No terminal, no LM Studio (the LLM runs in-process).
#
#   ./scripts/build_app.sh                build into dist/LocalFlow.app
#                                         (dev build: runs code from this
#                                         repo's venv; rebuild-free edits)
#   ./scripts/build_app.sh --install      also copy to /Applications
#   ./scripts/build_app.sh --standalone   fully self-contained bundle: its
#                                         own Python runtime + all deps
#                                         inside (~700 MB). Works on any
#                                         Apple Silicon Mac - nothing to
#                                         install first.
#   ./scripts/build_app.sh --standalone --dmg
#                                         also produce a drag-to-install
#                                         dist/LocalFlow-<version>.dmg
#
# The dev build launches the daemon from this repo's venv (absolute path
# baked in at build time); the standalone build embeds a relocatable CPython
# (python-build-standalone) so the target machine needs no Homebrew, no
# Python, nothing. Both log to ~/Library/Logs/LocalFlow.log and get their
# own permission identity: on first launch macOS asks for Microphone,
# Accessibility and Input Monitoring for "LocalFlow" - grant all three.

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$REPO/venv"
PY="$VENV/bin/python"
APP="$REPO/dist/LocalFlow.app"
VERSION="$("$PY" -c 'import localflow; print(localflow.__version__)')"

INSTALL=0
STANDALONE=0
DMG=0
for arg in "$@"; do
    case "$arg" in
        --install) INSTALL=1 ;;
        --standalone) STANDALONE=1 ;;
        --dmg) DMG=1 ;;
        *) echo "unknown option: $arg"; exit 1 ;;
    esac
done

[ -x "$VENV/bin/localflow" ] || { echo "venv missing; run: python3 -m venv venv && venv/bin/pip install -e '.[whispercpp,desktop,llm]'"; exit 1; }

echo "Building LocalFlow.app v$VERSION"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"

# ------------------------------------------------------------- Info.plist
cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key><string>LocalFlow</string>
    <key>CFBundleDisplayName</key><string>LocalFlow</string>
    <key>CFBundleIdentifier</key><string>dev.localflow.app</string>
    <key>CFBundleVersion</key><string>$VERSION</string>
    <key>CFBundleShortVersionString</key><string>$VERSION</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>CFBundleExecutable</key><string>LocalFlow</string>
    <key>CFBundleIconFile</key><string>LocalFlow</string>
    <key>LSMinimumSystemVersion</key><string>13.0</string>
    <key>LSUIElement</key><true/>
    <key>NSMicrophoneUsageDescription</key>
    <string>LocalFlow listens while you hold the dictation hotkey and transcribes your speech on-device.</string>
    <key>NSHighResolutionCapable</key><true/>
</dict>
</plist>
PLIST

# ---------------------------------------------------------------- payload
# The main executable is a small compiled binary that runs the daemon
# in-process through libpython (see scripts/app_stub.c for why: TCC ties
# permissions to the main executable's identity - shell launchers that
# exec another binary leave the toggle ON but the process untrusted).
STUB_DEFS=()

if [ "$STANDALONE" = 1 ]; then
    # Fully self-contained: a relocatable CPython runtime + every dependency
    # live inside the bundle. python-build-standalone is built for exactly
    # this (no baked absolute paths); Homebrew's Python is not relocatable.
    PYROOT="$APP/Contents/Resources/python"
    RUNTIME="$PYROOT/runtime"
    CACHE="${LOCALFLOW_BUILD_CACHE:-$HOME/Library/Caches/localflow-build}"
    mkdir -p "$CACHE" "$PYROOT"

    ARCH="$(uname -m)"
    case "$ARCH" in
        arm64) PBS_ARCH="aarch64-apple-darwin" ;;
        x86_64) PBS_ARCH="x86_64-apple-darwin" ;;
        *) echo "unsupported architecture: $ARCH"; exit 1 ;;
    esac

    # Resolve the runtime tarball: pin via LOCALFLOW_PYTHON_URL, or take the
    # newest CPython 3.13 install_only build from the latest release.
    PBS_URL="${LOCALFLOW_PYTHON_URL:-}"
    if [ -z "$PBS_URL" ]; then
        echo "  resolving python-build-standalone (cpython 3.13, $PBS_ARCH)..."
        PBS_URL="$(curl -fsSL https://api.github.com/repos/astral-sh/python-build-standalone/releases/latest \
            | "$PY" -c '
import json, sys
release = json.load(sys.stdin)
arch = sys.argv[1]
names = [a["browser_download_url"] for a in release["assets"]
         if a["name"].startswith("cpython-3.13.")
         and a["name"].endswith(f"{arch}-install_only.tar.gz")]
if not names:
    sys.exit("no matching cpython 3.13 asset in latest release")
print(names[0])' "$PBS_ARCH")"
    fi
    TARBALL="$CACHE/$(basename "$PBS_URL")"
    if [ ! -f "$TARBALL" ]; then
        echo "  downloading $(basename "$PBS_URL")..."
        curl -fL --retry 3 -o "$TARBALL.part" "$PBS_URL"
        mv "$TARBALL.part" "$TARBALL"
    fi

    echo "  unpacking Python runtime..."
    UNPACK="$(mktemp -d)"
    tar -xzf "$TARBALL" -C "$UNPACK"
    mv "$UNPACK/python" "$RUNTIME"
    rm -rf "$UNPACK"
    # Dead weight the daemon never imports (~90 MB of stdlib test suite etc.)
    rm -rf "$RUNTIME"/lib/python3.*/test \
           "$RUNTIME"/lib/python3.*/idlelib \
           "$RUNTIME"/lib/python3.*/turtledemo

    echo "  installing localflow + dependencies into the bundle..."
    "$RUNTIME/bin/python3" -m pip install --quiet --no-compile \
        --target "$PYROOT/site-packages" "$REPO[whispercpp,desktop,llm]"

    LIBPYTHON_NAME="$(cd "$RUNTIME/lib" && ls libpython3.*.dylib | head -1)"
    [ -n "$LIBPYTHON_NAME" ] || { echo "no libpython in runtime"; exit 1; }
    STUB_DEFS+=(
        -DLIBPYTHON_PATH="\"@RESOURCES@/python/runtime/lib/$LIBPYTHON_NAME\""
        -DLOCALFLOW_PYTHONPATH="\"@RESOURCES@/python/site-packages\""
        -DLOCALFLOW_PYTHONHOME="\"@RESOURCES@/python/runtime\""
    )

    # Bundle a Whisper model so dictation works offline on first launch
    # (no multi-minute silent download). "base" matches the default config;
    # override with LOCALFLOW_BUNDLE_WHISPER=small etc., or "none" to skip.
    WHISPER_MODEL="${LOCALFLOW_BUNDLE_WHISPER:-base}"
    if [ "$WHISPER_MODEL" != "none" ]; then
        GGML="$CACHE/ggml-$WHISPER_MODEL.bin"
        if [ ! -f "$GGML" ]; then
            echo "  downloading Whisper '$WHISPER_MODEL' model for the bundle..."
            curl -fL --retry 3 -o "$GGML.part" \
                "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-$WHISPER_MODEL.bin"
            mv "$GGML.part" "$GGML"
        fi
        mkdir -p "$APP/Contents/Resources/models"
        cp "$GGML" "$APP/Contents/Resources/models/"
    fi

    # Smoke test with the exact interpreter + path the stub will use.
    PYTHONPATH="$PYROOT/site-packages" "$RUNTIME/bin/python3" \
        -c "import localflow, mlx_lm, pywhispercpp, sounddevice" \
        || { echo "bundled runtime can't import the app - build broken"; exit 1; }
else
    SITE_PACKAGES="$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
    LIBPYTHON="$("$PY" -c 'import sysconfig; import os.path as p; print(p.join(sysconfig.get_config_var("LIBDIR"), "libpython3.13.dylib"))')"
    [ -f "$LIBPYTHON" ] || { echo "libpython not found at $LIBPYTHON"; exit 1; }
    STUB_DEFS+=(
        -DLIBPYTHON_PATH="\"$LIBPYTHON\""
        -DLOCALFLOW_PYTHONPATH="\"$SITE_PACKAGES:$REPO\""
    )
    # Smoke test: the stub hardwires "-m localflow.cli run"; verify the
    # import path works with the same interpreter the stub will use.
    PYTHONPATH="$SITE_PACKAGES:$REPO" "$PY" -c "import localflow" \
        || { echo "interpreter can't import localflow - check venv"; exit 1; }
fi

clang -O2 -Wall "${STUB_DEFS[@]}" \
    -o "$APP/Contents/MacOS/LocalFlow" "$REPO/scripts/app_stub.c"

# ------------------------------------------------------------------- icon
ICONSET="$REPO/dist/LocalFlow.iconset"
rm -rf "$ICONSET"; mkdir -p "$ICONSET"
"$PY" "$REPO/scripts/generate_icon.py" "$ICONSET/icon_512x512@2x.png" 1024 > /dev/null
for size in 16 32 128 256 512; do
    sips -z $size $size "$ICONSET/icon_512x512@2x.png" \
        --out "$ICONSET/icon_${size}x${size}.png" > /dev/null
    double=$((size * 2))
    sips -z $double $double "$ICONSET/icon_512x512@2x.png" \
        --out "$ICONSET/icon_${size}x${size}@2x.png" > /dev/null
done
iconutil -c icns "$ICONSET" -o "$APP/Contents/Resources/LocalFlow.icns"
rm -rf "$ICONSET"

# ------------------------------------------------------------------- sign
# Ad-hoc signatures give the bundle a stable identity for the TCC
# permission prompts (Accessibility / Input Monitoring / Microphone).
# Nested Mach-O files are signed first so the outer seal stays valid.
if [ "$STANDALONE" = 1 ]; then
    find "$APP/Contents/Resources/python" \
        \( -name "*.dylib" -o -name "*.so" \) -type f -print0 \
        | xargs -0 codesign --force -s - 2> /dev/null
    codesign --force -s - "$APP/Contents/Resources/python/runtime/bin/"* 2> /dev/null || true
fi
codesign --force -s - "$APP/Contents/MacOS/LocalFlow" 2> /dev/null
codesign --force -s - "$APP" 2> /dev/null

echo "Built: $APP"

if [ "$DMG" = 1 ]; then
    DMG_PATH="$REPO/dist/LocalFlow-$VERSION.dmg"
    STAGE="$(mktemp -d)"
    cp -R "$APP" "$STAGE/"
    ln -s /Applications "$STAGE/Applications"
    rm -f "$DMG_PATH"
    hdiutil create -volname "LocalFlow" -srcfolder "$STAGE" -ov \
        -format UDZO "$DMG_PATH" > /dev/null
    rm -rf "$STAGE"
    echo "Built: $DMG_PATH"
fi

if [ "$INSTALL" = 1 ]; then
    rm -rf "/Applications/LocalFlow.app"
    cp -R "$APP" /Applications/
    echo "Installed: /Applications/LocalFlow.app"
fi
