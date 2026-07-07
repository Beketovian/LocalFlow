#!/bin/bash
# Build LocalFlow.app - a double-clickable, menu-bar macOS app around the
# dictation daemon. No terminal, no LM Studio (the LLM runs in-process).
#
#   ./scripts/build_app.sh                build into dist/LocalFlow.app
#   ./scripts/build_app.sh --install      also copy to /Applications
#   ./scripts/build_app.sh --standalone   copy code + deps INTO the bundle
#                                         (~450 MB; survives moving/deleting
#                                         this repo, still needs Homebrew's
#                                         python@3.13 on the machine)
#
# The bundle launches the daemon from this repo's venv (absolute path baked
# in at build time), logs to ~/Library/Logs/LocalFlow.log, and gets its own
# permission identity: on first launch macOS will ask for Microphone,
# Accessibility and Input Monitoring for "LocalFlow" - grant all three.

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$REPO/venv"
PY="$VENV/bin/python"
APP="$REPO/dist/LocalFlow.app"
VERSION="$("$PY" -c 'import localflow; print(localflow.__version__)')"

INSTALL=0
STANDALONE=0
for arg in "$@"; do
    case "$arg" in
        --install) INSTALL=1 ;;
        --standalone) STANDALONE=1 ;;
        *) echo "unknown option: $arg"; exit 1 ;;
    esac
done

[ -x "$VENV/bin/localflow" ] || { echo "venv missing; run: python -m venv venv && venv/bin/pip install -e '.[whispercpp,desktop]' mlx-lm 'transformers<5'"; exit 1; }

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

# ------------------------------------------------------------------ stub
# The main executable is a small compiled binary that runs the daemon
# in-process through libpython (see scripts/app_stub.c for why: TCC ties
# permissions to the main executable's identity - shell launchers that
# exec another binary leave the toggle ON but the process untrusted).
SITE_PACKAGES="$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
LIBPYTHON="$("$PY" -c 'import sysconfig; import os.path as p; print(p.join(sysconfig.get_config_var("LIBDIR"), "libpython3.13.dylib"))')"
[ -f "$LIBPYTHON" ] || { echo "libpython not found at $LIBPYTHON"; exit 1; }

if [ "$STANDALONE" = 1 ]; then
    # Copy the code and every dependency into the bundle: the app keeps
    # working if this repo moves or the venv is rebuilt. Data files inside
    # Resources/ are fine with codesign (unlike Contents/ or MacOS/).
    echo "  copying dependencies into the bundle (standalone)..."
    PYROOT="$APP/Contents/Resources/python"
    mkdir -p "$PYROOT"
    rsync -a --exclude "__pycache__" "$SITE_PACKAGES/" "$PYROOT/site-packages/"
    rsync -a --exclude "__pycache__" "$REPO/localflow" "$PYROOT/app/"
    STUB_PYTHONPATH="@RESOURCES@/python/site-packages:@RESOURCES@/python/app"
else
    STUB_PYTHONPATH="$SITE_PACKAGES:$REPO"
fi

clang -O2 -Wall \
    -DLIBPYTHON_PATH="\"$LIBPYTHON\"" \
    -DLOCALFLOW_PYTHONPATH="\"$STUB_PYTHONPATH\"" \
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

# ------------------------------------------------------------- smoke test
# The stub hardwires "-m localflow.cli run"; verify the import path works
# with the same interpreter + PYTHONPATH the stub will use.
PYTHONPATH="$SITE_PACKAGES:$REPO" "$PY" -c "import localflow, mlx_lm" \
    || { echo "interpreter can't import localflow - check venv"; exit 1; }

# ------------------------------------------------------------------- sign
# Ad-hoc signature gives the bundle a stable identity for the TCC
# permission prompts (Accessibility / Input Monitoring / Microphone).
codesign --force -s - "$APP/Contents/MacOS/LocalFlow" 2> /dev/null
codesign --force -s - "$APP" 2> /dev/null

echo "Built: $APP"

if [ "$INSTALL" = 1 ]; then
    rm -rf "/Applications/LocalFlow.app"
    cp -R "$APP" /Applications/
    echo "Installed: /Applications/LocalFlow.app"
fi
