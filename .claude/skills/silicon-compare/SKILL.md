---
name: silicon-compare
description: Side-by-side comparison of two apps — diff entitlements, frameworks, CoreML models, Metal shaders, permissions, and attack surface. Useful for comparing competing apps or versions.
argument-hint: "<App1> <App2>"
---

# /silicon-compare

Compare two macOS applications side-by-side across every dimension.

Parse the arguments — expect two app names separated by space.

## Step 1: Find both apps

```bash
ARGS_STR="${ARGS}"
APP1_NAME=$(echo "$ARGS_STR" | awk '{print $1}')
APP2_NAME=$(echo "$ARGS_STR" | awk '{$1=""; print $0}' | sed 's/^ //')
APP1=$(find /Applications /System/Applications -maxdepth 1 -name "*${APP1_NAME}*" -type d 2>/dev/null | head -1)
APP2=$(find /Applications /System/Applications -maxdepth 1 -name "*${APP2_NAME}*" -type d 2>/dev/null | head -1)
echo "App 1: $APP1"
echo "App 2: $APP2"
```

## Step 2: Identity comparison

```bash
for APP in "$APP1" "$APP2"; do
    echo "=== $(basename "$APP" .app) ==="
    echo "  Bundle: $(defaults read "$APP/Contents/Info.plist" CFBundleIdentifier 2>/dev/null)"
    echo "  Version: $(defaults read "$APP/Contents/Info.plist" CFBundleShortVersionString 2>/dev/null)"
    BINARY=$(find "$APP/Contents/MacOS" -type f -perm +111 2>/dev/null | head -1)
    echo "  Arch: $(file "$BINARY" 2>/dev/null | grep -oE 'x86_64|arm64|universal' | sort -u | tr '\n' ' ')"
    echo "  Signing: $(codesign -dvv "$APP" 2>&1 | grep 'Authority=' | head -1 | sed 's/Authority=//')"
done
```

## Step 3: Entitlements diff

```bash
echo "=== Entitlements ==="
ENT1=$(codesign -d --entitlements :- "$APP1" 2>/dev/null | grep -oE 'com\.apple\.[a-zA-Z0-9._-]+' | sort -u)
ENT2=$(codesign -d --entitlements :- "$APP2" 2>/dev/null | grep -oE 'com\.apple\.[a-zA-Z0-9._-]+' | sort -u)

echo ""
echo "Only in $(basename "$APP1" .app):"
comm -23 <(echo "$ENT1") <(echo "$ENT2")

echo ""
echo "Only in $(basename "$APP2" .app):"
comm -13 <(echo "$ENT1") <(echo "$ENT2")

echo ""
echo "Shared:"
comm -12 <(echo "$ENT1") <(echo "$ENT2")
```

## Step 4: Framework diff

```bash
echo "=== Frameworks ==="
for APP in "$APP1" "$APP2"; do
    BINARY=$(find "$APP/Contents/MacOS" -type f -perm +111 2>/dev/null | head -1)
    # Also check main embedded framework
    FWBIN=$(find "$APP/Contents/Frameworks" -name "*.framework" -maxdepth 1 2>/dev/null | head -1)
    if [ -n "$FWBIN" ]; then
        name=$(basename "$FWBIN" .framework)
        FWBIN="$FWBIN/$name"
        [ -f "$FWBIN" ] || FWBIN="$(dirname "$FWBIN")/Versions/A/$name"
    fi

    echo ""
    echo "$(basename "$APP" .app):"
    priv=$(otool -L "$BINARY" "$FWBIN" 2>/dev/null | grep -c 'PrivateFrameworks' || echo 0)
    pub=$(otool -L "$BINARY" "$FWBIN" 2>/dev/null | grep '/System/Library/Frameworks' | grep -v Private | wc -l || echo 0)
    echo "  Public frameworks: $pub"
    echo "  Private frameworks: $priv"

    if [ "$priv" -gt 0 ]; then
        echo "  Private list:"
        otool -L "$BINARY" "$FWBIN" 2>/dev/null | grep 'PrivateFrameworks' | sed 's|.*/PrivateFrameworks/||;s|\.framework.*||' | sort -u | while read fw; do
            echo "    $fw"
        done
    fi
done
```

## Step 5: Embedded content comparison

```bash
echo "=== Embedded Content ==="
for APP in "$APP1" "$APP2"; do
    name=$(basename "$APP" .app)
    fwcount=$(ls "$APP/Contents/Frameworks/" 2>/dev/null | wc -l)
    models=$(find "$APP" -name "*.mlmodelc" -type d 2>/dev/null | wc -l)
    shaders=$(find "$APP" -name "*.metallib" 2>/dev/null | wc -l)
    xpc=$(find "$APP" -name "*.xpc" -type d 2>/dev/null | wc -l)
    extensions=$(find "$APP" -name "*.appex" 2>/dev/null | wc -l)
    dylibs=$(find "$APP" -name "*.dylib" 2>/dev/null | wc -l)

    echo ""
    echo "  $name:"
    echo "    Frameworks:  $fwcount"
    echo "    CoreML models: $models"
    echo "    Metal shaders: $shaders"
    echo "    XPC services:  $xpc"
    echo "    App extensions: $extensions"
    echo "    Dylibs:        $dylibs"
done
```

## Step 6: Data access comparison

```bash
echo "=== Data Access ==="
CATEGORIES="camera|microphone|audio-input|location|contacts|photos|calendar|bluetooth|usb|health|biometric|keychain"
for APP in "$APP1" "$APP2"; do
    name=$(basename "$APP" .app)
    echo ""
    echo "  $name:"
    codesign -d --entitlements :- "$APP" 2>/dev/null | grep -oE 'com\.apple\.[a-zA-Z0-9._-]+' | grep -iE "$CATEGORIES" | while read ent; do
        echo "    $ent"
    done
done
```

## Step 7: Tracking/Analytics comparison

```bash
echo "=== Tracking & Analytics ==="
for APP in "$APP1" "$APP2"; do
    name=$(basename "$APP" .app)
    BINARY=$(find "$APP/Contents/MacOS" -type f -perm +111 2>/dev/null | head -1)
    trackers=$(strings "$BINARY" 2>/dev/null | grep -ciE '(analytics|tracking|telemetry|amplitude|mixpanel|firebase|sentry|segment|bugsnag|crashlytics|appsflyer)' || echo 0)
    echo "  $name: $trackers tracking-related strings"
done
```

## How to present

Create a side-by-side comparison table:

| Dimension | App 1 | App 2 |
|-----------|-------|-------|
| Version | x.x.x | y.y.y |
| Architecture | ... | ... |
| Entitlements | N | N |
| Private frameworks | N | N |
| CoreML models | N | N |
| Metal shaders | N | N |
| Data access | list | list |
| Tracking strings | N | N |

Then highlight:
- **Unique entitlements** per app (what one requests that the other doesn't)
- **Privacy winner** (which app requests less access)
- **ML capabilities** (which uses more on-device AI)
- **Attack surface** (which has more private API usage)
- **Verdict**: which app is more privacy-friendly and why
