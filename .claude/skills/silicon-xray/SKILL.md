---
name: silicon-xray
description: X-ray any installed app — discover which private frameworks it uses, what entitlements it has, and scan its hidden APIs.
argument-hint: "<AppName>"
---

# /silicon-xray

Analyze any installed macOS application to discover its private framework usage, entitlements, and hidden APIs.

## Steps

### 1. Find the app binary
```bash
APP_NAME="${ARGS}"
# Search in Applications
APP_PATH=$(find /Applications /System/Applications -maxdepth 1 -name "*${APP_NAME}*" -type d 2>/dev/null | head -1)
[ -z "$APP_PATH" ] && echo "App not found: $APP_NAME" && exit 1
BINARY="$APP_PATH/Contents/MacOS/$(basename "$APP_PATH" .app)"
echo "App: $APP_PATH"
echo "Binary: $BINARY"
```

### 2. List linked frameworks (especially private ones)
```bash
echo "=== Linked Frameworks ==="
otool -L "$BINARY" 2>/dev/null | grep -v '^\t@rpath' | head -50
echo ""
echo "=== Private Frameworks ==="
otool -L "$BINARY" 2>/dev/null | grep 'PrivateFrameworks' | sed 's|.*/PrivateFrameworks/||;s|\.framework.*||' | sort
```

### 3. Read entitlements
```bash
echo "=== Entitlements ==="
codesign -d --entitlements :- "$APP_PATH" 2>/dev/null
```

### 4. Scan interesting private frameworks the app uses
For each interesting private framework found, run:
```bash
make tools/framework_scanner 2>/dev/null
tools/framework_scanner <FrameworkName> 2>&1 | grep -E '(CLASS:|Total)' | head -20
```

### 5. Check embedded frameworks inside the app bundle
```bash
ls "$APP_PATH/Contents/Frameworks/" 2>/dev/null | head -20
```

## How to present

Organize findings as:
- **App Identity**: name, binary, architecture
- **Entitlements**: what system permissions it requests (camera, location, JIT, etc.)
- **Private Frameworks**: list with counts, categorized (ML/AI, Security, Location, UI, etc.)
- **Interesting APIs**: scan 2-3 of the most unusual private frameworks and highlight notable classes
- **Security notes**: flag any unusual entitlements or suspicious framework usage
