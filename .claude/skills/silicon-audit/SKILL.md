---
name: silicon-audit
description: Full privacy/security audit of an installed app — scans frameworks, entitlements, CoreML models, embedded binaries, and flags anything suspicious.
argument-hint: "<AppName>"
---

# /silicon-audit

Run a comprehensive privacy and security audit of any macOS application.

## Steps

### 1. Find the app and its binary
```bash
APP_NAME="${ARGS}"
APP_PATH=$(find /Applications /System/Applications -maxdepth 1 -name "*${APP_NAME}*" -type d 2>/dev/null | head -1)
echo "App: $APP_PATH"
```

### 2. Entitlements (what permissions does it request?)
```bash
codesign -d --entitlements :- "$APP_PATH" 2>/dev/null | grep -oE 'com\.apple\.[a-zA-Z0-9._-]+' | sort -u
```

### 3. Private framework usage
```bash
BINARY=$(find "$APP_PATH/Contents/MacOS" -type f -perm +111 | head -1)
otool -L "$BINARY" 2>/dev/null | grep 'PrivateFrameworks' | sed 's|.*/PrivateFrameworks/||;s|\.framework.*||' | sort
```

### 4. Embedded CoreML models (is it running ML locally?)
```bash
find "$APP_PATH" -name "*.mlmodelc" -type d 2>/dev/null
```

### 5. Embedded frameworks and libraries
```bash
ls "$APP_PATH/Contents/Frameworks/" 2>/dev/null
```

### 6. Network entitlements and tracking indicators
```bash
codesign -d --entitlements :- "$APP_PATH" 2>/dev/null | grep -oE 'com\.apple\.[a-zA-Z0-9._-]+' | grep -iE '(network|http|socket|cloud|icloud|push|remote|analytics|tracking|telemetry|advertising)'
```

### 7. Data access entitlements
```bash
codesign -d --entitlements :- "$APP_PATH" 2>/dev/null | grep -oE 'com\.apple\.[a-zA-Z0-9._-]+' | grep -iE '(contacts|photos|calendar|location|camera|microphone|bluetooth|health|biometric|keychain|file|usb)'
```

## How to present

Write an audit report with sections:
- **Identity**: app name, developer, architecture
- **Permissions**: categorized as Low/Medium/High risk
- **Private APIs**: flag any unusual private framework usage
- **ML Models**: what AI runs on-device
- **Data Access**: what personal data it can reach
- **Network**: what outbound access it has
- **Verdict**: overall privacy assessment (Green/Yellow/Red)

Flag anything unusual. Compare against what the app claims to need for its functionality.
