---
name: silicon-deepxray
description: Complete deep analysis of any installed app — reverse engineers all frameworks, libraries, entitlements, CoreML models, Metal shaders, XPC services, embedded binaries, URL schemes, privacy trackers, and undocumented API usage. The most thorough app analysis possible without disassembly.
argument-hint: "<AppName>"
---

# /silicon-deepxray

Perform the most comprehensive reverse engineering analysis possible on any installed macOS application. This goes far beyond `/silicon-xray` — it maps every binary, framework, model, shader, service, and data access pattern.

Run each section and compile into a full technical report.

## Phase 1: Identity & Signing

```bash
APP_NAME="${ARGS}"
APP_PATH=$(find /Applications /System/Applications -maxdepth 1 -name "*${APP_NAME}*" -type d 2>/dev/null | head -1)
echo "=== App Identity ==="
echo "Path: $APP_PATH"
echo "Bundle ID: $(defaults read "$APP_PATH/Contents/Info.plist" CFBundleIdentifier 2>/dev/null)"
echo "Version: $(defaults read "$APP_PATH/Contents/Info.plist" CFBundleShortVersionString 2>/dev/null)"
echo "Min macOS: $(defaults read "$APP_PATH/Contents/Info.plist" LSMinimumSystemVersion 2>/dev/null)"

echo ""
echo "=== Code Signing ==="
codesign -dvv "$APP_PATH" 2>&1 | head -20

echo ""
echo "=== Binary Architecture ==="
BINARY=$(find "$APP_PATH/Contents/MacOS" -type f -perm +111 | head -1)
file "$BINARY" 2>/dev/null
```

## Phase 2: Entitlements (complete dump)

```bash
echo "=== ALL Entitlements ==="
codesign -d --entitlements :- "$APP_PATH" 2>/dev/null | grep -oE 'com\.apple\.[a-zA-Z0-9._-]+' | sort -u
```

Categorize every entitlement found:
- **Sandbox**: com.apple.security.* — what sandbox exceptions does it have?
- **Hardware**: camera, microphone, bluetooth, USB, location, NFC
- **Data**: contacts, photos, calendar, health, keychain, file access paths
- **Network**: client, server, VPN, socket delegation
- **System**: JIT, automation, virtualization, kernel extensions
- **Private**: com.apple.private.* — internal Apple APIs (flag these prominently)
- **Intelligence**: model*, intelligence*, generative* — AI/ML access

## Phase 3: Linked Frameworks (system)

```bash
echo "=== System Frameworks ==="
otool -L "$BINARY" 2>/dev/null | grep '/System/' | sed 's/\t//;s/ (.*//' | sort

echo ""
echo "=== Private Frameworks ==="
otool -L "$BINARY" 2>/dev/null | grep 'PrivateFrameworks' | sed 's|.*/PrivateFrameworks/||;s|\.framework.*||' | sort
```

For each private framework found, run:
```bash
make tools/framework_scanner 2>/dev/null
tools/framework_scanner <FrameworkName> 2>&1 | grep -E '(CLASS:|Total)' | head -15
```

## Phase 4: Embedded Frameworks & Libraries

```bash
echo "=== Embedded Frameworks ==="
ls "$APP_PATH/Contents/Frameworks/" 2>/dev/null

echo ""
echo "=== Embedded dylibs ==="
find "$APP_PATH" -name "*.dylib" 2>/dev/null

echo ""
echo "=== Embedded Plugins ==="
find "$APP_PATH" -name "*.appex" -o -name "*.plugin" -o -name "*.bundle" 2>/dev/null

echo ""
echo "=== Helper Binaries ==="
find "$APP_PATH" -type f -perm +111 -not -name "*.dylib" 2>/dev/null
```

For each embedded framework, check if it links to private APIs:
```bash
for fw in "$APP_PATH"/Contents/Frameworks/*.framework; do
    name=$(basename "$fw" .framework)
    binary="$fw/$name"
    [ -f "$binary" ] || binary="$fw/Versions/A/$name"
    [ -f "$binary" ] || continue
    priv=$(otool -L "$binary" 2>/dev/null | grep -c 'PrivateFrameworks' || echo 0)
    if [ "$priv" -gt 0 ]; then
        echo "$name uses $priv private frameworks"
    fi
done
```

## Phase 5: CoreML Models

```bash
echo "=== Embedded CoreML Models ==="
find "$APP_PATH" -name "*.mlmodelc" -type d 2>/dev/null | while read model; do
    name=$(basename "$model" .mlmodelc)
    size=$(du -sh "$model" 2>/dev/null | awk '{print $1}')
    echo "  [$size] $name"
    # Try to read metadata
    meta="$model/metadata.json"
    if [ -f "$meta" ]; then
        python3 -c "
import json
m = json.load(open('$meta'))
if isinstance(m, list): m = m[0]
print(f'    Type: {m.get(\"modelType\",{}).get(\"name\",\"?\")}')
print(f'    Author: {m.get(\"author\",\"?\")}')
inputs = m.get('inputSchema',[])
for i in inputs[:3]:
    print(f'    Input: {i.get(\"name\",\"?\")}: {i.get(\"formattedType\",\"?\")}')
outputs = m.get('outputSchema',[])
for o in outputs[:3]:
    print(f'    Output: {o.get(\"name\",\"?\")}: {o.get(\"formattedType\",\"?\")}')
" 2>/dev/null
    fi
done
```

## Phase 6: Metal Shaders

```bash
echo "=== Metal Shader Libraries ==="
find "$APP_PATH" -name "*.metallib" 2>/dev/null | while read lib; do
    size=$(ls -lh "$lib" | awk '{print $5}')
    echo "  [$size] $(basename $lib)"
done

echo ""
echo "=== Metal Shader Source (if included) ==="
find "$APP_PATH" -name "*.metal" 2>/dev/null | head -10
```

## Phase 7: XPC Services & IPC

```bash
echo "=== XPC Services ==="
find "$APP_PATH" -name "*.xpc" -type d 2>/dev/null | while read xpc; do
    name=$(basename "$xpc" .xpc)
    echo "  $name"
    # Check its entitlements too
    binary="$xpc/Contents/MacOS/$name"
    if [ -f "$binary" ]; then
        codesign -d --entitlements :- "$binary" 2>/dev/null | grep -oE 'com\.apple\.[a-zA-Z0-9._-]+' | grep -iE '(private|security|network)' | while read ent; do
            echo "    $ent"
        done
    fi
done

echo ""
echo "=== Mach Services (from Info.plist) ==="
defaults read "$APP_PATH/Contents/Info.plist" 2>/dev/null | grep -A2 'MachServices' || echo "  None declared"

echo ""
echo "=== Launch Services ==="
defaults read "$APP_PATH/Contents/Info.plist" LSEnvironment 2>/dev/null || true
```

## Phase 8: URL Schemes & App Links

```bash
echo "=== URL Schemes ==="
defaults read "$APP_PATH/Contents/Info.plist" CFBundleURLTypes 2>/dev/null || echo "  None"

echo ""
echo "=== Associated Domains ==="
codesign -d --entitlements :- "$APP_PATH" 2>/dev/null | grep -oE 'com\.apple\.developer\.associated-domains[a-zA-Z0-9._-]*' || echo "  None"

echo ""
echo "=== Document Types ==="
defaults read "$APP_PATH/Contents/Info.plist" CFBundleDocumentTypes 2>/dev/null | head -20 || echo "  None"
```

## Phase 9: Privacy Manifest & Tracking

```bash
echo "=== Privacy Manifest ==="
find "$APP_PATH" -name "PrivacyInfo.xcprivacy" 2>/dev/null | while read pf; do
    echo "  Found: $pf"
    plutil -p "$pf" 2>/dev/null
done

echo ""
echo "=== Tracking Domains (embedded strings) ==="
strings "$BINARY" 2>/dev/null | grep -iE '(analytics|tracking|telemetry|amplitude|mixpanel|firebase|sentry|bugsnag|segment|appsflyer|adjust\.com|branch\.io|facebook.*sdk|google.*analytics|crashlytics)' | sort -u | head -20

echo ""
echo "=== API Keys & Endpoints (embedded strings) ==="
strings "$BINARY" 2>/dev/null | grep -iE '(api\..*\.com|https://.*api|\.amazonaws\.com|\.cloudfront\.net|\.azure\.|\.googleapis\.)' | sort -u | head -20
```

## Phase 10: Resources & Data Formats

```bash
echo "=== Interesting Resources ==="
find "$APP_PATH/Contents/Resources" -maxdepth 2 -type f 2>/dev/null | sed 's/.*\.//' | sort | uniq -c | sort -rn | head -15

echo ""
echo "=== Database Schemas (if SQLite present) ==="
find "$APP_PATH" -name "*.sqlite" -o -name "*.db" -o -name "*.sqlite3" 2>/dev/null | head -5

echo ""
echo "=== Configuration Files ==="
find "$APP_PATH" -name "*.json" -o -name "*.yaml" -o -name "*.yml" -o -name "*.toml" 2>/dev/null | grep -v 'node_modules' | head -10

echo ""
echo "=== Certificates & Keys ==="
find "$APP_PATH" -name "*.pem" -o -name "*.cer" -o -name "*.p12" -o -name "*.key" 2>/dev/null | head -5
```

## Phase 11: Runtime Capabilities Check

```bash
echo "=== App Extensions ==="
find "$APP_PATH" -name "*.appex" 2>/dev/null | while read ext; do
    name=$(basename "$ext" .appex)
    echo "  $name"
    plutil -p "$ext/Contents/Info.plist" 2>/dev/null | grep -i 'NSExtensionPointIdentifier' | head -1
done

echo ""
echo "=== Background Modes ==="
defaults read "$APP_PATH/Contents/Info.plist" UIBackgroundModes 2>/dev/null || echo "  None"

echo ""
echo "=== App Groups (shared data) ==="
codesign -d --entitlements :- "$APP_PATH" 2>/dev/null | grep -oE 'com\.apple\.security\.application-groups[^<]*' || echo "  None"
```

## How to present the final report

Write a comprehensive report with:

### Summary Card
- App name, version, bundle ID
- Architecture (native ARM64 / Universal / Intel via Rosetta)
- Code signing authority (Apple / Developer / Ad-hoc)

### Permission Matrix
Table of every permission with risk level (Low/Medium/High/Critical)

### Private API Usage
List every private framework with class count and notable classes found

### ML/AI Capabilities
Every CoreML model with architecture, size, and inferred purpose

### Network Profile
API endpoints, tracking SDKs, analytics services found in strings

### Data Access Map
What personal data categories the app can access

### Security Assessment
- Sandbox status and exceptions
- JIT usage
- Private API risk
- Tracking exposure
- Overall verdict: GREEN (clean) / YELLOW (minor concerns) / ORANGE (notable risk) / RED (significant concerns)

### Recommendations
What the user should be aware of when using this app
