---
name: silicon-entitlements
description: Dump all entitlements of any binary, app, or daemon. Shows what private APIs and system resources it has access to.
argument-hint: "<path_to_app_or_binary>"
---

# /silicon-entitlements

Extract and analyze the entitlements (permissions) of any macOS binary or app bundle.

## Find and analyze

```bash
TARGET="${ARGS}"

# If it's an app name, find it
if [[ ! -f "$TARGET" ]] && [[ ! -d "$TARGET" ]]; then
    TARGET=$(find /Applications /System/Applications -maxdepth 1 -name "*${ARGS}*" -type d 2>/dev/null | head -1)
fi

echo "Target: $TARGET"

# Extract entitlements
codesign -d --entitlements :- "$TARGET" 2>/dev/null | grep -oE 'com\.apple\.[a-zA-Z0-9._-]+' | sort -u
```

## Categorize the entitlements

Group them by type:
- **Security**: `com.apple.security.*` — sandbox exceptions, file access, network
- **Private**: `com.apple.private.*` — internal Apple APIs, restricted access
- **Hardware**: `com.apple.ane.*`, `com.apple.iokit.*` — direct hardware access
- **Intelligence**: `com.apple.model*`, `com.apple.intelligence*`, `com.apple.generative*` — AI/ML
- **Data**: `com.apple.biome.*`, `com.apple.coreduet.*` — user data access

Highlight any unusual or powerful entitlements. Flag anything that grants broad system access.
