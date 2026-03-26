---
name: silicon-scan
description: Scan any macOS private framework to discover undocumented classes, methods, properties via ObjC runtime introspection.
argument-hint: "<FrameworkName> [class_filter] OR <category>"
---

Scan a macOS private framework for undocumented APIs.

## Usage

Scan a specific framework:
```bash
cd /Users/caiovicentino/Desktop/apis && make tools/framework_scanner 2>/dev/null && tools/framework_scanner ${ARGS}
```

Scan by category (gpu, ml, perf, hw, compute, audio, all):
```bash
cd /Users/caiovicentino/Desktop/apis && make tools/batch_scan 2>/dev/null && tools/batch_scan ${ARGS}
```

Deep-probe a specific class (two arguments: Framework ClassName):
```bash
cd /Users/caiovicentino/Desktop/apis && make tools/deep_probe 2>/dev/null && tools/deep_probe ${ARGS}
```

Highlight factory methods, interesting properties, action methods, and protocols. Skip boilerplate.
