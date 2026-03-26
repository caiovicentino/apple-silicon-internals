---
name: silicon-shaders
description: Dump and analyze all Metal shader libraries (.metallib) on the system. Shows which frameworks use GPU compute, their sizes, and categorizes by function (ML, rendering, image processing, etc).
argument-hint: "[framework_filter]"
---

# /silicon-shaders

Map all 265+ Metal shader libraries on the system. Shows what GPU compute each framework uses.

## Full dump (or filtered by name):

```bash
FILTER="${ARGS}"

echo "=== Metal Shader Libraries ==="
find /System/Library -name "*.metallib" 2>/dev/null | while read lib; do
    size=$(ls -lh "$lib" 2>/dev/null | awk '{print $5}')
    fw=$(echo "$lib" | grep -oE '[^/]+\.framework' | head -1)
    dir=$(dirname "$lib" | sed 's|.*/Resources||;s|^/||')
    name=$(basename "$lib")
    line="[$size] $fw → $name"
    if [ -n "$FILTER" ]; then
        echo "$line" | grep -i "$FILTER" && true
    else
        echo "  $line"
    fi
done | sort -t'[' -k2 -rh

echo ""
echo "=== Total ==="
find /System/Library -name "*.metallib" 2>/dev/null | wc -l
echo "metallib files"
```

## Categorize the results:

- **ML/AI**: Espresso (CoreML GPU kernels), MPS (MatMul, attention, FFT), MLCompute
- **Rendering**: QuartzCore (compositing), RenderBox, ShaderGraph, SceneKit, SpriteKit
- **Image**: CoreImage (filters), HDRProcessing, VideoProcessing
- **3D/AR**: CompositorServices, ModelIO, RealityKit
- **Raytracing**: Metal BVH builder
- **Debug**: GPUToolsReplay

Highlight the most interesting ones — especially Espresso (28MB of ML kernels that CoreML uses internally) and any unexpected frameworks with GPU compute.

If the user asks about a specific metallib, try to find what shaders are inside:
```bash
# Metal shader function names (if the metallib has symbols)
xcrun metal-objdump --disassemble-symbols "$METALLIB_PATH" 2>/dev/null | grep '^_' | head -20
```
