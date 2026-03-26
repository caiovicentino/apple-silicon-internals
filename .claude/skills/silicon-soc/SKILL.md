---
name: silicon-soc
description: Show complete Apple Silicon SoC hardware map — CPU cores, GPU class, memory, feature flags, IOReport channel count.
---

Display a complete hardware map of the current Apple Silicon SoC.

Run all of these and compile into a hardware spec sheet:

```bash
# SoC and CPU
sysctl machdep.cpu.brand_string hw.nperflevels hw.physicalcpu hw.memsize
for i in 0 1 2; do sysctl hw.perflevel${i}.name hw.perflevel${i}.physicalcpu hw.perflevel${i}.l2cachesize 2>/dev/null; done

# Feature flags
sysctl hw.optional.arm.FEAT_SME hw.optional.arm.FEAT_SME2 hw.optional.arm.FEAT_BF16 hw.optional.arm.FEAT_I8MM hw.optional.arm.sme_max_svl_b 2>/dev/null

# GPU
make probes/gpu_probe 2>/dev/null && probes/gpu_probe 2>&1 | head -40

# IOReport channels
make tools/full_soc_map 2>/dev/null && tools/full_soc_map 2>&1 | grep -E '(═══|TOTAL)'

# Platform ID
ioreg -l -w0 2>/dev/null | grep -E '"(compatible|platform-name|model)"' | head -5
```

Present as an organized spec sheet: SoC identity, CPU layout, GPU details, memory, features, telemetry coverage.
