---
name: silicon-power
description: Read real-time power consumption of every Apple Silicon subsystem (CPU cores, GPU, ANE, DRAM, Display, PCIe). No sudo required.
argument-hint: "[samples] [interval_ms]"
---

Run the SoC power monitor to read energy data from all subsystems via IOReport.

Build if needed, then run:
```bash
make probes/soc_power 2>/dev/null && probes/soc_power ${ARGS:-3 1000}
```

The first argument is number of samples (default 3), second is interval in ms (default 1000).

## How to interpret the output

- **Energy Model**: per-core energy in mJ. ECPU = efficiency cores, PCPU = performance cores.
- **GPU Stats**: GPU utilization. GTP Idle / FRG Idle = idle cycles.
- **CPU Stats**: P-state residency. IDLE = sleeping. V0P7 = lowest freq, V7P0 = highest.
- **AMC Stats**: Memory bandwidth in bytes per subsystem (CPU, GPU, ANE, Display).

Report what each subsystem is doing and highlight anything unusual.
