---
name: silicon-watch
description: Live monitoring of Apple Silicon — stream CPU frequency, GPU thermals, power consumption, and memory bandwidth.
argument-hint: "[samples] [interval_ms]"
---

Start live monitoring of the SoC with real-time data.

```bash
make probes/soc_power 2>/dev/null && probes/soc_power ${ARGS:-5 1000}
```

First argument: number of samples (default 5). Second: interval in ms (default 1000).

Watch for: CPU energy rising (heavy CPU use), GPU energy (compute/rendering), DRAM bandwidth (memory-intensive), throttle counters (thermal throttling), P-state distribution (which frequency cores are running at).

Report trends and anomalies.
