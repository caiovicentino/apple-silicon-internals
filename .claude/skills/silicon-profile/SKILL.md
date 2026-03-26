---
name: silicon-profile
description: Profile the energy cost of a command. Measures power across all SoC subsystems before and during execution.
argument-hint: "<command to profile>"
---

Profile the energy cost of running a command on Apple Silicon.

## Steps

1. Build tools:
```bash
make probes/soc_power 2>/dev/null
```

2. Take a baseline sample (idle):
```bash
probes/soc_power 1 500 2>&1
```

3. Run the command while sampling energy in background:
```bash

probes/soc_power 100 200 > /tmp/silicon_profile_power.txt 2>&1 &
POWER_PID=$!
time ${ARGS}
kill $POWER_PID 2>/dev/null; wait $POWER_PID 2>/dev/null
```

4. Read energy data:
```bash
cat /tmp/silicon_profile_power.txt
```

Compare baseline vs during-execution. Report: total CPU energy (E-core vs P-core breakdown), GPU energy, ANE energy, DRAM energy, and which cores were most active.
