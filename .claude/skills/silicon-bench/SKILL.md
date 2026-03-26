---
name: silicon-bench
description: Benchmark Apple Silicon compute — CPU (SME2), GPU (Metal), model capacity analysis, and real transformer layer performance.
argument-hint: "[matrix_size]"
---

Run performance benchmarks on Apple Silicon.

## Quick compute benchmark (CPU vs GPU):
```bash
make benchmarks/inference_engine 2>/dev/null && benchmarks/inference_engine ${ARGS:-512}
```

## Model capacity analysis:
```bash
make benchmarks/capacity_test 2>/dev/null && benchmarks/capacity_test
```

## Real transformer architecture benchmark:
```bash
make benchmarks/real_model_bench 2>/dev/null && benchmarks/real_model_bench
```

Focus on practical answers: can model X fit? How fast? CPU or GPU? Note these are synthetic benchmarks — real performance depends on quantization and memory bandwidth.
