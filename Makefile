CC = clang
CFLAGS = -fobjc-arc -Wall -O2
LIBS = -framework Foundation -framework IOKit -lobjc -ldl
METAL_LIBS = -framework Foundation -framework Metal -framework IOKit -lobjc -ldl
MPS_LIBS = -framework Foundation -framework Metal -framework MetalPerformanceShaders \
           -framework MetalPerformanceShadersGraph -framework IOKit -lobjc -ldl
ACC_LIBS = -framework Foundation -framework Accelerate -lobjc -ldl -DACCELERATE_NEW_LAPACK

.PHONY: all tools probes pocs benchmarks clean soc-map power bench

all: tools probes pocs benchmarks

# ═══ Discovery Tools ═══
tools: tools/framework_scanner tools/deep_probe tools/batch_scan tools/full_soc_map

tools/framework_scanner: tools/framework_scanner.m
	$(CC) $(CFLAGS) -o $@ $< $(LIBS)

tools/deep_probe: tools/deep_probe.m
	$(CC) $(CFLAGS) -o $@ $< $(LIBS)

tools/batch_scan: tools/batch_scan.m
	$(CC) $(CFLAGS) -o $@ $< $(LIBS)

tools/full_soc_map: tools/full_soc_map.m
	$(CC) $(CFLAGS) -o $@ $< $(LIBS)

# ═══ Hardware Probes ═══
probes: probes/clpc_probe probes/gpu_probe probes/intelligence_probe probes/soc_power

probes/clpc_probe: probes/clpc_probe.m
	$(CC) $(CFLAGS) -o $@ $< $(LIBS)

probes/gpu_probe: probes/gpu_probe.m
	$(CC) $(CFLAGS) -o $@ $< $(METAL_LIBS)

probes/intelligence_probe: probes/intelligence_probe.m
	$(CC) $(CFLAGS) -o $@ $< $(LIBS)

probes/soc_power: probes/soc_power.m
	$(CC) $(CFLAGS) -o $@ $< $(LIBS)

# ═══ Proof of Concepts ═══
pocs: pocs/metal_tensor_poc pocs/ane_direct pocs/gamemode_ctl pocs/mtl4ml_engine

pocs/metal_tensor_poc: pocs/metal_tensor_poc.m
	$(CC) $(CFLAGS) -o $@ $< $(METAL_LIBS)

pocs/ane_direct: pocs/ane_direct.m
	$(CC) $(CFLAGS) -o $@ $< $(LIBS) -framework IOSurface

pocs/gamemode_ctl: pocs/gamemode_ctl.m
	$(CC) $(CFLAGS) -o $@ $< $(LIBS)

pocs/mtl4ml_engine: pocs/mtl4ml_engine.m
	$(CC) $(CFLAGS) -o $@ $< $(MPS_LIBS) -DACCELERATE_NEW_LAPACK

# ═══ Benchmarks ═══
benchmarks: benchmarks/inference_engine benchmarks/capacity_test benchmarks/real_model_bench \
            benchmarks/advantage_test benchmarks/mps_quant_bench benchmarks/cpu_attn_bench \
            benchmarks/kvcache_compress

benchmarks/inference_engine: benchmarks/inference_engine.m
	$(CC) $(CFLAGS) -o $@ $< $(ACC_LIBS) -framework CoreML -framework IOSurface -framework Metal

benchmarks/capacity_test: benchmarks/capacity_test.m
	$(CC) $(CFLAGS) -o $@ $< $(ACC_LIBS) -framework Metal

benchmarks/real_model_bench: benchmarks/real_model_bench.m
	$(CC) $(CFLAGS) -o $@ $< $(MPS_LIBS) -framework Accelerate -DACCELERATE_NEW_LAPACK

benchmarks/advantage_test: benchmarks/advantage_test.m
	$(CC) $(CFLAGS) -o $@ $< $(METAL_LIBS) -framework IOSurface -framework MetalPerformanceShaders

benchmarks/mps_quant_bench: benchmarks/mps_quant_bench.m
	$(CC) $(CFLAGS) -o $@ $< $(MPS_LIBS)

benchmarks/cpu_attn_bench: benchmarks/cpu_attn_bench.m
	$(CC) $(CFLAGS) -o $@ $< $(ACC_LIBS)

benchmarks/kvcache_compress: benchmarks/kvcache_compress.m
	$(CC) $(CFLAGS) -o $@ $< $(ACC_LIBS)

# ═══ Quick Targets ═══
soc-map: tools/full_soc_map
	tools/full_soc_map

power: probes/soc_power
	probes/soc_power 5 1000

bench: benchmarks/real_model_bench
	benchmarks/real_model_bench

scan-all: tools/batch_scan
	tools/batch_scan all

clean:
	find tools probes pocs benchmarks -type f ! -name '*.m' -delete 2>/dev/null || true
