# llama.cpp Issue Draft

**Title**: Metal tensor API test results on M4 / macOS 26 Tahoe

**Labels**: `metal`, `apple-silicon`, `testing`

---

**Hardware**: Mac mini M4 (Mac16,10), 16 GB, macOS 26.3.1 Tahoe (Build 25D771280a)

**Model**: Qwen3.5-4B Q8_0 (4.16 GiB)

**Build**: b8500-342d6125b (Homebrew ggml 0.9.8)

## Summary

Tested `GGML_METAL_TENSOR_ENABLE=1` on M4 / macOS 26 Tahoe to see if the tensor API provides any benefit on this chip. The gating behind M5/A19 appears correct — no performance gain, slight regression on prompt processing.

Sharing the data in case it's useful for future M5 testing or macOS 26 compatibility.

## Results

### Baseline (default, tensor API disabled)

```
ggml_metal_device_init: tensor API disabled for pre-M5 and pre-A19 devices
ggml_metal_device_init: has tensor            = false
ggml_metal_device_init: GPU family: MTLGPUFamilyMetal4  (5002)

| qwen35 4B Q8_0 | 4.16 GiB | 4.21 B | BLAS,MTL | 10 | pp512  | 404.39 ± 1.41 |
| qwen35 4B Q8_0 | 4.16 GiB | 4.21 B | BLAS,MTL | 10 | tg128  |  18.91 ± 0.15 |
```

### With `GGML_METAL_TENSOR_ENABLE=1`

```
ggml_metal_device_init: testing tensor API for f16 support
ggml_metal_device_init: testing tensor API for bfloat support
ggml_metal_device_init: has tensor            = true
ggml_metal_library_init: loaded in 6.344 sec   (vs 0.024 sec without — shader recompilation)

| qwen35 4B Q8_0 | 4.16 GiB | 4.21 B | BLAS,MTL | 10 | pp512  | 383.65 ± 5.28 |
| qwen35 4B Q8_0 | 4.16 GiB | 4.21 B | BLAS,MTL | 10 | tg128  |  19.09 ± 0.25 |
```

### Comparison

| Test | Default | Tensor ON | Delta |
|------|---------|-----------|-------|
| pp512 | 404.4 tok/s | 383.7 tok/s | **-5.1%** |
| tg128 | 18.91 tok/s | 19.09 tok/s | **+1.0%** (within noise) |
| Shader load | 0.024s | 6.344s | 264x slower (recompilation) |

## Notes

- The tensor API compiles and runs without errors on M4 / macOS 26
- f16 and bfloat tensor support both test successfully
- `MTLGPUFamilyMetal4 (5002)` is reported
- The 6.3s shader load penalty on first run is significant
- pp regression may be from non-optimized tensor kernel paths for this GPU generation
- tg is unchanged (memory-bandwidth-bound, not compute-bound)

## Conclusion

The M5/A19 gating decision in `ggml_metal_device_init` appears correct for current performance. The tensor API works on M4 but doesn't help. This data may be useful as a baseline when M5 hardware becomes available.

## Additional macOS 26 observations

- `MTLGPUFamilyMetal4 (5002)` is now reported (may be new to macOS 26)
- `[MTLDevice supportsTensors]` returns YES on M4 via ObjC runtime
- `[MTLDevice supportsMachineLearningCommandEncoders]` returns YES
- 93 MTL4 classes are loaded in the runtime
- `MTLTensorDescriptor` / `MTLTensorExtents` are functional and can create `AGXG16GFamilyTensor` objects
