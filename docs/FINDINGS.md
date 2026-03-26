# macOS Private API Reverse Engineering — Findings & Working PoCs

Reverse engineering approach based on [maderix/ANE](https://github.com/maderix/ANE).
Target: Apple M4, macOS 15+

## Working PoCs

### 1. Metal Tensor API (WORKS)

**`metal_tensor_poc`** — Creates real GPU tensors via undocumented Metal API.

```
Device: Apple M4 (AGXG16GDevice)
supportsTensors: YES
supportsMachineLearningCommandEncoders: YES

MTLTensorDescriptor → set dimensions via MTLTensorExtents(rank:3, [4, 256, 256])
→ device.newTensorWithDescriptor:error: → AGXG16GFamilyTensor
   allocatedSize = 1,048,576 bytes
   dataType = Float32
   Supports: reshape, slice, view creation
```

Key discovered APIs:
- `MTLTensorDescriptor` — 27 methods, 10 properties (dimensions, strides, dataType, usage, storageMode)
- `MTLTensorExtents` — `initWithRank:values:` creates dimension/stride specs
- `[MTLDevice newTensorWithDescriptor:error:]` — Creates `AGXG16GFamilyTensor`
- `[MTLDevice newTensorWithBuffer:descriptor:offset:strides:error:]` — Tensor backed by existing buffer
- `[MTLDevice tensorSizeAndAlignWithDescriptor:]` — Query size/alignment
- `[MTLDevice supportsTensors]` / `[MTLDevice supportsMachineLearningCommandEncoders]`
- Tensor operations: `replaceSlice:withBytes:strides:`, `getBytes:strides:fromSlice:`, `newTensorViewWithReshapedDescriptor:error:`, `newTensorViewWithSlice:error:`

### 2. SoC Power Monitor via IOReport (WORKS)

**`soc_power`** — Real-time power monitoring of every M4 subsystem.

```
Energy Model: ECPU0-5 (6 E-cores), PCPU0-3 (4 P-cores)
              GPU, GPU SRAM, ISP, AVE, MSR, AMCC, DCS, DRAM, DISP, SOC_AON
GPU Stats: GTP Idle, FRG Idle, Channel utilization, submissions, throttle counters
```

Uses `libIOReport.dylib` (same as `powermetrics`) loaded via dlsym:
- `IOReportCopyChannelsInGroup()` — Subscribe to "Energy Model", "CPU Stats", "GPU Stats"
- `IOReportCreateSubscription()` → `IOReportCreateSamples()` → `IOReportCreateSamplesDelta()`
- Per-core energy in nanojoules, converted to mW
- All 10+ SoC power domains visible

### 3. CLPC Game Mode Controller (COMPILES, needs entitlements)

**`gamemode_ctl`** — Toggle Game Mode via private CLPC API.

Discovered methods on `CLPCPolicyClient`:
- `setGameMode:options:error:` — Enable/disable Game Mode
- `isLowPowerModeCandidate:error:` — Query LPM eligibility
- `setCLPCTrialID:error:` — A/B test experiments

Requires `com.apple.private.clpc` entitlement for full access.

### 4. ANE IOSurface I/O (WORKS)

**`ane_direct`** — Demonstrates zero-copy ANE I/O path.

- `_ANEIOSurfaceObject` wraps `IOSurfaceRef` for ANE data transfer
- `_ANEInMemoryModelDescriptor.modelWithMILText:weights:optionsPlist:` — Creates model descriptor
- `_ANEInMemoryModel.compileWithQoS:options:error:` — Reaches the ANE compiler
- MIL compilation requires CoreML-generated MIL format (not freeform text)

## Key Discoveries

### GPU Architecture (M4)

```
AGXG16GDevice → AGXG16GFamilyDevice → IOGPUMetalDevice → _MTLDevice → NSObject
Instance size: 1,040 bytes, 35 ivars, 92 methods, 23 properties
```

Internal ivars: `_deviceRef`, `_acceleratorPort`, `_textureRam`, `_videoRam`,
`_sharedMemorySize`, `_accelID`, `_peerGroupID`, `_maxTransferRate`

### Metal 4 API Surface (93 classes)

- `IOGPUMetal4MachineLearningCommandEncoder` — Dedicated ML command encoder
- `MTL4MachineLearningPipelineDescriptor` / `_MTL4MachineLearningPipelineState`
- `IOGPUMetal4CommandQueue`, `IOGPUMetal4CommandBuffer`, `IOGPUMetal4CommandAllocator`
- `IOGPUMetal4ComputeCommandEncoder`, `IOGPUMetal4RenderCommandEncoder`
- Full compiler pipeline: `_MTL4Compiler`, `_MTL4CompilerTask`, `MTL4CompilerDescriptor`
- Archives: `_MTL4Archive`, `MTL4PipelineDataSetSerializer`

### CLPC Performance Controller

`CLPCReportingClient` (1,208 bytes) — Direct SoC telemetry:
- `num_cpu_clusters`, `num_ane_clusters`, `num_package_zones`, `num_cpu_cores`
- 11 schema types for different metric categories
- IOReport subscription-based sampling (`readStats:`, `readDeltaStats:`)
- Requires kernel driver connection (`agx_service`)

### PerfPower Metrics (41 system + 39 per-process)

System: `cpuPower`, `gpuPower`, `anePower`, `dramPower`, `displayPower`, `wifiPower`,
        `batteryTemperature`, `skinTemperature`, `thermalPressure`, `displayFPS`,
        `edrHeadroom`, `cpuEnergy`, `gpuEnergy`, `gpuSRAMEnergy`, `aneEnergy`,
        `dramBytes`, `aneDCSBytes`, `aneFabricBytes`

Per-process: `cpuCost`, `gpuCost`, `gpuTime`, `aneEnergy`, `aneTime`,
             `cpuInstructions`, `bytesRead/Written`, QoS breakdown (7 levels),
             `displayPower`, `networkCost`, `wifiIn/Out`

### Apple Intelligence Infrastructure

- `PrivateMLClient.PrivateMLClient` — Main PCC inference client
- `PrivateCloudCompute.TC2Client` — Trusted Cloud Compute v2
- `Apple_Cloudml_Inference_Tie_GenerateRequest` — Inference request protobuf
- `GDComputeOrchestration` — Compute orchestration with view management

### NeuralNetworks Framework

Training primitives: `SoftmaxCrossEntropyOperation`, `BatchNormOperation`,
`BackpropQueue`, `GradientAccumulator`, `LazyTensorFunctionBuilder`,
`EspressoTensorStorage`, `ExecutionContext`

## Tools

```bash
make all                             # Build everything

# Discovery
./framework_scanner <Name> [filter]  # Scan any framework
./deep_probe <Framework> <Class>     # Deep introspection
./batch_scan <gpu|ml|perf|hw|all>    # Batch scan by category

# Probes
./gpu_probe                          # GPU internals, Metal 4, tensors
./clpc_probe                         # CLPC performance controller
./intelligence_probe                 # Apple Intelligence APIs

# Working PoCs
./metal_tensor_poc                   # Create GPU tensors via private API
./soc_power [samples] [interval_ms]  # Real-time SoC power monitoring
./gamemode_ctl <status|on|off>       # Game Mode control (needs entitlements)
./ane_direct                         # ANE direct access demo
```

## Methodology

Same runtime introspection as the ANE repo:
1. `dlopen()` private frameworks from `/System/Library/PrivateFrameworks/`
2. `objc_copyClassList()` + `class_getImageName()` to find classes per framework
3. `class_copyMethodList()` / `class_copyPropertyList()` / `class_copyIvarList()`
4. `objc_msgSend()` dynamic dispatch to call discovered methods
5. `dlsym()` for C functions in private libraries (`libIOReport.dylib`)
6. `IOSurfaceCreate()` for zero-copy I/O (ANE pattern)

## Statistics

- 2,152 private frameworks scanned on macOS
- 657+ classes discovered across ML/Intelligence frameworks
- 93 Metal 4 classes found in runtime
- 47 IOGPU classes with full method dumps
- 34 ANE classes fully enumerated
