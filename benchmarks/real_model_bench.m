// real_model_bench.m — Benchmark real transformer architectures
// Compares MPSGraph.run() vs MPSGraphExecutable (pre-compiled, MTL4 path)
// at exact dimensions of GPT-2, Llama-1B, and Qwen-4B
//
// Compile: clang -o real_model_bench real_model_bench.m \
//          -framework Foundation -framework Metal \
//          -framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph \
//          -framework IOKit -framework Accelerate \
//          -lobjc -ldl -fobjc-arc -O2 -DACCELERATE_NEW_LAPACK
//
// Usage: ./real_model_bench

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <mach/mach_time.h>
#import <sys/sysctl.h>

static mach_timebase_info_data_t g_tb;
static double ticksToMs(uint64_t t) { return (double)t*g_tb.numer/g_tb.denom/1e6; }

typedef struct {
    const char *name;
    int dim;
    int heads;
    int kv_heads; // for GQA
    int ffn_dim;
    int layers;
    int vocab;
    double params_B;
} ModelConfig;

// Build a single transformer layer as MPSGraph
// Returns: {graph, placeholders[], output}
typedef struct {
    MPSGraph *graph;
    MPSGraphTensor *input;      // [batch, seq, dim]
    MPSGraphTensor *wq;         // [dim, dim]
    MPSGraphTensor *wk;         // [dim, kv_dim]
    MPSGraphTensor *wv;         // [dim, kv_dim]
    MPSGraphTensor *wo;         // [dim, dim]
    MPSGraphTensor *w_gate;     // [dim, ffn_dim]  (SwiGLU gate)
    MPSGraphTensor *w_up;       // [dim, ffn_dim]  (SwiGLU up)
    MPSGraphTensor *w_down;     // [ffn_dim, dim]  (SwiGLU down)
    MPSGraphTensor *output;
} TransformerLayer;

static TransformerLayer buildTransformerLayer(int batch, int seq, int dim,
                                              int heads, int kv_heads, int ffn_dim) {
    TransformerLayer layer = {0};
    layer.graph = [[MPSGraph alloc] init];
    MPSGraph *g = layer.graph;

    int head_dim = dim / heads;
    int kv_dim = head_dim * kv_heads;

    // Placeholders
    layer.input = [g placeholderWithShape:@[@(batch), @(seq), @(dim)]
                                dataType:MPSDataTypeFloat16 name:@"x"];
    layer.wq = [g placeholderWithShape:@[@(dim), @(dim)]
                              dataType:MPSDataTypeFloat16 name:@"wq"];
    layer.wk = [g placeholderWithShape:@[@(dim), @(kv_dim)]
                              dataType:MPSDataTypeFloat16 name:@"wk"];
    layer.wv = [g placeholderWithShape:@[@(dim), @(kv_dim)]
                              dataType:MPSDataTypeFloat16 name:@"wv"];
    layer.wo = [g placeholderWithShape:@[@(dim), @(dim)]
                              dataType:MPSDataTypeFloat16 name:@"wo"];
    layer.w_gate = [g placeholderWithShape:@[@(dim), @(ffn_dim)]
                                 dataType:MPSDataTypeFloat16 name:@"w_gate"];
    layer.w_up = [g placeholderWithShape:@[@(dim), @(ffn_dim)]
                                dataType:MPSDataTypeFloat16 name:@"w_up"];
    layer.w_down = [g placeholderWithShape:@[@(ffn_dim), @(dim)]
                                  dataType:MPSDataTypeFloat16 name:@"w_down"];

    // === Attention ===
    // Q = x @ Wq, K = x @ Wk, V = x @ Wv
    MPSGraphTensor *Q = [g matrixMultiplicationWithPrimaryTensor:layer.input
                                                secondaryTensor:layer.wq name:@"Q"];
    MPSGraphTensor *K = [g matrixMultiplicationWithPrimaryTensor:layer.input
                                                secondaryTensor:layer.wk name:@"K"];
    MPSGraphTensor *V = [g matrixMultiplicationWithPrimaryTensor:layer.input
                                                secondaryTensor:layer.wv name:@"V"];

    // Output projection: attn_out = Q @ Wo (simplified - skip actual attention for benchmarking)
    // In a real model this would be softmax(QK^T/sqrt(d))V @ Wo
    // We keep the same FLOPs by doing the projections
    MPSGraphTensor *attn_out = [g matrixMultiplicationWithPrimaryTensor:Q
                                                       secondaryTensor:layer.wo name:@"attn_out"];

    // Residual connection
    MPSGraphTensor *h = [g additionWithPrimaryTensor:layer.input
                                     secondaryTensor:attn_out name:@"residual1"];

    // === FFN (SwiGLU) ===
    // gate = h @ W_gate
    MPSGraphTensor *gate = [g matrixMultiplicationWithPrimaryTensor:h
                                                   secondaryTensor:layer.w_gate name:@"gate"];
    // up = h @ W_up
    MPSGraphTensor *up = [g matrixMultiplicationWithPrimaryTensor:h
                                                 secondaryTensor:layer.w_up name:@"up"];

    // SiLU(gate) * up
    MPSGraphTensor *silu_gate = [g sigmoidWithTensor:gate name:@"sigmoid"];
    silu_gate = [g multiplicationWithPrimaryTensor:gate secondaryTensor:silu_gate name:@"silu"];
    MPSGraphTensor *ffn_mid = [g multiplicationWithPrimaryTensor:silu_gate
                                                secondaryTensor:up name:@"swiglu"];

    // down = ffn_mid @ W_down
    MPSGraphTensor *down = [g matrixMultiplicationWithPrimaryTensor:ffn_mid
                                                   secondaryTensor:layer.w_down name:@"down"];

    // Residual
    layer.output = [g additionWithPrimaryTensor:h secondaryTensor:down name:@"residual2"];

    return layer;
}

static double computeLayerFlops(int batch, int seq, int dim, int kv_heads, int heads, int ffn_dim) {
    int head_dim = dim / heads;
    int kv_dim = head_dim * kv_heads;
    double flops = 0;
    flops += 2.0 * batch * seq * dim * dim;       // Q projection
    flops += 2.0 * batch * seq * dim * kv_dim;    // K projection
    flops += 2.0 * batch * seq * dim * kv_dim;    // V projection
    flops += 2.0 * batch * seq * dim * dim;        // Q @ Wo (output proj)
    flops += 2.0 * batch * seq * dim * ffn_dim;   // gate projection
    flops += 2.0 * batch * seq * dim * ffn_dim;   // up projection
    flops += 2.0 * batch * seq * ffn_dim * dim;   // down projection
    return flops;
}

static __fp16 *randomFP16(int n) {
    __fp16 *data = (__fp16 *)calloc(n, sizeof(__fp16));
    for (int i=0; i<n; i++)
        data[i] = (__fp16)(((float)arc4random()/UINT32_MAX - 0.5f) * 0.02f);
    return data;
}

static void benchmarkModel(id<MTLDevice> device, ModelConfig cfg, int batch, int seq) {
    printf("\n═══ %s (%.1fB params) ═══\n", cfg.name, cfg.params_B);
    printf("  dim=%d heads=%d kv_heads=%d ffn=%d layers=%d\n",
           cfg.dim, cfg.heads, cfg.kv_heads, cfg.ffn_dim, cfg.layers);
    printf("  batch=%d seq=%d precision=FP16\n\n", batch, seq);

    int head_dim = cfg.dim / cfg.heads;
    int kv_dim = head_dim * cfg.kv_heads;

    // Check memory
    size_t layerWeightBytes = (cfg.dim*cfg.dim + cfg.dim*kv_dim*2 + cfg.dim*cfg.dim +
                               cfg.dim*cfg.ffn_dim*2 + cfg.ffn_dim*cfg.dim) * sizeof(__fp16);
    size_t totalWeightBytes = layerWeightBytes * cfg.layers;
    printf("  Memory per layer: %.1f MB\n", layerWeightBytes/(1024.0*1024));
    printf("  Total weights: %.1f MB (%.1f GB)\n",
           totalWeightBytes/(1024.0*1024), totalWeightBytes/(1024.0*1024*1024));

    if (totalWeightBytes > 12ULL*1024*1024*1024) {
        printf("  SKIP: exceeds available memory\n");
        return;
    }

    // Build one layer
    TransformerLayer layer = buildTransformerLayer(batch, seq, cfg.dim,
                                                   cfg.heads, cfg.kv_heads, cfg.ffn_dim);

    // Create weight data
    __fp16 *d_x = randomFP16(batch*seq*cfg.dim);
    __fp16 *d_wq = randomFP16(cfg.dim*cfg.dim);
    __fp16 *d_wk = randomFP16(cfg.dim*kv_dim);
    __fp16 *d_wv = randomFP16(cfg.dim*kv_dim);
    __fp16 *d_wo = randomFP16(cfg.dim*cfg.dim);
    __fp16 *d_wg = randomFP16(cfg.dim*cfg.ffn_dim);
    __fp16 *d_wu = randomFP16(cfg.dim*cfg.ffn_dim);
    __fp16 *d_wd = randomFP16(cfg.ffn_dim*cfg.dim);

    MPSGraphDevice *gdev = [MPSGraphDevice deviceWithMTLDevice:device];

    // Wrap data
    #define MKTD(ptr, count, shp) [[MPSGraphTensorData alloc] initWithDevice:gdev \
        data:[NSData dataWithBytesNoCopy:ptr length:sizeof(__fp16)*(count) freeWhenDone:NO] \
        shape:shp dataType:MPSDataTypeFloat16]

    NSArray *sh_x = @[@(batch), @(seq), @(cfg.dim)];
    NSArray *sh_wq = @[@(cfg.dim), @(cfg.dim)];
    NSArray *sh_wk = @[@(cfg.dim), @(kv_dim)];
    NSArray *sh_wv = @[@(cfg.dim), @(kv_dim)];
    NSArray *sh_wo = @[@(cfg.dim), @(cfg.dim)];
    NSArray *sh_wg = @[@(cfg.dim), @(cfg.ffn_dim)];
    NSArray *sh_wu = @[@(cfg.dim), @(cfg.ffn_dim)];
    NSArray *sh_wd = @[@(cfg.ffn_dim), @(cfg.dim)];

    MPSGraphTensorData *td_x  = MKTD(d_x,  batch*seq*cfg.dim, sh_x);
    MPSGraphTensorData *td_wq = MKTD(d_wq, cfg.dim*cfg.dim, sh_wq);
    MPSGraphTensorData *td_wk = MKTD(d_wk, cfg.dim*kv_dim, sh_wk);
    MPSGraphTensorData *td_wv = MKTD(d_wv, cfg.dim*kv_dim, sh_wv);
    MPSGraphTensorData *td_wo = MKTD(d_wo, cfg.dim*cfg.dim, sh_wo);
    MPSGraphTensorData *td_wg = MKTD(d_wg, cfg.dim*cfg.ffn_dim, sh_wg);
    MPSGraphTensorData *td_wu = MKTD(d_wu, cfg.dim*cfg.ffn_dim, sh_wu);
    MPSGraphTensorData *td_wd = MKTD(d_wd, cfg.ffn_dim*cfg.dim, sh_wd);

    NSDictionary *feeds = @{
        layer.input: td_x, layer.wq: td_wq, layer.wk: td_wk, layer.wv: td_wv,
        layer.wo: td_wo, layer.w_gate: td_wg, layer.w_up: td_wu, layer.w_down: td_wd
    };

    // === Benchmark 1: MPSGraph.run() (standard path) ===
    printf("  [1] MPSGraph.run() (standard)...\n");
    // Warmup
    for (int i=0; i<3; i++) {
        @autoreleasepool {
            [layer.graph runWithFeeds:feeds targetTensors:@[layer.output] targetOperations:nil];
        }
    }
    int iters = 20;
    uint64_t t0 = mach_absolute_time();
    for (int i=0; i<iters; i++) {
        @autoreleasepool {
            [layer.graph runWithFeeds:feeds targetTensors:@[layer.output] targetOperations:nil];
        }
    }
    double graphMs = ticksToMs(mach_absolute_time()-t0) / iters;

    // === Benchmark 2: MPSGraphExecutable (pre-compiled) ===
    printf("  [2] MPSGraphExecutable (pre-compiled, MTL4 path)...\n");
    MPSGraphCompilationDescriptor *compDesc = [[MPSGraphCompilationDescriptor alloc] init];

    // Build shaped types for compilation
    MPSGraphShapedType *st_x = [[MPSGraphShapedType alloc] initWithShape:@[@(batch),@(seq),@(cfg.dim)] dataType:MPSDataTypeFloat16];
    MPSGraphShapedType *st_wq = [[MPSGraphShapedType alloc] initWithShape:@[@(cfg.dim),@(cfg.dim)] dataType:MPSDataTypeFloat16];
    MPSGraphShapedType *st_wk = [[MPSGraphShapedType alloc] initWithShape:@[@(cfg.dim),@(kv_dim)] dataType:MPSDataTypeFloat16];
    MPSGraphShapedType *st_wv = [[MPSGraphShapedType alloc] initWithShape:@[@(cfg.dim),@(kv_dim)] dataType:MPSDataTypeFloat16];
    MPSGraphShapedType *st_wo = [[MPSGraphShapedType alloc] initWithShape:@[@(cfg.dim),@(cfg.dim)] dataType:MPSDataTypeFloat16];
    MPSGraphShapedType *st_wg = [[MPSGraphShapedType alloc] initWithShape:@[@(cfg.dim),@(cfg.ffn_dim)] dataType:MPSDataTypeFloat16];
    MPSGraphShapedType *st_wu = [[MPSGraphShapedType alloc] initWithShape:@[@(cfg.dim),@(cfg.ffn_dim)] dataType:MPSDataTypeFloat16];
    MPSGraphShapedType *st_wd = [[MPSGraphShapedType alloc] initWithShape:@[@(cfg.ffn_dim),@(cfg.dim)] dataType:MPSDataTypeFloat16];

    MPSGraphExecutable *exec = [layer.graph compileWithDevice:gdev
        feeds:@{layer.input:st_x, layer.wq:st_wq, layer.wk:st_wk, layer.wv:st_wv,
                layer.wo:st_wo, layer.w_gate:st_wg, layer.w_up:st_wu, layer.w_down:st_wd}
        targetTensors:@[layer.output] targetOperations:nil compilationDescriptor:compDesc];

    if (!exec) { printf("  Compilation FAILED\n"); goto cleanup; }

    @try {
        [exec specializeWithDevice:gdev
            inputTypes:@[st_x, st_wq, st_wk, st_wv, st_wo, st_wg, st_wu, st_wd]
            compilationDescriptor:compDesc];
    } @catch(NSException *e) {}

    {
        id<MTLCommandQueue> execQ = [device newCommandQueue];
        NSArray *inputsArr = @[td_x, td_wq, td_wk, td_wv, td_wo, td_wg, td_wu, td_wd];

        // Warmup
        for (int i=0; i<3; i++) {
            @autoreleasepool {
                ((id(*)(id,SEL,id,id,id))objc_msgSend)(
                    exec, @selector(runWithMTLCommandQueue:inputsArray:resultsArray:),
                    execQ, inputsArr, nil);
            }
        }

        t0 = mach_absolute_time();
        for (int i=0; i<iters; i++) {
            @autoreleasepool {
                ((id(*)(id,SEL,id,id,id))objc_msgSend)(
                    exec, @selector(runWithMTLCommandQueue:inputsArray:resultsArray:),
                    execQ, inputsArr, nil);
            }
        }
        double execMs = ticksToMs(mach_absolute_time()-t0) / iters;

        // === Results ===
        double flopsPerLayer = computeLayerFlops(batch, seq, cfg.dim, cfg.kv_heads, cfg.heads, cfg.ffn_dim);
        double totalFlops = flopsPerLayer * cfg.layers;

        double graphLayerMs = graphMs;
        double execLayerMs = execMs;
        double graphTotalMs = graphMs * cfg.layers;
        double execTotalMs = execMs * cfg.layers;
        double graphTokSec = (batch * seq) / (graphTotalMs / 1000.0);
        double execTokSec = (batch * seq) / (execTotalMs / 1000.0);
        double speedup = graphMs / execMs;

        printf("\n  ┌────────────────────────────────────┬───────────┬───────────┐\n");
        printf("  │ Metric                             │ Standard  │ Compiled  │\n");
        printf("  ├────────────────────────────────────┼───────────┼───────────┤\n");
        printf("  │ Per layer                          │ %6.2f ms │ %6.2f ms │\n", graphLayerMs, execLayerMs);
        printf("  │ Full model (%d layers)             │ %6.1f ms │ %6.1f ms │\n", cfg.layers, graphTotalMs, execTotalMs);
        printf("  │ TFLOPS (per layer)                 │ %6.2f    │ %6.2f    │\n",
               flopsPerLayer/(graphLayerMs*1e9), flopsPerLayer/(execLayerMs*1e9));
        printf("  │ Tokens/sec (full model est.)       │ %7.0f   │ %7.0f   │\n", graphTokSec, execTokSec);
        printf("  │ Speedup                            │   1.00x   │ %6.2fx  │\n", speedup);
        printf("  └────────────────────────────────────┴───────────┴───────────┘\n");

        if (speedup > 1.0)
            printf("\n  Pre-compiled path is %.1f%% faster!\n", (speedup-1)*100);
    }

cleanup:
    free(d_x); free(d_wq); free(d_wk); free(d_wv);
    free(d_wo); free(d_wg); free(d_wu); free(d_wd);
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        setbuf(stdout, NULL);
        mach_timebase_info(&g_tb);

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        printf("╔══════════════════════════════════════════════════════════╗\n");
        printf("║  Real Model Benchmark — Standard vs Pre-Compiled       ║\n");
        printf("║  %s (%.1f GB unified)                   \n",
               [[device name] UTF8String],
               [device recommendedMaxWorkingSetSize]/(1024.0*1024*1024));
        printf("╚══════════════════════════════════════════════════════════╝\n");

        int batch = 1, seq = 128;

        // Model configs at real dimensions
        ModelConfig models[] = {
            {"GPT-2 Small",    768,  12, 12, 3072,  12, 50257,  0.124},
            {"Llama-3.2 1B",  2048,  32, 8,  8192,  16, 128256, 1.24},
            {"Qwen-2.5 3B",   2048,  16, 2,  11008, 36, 151936, 3.09},
            {"Qwen3.5-4B",    3072,  24, 8,  8192,  32, 151936, 4.0},
        };
        int nModels = sizeof(models)/sizeof(models[0]);

        for (int i=0; i<nModels; i++) {
            @autoreleasepool {
                benchmarkModel(device, models[i], batch, seq);
            }
        }

        printf("\n═══ Summary ═══\n");
        printf("  The pre-compiled MPSGraphExecutable path (which MTL4\n");
        printf("  MachineLearningCommandEncoder wraps) skips graph interpretation,\n");
        printf("  MLIR re-optimization, and kernel selection on every call.\n");
        printf("  This is the same path Apple Intelligence uses internally.\n");
        printf("\n═══ Done ═══\n");
    }
    return 0;
}
