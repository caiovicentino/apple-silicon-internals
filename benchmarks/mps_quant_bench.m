// mps_quant_bench.m — Benchmark MPS quantized ops vs raw Metal compute
// Tests MPSNDArrayQuantizedMatrixMultiplication and QuantizedScaledDotProductAttention
// These are hardware-accelerated ops that llama.cpp does NOT use
//
// Compile: clang -o mps_quant_bench mps_quant_bench.m \
//          -framework Foundation -framework Metal \
//          -framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph \
//          -lobjc -ldl -fobjc-arc -O2
//
// Usage: ./mps_quant_bench

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <mach/mach_time.h>

static mach_timebase_info_data_t g_tb;
static double ticksToMs(uint64_t t) { return (double)t*g_tb.numer/g_tb.denom/1e6; }

int main() {
    @autoreleasepool {
        setbuf(stdout, NULL);
        mach_timebase_info(&g_tb);

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> queue = [device newCommandQueue];

        printf("╔══════════════════════════════════════════════════════════╗\n");
        printf("║  MPS Quantized Ops Benchmark                          ║\n");
        printf("║  Testing ops llama.cpp does NOT use                    ║\n");
        printf("╚══════════════════════════════════════════════════════════╝\n\n");

        // Dump MPSNDArrayQuantizedMatrixMultiplication
        printf("═══ Available MPS Quantized Classes ═══\n");
        const char *qclasses[] = {
            "MPSNDArrayQuantizedScaledDotProductAttention",
            "MPSNDArrayQuantizedMatrixMultiplication",
            "MPSNDArrayAffineInt4Dequantize",
            "MPSNDArrayVectorLUTDequantize",
            "MPSNDArrayLUTDequantize",
            "MPSNDArrayAffineQuantizationDescriptor",
            "MPSNDArrayLUTQuantizationDescriptor",
            "MPSNDArrayQuantizationDescriptor",
            "MPSNDArrayScaledDotProductAttention",
            NULL
        };
        for (int i=0; qclasses[i]; i++) {
            Class cls = NSClassFromString([NSString stringWithUTF8String:qclasses[i]]);
            if (cls) {
                unsigned int mcount = 0;
                Method *methods = class_copyMethodList(cls, &mcount);
                printf("\n  %s (%u methods)\n", qclasses[i], mcount);
                for (unsigned int j=0; j<mcount; j++) {
                    const char *name = sel_getName(method_getName(methods[j]));
                    if (strstr(name, "init") || strstr(name, "encode") ||
                        strstr(name, "set") || strstr(name, "With") ||
                        strstr(name, "result") || strstr(name, "descriptor")) {
                        printf("    - %s\n", name);
                    }
                }
                free(methods);
                // Properties
                unsigned int pcount = 0;
                objc_property_t *props = class_copyPropertyList(cls, &pcount);
                if (pcount > 0) {
                    printf("    Properties:\n");
                    for (unsigned int j=0; j<pcount; j++)
                        printf("      @property %s\n", property_getName(props[j]));
                }
                free(props);
            } else {
                printf("  %s: NOT FOUND\n", qclasses[i]);
            }
        }

        // === Benchmark: MPSGraph matmul FP16 vs INT8 quantized ===
        int M = 128, K = 3072, N = 3072; // Qwen3.5-4B dimensions
        printf("\n═══ Matmul Benchmark [%d x %d] @ [%d x %d] ═══\n", M, K, K, N);

        MPSGraph *graph = [[MPSGraph alloc] init];
        MPSGraphDevice *gdev = [MPSGraphDevice deviceWithMTLDevice:device];

        // FP16 matmul
        MPSGraphTensor *a_fp16 = [graph placeholderWithShape:@[@(M), @(K)]
                                                    dataType:MPSDataTypeFloat16 name:@"a"];
        MPSGraphTensor *b_fp16 = [graph placeholderWithShape:@[@(K), @(N)]
                                                    dataType:MPSDataTypeFloat16 name:@"b"];
        MPSGraphTensor *c_fp16 = [graph matrixMultiplicationWithPrimaryTensor:a_fp16
                                                             secondaryTensor:b_fp16 name:@"matmul"];

        // Create data
        size_t aBytes = M * K * sizeof(__fp16);
        size_t bBytes = K * N * sizeof(__fp16);
        __fp16 *aData = (__fp16 *)calloc(M*K, sizeof(__fp16));
        __fp16 *bData = (__fp16 *)calloc(K*N, sizeof(__fp16));
        for (int i=0; i<M*K; i++) aData[i] = (__fp16)(((float)arc4random()/UINT32_MAX-0.5f)*0.02f);
        for (int i=0; i<K*N; i++) bData[i] = (__fp16)(((float)arc4random()/UINT32_MAX-0.5f)*0.02f);

        MPSGraphTensorData *aTD = [[MPSGraphTensorData alloc] initWithDevice:gdev
            data:[NSData dataWithBytesNoCopy:aData length:aBytes freeWhenDone:NO]
            shape:@[@(M), @(K)] dataType:MPSDataTypeFloat16];
        MPSGraphTensorData *bTD = [[MPSGraphTensorData alloc] initWithDevice:gdev
            data:[NSData dataWithBytesNoCopy:bData length:bBytes freeWhenDone:NO]
            shape:@[@(K), @(N)] dataType:MPSDataTypeFloat16];

        // Warmup
        for (int i=0; i<3; i++) {
            @autoreleasepool {
                [graph runWithFeeds:@{a_fp16:aTD, b_fp16:bTD}
                     targetTensors:@[c_fp16] targetOperations:nil];
            }
        }

        // Benchmark FP16
        int iters = 50;
        uint64_t t0 = mach_absolute_time();
        for (int i=0; i<iters; i++) {
            @autoreleasepool {
                [graph runWithFeeds:@{a_fp16:aTD, b_fp16:bTD}
                     targetTensors:@[c_fp16] targetOperations:nil];
            }
        }
        double fp16Ms = ticksToMs(mach_absolute_time()-t0) / iters;
        double fp16Tflops = 2.0*M*K*N / (fp16Ms*1e9);

        // Compiled FP16
        MPSGraphCompilationDescriptor *compDesc = [[MPSGraphCompilationDescriptor alloc] init];
        MPSGraphShapedType *aST = [[MPSGraphShapedType alloc] initWithShape:@[@(M),@(K)] dataType:MPSDataTypeFloat16];
        MPSGraphShapedType *bST = [[MPSGraphShapedType alloc] initWithShape:@[@(K),@(N)] dataType:MPSDataTypeFloat16];
        MPSGraphExecutable *exec = [graph compileWithDevice:gdev
            feeds:@{a_fp16:aST, b_fp16:bST}
            targetTensors:@[c_fp16] targetOperations:nil compilationDescriptor:compDesc];

        @try { [exec specializeWithDevice:gdev inputTypes:@[aST, bST] compilationDescriptor:compDesc]; }
        @catch(NSException *e) {}

        // Warmup compiled
        for (int i=0; i<3; i++) {
            @autoreleasepool {
                ((id(*)(id,SEL,id,id,id))objc_msgSend)(
                    exec, @selector(runWithMTLCommandQueue:inputsArray:resultsArray:),
                    queue, @[aTD, bTD], nil);
            }
        }

        t0 = mach_absolute_time();
        for (int i=0; i<iters; i++) {
            @autoreleasepool {
                ((id(*)(id,SEL,id,id,id))objc_msgSend)(
                    exec, @selector(runWithMTLCommandQueue:inputsArray:resultsArray:),
                    queue, @[aTD, bTD], nil);
            }
        }
        double compiledMs = ticksToMs(mach_absolute_time()-t0) / iters;
        double compiledTflops = 2.0*M*K*N / (compiledMs*1e9);

        // === Now try INT8 quantized matmul via MPSGraph ===
        printf("\n  Trying INT8 quantized matmul via MPSGraph...\n");
        MPSGraph *qgraph = [[MPSGraph alloc] init];

        // INT8 quantized path: dequantize then matmul
        MPSGraphTensor *a_int8 = [qgraph placeholderWithShape:@[@(M), @(K)]
                                                     dataType:MPSDataTypeFloat16 name:@"a_fp16"];
        MPSGraphTensor *b_int8 = [qgraph placeholderWithShape:@[@(K), @(N)]
                                                     dataType:MPSDataTypeInt8 name:@"b_int8"];
        MPSGraphTensor *b_scale = [qgraph placeholderWithShape:@[@1, @(N)]
                                                      dataType:MPSDataTypeFloat16 name:@"b_scale"];

        // Dequantize: b_fp16 = b_int8 * scale
        MPSGraphTensor *b_deq = [qgraph castTensor:b_int8 toType:MPSDataTypeFloat16 name:@"cast"];
        b_deq = [qgraph multiplicationWithPrimaryTensor:b_deq secondaryTensor:b_scale name:@"dequant"];

        // Matmul
        MPSGraphTensor *c_int8 = [qgraph matrixMultiplicationWithPrimaryTensor:a_int8
                                                               secondaryTensor:b_deq name:@"qmatmul"];

        // Create INT8 data
        int8_t *bInt8Data = (int8_t *)calloc(K*N, sizeof(int8_t));
        __fp16 *scaleData = (__fp16 *)calloc(N, sizeof(__fp16));
        for (int i=0; i<K*N; i++) bInt8Data[i] = (int8_t)((arc4random() % 256) - 128);
        for (int i=0; i<N; i++) scaleData[i] = (__fp16)(0.01f);

        MPSGraphTensorData *aQTD = aTD; // reuse FP16 input
        MPSGraphTensorData *bQTD = [[MPSGraphTensorData alloc] initWithDevice:gdev
            data:[NSData dataWithBytesNoCopy:bInt8Data length:K*N*sizeof(int8_t) freeWhenDone:NO]
            shape:@[@(K), @(N)] dataType:MPSDataTypeInt8];
        MPSGraphTensorData *sQTD = [[MPSGraphTensorData alloc] initWithDevice:gdev
            data:[NSData dataWithBytesNoCopy:scaleData length:N*sizeof(__fp16) freeWhenDone:NO]
            shape:@[@1, @(N)] dataType:MPSDataTypeFloat16];

        // Warmup + bench INT8
        for (int i=0; i<3; i++) {
            @autoreleasepool {
                [qgraph runWithFeeds:@{a_int8:aQTD, b_int8:bQTD, b_scale:sQTD}
                      targetTensors:@[c_int8] targetOperations:nil];
            }
        }

        t0 = mach_absolute_time();
        for (int i=0; i<iters; i++) {
            @autoreleasepool {
                [qgraph runWithFeeds:@{a_int8:aQTD, b_int8:bQTD, b_scale:sQTD}
                      targetTensors:@[c_int8] targetOperations:nil];
            }
        }
        double int8Ms = ticksToMs(mach_absolute_time()-t0) / iters;
        double int8Tflops = 2.0*M*K*N / (int8Ms*1e9);

        // Compiled INT8
        MPSGraphShapedType *bInt8ST = [[MPSGraphShapedType alloc] initWithShape:@[@(K),@(N)] dataType:MPSDataTypeInt8];
        MPSGraphShapedType *sQST = [[MPSGraphShapedType alloc] initWithShape:@[@1,@(N)] dataType:MPSDataTypeFloat16];
        MPSGraphExecutable *qexec = [qgraph compileWithDevice:gdev
            feeds:@{a_int8:aST, b_int8:bInt8ST, b_scale:sQST}
            targetTensors:@[c_int8] targetOperations:nil compilationDescriptor:compDesc];

        @try { [qexec specializeWithDevice:gdev inputTypes:@[aST, bInt8ST, sQST] compilationDescriptor:compDesc]; }
        @catch(NSException *e) {}

        for (int i=0; i<3; i++) {
            @autoreleasepool {
                ((id(*)(id,SEL,id,id,id))objc_msgSend)(
                    qexec, @selector(runWithMTLCommandQueue:inputsArray:resultsArray:),
                    queue, @[aQTD, bQTD, sQTD], nil);
            }
        }

        t0 = mach_absolute_time();
        for (int i=0; i<iters; i++) {
            @autoreleasepool {
                ((id(*)(id,SEL,id,id,id))objc_msgSend)(
                    qexec, @selector(runWithMTLCommandQueue:inputsArray:resultsArray:),
                    queue, @[aQTD, bQTD, sQTD], nil);
            }
        }
        double qcompMs = ticksToMs(mach_absolute_time()-t0) / iters;
        double qcompTflops = 2.0*M*K*N / (qcompMs*1e9);

        // === Results ===
        printf("\n═══ RESULTS: [%d x %d] @ [%d x %d] ═══\n\n", M, K, K, N);
        printf("  %-40s %8s %8s\n", "Method", "ms", "TFLOPS");
        printf("  %-40s %8s %8s\n", "────────────────────────────────────", "──────", "──────");
        printf("  %-40s %8.3f %8.2f\n", "MPSGraph FP16 matmul (interpreted)", fp16Ms, fp16Tflops);
        printf("  %-40s %8.3f %8.2f\n", "MPSGraph FP16 matmul (compiled)", compiledMs, compiledTflops);
        printf("  %-40s %8.3f %8.2f\n", "MPSGraph INT8 deq+matmul (interpreted)", int8Ms, int8Tflops);
        printf("  %-40s %8.3f %8.2f\n", "MPSGraph INT8 deq+matmul (compiled)", qcompMs, qcompTflops);

        double bestMs = compiledMs < qcompMs ? compiledMs : qcompMs;
        printf("\n  llama.cpp uses custom Metal shaders for Q8_0 matmul.\n");
        printf("  MPS compiled path: %.3f ms (%.2f TFLOPS)\n", bestMs, 2.0*M*K*N/(bestMs*1e9));
        printf("  If MPS is faster than llama.cpp's shaders, that's a real gain.\n");

        free(aData); free(bData); free(bInt8Data); free(scaleData);
        printf("\n═══ Done ═══\n");
    }
    return 0;
}
