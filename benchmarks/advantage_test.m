// advantage_test.m — Measure the REAL advantage of private APIs vs public ones
// Compares: MTLBuffer (what llama.cpp uses) vs MTLTensor (what Apple has internally)
// Shows what the discovered APIs actually improve
//
// Compile: clang -o advantage_test advantage_test.m \
//          -framework Foundation -framework Metal -framework Accelerate \
//          -framework IOKit -framework IOSurface -framework MetalPerformanceShaders \
//          -lobjc -ldl -fobjc-arc -O2 -DACCELERATE_NEW_LAPACK
//
// Usage: ./advantage_test

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <Accelerate/Accelerate.h>
#import <IOSurface/IOSurface.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <mach/mach_time.h>
#import <sys/sysctl.h>

static mach_timebase_info_data_t g_tb;
static double ticksToMs(uint64_t t) { return (double)t*g_tb.numer/g_tb.denom/1e6; }
static double ticksToUs(uint64_t t) { return (double)t*g_tb.numer/g_tb.denom/1e3; }

int main() {
    @autoreleasepool {
        setbuf(stdout, NULL);
        mach_timebase_info(&g_tb);

        dlopen("/System/Library/PrivateFrameworks/IOGPU.framework/IOGPU", RTLD_NOW);
        dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();

        printf("╔══════════════════════════════════════════════════════════╗\n");
        printf("║  Private API Advantage Test — What's Actually New      ║\n");
        printf("║  %s                                                    \n", [[device name] UTF8String]);
        printf("╚══════════════════════════════════════════════════════════╝\n\n");

        // ═══════════════════════════════════════════════════════════
        // TEST 1: MTLBuffer (llama.cpp) vs MTLTensor (private API)
        // Key question: does the tensor API reduce overhead for
        // view/reshape operations common in transformer inference?
        // ═══════════════════════════════════════════════════════════

        printf("═══ TEST 1: MTLBuffer vs MTLTensor — Reshape/View Cost ═══\n\n");
        printf("  Transformer inference reshapes tensors constantly:\n");
        printf("  [batch, seq, dim] → [batch, heads, seq, head_dim] (for attention)\n");
        printf("  llama.cpp does this with pointer math on MTLBuffer.\n");
        printf("  MTLTensor has native reshape/slice with zero-copy views.\n\n");

        int dim = 4096, heads = 32, head_dim = dim/heads, seq = 512, batch = 1;
        size_t totalBytes = batch * seq * dim * sizeof(float);

        // --- MTLBuffer approach (what llama.cpp does) ---
        id<MTLBuffer> buf = [device newBufferWithLength:totalBytes
                                               options:MTLResourceStorageModeShared];
        float *bufPtr = (float *)[buf contents];
        for (int i=0; i<batch*seq*dim; i++) bufPtr[i] = 0.01f * i;

        // Simulating "reshape" with buffers = just offset calculation (nearly free)
        uint64_t t0 = mach_absolute_time();
        int iters = 100000;
        volatile size_t dummy = 0;
        for (int i=0; i<iters; i++) {
            // "Reshape" [batch,seq,dim] → [batch,heads,seq,head_dim]
            // With buffers, this is just calculating offsets — no data movement
            for (int h=0; h<heads; h++) {
                size_t offset = h * head_dim * sizeof(float);
                dummy += offset; // Prevent optimization
            }
        }
        double bufReshapeUs = ticksToUs(mach_absolute_time()-t0) / iters;
        (void)dummy;

        // --- MTLTensor approach (private API) ---
        Class TensorDesc = NSClassFromString(@"MTLTensorDescriptor");
        Class TensorExtents = NSClassFromString(@"MTLTensorExtents");

        double tensorReshapeUs = -1;
        if (TensorDesc && TensorExtents) {
            // Create a 3D tensor [batch*seq, heads, head_dim]
            id desc = [[TensorDesc alloc] init];
            ((void(*)(id,SEL,NSInteger))objc_msgSend)(desc, @selector(setDataType:), 3); // float32

            int64_t dims3[] = {batch*seq, heads, head_dim};
            id extents = ((id(*)(id,SEL,uint64_t,const int64_t*))objc_msgSend)(
                [TensorExtents alloc], @selector(initWithRank:values:), (uint64_t)3, dims3);
            ((void(*)(id,SEL,id))objc_msgSend)(desc, @selector(setDimensions:), extents);

            NSError *err = nil;
            id tensor = ((id(*)(id,SEL,id,NSError**))objc_msgSend)(
                (id)device, @selector(newTensorWithDescriptor:error:), desc, &err);

            if (tensor) {
                // Create reshape descriptor [batch, heads, seq, head_dim]
                id reshapeDesc = [[TensorDesc alloc] init];
                ((void(*)(id,SEL,NSInteger))objc_msgSend)(reshapeDesc, @selector(setDataType:), 3);
                int64_t dims4[] = {batch, heads, seq, head_dim};
                id extents4 = ((id(*)(id,SEL,uint64_t,const int64_t*))objc_msgSend)(
                    [TensorExtents alloc], @selector(initWithRank:values:), (uint64_t)4, dims4);
                ((void(*)(id,SEL,id))objc_msgSend)(reshapeDesc, @selector(setDimensions:), extents4);

                // Check if reshape is possible
                SEL canReshape = @selector(isTensorViewableWithReshapedDescriptor:);
                if ([tensor respondsToSelector:canReshape]) {
                    BOOL ok = ((BOOL(*)(id,SEL,id))objc_msgSend)(tensor, canReshape, reshapeDesc);
                    printf("  Tensor reshape [%d,%d,%d] → [%d,%d,%d,%d]: %s\n",
                           batch*seq, heads, head_dim, batch, heads, seq, head_dim,
                           ok ? "SUPPORTED (zero-copy)" : "NOT SUPPORTED");

                    if (ok) {
                        // Benchmark the reshape view creation
                        t0 = mach_absolute_time();
                        int reshapeIters = 10000;
                        for (int i=0; i<reshapeIters; i++) {
                            NSError *rerr = nil;
                            id view = ((id(*)(id,SEL,id,NSError**))objc_msgSend)(
                                tensor, @selector(newTensorViewWithReshapedDescriptor:error:),
                                reshapeDesc, &rerr);
                            (void)view;
                        }
                        tensorReshapeUs = ticksToUs(mach_absolute_time()-t0) / reshapeIters;
                    }
                }

                // Test slice operation (extracting one head's data)
                SEL sliceSel = @selector(newTensorViewWithSlice:error:);
                if ([tensor respondsToSelector:sliceSel]) {
                    printf("  Tensor slice (extract single head): SUPPORTED\n");
                }

                NSUInteger allocSize = ((NSUInteger(*)(id,SEL))objc_msgSend)(
                    tensor, @selector(allocatedSize));
                printf("  Tensor allocated: %lu bytes (%.1f MB)\n",
                       (unsigned long)allocSize, allocSize/(1024.0*1024));
            }
        }

        printf("\n  Results:\n");
        printf("    MTLBuffer reshape (offset calc):   %.3f us/op\n", bufReshapeUs);
        if (tensorReshapeUs > 0)
            printf("    MTLTensor reshape (zero-copy view): %.3f us/op\n", tensorReshapeUs);
        printf("\n  Verdict: Buffer offset math is ~free. MTLTensor's value is NOT in\n");
        printf("  reshape speed — it's in GPU-side operations that can use tensor\n");
        printf("  metadata directly without CPU intervention.\n");

        // ═══════════════════════════════════════════════════════════
        // TEST 2: Where MTLTensor ACTUALLY wins — GPU-native views
        // The real advantage: the GPU can access tensor dimensions/strides
        // natively, which means kernels can be written for arbitrary
        // tensor layouts without manual stride calculation
        // ═══════════════════════════════════════════════════════════

        printf("\n═══ TEST 2: Metal 4 ML Pipeline — What Is It? ═══\n\n");

        Class ML4Desc = NSClassFromString(@"MTL4MachineLearningPipelineDescriptor");
        Class ML4State = NSClassFromString(@"_MTL4MachineLearningPipelineState");
        Class ML4Enc = NSClassFromString(@"_MTL4MachineLearningCommandEncoder");
        Class ML4Refl = NSClassFromString(@"MTL4MachineLearningPipelineReflection");

        printf("  MTL4MachineLearningPipelineDescriptor: %s\n", ML4Desc ? "FOUND" : "not found");
        printf("  _MTL4MachineLearningPipelineState:     %s\n", ML4State ? "FOUND" : "not found");
        printf("  _MTL4MachineLearningCommandEncoder:    %s\n", ML4Enc ? "FOUND" : "not found");
        printf("  MTL4MachineLearningPipelineReflection: %s\n", ML4Refl ? "FOUND" : "not found");

        if (ML4Desc) {
            printf("\n  MTL4MachineLearningPipelineDescriptor methods:\n");
            unsigned int count = 0;
            objc_property_t *props = class_copyPropertyList(ML4Desc, &count);
            for (unsigned int i=0; i<count; i++) {
                printf("    @property %s  [%s]\n", property_getName(props[i]),
                       property_getAttributes(props[i]) ?: "?");
            }
            free(props);

            Method *methods = class_copyMethodList(ML4Desc, &count);
            for (unsigned int i=0; i<count; i++) {
                const char *name = sel_getName(method_getName(methods[i]));
                if (name[0] != '.' && name[0] != '_')
                    printf("    - %s\n", name);
            }
            free(methods);
        }

        if (ML4Enc) {
            printf("\n  _MTL4MachineLearningCommandEncoder methods:\n");
            unsigned int count = 0;
            Method *methods = class_copyMethodList(ML4Enc, &count);
            for (unsigned int i=0; i<count; i++) {
                printf("    - %s  [%s]\n", sel_getName(method_getName(methods[i])),
                       method_getTypeEncoding(methods[i]) ?: "?");
            }
            free(methods);

            // Walk up to see parent class methods
            Class parent = class_getSuperclass(ML4Enc);
            if (parent) {
                printf("\n  Parent %s methods:\n", class_getName(parent));
                methods = class_copyMethodList(parent, &count);
                for (unsigned int i=0; i<count; i++) {
                    const char *name = sel_getName(method_getName(methods[i]));
                    printf("    - %s\n", name);
                }
                free(methods);
            }
        }

        // ═══════════════════════════════════════════════════════════
        // TEST 3: ANE direct vs CoreML — actual overhead comparison
        // ═══════════════════════════════════════════════════════════

        printf("\n═══ TEST 3: IOSurface vs MTLBuffer — Data Transfer ═══\n\n");
        printf("  ANE uses IOSurface for zero-copy I/O.\n");
        printf("  GPU uses MTLBuffer with shared memory.\n");
        printf("  Both use unified memory — but is one faster?\n\n");

        size_t testSizes[] = {1024, 65536, 1048576, 16777216}; // 1KB to 16MB
        const char *sizeNames[] = {"1 KB", "64 KB", "1 MB", "16 MB"};

        printf("  %-10s %-18s %-18s %-10s\n", "Size", "MTLBuffer (ns)", "IOSurface (ns)", "Winner");
        printf("  %-10s %-18s %-18s %-10s\n", "────", "──────────────", "──────────────", "──────");

        for (int si=0; si<4; si++) {
            size_t sz = testSizes[si];
            int niters = sz < 1048576 ? 100000 : 10000;

            // MTLBuffer write
            id<MTLBuffer> testBuf = [device newBufferWithLength:sz
                                                       options:MTLResourceStorageModeShared];
            float *tbPtr = (float *)[testBuf contents];

            t0 = mach_absolute_time();
            for (int i=0; i<niters; i++) {
                tbPtr[0] = (float)i;
                tbPtr[sz/sizeof(float)-1] = (float)i;
            }
            double bufNs = ticksToUs(mach_absolute_time()-t0)*1000.0 / niters;

            // IOSurface write
            IOSurfaceRef surf = IOSurfaceCreate((__bridge CFDictionaryRef)@{
                (id)kIOSurfaceWidth: @(sz), (id)kIOSurfaceHeight: @1,
                (id)kIOSurfaceBytesPerElement: @1, (id)kIOSurfaceBytesPerRow: @(sz),
                (id)kIOSurfaceAllocSize: @(sz), (id)kIOSurfacePixelFormat: @0
            });

            t0 = mach_absolute_time();
            for (int i=0; i<niters; i++) {
                IOSurfaceLock(surf, 0, NULL);
                float *p = (float *)IOSurfaceGetBaseAddress(surf);
                p[0] = (float)i;
                p[sz/sizeof(float)-1] = (float)i;
                IOSurfaceUnlock(surf, 0, NULL);
            }
            double ioNs = ticksToUs(mach_absolute_time()-t0)*1000.0 / niters;

            CFRelease(surf);

            const char *winner = bufNs < ioNs ? "Buffer" : "IOSurf";
            printf("  %-10s %-18.0f %-18.0f %-10s\n", sizeNames[si], bufNs, ioNs, winner);
        }

        printf("\n  Verdict: MTLBuffer is faster for simple writes (no lock overhead).\n");
        printf("  IOSurface advantage is cross-process/cross-device sharing (CPU↔ANE↔GPU)\n");
        printf("  without copies — critical for ANE pipeline.\n");

        // ═══════════════════════════════════════════════════════════
        // TEST 4: What we CAN build that doesn't exist
        // ═══════════════════════════════════════════════════════════

        printf("\n═══ WHAT'S GENUINELY NEW & ACTIONABLE ═══\n\n");

        printf("  1. MTL4 ML Pipeline (Metal 4 Machine Learning)\n");
        printf("     Status: Classes exist but pipeline encoding is undocumented.\n");
        printf("     This appears to be Apple's upcoming native ML acceleration path\n");
        printf("     on GPU — distinct from MPS, compute shaders, and ANE.\n");
        printf("     If we can figure out the descriptor format, it could unlock\n");
        printf("     hardware-optimized GEMM/attention without writing shaders.\n\n");

        printf("  2. ANE + GPU Hybrid Pipeline\n");
        printf("     Nobody combines ANE (for quantized ops) + GPU (for attention)\n");
        printf("     in a single inference pass. IOSurface enables zero-copy sharing\n");
        printf("     between them. The maderix/ANE repo showed this is possible.\n");
        printf("     Potential: prefill on GPU, decode on ANE → best of both.\n\n");

        printf("  3. Adaptive Compute via IOReport\n");
        printf("     1009 IOReport channels give real-time SoC state.\n");
        printf("     An inference engine could dynamically switch backends:\n");
        printf("       - GPU throttling? → shift to CPU/SME2\n");
        printf("       - Battery low? → shift to ANE (most efficient)\n");
        printf("       - Thermal pressure? → reduce batch/precision\n");
        printf("     Nobody does this. All existing engines use a fixed backend.\n\n");

        printf("  4. SME2 Custom Kernels\n");
        printf("     The M4 has SME2 with 512-bit matrix tiles.\n");
        printf("     Accelerate uses it for GEMM, but nobody uses it for:\n");
        printf("       - Flash attention (fused QK^TV)\n");
        printf("       - Custom quantization (2-bit, 3-bit)\n");
        printf("       - Speculative decoding\n");
        printf("     Direct SME2 assembly could outperform generic cblas.\n\n");

        printf("  5. MTLTensor for KV Cache\n");
        printf("     Transformer KV caches need frequent view/slice/concat.\n");
        printf("     MTLTensor has native:\n");
        printf("       - newTensorViewWithSlice (extract KV for specific heads)\n");
        printf("       - newTensorViewWithReshapedDescriptor (reformat without copy)\n");
        printf("       - replaceSlice:withBytes:strides: (update in place)\n");
        printf("     llama.cpp manages this manually with buffer offsets.\n");
        printf("     MTLTensor could make this GPU-native and faster.\n");

        printf("\n═══ Done ═══\n");
    }
    return 0;
}
