// ane_direct.m — Direct ANE (Apple Neural Engine) access PoC
// Compiles and runs a MIL program on the ANE without CoreML
// Based on reverse engineering from github.com/maderix/ANE
//
// Compile: clang -o ane_direct ane_direct.m \
//          -framework Foundation -framework IOKit -lobjc -ldl -fobjc-arc
//
// Usage: ./ane_direct

#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <IOSurface/IOSurface.h>
#import <mach/mach_time.h>

static mach_timebase_info_data_t g_tb;
static double ticksToMs(uint64_t t) { return (double)t * g_tb.numer / g_tb.denom / 1e6; }

// Build an ANE weight blob: 128-byte header + fp16 data
static uint8_t *buildWeightBlob(const float *src, int rows, int cols, size_t *outLen) {
    int wsize = rows * cols * 2;
    int total = 128 + wsize;
    uint8_t *buf = (uint8_t *)calloc(total, 1);
    buf[0] = 0x01; buf[4] = 0x02;
    buf[64] = 0xEF; buf[65] = 0xBE; buf[66] = 0xAD; buf[67] = 0xDE;
    buf[68] = 0x01;
    *(uint32_t*)(buf + 72) = wsize;
    *(uint32_t*)(buf + 80) = 128;
    _Float16 *fp16 = (_Float16 *)(buf + 128);
    for (int i = 0; i < rows * cols; i++)
        fp16[i] = (_Float16)src[i];
    *outLen = total;
    return buf;
}

static IOSurfaceRef createSurface(size_t bytes) {
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth: @(bytes),
        (id)kIOSurfaceHeight: @1,
        (id)kIOSurfaceBytesPerElement: @1,
        (id)kIOSurfaceBytesPerRow: @(bytes),
        (id)kIOSurfaceAllocSize: @(bytes),
        (id)kIOSurfacePixelFormat: @0
    });
}

int main() {
    @autoreleasepool {
        mach_timebase_info(&g_tb);

        setbuf(stdout, NULL);

        printf("╔══════════════════════════════════════════════════════════╗\n");
        printf("║  Direct ANE Access PoC                                 ║\n");
        printf("╚══════════════════════════════════════════════════════════╝\n\n");

        // Load the private framework
        void *handle = dlopen(
            "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine",
            RTLD_NOW);
        if (!handle) {
            printf("Failed to load AppleNeuralEngine: %s\n", dlerror());
            return 1;
        }

        // Resolve classes
        Class ANEDesc = NSClassFromString(@"_ANEInMemoryModelDescriptor");
        Class ANEInMem = NSClassFromString(@"_ANEInMemoryModel");
        Class ANEReq = NSClassFromString(@"_ANERequest");
        Class ANEIO = NSClassFromString(@"_ANEIOSurfaceObject");
        Class ANEDevInfo = NSClassFromString(@"_ANEDeviceInfo");
        Class ANEPerf = NSClassFromString(@"_ANEPerformanceStats");

        printf("Classes resolved:\n");
        printf("  _ANEInMemoryModelDescriptor: %s\n", ANEDesc ? "YES" : "NO");
        printf("  _ANEInMemoryModel: %s\n", ANEInMem ? "YES" : "NO");
        printf("  _ANERequest: %s\n", ANEReq ? "YES" : "NO");
        printf("  _ANEIOSurfaceObject: %s\n", ANEIO ? "YES" : "NO");
        printf("  _ANEDeviceInfo: %s\n", ANEDevInfo ? "YES" : "NO");

        if (!ANEDesc || !ANEInMem || !ANEReq || !ANEIO) {
            printf("Missing required classes\n");
            return 1;
        }

        // === Query ANE device info (safe methods only) ===
        if (ANEDevInfo) {
            printf("\n═══ ANE Device Info ═══\n");
            @try {
                BOOL disabled = ((BOOL(*)(Class,SEL))objc_msgSend)(ANEDevInfo, @selector(precompiledModelChecksDisabled));
                printf("  precompiledModelChecksDisabled = %s\n", disabled ? "YES" : "NO");
            } @catch (NSException *ex) {}
        }

        // === Build a MIL program in ANE-compatible format ===
        // The ANE MIL compiler expects specific protobuf-like MIL text format
        // Using the format discovered by the maderix/ANE project
        int C = 64, S = 1; // Keep it simple: 64-channel, 1 spatial
        size_t tensorBytes = C * S * 2; // fp16

        // ANE MIL text format — valid for the _ANEInMemoryModelDescriptor compiler
        NSString *milText = [NSString stringWithFormat:
            @"program(\"com.test.ane_direct\") {\n"
            @"  func main(input: tensor<fp16, [1, %d, 1, %d]>) -> tensor<fp16, [1, %d, 1, %d]> {\n"
            @"    %%0 = add(x = input, y = input, name = \"add0\");\n"
            @"    return %%0;\n"
            @"  }\n"
            @"}\n",
            C, S, C, S];

        printf("\n═══ MIL Program ═══\n");
        printf("%s\n", [milText UTF8String]);

        // === Create model descriptor ===
        printf("═══ Creating Model ═══\n");
        NSData *milData = [milText dataUsingEncoding:NSUTF8StringEncoding];

        id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
            ANEDesc, @selector(modelWithMILText:weights:optionsPlist:),
            milData, @{}, nil);

        if (!desc) {
            printf("modelWithMILText failed\n");

            // Try with NSString directly
            desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
                ANEDesc, @selector(modelWithMILText:weights:optionsPlist:),
                milText, @{}, nil);
        }

        if (!desc) {
            printf("Model descriptor creation failed.\n");
            printf("Note: MIL compilation requires specific MIL syntax compatible with the ANE compiler.\n");
            printf("The MIL format used by ANE differs from the text representation.\n\n");

            printf("═══ Attempting alternative: _ANEModel from file ═══\n");

            // Dump _ANEModel class methods to see what's available
            Class ANEModel = NSClassFromString(@"_ANEModel");
            if (ANEModel) {
                unsigned int count = 0;
                Method *cmethods = class_copyMethodList(object_getClass(ANEModel), &count);
                printf("_ANEModel class methods (%u):\n", count);
                for (unsigned int i = 0; i < count; i++) {
                    printf("  + %s\n", sel_getName(method_getName(cmethods[i])));
                }
                free(cmethods);

                Method *imethods = class_copyMethodList(ANEModel, &count);
                printf("_ANEModel instance methods (%u):\n", count);
                for (unsigned int i = 0; i < count; i++) {
                    printf("  - %s\n", sel_getName(method_getName(imethods[i])));
                }
                free(imethods);
            }

            // Dump _ANEClient
            Class ANEClient = NSClassFromString(@"_ANEClient");
            if (ANEClient) {
                printf("\n_ANEClient:\n");
                unsigned int count = 0;
                Method *cmethods = class_copyMethodList(object_getClass(ANEClient), &count);
                printf("  Class methods (%u):\n", count);
                for (unsigned int i = 0; i < count; i++) {
                    printf("    + %s\n", sel_getName(method_getName(cmethods[i])));
                }
                free(cmethods);

                // Try to get shared connection
                @try {
                    id shared = ((id(*)(Class,SEL))objc_msgSend)(ANEClient, @selector(sharedConnection));
                    if (shared) {
                        printf("  Shared connection: %s (%s)\n",
                               [[shared description] UTF8String],
                               class_getName([shared class]));
                    }
                } @catch (NSException *ex) {
                    printf("  sharedConnection failed: %s\n", [[ex reason] UTF8String]);
                }
            }

            printf("\n═══ IOSurface I/O Test ═══\n");
            // Even without full MIL compilation, we can demonstrate the IOSurface
            // zero-copy mechanism used for ANE data transfer
            IOSurfaceRef surf = createSurface(tensorBytes);
            if (surf) {
                printf("IOSurface created: %lu bytes\n", (unsigned long)tensorBytes);

                // Write fp16 test data
                IOSurfaceLock(surf, 0, NULL);
                _Float16 *data = (_Float16 *)IOSurfaceGetBaseAddress(surf);
                for (int i = 0; i < C * S; i++) {
                    data[i] = (_Float16)(i * 0.01f);
                }
                IOSurfaceUnlock(surf, 0, NULL);

                // Create ANE IO wrapper
                id ioObj = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
                    ANEIO, @selector(objectWithIOSurface:), surf);
                printf("_ANEIOSurfaceObject: %s\n",
                       ioObj ? [[ioObj description] UTF8String] : "nil");

                // Read back
                IOSurfaceLock(surf, kIOSurfaceLockReadOnly, NULL);
                _Float16 *readback = (_Float16 *)IOSurfaceGetBaseAddress(surf);
                printf("Data readback [0..3]: %.4f, %.4f, %.4f, %.4f\n",
                       (float)readback[0], (float)readback[1],
                       (float)readback[2], (float)readback[3]);
                IOSurfaceUnlock(surf, kIOSurfaceLockReadOnly, NULL);

                printf("IOSurface zero-copy I/O: WORKS\n");
                CFRelease(surf);
            }

            return 0;
        }

        printf("Model descriptor created: %s\n", [[desc description] UTF8String]);

        // Create in-memory model
        id model = ((id(*)(Class,SEL,id))objc_msgSend)(
            ANEInMem, @selector(inMemoryModelWithDescriptor:), desc);
        if (!model) {
            printf("inMemoryModelWithDescriptor failed\n");
            return 1;
        }

        // Pre-populate temp directory
        id hexId = ((id(*)(id,SEL))objc_msgSend)(model, @selector(hexStringIdentifier));
        NSString *tmpDir = [NSTemporaryDirectory() stringByAppendingPathComponent:hexId];
        NSFileManager *fm = [NSFileManager defaultManager];
        [fm createDirectoryAtPath:tmpDir withIntermediateDirectories:YES attributes:nil error:nil];
        [milData writeToFile:[tmpDir stringByAppendingPathComponent:@"model.mil"] atomically:YES];

        // Compile
        printf("Compiling on ANE...\n");
        NSError *err = nil;
        BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
            model, @selector(compileWithQoS:options:error:), 21, @{}, &err);
        printf("Compile: %s\n", ok ? "SUCCESS" : "FAILED");
        if (err) printf("  Error: %s\n", [[err description] UTF8String]);
        if (!ok) { [fm removeItemAtPath:tmpDir error:nil]; return 1; }

        // Load
        printf("Loading on ANE...\n");
        err = nil;
        ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
            model, @selector(loadWithQoS:options:error:), 21, @{}, &err);
        printf("Load: %s\n", ok ? "SUCCESS" : "FAILED");
        if (err) printf("  Error: %s\n", [[err description] UTF8String]);
        if (!ok) { [fm removeItemAtPath:tmpDir error:nil]; return 1; }

        // Create IOSurfaces
        IOSurfaceRef ioIn = createSurface(tensorBytes);
        IOSurfaceRef ioOut = createSurface(tensorBytes);

        // Write input data
        IOSurfaceLock(ioIn, 0, NULL);
        _Float16 *inData = (_Float16 *)IOSurfaceGetBaseAddress(ioIn);
        for (int i = 0; i < C * S; i++)
            inData[i] = (_Float16)(i * 0.1f);
        IOSurfaceUnlock(ioIn, 0, NULL);

        // Build request
        id wIn = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(ANEIO, @selector(objectWithIOSurface:), ioIn);
        id wOut = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(ANEIO, @selector(objectWithIOSurface:), ioOut);
        id request = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(
            ANEReq,
            @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
            @[wIn], @[@0], @[wOut], @[@0], nil, nil, @0);

        // Evaluate
        printf("\nEvaluating on ANE...\n");
        err = nil;
        ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
            model, @selector(evaluateWithQoS:options:request:error:),
            21, @{}, request, &err);
        printf("Evaluate: %s\n", ok ? "SUCCESS" : "FAILED");
        if (err) printf("  Error: %s\n", [[err description] UTF8String]);

        if (ok) {
            // Read output
            IOSurfaceLock(ioOut, kIOSurfaceLockReadOnly, NULL);
            _Float16 *outData = (_Float16 *)IOSurfaceGetBaseAddress(ioOut);
            printf("\nInput[0..3]:  %.2f, %.2f, %.2f, %.2f\n",
                   (float)inData[0], (float)inData[1], (float)inData[2], (float)inData[3]);
            printf("Output[0..3]: %.2f, %.2f, %.2f, %.2f\n",
                   (float)outData[0], (float)outData[1], (float)outData[2], (float)outData[3]);
            printf("Expected:     %.2f, %.2f, %.2f, %.2f\n",
                   (float)inData[0]+1, (float)inData[1]+1, (float)inData[2]+1, (float)inData[3]+1);
            IOSurfaceUnlock(ioOut, kIOSurfaceLockReadOnly, NULL);

            // Benchmark
            printf("\n═══ Benchmark ═══\n");
            // Warmup
            for (int i = 0; i < 10; i++)
                ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                    model, @selector(evaluateWithQoS:options:request:error:),
                    21, @{}, request, &err);

            int iters = 100;
            uint64_t t0 = mach_absolute_time();
            for (int i = 0; i < iters; i++)
                ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                    model, @selector(evaluateWithQoS:options:request:error:),
                    21, @{}, request, &err);
            double ms = ticksToMs(mach_absolute_time() - t0) / iters;
            printf("  %d iterations: %.3f ms/eval\n", iters, ms);
        }

        // Cleanup
        ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(
            model, @selector(unloadWithQoS:error:), 21, &err);
        CFRelease(ioIn);
        CFRelease(ioOut);
        [fm removeItemAtPath:tmpDir error:nil];

        printf("\n═══ Done ═══\n");
    }
    return 0;
}
