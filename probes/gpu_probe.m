// gpu_probe.m — Probe Apple GPU internals via private IOGPU/Metal APIs
// Discovers hidden GPU capabilities, performance counters, and tensor ops
//
// Compile: clang -o gpu_probe gpu_probe.m \
//          -framework Foundation -framework Metal -framework IOKit \
//          -lobjc -ldl -fobjc-arc
//
// Usage: ./gpu_probe

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>

static void dumpHiddenMethods(id obj, const char *label) {
    Class cls = [obj class];
    unsigned int count = 0;
    Method *methods = class_copyMethodList(cls, &count);
    printf("\n  %s (%s) — %u methods:\n", label, class_getName(cls), count);
    for (unsigned int i = 0; i < count; i++) {
        SEL sel = method_getName(methods[i]);
        const char *name = sel_getName(sel);
        // Show interesting/undocumented methods
        if (strstr(name, "max") || strstr(name, "supports") || strstr(name, "is") ||
            strstr(name, "count") || strstr(name, "size") || strstr(name, "family") ||
            strstr(name, "gpu") || strstr(name, "GPU") || strstr(name, "clock") ||
            strstr(name, "memory") || strstr(name, "Memory") || strstr(name, "bandwidth") ||
            strstr(name, "ane") || strstr(name, "ANE") || strstr(name, "neural") ||
            strstr(name, "tensor") || strstr(name, "Tensor") || strstr(name, "ml") ||
            strstr(name, "ML") || strstr(name, "perf") || strstr(name, "Perf") ||
            strstr(name, "counter") || strstr(name, "Counter") ||
            strstr(name, "compute") || strstr(name, "ray") || strstr(name, "mesh") ||
            strstr(name, "tile") || strstr(name, "simd") || strstr(name, "SIMD") ||
            strstr(name, "vendor") || strstr(name, "device") || strstr(name, "chip") ||
            strstr(name, "architecture") || strstr(name, "feature") ||
            strstr(name, "kern") || strstr(name, "heap") || strstr(name, "cache") ||
            strstr(name, "residency") || strstr(name, "sparse")) {
            printf("    - %s\n", name);
        }
    }
    free(methods);
}

static void tryCallBoolGetter(id obj, const char *selName) {
    SEL sel = sel_registerName(selName);
    if ([obj respondsToSelector:sel]) {
        @try {
            BOOL val = ((BOOL(*)(id,SEL))objc_msgSend)(obj, sel);
            printf("    %s = %s\n", selName, val ? "YES" : "NO");
        } @catch (NSException *ex) {}
    }
}

static void tryCallIntGetter(id obj, const char *selName) {
    SEL sel = sel_registerName(selName);
    if ([obj respondsToSelector:sel]) {
        @try {
            NSUInteger val = ((NSUInteger(*)(id,SEL))objc_msgSend)(obj, sel);
            printf("    %s = %lu\n", selName, (unsigned long)val);
        } @catch (NSException *ex) {}
    }
}

int main() {
    @autoreleasepool {
        // Load private GPU frameworks
        dlopen("/System/Library/PrivateFrameworks/IOGPU.framework/IOGPU", RTLD_NOW);
        dlopen("/System/Library/PrivateFrameworks/GPURawCounter.framework/GPURawCounter", RTLD_NOW);
        dlopen("/System/Library/PrivateFrameworks/MetalTools.framework/MetalTools", RTLD_NOW);
        dlopen("/System/Library/PrivateFrameworks/AGXGPURawCounter.framework/AGXGPURawCounter", RTLD_NOW);

        // Get Metal device
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            printf("No Metal device found\n");
            return 1;
        }

        printf("╔══════════════════════════════════════════════════════════╗\n");
        printf("║  Apple GPU Deep Probe                                  ║\n");
        printf("╚══════════════════════════════════════════════════════════╝\n\n");

        // === Standard Metal device info ===
        printf("═══ Standard Device Info ═══\n");
        printf("  Name: %s\n", [[device name] UTF8String]);
        printf("  Registry ID: %llu\n", [device registryID]);
        printf("  Max buffer length: %lu MB\n", (unsigned long)[device maxBufferLength] / (1024*1024));
        printf("  Max threads per threadgroup: (%lu, %lu, %lu)\n",
               (unsigned long)[device maxThreadsPerThreadgroup].width,
               (unsigned long)[device maxThreadsPerThreadgroup].height,
               (unsigned long)[device maxThreadsPerThreadgroup].depth);
        printf("  Recommended max working set: %lu MB\n",
               (unsigned long)[device recommendedMaxWorkingSetSize] / (1024*1024));

        // === Hidden device properties ===
        printf("\n═══ Hidden Device Properties ═══\n");
        Class cls = [device class];
        printf("  Actual class: %s\n", class_getName(cls));

        // Walk the class hierarchy
        Class c = cls;
        printf("  Class hierarchy: ");
        while (c) {
            printf("%s", class_getName(c));
            c = class_getSuperclass(c);
            if (c) printf(" → ");
        }
        printf("\n");

        // Try undocumented properties
        printf("\n═══ Undocumented Capabilities ═══\n");

        // GPU architecture info
        tryCallIntGetter((id)device, "gpuFamily");
        tryCallIntGetter((id)device, "gpuFamilyVersion");
        tryCallIntGetter((id)device, "vendorID");
        tryCallIntGetter((id)device, "deviceID");
        tryCallIntGetter((id)device, "revisionID");
        tryCallIntGetter((id)device, "metalGPUFamily");

        // Memory info
        tryCallIntGetter((id)device, "currentAllocatedSize");
        tryCallIntGetter((id)device, "maxTransferRate");
        tryCallIntGetter((id)device, "memorySize");

        // Compute info
        tryCallIntGetter((id)device, "maxComputeUnits");
        tryCallIntGetter((id)device, "maxTotalThreadsPerThreadgroup");
        tryCallIntGetter((id)device, "maxThreadgroupMemoryLength");
        tryCallIntGetter((id)device, "maxSIMDGroupWidth");
        tryCallIntGetter((id)device, "simdGroupWidth");

        // Feature flags
        printf("\n═══ Feature Flags ═══\n");
        tryCallBoolGetter((id)device, "supportsBCTextureCompression");
        tryCallBoolGetter((id)device, "supportsQueryTextureLOD");
        tryCallBoolGetter((id)device, "supportsPullModelInterpolation");
        tryCallBoolGetter((id)device, "supportsShaderBarycentricCoordinates");
        tryCallBoolGetter((id)device, "supports32BitFloatFiltering");
        tryCallBoolGetter((id)device, "supports32BitMSAA");
        tryCallBoolGetter((id)device, "supportsRaytracing");
        tryCallBoolGetter((id)device, "supportsMeshShaders");
        tryCallBoolGetter((id)device, "supportsFunctionPointers");
        tryCallBoolGetter((id)device, "supportsDynamicLibraries");
        tryCallBoolGetter((id)device, "supportsPrimitiveMotionBlur");
        tryCallBoolGetter((id)device, "supportsRaytracingFromRender");
        tryCallBoolGetter((id)device, "supportsResidencySets");

        // Hidden capabilities
        printf("\n═══ Hidden Capabilities ═══\n");
        tryCallBoolGetter((id)device, "supportsTensorCores");
        tryCallBoolGetter((id)device, "supportsANE");
        tryCallBoolGetter((id)device, "supportsNeuralEngine");
        tryCallBoolGetter((id)device, "supportsMLCompute");
        tryCallBoolGetter((id)device, "supportsInfiniteBuffers");
        tryCallBoolGetter((id)device, "supportsCooperativeMatrices");
        tryCallBoolGetter((id)device, "supportsHalfPrecision");
        tryCallBoolGetter((id)device, "supportsBFloat16");
        tryCallBoolGetter((id)device, "supportsInt8");
        tryCallBoolGetter((id)device, "supportsInt4");
        tryCallBoolGetter((id)device, "supportsSparseTextures");
        tryCallBoolGetter((id)device, "supportsSparseHeaps");
        tryCallBoolGetter((id)device, "supportsLosslessColorAttachments");
        tryCallBoolGetter((id)device, "supportsDepthClipMode");
        tryCallBoolGetter((id)device, "supportsUnifiedMemory");
        tryCallBoolGetter((id)device, "isLowPower");
        tryCallBoolGetter((id)device, "isRemovable");
        tryCallBoolGetter((id)device, "isHeadless");
        tryCallBoolGetter((id)device, "hasUnifiedMemory");

        // === Dump ALL methods on the real device class ===
        dumpHiddenMethods((id)device, "Metal Device");

        // === Try to access IOGPUMemoryInfo ===
        printf("\n═══ IOGPUMemoryInfo ═══\n");
        Class memInfoClass = NSClassFromString(@"IOGPUMemoryInfo");
        if (memInfoClass) {
            // List class methods to find creation method
            unsigned int count = 0;
            Method *cmethods = class_copyMethodList(object_getClass(memInfoClass), &count);
            printf("  Class methods:\n");
            for (unsigned int i = 0; i < count; i++)
                printf("    + %s\n", sel_getName(method_getName(cmethods[i])));
            free(cmethods);

            Method *imethods = class_copyMethodList(memInfoClass, &count);
            printf("  Instance methods:\n");
            for (unsigned int i = 0; i < count; i++)
                printf("    - %s\n", sel_getName(method_getName(imethods[i])));
            free(imethods);
        }

        // === Try GPURawCounter ===
        printf("\n═══ GPURawCounter ═══\n");
        Class counterClass = NSClassFromString(@"_GPURawCounterSource");
        if (counterClass) {
            unsigned int count = 0;
            Method *methods = class_copyMethodList(counterClass, &count);
            printf("  _GPURawCounterSource instance methods (%u):\n", count);
            for (unsigned int i = 0; i < count; i++) {
                SEL sel = method_getName(methods[i]);
                printf("    - %s\n", sel_getName(sel));
            }
            free(methods);

            // Properties
            objc_property_t *props = class_copyPropertyList(counterClass, &count);
            printf("  Properties (%u):\n", count);
            for (unsigned int i = 0; i < count; i++) {
                printf("    @property %s  [%s]\n",
                       property_getName(props[i]),
                       property_getAttributes(props[i]) ?: "?");
            }
            free(props);
        }

        // === MTL4 Machine Learning ===
        printf("\n═══ Metal 4 Machine Learning ═══\n");
        Class ml4Encoder = NSClassFromString(@"IOGPUMetal4MachineLearningCommandEncoder");
        if (ml4Encoder) {
            printf("  IOGPUMetal4MachineLearningCommandEncoder EXISTS!\n");
            unsigned int count = 0;
            Method *methods = class_copyMethodList(ml4Encoder, &count);
            printf("  Methods (%u):\n", count);
            for (unsigned int i = 0; i < count; i++) {
                SEL sel = method_getName(methods[i]);
                printf("    - %s  [%s]\n", sel_getName(sel),
                       method_getTypeEncoding(methods[i]) ?: "?");
            }
            free(methods);
        } else {
            printf("  Not found on this device\n");
        }

        // === Metal Tensor support ===
        printf("\n═══ IOGPUMetalTensor ═══\n");
        Class tensorClass = NSClassFromString(@"IOGPUMetalTensor");
        if (tensorClass) {
            printf("  IOGPUMetalTensor EXISTS!\n");
            unsigned int count = 0;
            objc_property_t *props = class_copyPropertyList(tensorClass, &count);
            printf("  Properties (%u):\n", count);
            for (unsigned int i = 0; i < count; i++) {
                printf("    @property %s\n", property_getName(props[i]));
            }
            free(props);

            Method *methods = class_copyMethodList(tensorClass, &count);
            printf("  Instance methods (%u):\n", count);
            for (unsigned int i = 0; i < count; i++) {
                printf("    - %s\n", sel_getName(method_getName(methods[i])));
            }
            free(methods);
        }

        printf("\n═══ Done ═══\n");
    }
    return 0;
}
