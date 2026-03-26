// batch_scan.m — Scan multiple private frameworks and output structured report
// Targets the most interesting hardware/compute/ML frameworks
//
// Usage: ./batch_scan [category]
//   Categories: gpu, ml, perf, hw, compute, audio, all
//
// Compile: clang -o batch_scan batch_scan.m \
//          -framework Foundation -lobjc -ldl -fobjc-arc

#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>

typedef struct {
    const char *name;
    const char *category;
    const char *description;
} FrameworkInfo;

static FrameworkInfo g_frameworks[] = {
    // GPU / Graphics
    {"AccelerateGPU",           "gpu",     "GPU acceleration primitives"},
    {"IOGPU",                   "gpu",     "Low-level GPU I/O interface"},
    {"IOAccelerator",           "gpu",     "Hardware accelerator interface"},
    {"AGXGPURawCounter",        "gpu",     "Apple GPU raw performance counters"},
    {"GPUCompiler",             "gpu",     "GPU shader compiler"},
    {"GPUInfo",                 "gpu",     "GPU hardware info queries"},
    {"GPURawCounter",           "gpu",     "GPU performance counters"},
    {"GPUSupport",              "gpu",     "GPU support utilities"},
    {"GPUWrangler",             "gpu",     "GPU device management"},
    {"MetalTools",              "gpu",     "Metal debugging/profiling tools"},
    {"WebGPU",                  "gpu",     "WebGPU implementation"},
    {"IOSurfaceAccelerator",    "gpu",     "IOSurface hardware acceleration"},

    // ML / AI
    {"AppleNeuralEngine",       "ml",      "ANE private API"},
    {"CipherML",                "ml",      "ML cipher/encryption"},
    {"CoreMLOdie",              "ml",      "CoreML Odie backend"},
    {"MLCompilerRuntime",       "ml",      "ML compiler runtime"},
    {"MLCompilerServices",      "ml",      "ML compiler services"},
    {"MLIR_ML",                 "ml",      "MLIR for ML"},
    {"MLRuntime",               "ml",      "ML runtime execution"},
    {"NeuralNetworks",          "ml",      "Neural network primitives"},
    {"IntelligencePlatformCompute", "ml",  "Apple Intelligence compute"},
    {"IntelligenceEngine",      "ml",      "Apple Intelligence engine"},
    {"MediaML",                 "ml",      "Media ML processing"},
    {"ProactiveML",             "ml",      "Proactive ML features"},
    {"PrivateMLClient",         "ml",      "Private ML client API"},
    {"RemoteCoreML",            "ml",      "Remote CoreML execution"},
    {"GraphCompute",            "ml",      "Graph computation framework"},
    {"GraphComputeRT",          "ml",      "Graph compute runtime"},
    {"MLAssetIO",               "ml",      "ML asset I/O"},
    {"MLModelSpecification",    "ml",      "ML model spec handling"},
    {"SensitiveContentAnalysisML", "ml",   "Content analysis ML"},
    {"ComputeSafeguards",       "ml",      "Compute safety guardrails"},

    // Performance / Power
    {"kperf",                   "perf",    "Kernel performance counters"},
    {"kperfdata",               "perf",    "Kernel perf data reader"},
    {"perfdata",                "perf",    "Performance data framework"},
    {"PerformanceAnalysis",     "perf",    "Performance analysis tools"},
    {"PerformanceControlKit",   "perf",    "Performance control API"},
    {"PerformanceTrace",        "perf",    "Performance tracing"},
    {"PerfPowerMetricMonitor",  "perf",    "Perf/power metric monitor"},
    {"PerfPowerServicesReader", "perf",    "Perf/power data reader"},
    {"PowerLog",                "perf",    "Power logging"},
    {"PowerExperience",         "perf",    "Power experience management"},
    {"PowerlogCore",            "perf",    "Power log core"},
    {"LowPowerMode",           "perf",    "Low power mode control"},

    // Hardware
    {"BiometricKit",            "hw",      "Biometric (Touch/Face ID)"},
    {"HID",                     "hw",      "Human Interface Device"},
    {"HIDAnalytics",            "hw",      "HID analytics"},
    {"SensorAccess",            "hw",      "Sensor access API"},
    {"CoreMotionAlgorithms",    "hw",      "Motion processing algorithms"},
    {"DisplayServices",         "hw",      "Display management"},
    {"BluetoothManager",       "hw",      "Bluetooth management"},
    {"CoreWiFi",               "hw",      "WiFi internals"},
    {"MicroLocation",          "hw",      "Ultra-wideband location"},

    // Compute
    {"AGXCompilerCore",         "compute", "AGX GPU compiler core"},
    {"JetEngine",               "compute", "Jet engine compute"},
    {"CascadeEngine",           "compute", "Cascade compute engine"},
    {"ReplicatorEngine",        "compute", "Replicator engine"},
    {"ArchetypeEngine",         "compute", "Archetype engine"},

    // Audio
    {"AudioDSPAnalysis",        "audio",   "Audio DSP analysis"},
    {"AudioDSPGraph",           "audio",   "Audio DSP compute graph"},
    {"AudioDSPManager",         "audio",   "Audio DSP management"},

    {NULL, NULL, NULL}
};

static int scanFramework(const char *name) {
    NSString *path = [NSString stringWithFormat:
        @"/System/Library/PrivateFrameworks/%s.framework/%s", name, name];
    void *handle = dlopen([path UTF8String], RTLD_NOW);
    if (!handle) {
        // Try Versions/A/ path
        path = [NSString stringWithFormat:
            @"/System/Library/PrivateFrameworks/%s.framework/Versions/A/%s", name, name];
        handle = dlopen([path UTF8String], RTLD_NOW);
    }
    if (!handle) return -1;

    unsigned int classCount = 0;
    Class *classes = objc_copyClassList(&classCount);
    NSString *fwStr = [NSString stringWithUTF8String:name];
    int found = 0;

    for (unsigned int i = 0; i < classCount; i++) {
        const char *image = class_getImageName(classes[i]);
        if (!image) continue;
        if (!strstr(image, name)) continue;

        const char *className = class_getName(classes[i]);
        Class super = class_getSuperclass(classes[i]);

        unsigned int mCount = 0;
        Method *methods = class_copyMethodList(classes[i], &mCount);
        unsigned int cmCount = 0;
        Method *cmethods = class_copyMethodList(object_getClass(classes[i]), &cmCount);
        unsigned int pCount = 0;
        objc_property_t *props = class_copyPropertyList(classes[i], &pCount);

        printf("  %s", className);
        if (super) printf(" : %s", class_getName(super));
        printf("  [%u class, %u instance methods, %u properties]\n",
               cmCount, mCount, pCount);

        // Print the most interesting class methods (factory methods)
        for (unsigned int j = 0; j < cmCount; j++) {
            SEL sel = method_getName(cmethods[j]);
            const char *selName = sel_getName(sel);
            // Show factory/creation/shared methods
            if (strstr(selName, "shared") || strstr(selName, "default") ||
                strstr(selName, "With") || strstr(selName, "with") ||
                strstr(selName, "create") || strstr(selName, "new") ||
                strstr(selName, "model") || strstr(selName, "device") ||
                strstr(selName, "compile") || strstr(selName, "load") ||
                strstr(selName, "request") || strstr(selName, "surface") ||
                strstr(selName, "alloc")) {
                printf("      + %s\n", selName);
            }
        }

        free(methods);
        free(cmethods);
        free(props);
        found++;
    }

    free(classes);
    dlclose(handle);
    return found;
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        const char *category = argc > 1 ? argv[1] : "all";
        BOOL showAll = strcmp(category, "all") == 0;

        printf("╔══════════════════════════════════════════════════════════╗\n");
        printf("║  macOS Private Framework Scanner                       ║\n");
        printf("║  Category: %-44s ║\n", category);
        printf("╚══════════════════════════════════════════════════════════╝\n\n");

        int totalFrameworks = 0;
        int totalClasses = 0;

        for (int i = 0; g_frameworks[i].name; i++) {
            if (!showAll && strcmp(g_frameworks[i].category, category) != 0)
                continue;

            printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
            printf("FRAMEWORK: %s\n", g_frameworks[i].name);
            printf("  Category: %s\n", g_frameworks[i].category);
            printf("  Desc: %s\n", g_frameworks[i].description);

            int n = scanFramework(g_frameworks[i].name);
            if (n < 0) {
                printf("  ⚠ Failed to load (may not exist on this macOS version)\n");
            } else {
                printf("  → %d classes found\n", n);
                totalClasses += n;
                totalFrameworks++;
            }
            printf("\n");
        }

        printf("═══════════════════════════════════════════\n");
        printf("SUMMARY: %d frameworks loaded, %d total classes\n",
               totalFrameworks, totalClasses);
    }
    return 0;
}
