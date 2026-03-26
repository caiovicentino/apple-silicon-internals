// intelligence_probe.m — Probe Apple Intelligence private APIs
// Explores IntelligencePlatformCompute, PrivateMLClient, and related frameworks
//
// Compile: clang -o intelligence_probe intelligence_probe.m \
//          -framework Foundation -lobjc -ldl -fobjc-arc
//
// Usage: ./intelligence_probe

#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>

static void scanAllClasses(const char *fwName) {
    unsigned int classCount = 0;
    Class *classes = objc_copyClassList(&classCount);

    for (unsigned int i = 0; i < classCount; i++) {
        const char *image = class_getImageName(classes[i]);
        if (!image || !strstr(image, fwName)) continue;

        const char *className = class_getName(classes[i]);
        Class super = class_getSuperclass(classes[i]);

        printf("\n  ┌─ %s", className);
        if (super) printf(" : %s", class_getName(super));
        printf("\n");

        // Properties
        unsigned int pCount = 0;
        objc_property_t *props = class_copyPropertyList(classes[i], &pCount);
        for (unsigned int j = 0; j < pCount; j++) {
            printf("  │  @property %s\n", property_getName(props[j]));
        }
        free(props);

        // Class methods
        unsigned int cmCount = 0;
        Method *cmethods = class_copyMethodList(object_getClass(classes[i]), &cmCount);
        for (unsigned int j = 0; j < cmCount; j++) {
            printf("  │  + %s\n", sel_getName(method_getName(cmethods[j])));
        }
        free(cmethods);

        // Instance methods (abbreviated)
        unsigned int imCount = 0;
        Method *imethods = class_copyMethodList(classes[i], &imCount);
        for (unsigned int j = 0; j < imCount; j++) {
            const char *name = sel_getName(method_getName(imethods[j]));
            // Skip boilerplate
            if (name[0] == '.' || name[0] == '_') continue;
            printf("  │  - %s\n", name);
        }
        free(imethods);

        printf("  └─\n");
    }
    free(classes);
}

int main() {
    @autoreleasepool {
        printf("╔══════════════════════════════════════════════════════════╗\n");
        printf("║  Apple Intelligence / Private ML API Probe             ║\n");
        printf("╚══════════════════════════════════════════════════════════╝\n\n");

        // Load frameworks
        typedef struct {
            const char *name;
            const char *path;
            const char *desc;
        } FWEntry;

        FWEntry fws[] = {
            {"IntelligencePlatformCompute",
             "/System/Library/PrivateFrameworks/IntelligencePlatformCompute.framework/IntelligencePlatformCompute",
             "Apple Intelligence compute orchestration"},
            {"IntelligenceEngine",
             "/System/Library/PrivateFrameworks/IntelligenceEngine.framework/IntelligenceEngine",
             "Apple Intelligence inference engine"},
            {"IntelligenceTasksEngine",
             "/System/Library/PrivateFrameworks/IntelligenceTasksEngine.framework/IntelligenceTasksEngine",
             "Apple Intelligence task scheduling"},
            {"IntelligencePlatformLibrary",
             "/System/Library/PrivateFrameworks/IntelligencePlatformLibrary.framework/IntelligencePlatformLibrary",
             "Apple Intelligence platform library"},
            {"PrivateMLClient",
             "/System/Library/PrivateFrameworks/PrivateMLClient.framework/PrivateMLClient",
             "Private ML inference client"},
            {"PrivateCloudCompute",
             "/System/Library/PrivateFrameworks/PrivateCloudCompute.framework/PrivateCloudCompute",
             "Private Cloud Compute (PCC) client"},
            {"GraphCompute",
             "/System/Library/PrivateFrameworks/GraphCompute.framework/GraphCompute",
             "Compute graph execution"},
            {"GraphComputeRT",
             "/System/Library/PrivateFrameworks/GraphComputeRT.framework/GraphComputeRT",
             "Compute graph runtime"},
            {"MLAssetIO",
             "/System/Library/PrivateFrameworks/MLAssetIO.framework/MLAssetIO",
             "ML asset I/O"},
            {"NeuralNetworks",
             "/System/Library/PrivateFrameworks/NeuralNetworks.framework/NeuralNetworks",
             "Neural network primitives"},
            {"ComputeSafeguards",
             "/System/Library/PrivateFrameworks/ComputeSafeguards.framework/ComputeSafeguards",
             "Compute safety guardrails"},
            {"MediaML",
             "/System/Library/PrivateFrameworks/MediaML.framework/MediaML",
             "Media ML processing"},
            {"MLIR_ML",
             "/System/Library/PrivateFrameworks/MLIR_ML.framework/MLIR_ML",
             "MLIR for ML"},
            {NULL, NULL, NULL}
        };

        for (int i = 0; fws[i].name; i++) {
            printf("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
            printf("FRAMEWORK: %s\n", fws[i].name);
            printf("  %s\n", fws[i].desc);

            void *handle = dlopen(fws[i].path, RTLD_NOW);
            if (!handle) {
                printf("  ✗ Failed to load: %s\n", dlerror());
                continue;
            }
            printf("  ✓ Loaded\n");

            scanAllClasses(fws[i].name);
        }

        printf("\n═══════════════════════════════════════════\n");
        printf("Done. Check the output for interesting APIs!\n");
    }
    return 0;
}
