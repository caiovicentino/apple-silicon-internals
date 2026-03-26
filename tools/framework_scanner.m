// framework_scanner.m — Scan private frameworks for classes and methods
// Based on the reverse engineering approach from github.com/maderix/ANE
//
// Usage: ./framework_scanner <FrameworkName> [class_filter]
// Example: ./framework_scanner AppleNeuralEngine
//          ./framework_scanner IOGPU GPU
//
// Compile: clang -o framework_scanner framework_scanner.m \
//          -framework Foundation -lobjc -ldl -fobjc-arc

#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>

static void printMethodSignature(Method m, BOOL isClass) {
    SEL sel = method_getName(m);
    const char *enc = method_getTypeEncoding(m);
    int nargs = method_getNumberOfArguments(m);
    printf("    %c %s\n", isClass ? '+' : '-', sel_getName(sel));
    printf("      args: %d  encoding: %s\n", nargs, enc ? enc : "?");
}

static void printProperties(Class cls) {
    unsigned int count = 0;
    objc_property_t *props = class_copyPropertyList(cls, &count);
    if (count > 0) {
        printf("    Properties (%u):\n", count);
        for (unsigned int i = 0; i < count; i++) {
            const char *name = property_getName(props[i]);
            const char *attrs = property_getAttributes(props[i]);
            printf("      @property %s  [%s]\n", name, attrs ? attrs : "?");
        }
    }
    free(props);
}

static void printProtocols(Class cls) {
    unsigned int count = 0;
    Protocol * __unsafe_unretained *protos = class_copyProtocolList(cls, &count);
    if (count > 0) {
        printf("    Protocols (%u):\n", count);
        for (unsigned int i = 0; i < count; i++) {
            printf("      <%s>\n", protocol_getName(protos[i]));
        }
    }
    free(protos);
}

static void printIvars(Class cls) {
    unsigned int count = 0;
    Ivar *ivars = class_copyIvarList(cls, &count);
    if (count > 0) {
        printf("    Ivars (%u):\n", count);
        for (unsigned int i = 0; i < count; i++) {
            const char *name = ivar_getName(ivars[i]);
            const char *type = ivar_getTypeEncoding(ivars[i]);
            printf("      %s : %s\n", name ? name : "?", type ? type : "?");
        }
    }
    free(ivars);
}

static void scanClass(Class cls) {
    const char *name = class_getName(cls);
    Class super = class_getSuperclass(cls);
    printf("\n══════════════════════════════════════════\n");
    printf("CLASS: %s\n", name);
    if (super) printf("  superclass: %s\n", class_getName(super));
    printf("  size: %zu bytes\n", class_getInstanceSize(cls));

    printProtocols(cls);
    printProperties(cls);
    printIvars(cls);

    // Class methods
    unsigned int count = 0;
    Method *methods = class_copyMethodList(object_getClass(cls), &count);
    if (count > 0) {
        printf("  Class methods (%u):\n", count);
        for (unsigned int i = 0; i < count; i++)
            printMethodSignature(methods[i], YES);
    }
    free(methods);

    // Instance methods
    methods = class_copyMethodList(cls, &count);
    if (count > 0) {
        printf("  Instance methods (%u):\n", count);
        for (unsigned int i = 0; i < count; i++)
            printMethodSignature(methods[i], NO);
    }
    free(methods);
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc < 2) {
            printf("Usage: %s <FrameworkName> [class_filter]\n", argv[0]);
            printf("Example: %s AppleNeuralEngine\n", argv[0]);
            printf("         %s IOGPU GPU\n", argv[0]);
            printf("\nScans a private framework and dumps all classes, methods,\n");
            printf("properties, ivars, and protocols.\n");
            return 1;
        }

        const char *fwName = argv[1];
        const char *filter = argc > 2 ? argv[2] : NULL;

        // Try multiple paths
        NSString *paths[] = {
            [NSString stringWithFormat:@"/System/Library/PrivateFrameworks/%s.framework/%s", fwName, fwName],
            [NSString stringWithFormat:@"/System/Library/Frameworks/%s.framework/%s", fwName, fwName],
            [NSString stringWithFormat:@"/System/Library/PrivateFrameworks/%s.framework/Versions/A/%s", fwName, fwName],
        };

        void *handle = NULL;
        for (int i = 0; i < 3; i++) {
            handle = dlopen([paths[i] UTF8String], RTLD_NOW);
            if (handle) {
                printf("Loaded: %s\n", [paths[i] UTF8String]);
                break;
            }
        }

        if (!handle) {
            printf("Failed to load framework '%s'\n", fwName);
            printf("  dlerror: %s\n", dlerror());
            return 1;
        }

        // Enumerate all classes
        unsigned int classCount = 0;
        Class *classes = objc_copyClassList(&classCount);
        printf("Total ObjC classes in runtime: %u\n", classCount);

        // Filter classes related to the framework
        NSString *filterStr = filter ? [NSString stringWithUTF8String:filter] : nil;
        int found = 0;

        for (unsigned int i = 0; i < classCount; i++) {
            const char *className = class_getName(classes[i]);
            if (!className) continue;

            // Get the image (dylib) this class belongs to
            const char *image = class_getImageName(classes[i]);
            if (!image) continue;

            NSString *imageStr = [NSString stringWithUTF8String:image];
            NSString *fwStr = [NSString stringWithUTF8String:fwName];

            // Check if class belongs to our framework
            if ([imageStr containsString:fwStr]) {
                NSString *classStr = [NSString stringWithUTF8String:className];

                // Apply additional filter if provided
                if (filterStr && ![classStr localizedCaseInsensitiveContainsString:filterStr]) {
                    continue;
                }

                scanClass(classes[i]);
                found++;
            }
        }

        free(classes);
        printf("\n══════════════════════════════════════════\n");
        printf("Total classes found in %s: %d\n", fwName, found);

        dlclose(handle);
    }
    return 0;
}
