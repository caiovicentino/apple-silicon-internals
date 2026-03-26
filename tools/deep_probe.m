// deep_probe.m — Deep exploration of specific private API classes
// Attempts to instantiate objects and call safe getter methods
//
// Usage: ./deep_probe <FrameworkName> <ClassName>
// Example: ./deep_probe IOGPU IOGPUDevice
//
// Compile: clang -o deep_probe deep_probe.m \
//          -framework Foundation -framework IOKit -lobjc -ldl -fobjc-arc

#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>

// Check if a selector is likely a safe "getter" (no side effects)
static BOOL isSafeGetter(const char *selName) {
    // Skip init/alloc/dealloc/set/remove/delete/create/start/stop/reset
    if (strncmp(selName, "init", 4) == 0) return NO;
    if (strncmp(selName, "alloc", 5) == 0) return NO;
    if (strncmp(selName, "dealloc", 7) == 0) return NO;
    if (strncmp(selName, "set", 3) == 0 && selName[3] >= 'A' && selName[3] <= 'Z') return NO;
    if (strncmp(selName, "remove", 6) == 0) return NO;
    if (strncmp(selName, "delete", 6) == 0) return NO;
    if (strncmp(selName, "create", 6) == 0) return NO;
    if (strncmp(selName, "start", 5) == 0) return NO;
    if (strncmp(selName, "stop", 4) == 0) return NO;
    if (strncmp(selName, "reset", 5) == 0) return NO;
    if (strncmp(selName, "release", 7) == 0) return NO;
    if (strncmp(selName, "retain", 6) == 0) return NO;
    if (strncmp(selName, "add", 3) == 0 && selName[3] >= 'A') return NO;
    if (strncmp(selName, "load", 4) == 0) return NO;
    if (strncmp(selName, "unload", 6) == 0) return NO;
    if (strncmp(selName, "compile", 7) == 0) return NO;
    if (strncmp(selName, "evaluate", 8) == 0) return NO;
    if (strncmp(selName, "write", 5) == 0) return NO;
    if (strncmp(selName, "send", 4) == 0) return NO;
    if (strncmp(selName, "connect", 7) == 0) return NO;
    if (strncmp(selName, "disconnect", 10) == 0) return NO;
    if (strncmp(selName, "open", 4) == 0) return NO;
    if (strncmp(selName, "close", 5) == 0) return NO;
    if (strncmp(selName, "perform", 7) == 0) return NO;
    if (strncmp(selName, "execute", 7) == 0) return NO;
    if (strncmp(selName, "dispatch", 8) == 0) return NO;
    if (strncmp(selName, "invalidate", 10) == 0) return NO;
    if (strncmp(selName, "cancel", 6) == 0) return NO;
    if (strncmp(selName, "suspend", 7) == 0) return NO;
    if (strncmp(selName, "resume", 6) == 0) return NO;
    if (strncmp(selName, "flush", 5) == 0) return NO;
    if (strncmp(selName, "clear", 5) == 0) return NO;
    if (strncmp(selName, "destroy", 7) == 0) return NO;
    if (strncmp(selName, "free", 4) == 0) return NO;
    if (strncmp(selName, "update", 6) == 0) return NO;
    if (strncmp(selName, "register", 8) == 0) return NO;
    if (strncmp(selName, "unregister", 10) == 0) return NO;
    if (strncmp(selName, "enable", 6) == 0) return NO;
    if (strncmp(selName, "disable", 7) == 0) return NO;
    if (strncmp(selName, "configure", 9) == 0) return NO;
    if (strncmp(selName, ".cxx_destruct", 13) == 0) return NO;
    if (strncmp(selName, "_", 1) == 0 && strncmp(selName, "_is", 3) != 0) return NO;

    // Only call no-argument methods
    if (strchr(selName, ':') != NULL) return NO;

    return YES;
}

static void probeClassMethods(Class cls) {
    printf("\n=== Class Method Probing ===\n");
    unsigned int count = 0;
    Method *methods = class_copyMethodList(object_getClass(cls), &count);
    for (unsigned int i = 0; i < count; i++) {
        SEL sel = method_getName(methods[i]);
        const char *name = sel_getName(sel);
        int nargs = method_getNumberOfArguments(methods[i]);

        // Only probe class methods that take no args (besides self and _cmd)
        if (nargs == 2 && isSafeGetter(name)) {
            printf("  + %s => ", name);
            @try {
                id result = ((id(*)(Class,SEL))objc_msgSend)(cls, sel);
                if (result) {
                    printf("%s: %s\n", class_getName([result class]),
                           [[result description] UTF8String]);
                } else {
                    printf("nil\n");
                }
            } @catch (NSException *ex) {
                printf("EXCEPTION: %s\n", [[ex reason] UTF8String]);
            }
        }
    }
    free(methods);
}

static void probeInstance(id obj) {
    if (!obj) return;
    Class cls = [obj class];
    printf("\n=== Instance Probing: %s ===\n", class_getName(cls));

    unsigned int count = 0;
    Method *methods = class_copyMethodList(cls, &count);
    for (unsigned int i = 0; i < count; i++) {
        SEL sel = method_getName(methods[i]);
        const char *name = sel_getName(sel);
        int nargs = method_getNumberOfArguments(methods[i]);

        if (nargs == 2 && isSafeGetter(name)) {
            printf("  - %s => ", name);
            @try {
                id result = ((id(*)(id,SEL))objc_msgSend)(obj, sel);
                if (result) {
                    printf("%s: %s\n", class_getName([result class]),
                           [[result description] UTF8String]);
                } else {
                    printf("nil\n");
                }
            } @catch (NSException *ex) {
                printf("EXCEPTION: %s\n", [[ex reason] UTF8String]);
            }
        }
    }
    free(methods);
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc < 3) {
            printf("Usage: %s <FrameworkName> <ClassName>\n", argv[0]);
            printf("Example: %s IOGPU IOGPUDevice\n", argv[0]);
            printf("\nDeeply probes a private class: dumps structure and tries\n");
            printf("to call safe getter methods.\n");
            return 1;
        }

        const char *fwName = argv[1];
        const char *className = argv[2];

        // Load framework
        NSString *path = [NSString stringWithFormat:
            @"/System/Library/PrivateFrameworks/%s.framework/%s", fwName, fwName];
        void *handle = dlopen([path UTF8String], RTLD_NOW);
        if (!handle) {
            // Try public framework
            path = [NSString stringWithFormat:
                @"/System/Library/Frameworks/%s.framework/%s", fwName, fwName];
            handle = dlopen([path UTF8String], RTLD_NOW);
        }
        if (!handle) {
            printf("Failed to load: %s\n", dlerror());
            return 1;
        }
        printf("Loaded: %s\n", [path UTF8String]);

        Class cls = NSClassFromString([NSString stringWithUTF8String:className]);
        if (!cls) {
            printf("Class '%s' not found\n", className);
            return 1;
        }

        // === Full class dump ===
        printf("\n══════════════════════════════════════════\n");
        printf("CLASS: %s\n", className);
        Class super = class_getSuperclass(cls);
        printf("  Superclass: %s\n", super ? class_getName(super) : "none");
        printf("  Instance size: %zu bytes\n", class_getInstanceSize(cls));

        // Protocols
        unsigned int pCount = 0;
        Protocol * __unsafe_unretained *protos = class_copyProtocolList(cls, &pCount);
        if (pCount > 0) {
            printf("  Protocols:\n");
            for (unsigned int i = 0; i < pCount; i++)
                printf("    <%s>\n", protocol_getName(protos[i]));
        }
        free(protos);

        // Ivars
        unsigned int iCount = 0;
        Ivar *ivars = class_copyIvarList(cls, &iCount);
        if (iCount > 0) {
            printf("  Ivars (%u):\n", iCount);
            for (unsigned int i = 0; i < iCount; i++) {
                printf("    %s : %s (offset %ld)\n",
                       ivar_getName(ivars[i]),
                       ivar_getTypeEncoding(ivars[i]) ?: "?",
                       (long)ivar_getOffset(ivars[i]));
            }
        }
        free(ivars);

        // Properties
        unsigned int propCount = 0;
        objc_property_t *props = class_copyPropertyList(cls, &propCount);
        if (propCount > 0) {
            printf("  Properties (%u):\n", propCount);
            for (unsigned int i = 0; i < propCount; i++) {
                printf("    @property %s  [%s]\n",
                       property_getName(props[i]),
                       property_getAttributes(props[i]) ?: "?");
            }
        }
        free(props);

        // Class methods
        unsigned int cmCount = 0;
        Method *cmethods = class_copyMethodList(object_getClass(cls), &cmCount);
        printf("  Class methods (%u):\n", cmCount);
        for (unsigned int i = 0; i < cmCount; i++) {
            SEL s = method_getName(cmethods[i]);
            printf("    + %s  (args: %d, enc: %s)\n",
                   sel_getName(s),
                   method_getNumberOfArguments(cmethods[i]),
                   method_getTypeEncoding(cmethods[i]) ?: "?");
        }
        free(cmethods);

        // Instance methods
        unsigned int imCount = 0;
        Method *imethods = class_copyMethodList(cls, &imCount);
        printf("  Instance methods (%u):\n", imCount);
        for (unsigned int i = 0; i < imCount; i++) {
            SEL s = method_getName(imethods[i]);
            printf("    - %s  (args: %d, enc: %s)\n",
                   sel_getName(s),
                   method_getNumberOfArguments(imethods[i]),
                   method_getTypeEncoding(imethods[i]) ?: "?");
        }
        free(imethods);

        // === Probe class methods ===
        probeClassMethods(cls);

        // === Try to create instance via common patterns ===
        printf("\n=== Instantiation attempts ===\n");
        id instance = nil;

        // Try sharedInstance / shared / defaultManager patterns
        SEL sharedSels[] = {
            @selector(sharedInstance),
            @selector(shared),
            @selector(defaultManager),
            @selector(sharedManager),
            @selector(currentDevice),
            @selector(defaultDevice),
        };
        const char *sharedNames[] = {
            "sharedInstance", "shared", "defaultManager",
            "sharedManager", "currentDevice", "defaultDevice"
        };

        for (int i = 0; i < 6; i++) {
            if ([cls respondsToSelector:sharedSels[i]]) {
                printf("  Trying +%s...\n", sharedNames[i]);
                @try {
                    instance = ((id(*)(Class,SEL))objc_msgSend)(cls, sharedSels[i]);
                    if (instance) {
                        printf("  SUCCESS: %s\n", [[instance description] UTF8String]);
                        break;
                    }
                } @catch (NSException *ex) {
                    printf("  EXCEPTION: %s\n", [[ex reason] UTF8String]);
                }
            }
        }

        // Try alloc+init as last resort
        if (!instance) {
            printf("  Trying [[%s alloc] init]...\n", className);
            @try {
                instance = [[cls alloc] init];
                if (instance) {
                    printf("  SUCCESS: %s\n", [[instance description] UTF8String]);
                }
            } @catch (NSException *ex) {
                printf("  EXCEPTION: %s\n", [[ex reason] UTF8String]);
            }
        }

        // Probe the instance
        if (instance) {
            probeInstance(instance);
        }

        dlclose(handle);
    }
    return 0;
}
