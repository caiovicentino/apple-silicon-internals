// gamemode_ctl.m — Control Apple Game Mode via private CLPC API
// Toggle Game Mode programmatically on Apple Silicon
//
// Compile: clang -o gamemode_ctl gamemode_ctl.m \
//          -framework Foundation -framework IOKit -lobjc -ldl -fobjc-arc
//
// Usage: ./gamemode_ctl status
//        ./gamemode_ctl on
//        ./gamemode_ctl off

#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc < 2) {
            printf("Usage: %s <status|on|off>\n", argv[0]);
            printf("\nControls Apple Game Mode via private CLPC API\n");
            printf("  status  — Check current game mode eligibility\n");
            printf("  on      — Enable game mode\n");
            printf("  off     — Disable game mode\n");
            return 1;
        }

        const char *action = argv[1];

        void *handle = dlopen(
            "/System/Library/PrivateFrameworks/PerformanceControlKit.framework/PerformanceControlKit",
            RTLD_NOW);
        if (!handle) {
            printf("Failed to load PerformanceControlKit: %s\n", dlerror());
            return 1;
        }

        Class PolicyIF = NSClassFromString(@"CLPCPolicyInterface");
        if (!PolicyIF) {
            printf("CLPCPolicyInterface not found\n");
            return 1;
        }

        // Create policy client
        NSError *error = nil;
        id client = ((id(*)(Class,SEL,NSError**))objc_msgSend)(
            PolicyIF, @selector(createClient:), &error);
        if (!client) {
            printf("Failed to create CLPC policy client: %s\n",
                   error ? [[error description] UTF8String] : "unknown");
            return 1;
        }

        printf("CLPC Policy Client: %s\n\n", class_getName([client class]));

        // Dump available methods
        unsigned int count = 0;
        Method *methods = class_copyMethodList([client class], &count);
        printf("Available methods:\n");
        for (unsigned int i = 0; i < count; i++) {
            const char *name = sel_getName(method_getName(methods[i]));
            if (name[0] != '.' && name[0] != '_')
                printf("  - %s\n", name);
        }
        free(methods);
        printf("\n");

        if (strcmp(action, "status") == 0) {
            // Check low power mode candidacy
            @try {
                NSError *err = nil;
                BOOL isCandidate = ((BOOL(*)(id,SEL,NSError**))objc_msgSend)(
                    client, @selector(isLowPowerModeCandidate:error:), &err);
                // Note: this method's signature suggests it takes an error ptr
                printf("Low Power Mode Candidate: %s\n", isCandidate ? "YES" : "NO");
                if (err) printf("  (error: %s)\n", [[err description] UTF8String]);
            } @catch (NSException *ex) {
                printf("Exception checking LPM: %s\n", [[ex reason] UTF8String]);
            }
        }
        else if (strcmp(action, "on") == 0 || strcmp(action, "off") == 0) {
            BOOL enable = strcmp(action, "on") == 0;
            printf("Setting Game Mode: %s\n", enable ? "ON" : "OFF");

            @try {
                NSError *err = nil;
                // setGameMode:options:error: takes (BOOL, NSDictionary*, NSError**)
                // Try with gameMode = 1 (on) or 0 (off)
                BOOL ok = ((BOOL(*)(id,SEL,BOOL,id,NSError**))objc_msgSend)(
                    client, @selector(setGameMode:options:error:),
                    enable, @{}, &err);
                printf("Result: %s\n", ok ? "SUCCESS" : "FAILED");
                if (err) printf("Error: %s\n", [[err description] UTF8String]);
            } @catch (NSException *ex) {
                printf("Exception: %s\n", [[ex reason] UTF8String]);
                printf("\nNote: Game Mode control may require specific entitlements.\n");
                printf("This API is used by the system Game Mode service.\n");
            }
        }

        // Also dump CLPCPolicyClient ivars for context
        printf("\n═══ CLPCPolicyClient internals ═══\n");
        unsigned int iCount = 0;
        Ivar *ivars = class_copyIvarList([client class], &iCount);
        for (unsigned int i = 0; i < iCount; i++) {
            const char *name = ivar_getName(ivars[i]);
            const char *type = ivar_getTypeEncoding(ivars[i]);
            if (name) printf("  %s : %s\n", name, type ? type : "?");
        }
        free(ivars);

        // Walk up to CLPCUserClient
        Class superCls = class_getSuperclass([client class]);
        if (superCls) {
            printf("\n═══ %s internals ═══\n", class_getName(superCls));
            ivars = class_copyIvarList(superCls, &iCount);
            for (unsigned int i = 0; i < iCount; i++) {
                const char *name = ivar_getName(ivars[i]);
                const char *type = ivar_getTypeEncoding(ivars[i]);
                if (name) printf("  %s : %s\n", name, type ? type : "?");
            }
            free(ivars);

            methods = class_copyMethodList(superCls, &count);
            printf("  Methods (%u):\n", count);
            for (unsigned int i = 0; i < count; i++) {
                const char *name = sel_getName(method_getName(methods[i]));
                if (name[0] != '.' && name[0] != '_')
                    printf("    - %s\n", name);
            }
            free(methods);
        }

        dlclose(handle);
    }
    return 0;
}
