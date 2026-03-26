// clpc_probe.m — Probe Apple Silicon CLPC (Core Local Performance Controller)
// Reads CPU/GPU/ANE performance metrics via private PerformanceControlKit APIs
//
// This exposes the same data that powermetrics/Activity Monitor uses internally
//
// Compile: clang -o clpc_probe clpc_probe.m \
//          -framework Foundation -framework IOKit -lobjc -ldl -fobjc-arc
//
// Usage: ./clpc_probe

#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>

int main() {
    @autoreleasepool {
        // Load the private framework
        void *handle = dlopen(
            "/System/Library/PrivateFrameworks/PerformanceControlKit.framework/PerformanceControlKit",
            RTLD_NOW);
        if (!handle) {
            printf("Failed to load PerformanceControlKit: %s\n", dlerror());
            return 1;
        }
        printf("✓ Loaded PerformanceControlKit\n\n");

        // Get the CLPCReportingInterface class
        Class ReportingIF = NSClassFromString(@"CLPCReportingInterface");
        Class ReportingClient = NSClassFromString(@"CLPCReportingClient");
        Class PolicyIF = NSClassFromString(@"CLPCPolicyInterface");
        Class PolicyClient = NSClassFromString(@"CLPCPolicyClient");
        Class StatSelection = NSClassFromString(@"CLPCReportingStatSelection");
        Class Schema = NSClassFromString(@"CLPCReportingSchema");
        Class SchemaColumn = NSClassFromString(@"CLPCReportingSchemaColumn");

        printf("Classes found:\n");
        printf("  CLPCReportingInterface: %s\n", ReportingIF ? "YES" : "NO");
        printf("  CLPCReportingClient: %s\n", ReportingClient ? "YES" : "NO");
        printf("  CLPCPolicyInterface: %s\n", PolicyIF ? "YES" : "NO");
        printf("  CLPCPolicyClient: %s\n", PolicyClient ? "YES" : "NO");
        printf("  CLPCReportingStatSelection: %s\n", StatSelection ? "YES" : "NO");
        printf("  CLPCReportingSchema: %s\n", Schema ? "YES" : "NO");
        printf("  CLPCReportingSchemaColumn: %s\n", SchemaColumn ? "YES" : "NO");

        // === Create a reporting client ===
        printf("\n═══ Creating CLPC Reporting Client ═══\n");
        NSError *error = nil;

        // Use createClient: class method
        id client = ((id(*)(Class,SEL,NSError**))objc_msgSend)(
            ReportingIF, @selector(createClient:), &error);
        if (error) {
            printf("createClient error: %s\n", [[error description] UTF8String]);
        }
        if (client) {
            printf("✓ Client created: %s\n", [[client description] UTF8String]);
            printf("  Class: %s\n", class_getName([client class]));

            // Get supported stats
            printf("\n═══ Supported Stats ═══\n");
            @try {
                id stats = ((id(*)(id,SEL))objc_msgSend)(client, @selector(supportedStats));
                if (stats) {
                    printf("  Supported stats: %s\n", [[stats description] UTF8String]);
                }
            } @catch (NSException *ex) {
                printf("  Exception: %s\n", [[ex reason] UTF8String]);
            }

            // Get enabled stats
            @try {
                id enabled = ((id(*)(id,SEL))objc_msgSend)(client, @selector(enabledStats));
                if (enabled) {
                    printf("  Enabled stats: %s\n", [[enabled description] UTF8String]);
                }
            } @catch (NSException *ex) {
                printf("  Exception: %s\n", [[ex reason] UTF8String]);
            }

            // Try to enable all supported stats
            printf("\n═══ Enabling all stats ═══\n");
            @try {
                id stats = ((id(*)(id,SEL))objc_msgSend)(client, @selector(supportedStats));
                if (stats) {
                    NSError *enableErr = nil;
                    BOOL ok = ((BOOL(*)(id,SEL,id,NSError**))objc_msgSend)(
                        client, @selector(enableStats:error:), stats, &enableErr);
                    printf("  enableStats: %s\n", ok ? "YES" : "NO");
                    if (enableErr) printf("  error: %s\n", [[enableErr description] UTF8String]);
                }
            } @catch (NSException *ex) {
                printf("  Exception: %s\n", [[ex reason] UTF8String]);
            }

            // Read stats
            printf("\n═══ Reading Stats ═══\n");
            @try {
                NSError *readErr = nil;
                id result = ((id(*)(id,SEL,NSError**))objc_msgSend)(
                    client, @selector(readStats:), &readErr);
                if (result) {
                    printf("  Stats result class: %s\n", class_getName([result class]));
                    printf("  Stats: %s\n", [[result description] UTF8String]);
                }
                if (readErr) printf("  error: %s\n", [[readErr description] UTF8String]);
            } @catch (NSException *ex) {
                printf("  Exception: %s\n", [[ex reason] UTF8String]);
            }

            // Try readDeltaStats
            printf("\n═══ Reading Delta Stats ═══\n");
            @try {
                // First read to set baseline
                NSError *readErr = nil;
                ((id(*)(id,SEL,NSError**))objc_msgSend)(
                    client, @selector(readStats:), &readErr);

                // Sleep briefly
                usleep(100000); // 100ms

                // Read delta
                readErr = nil;
                id delta = ((id(*)(id,SEL,NSError**))objc_msgSend)(
                    client, @selector(readDeltaStats:), &readErr);
                if (delta) {
                    printf("  Delta result class: %s\n", class_getName([delta class]));

                    // Enumerate the result
                    if ([delta isKindOfClass:[NSDictionary class]]) {
                        NSDictionary *dict = (NSDictionary *)delta;
                        for (id key in dict) {
                            id value = dict[key];
                            printf("    [%s] = %s\n",
                                   [[key description] UTF8String],
                                   [[value description] UTF8String]);
                        }
                    } else {
                        printf("  Delta: %s\n", [[delta description] UTF8String]);
                    }
                }
                if (readErr) printf("  error: %s\n", [[readErr description] UTF8String]);
            } @catch (NSException *ex) {
                printf("  Exception: %s\n", [[ex reason] UTF8String]);
            }
        } else {
            printf("✗ Failed to create reporting client\n");
        }

        // === Try Policy Client (CPU frequency/QoS control) ===
        printf("\n═══ Creating CLPC Policy Client ═══\n");
        error = nil;
        id policyClient = ((id(*)(Class,SEL,NSError**))objc_msgSend)(
            PolicyIF, @selector(createClient:), &error);
        if (policyClient) {
            printf("✓ Policy client created: %s\n", class_getName([policyClient class]));

            // Dump all methods
            unsigned int count = 0;
            Method *methods = class_copyMethodList([policyClient class], &count);
            printf("  Methods (%u):\n", count);
            for (unsigned int i = 0; i < count; i++) {
                printf("    - %s\n", sel_getName(method_getName(methods[i])));
            }
            free(methods);
        }
        if (error) {
            printf("  error: %s\n", [[error description] UTF8String]);
        }

        // === Dump CLPCReportingStatSelection methods ===
        if (StatSelection) {
            printf("\n═══ CLPCReportingStatSelection ═══\n");
            unsigned int count = 0;
            Method *methods = class_copyMethodList(StatSelection, &count);
            printf("  Instance methods (%u):\n", count);
            for (unsigned int i = 0; i < count; i++) {
                printf("    - %s  [%s]\n",
                       sel_getName(method_getName(methods[i])),
                       method_getTypeEncoding(methods[i]) ?: "?");
            }
            free(methods);

            methods = class_copyMethodList(object_getClass(StatSelection), &count);
            printf("  Class methods (%u):\n", count);
            for (unsigned int i = 0; i < count; i++) {
                printf("    + %s\n", sel_getName(method_getName(methods[i])));
            }
            free(methods);
        }

        // === Dump Schema and SchemaColumn ===
        if (Schema) {
            printf("\n═══ CLPCReportingSchema ═══\n");
            unsigned int count = 0;
            Method *methods = class_copyMethodList(Schema, &count);
            for (unsigned int i = 0; i < count; i++) {
                printf("    - %s  [%s]\n",
                       sel_getName(method_getName(methods[i])),
                       method_getTypeEncoding(methods[i]) ?: "?");
            }
            free(methods);
        }

        if (SchemaColumn) {
            printf("\n═══ CLPCReportingSchemaColumn ═══\n");
            unsigned int count = 0;
            objc_property_t *props = class_copyPropertyList(SchemaColumn, &count);
            for (unsigned int i = 0; i < count; i++) {
                printf("    @property %s  [%s]\n",
                       property_getName(props[i]),
                       property_getAttributes(props[i]) ?: "?");
            }
            free(props);
        }

        dlclose(handle);
    }
    return 0;
}
