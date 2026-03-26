// soc_power.m — Read Apple Silicon power data via IOReport
// This is the same underlying mechanism used by powermetrics, Activity Monitor,
// and the PerfPowerMetricMonitor framework we discovered.
//
// Compile: clang -o soc_power soc_power.m \
//          -framework Foundation -framework IOKit -lobjc -fobjc-arc
//
// Usage: ./soc_power [samples] [interval_ms]
//   ./soc_power          # 10 samples, 1000ms
//   ./soc_power 5 500    # 5 samples, 500ms

#import <Foundation/Foundation.h>
#import <IOKit/IOKitLib.h>
#import <objc/runtime.h>
#import <dlfcn.h>

// IOReport types
typedef struct __IOReportSubscriptionCF *IOReportSubscriptionRef;
typedef IOReportSubscriptionRef (*IOReportCreateSubscription_t)(
    void*, CFMutableDictionaryRef, CFMutableDictionaryRef*, uint64_t, CFTypeRef);
typedef CFDictionaryRef (*IOReportCreateSamples_t)(
    IOReportSubscriptionRef, CFMutableDictionaryRef, CFTypeRef);
typedef CFDictionaryRef (*IOReportCreateSamplesDelta_t)(
    CFDictionaryRef, CFDictionaryRef, CFTypeRef);
typedef void (*IOReportIterate_t)(CFDictionaryRef, int(^)(CFDictionaryRef));
typedef CFMutableDictionaryRef (*IOReportCopyChannelsInGroup_t)(
    CFStringRef, CFStringRef, uint64_t, uint64_t, uint64_t);
typedef void (*IOReportMergeChannels_t)(
    CFMutableDictionaryRef, CFMutableDictionaryRef, CFTypeRef);
typedef int32_t (*IOReportChannelGetFormat_t)(CFDictionaryRef);
typedef int64_t (*IOReportSimpleGetIntegerValue_t)(CFDictionaryRef, int32_t);
typedef CFStringRef (*IOReportChannelGetChannelName_t)(CFDictionaryRef);
typedef CFStringRef (*IOReportChannelGetGroup_t)(CFDictionaryRef);
typedef CFStringRef (*IOReportChannelGetSubGroup_t)(CFDictionaryRef);
typedef CFStringRef (*IOReportChannelGetUnitLabel_t)(CFDictionaryRef);

// Global function pointers
static IOReportCreateSubscription_t pIOReportCreateSubscription;
static IOReportCreateSamples_t pIOReportCreateSamples;
static IOReportCreateSamplesDelta_t pIOReportCreateSamplesDelta;
static IOReportIterate_t pIOReportIterate;
static IOReportCopyChannelsInGroup_t pIOReportCopyChannelsInGroup;
static IOReportMergeChannels_t pIOReportMergeChannels;
static IOReportChannelGetFormat_t pIOReportChannelGetFormat;
static IOReportSimpleGetIntegerValue_t pIOReportSimpleGetIntegerValue;
static IOReportChannelGetChannelName_t pIOReportChannelGetChannelName;
static IOReportChannelGetGroup_t pIOReportChannelGetGroup;
static IOReportChannelGetSubGroup_t pIOReportChannelGetSubGroup;
static IOReportChannelGetUnitLabel_t pIOReportChannelGetUnitLabel;

// Convenience macros
#define IOReportCreateSubscription pIOReportCreateSubscription
#define IOReportCreateSamples pIOReportCreateSamples
#define IOReportCreateSamplesDelta pIOReportCreateSamplesDelta
#define IOReportIterate pIOReportIterate
#define IOReportCopyChannelsInGroup pIOReportCopyChannelsInGroup
#define IOReportMergeChannels pIOReportMergeChannels
#define IOReportChannelGetFormat pIOReportChannelGetFormat
#define IOReportSimpleGetIntegerValue pIOReportSimpleGetIntegerValue
#define IOReportChannelGetChannelName pIOReportChannelGetChannelName
#define IOReportChannelGetGroup pIOReportChannelGetGroup
#define IOReportChannelGetSubGroup pIOReportChannelGetSubGroup
#define IOReportChannelGetUnitLabel pIOReportChannelGetUnitLabel

#define kIOReportFormatSimple     1
#define kIOReportFormatState      2
#define kIOReportFormatHistogram  3

static int loadIOReport(void) {
    void *h = dlopen("/usr/lib/libIOReport.dylib", RTLD_NOW);
    if (!h) { printf("Failed to load libIOReport: %s\n", dlerror()); return -1; }
    #define LOAD(name) p##name = (name##_t)dlsym(h, #name); \
        if (!p##name) { printf("Missing: %s\n", #name); return -1; }
    LOAD(IOReportCreateSubscription)
    LOAD(IOReportCreateSamples)
    LOAD(IOReportCreateSamplesDelta)
    LOAD(IOReportIterate)
    LOAD(IOReportCopyChannelsInGroup)
    LOAD(IOReportMergeChannels)
    LOAD(IOReportChannelGetFormat)
    LOAD(IOReportSimpleGetIntegerValue)
    LOAD(IOReportChannelGetChannelName)
    LOAD(IOReportChannelGetGroup)
    LOAD(IOReportChannelGetSubGroup)
    LOAD(IOReportChannelGetUnitLabel)
    #undef LOAD
    return 0;
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        setbuf(stdout, NULL);
        int maxSamples = argc > 1 ? atoi(argv[1]) : 10;
        int intervalMs = argc > 2 ? atoi(argv[2]) : 1000;
        double interval = intervalMs / 1000.0;

        if (loadIOReport() != 0) return 1;

        printf("╔══════════════════════════════════════════════════════════╗\n");
        printf("║  Apple Silicon SoC Power Monitor (IOReport)            ║\n");
        printf("╚══════════════════════════════════════════════════════════╝\n\n");

        // Subscribe to energy model channels
        CFMutableDictionaryRef energyCh = IOReportCopyChannelsInGroup(
            CFSTR("Energy Model"), NULL, 0, 0, 0);

        // Also get CPU stats
        CFMutableDictionaryRef cpuCh = IOReportCopyChannelsInGroup(
            CFSTR("CPU Stats"), NULL, 0, 0, 0);

        // GPU stats
        CFMutableDictionaryRef gpuCh = IOReportCopyChannelsInGroup(
            CFSTR("GPU Stats"), NULL, 0, 0, 0);

        // Merge all channels
        CFMutableDictionaryRef allChannels = energyCh;
        if (cpuCh) { IOReportMergeChannels(allChannels, cpuCh, NULL); CFRelease(cpuCh); }
        if (gpuCh) { IOReportMergeChannels(allChannels, gpuCh, NULL); CFRelease(gpuCh); }

        if (!allChannels) {
            printf("Failed to get IOReport channels\n");
            return 1;
        }

        // Create subscription
        CFMutableDictionaryRef subbedChannels = NULL;
        IOReportSubscriptionRef sub = IOReportCreateSubscription(
            NULL, allChannels, &subbedChannels, 0, NULL);

        if (!sub) {
            printf("Failed to create IOReport subscription\n");
            CFRelease(allChannels);
            return 1;
        }

        printf("IOReport subscription created. Collecting %d samples at %dms interval...\n\n",
               maxSamples, intervalMs);

        // Take initial sample
        CFDictionaryRef prevSamples = IOReportCreateSamples(sub, subbedChannels, NULL);
        if (!prevSamples) {
            printf("Failed to create initial samples\n");
            return 1;
        }

        for (int s = 0; s < maxSamples; s++) {
            usleep(intervalMs * 1000);

            // Take new sample
            CFDictionaryRef curSamples = IOReportCreateSamples(sub, subbedChannels, NULL);
            if (!curSamples) continue;

            // Compute delta
            CFDictionaryRef delta = IOReportCreateSamplesDelta(prevSamples, curSamples, NULL);
            CFRelease(prevSamples);
            prevSamples = curSamples;

            if (!delta) continue;

            printf("\033[H\033[J"); // Clear screen
            printf("╔══════════════════════════════════════════════════════════╗\n");
            printf("║  SoC Power — Sample %d/%d  (interval: %dms)              \n",
                   s + 1, maxSamples, intervalMs);
            printf("╚══════════════════════════════════════════════════════════╝\n\n");

            // Iterate channels and display
            __block int energyCount = 0;
            __block int cpuCount = 0;

            printf("═══ Energy Model (mW) ═══\n");

            IOReportIterate(delta, ^(CFDictionaryRef channel) {
                CFStringRef group = IOReportChannelGetGroup(channel);
                CFStringRef name = IOReportChannelGetChannelName(channel);
                CFStringRef unit = IOReportChannelGetUnitLabel(channel);
                int32_t format = IOReportChannelGetFormat(channel);

                if (!group || !name) return 0;

                NSString *groupStr = (__bridge NSString *)group;
                NSString *nameStr = (__bridge NSString *)name;
                NSString *unitStr = unit ? (__bridge NSString *)unit : @"";

                if ([groupStr isEqualToString:@"Energy Model"] && format == kIOReportFormatSimple) {
                    int64_t val = IOReportSimpleGetIntegerValue(channel, 0);
                    if (val != 0) {
                        // Convert from nJ to mW (nJ per interval -> mW)
                        double mW = (double)val / (interval * 1e6);
                        printf("  %-30s %10.1f mW", [nameStr UTF8String], mW);

                        // Bar
                        int bars = (int)(mW / 200);
                        if (bars > 30) bars = 30;
                        printf("  ");
                        for (int b = 0; b < bars; b++) printf("█");
                        printf("\n");
                        energyCount++;
                    }
                }
                return 0;
            });

            if (energyCount == 0) {
                printf("  (no energy data - may need elevated privileges)\n");
            }

            // Show CPU/GPU stats
            printf("\n═══ CPU/GPU Stats ═══\n");
            IOReportIterate(delta, ^(CFDictionaryRef channel) {
                CFStringRef group = IOReportChannelGetGroup(channel);
                CFStringRef name = IOReportChannelGetChannelName(channel);
                int32_t format = IOReportChannelGetFormat(channel);

                if (!group || !name) return 0;
                NSString *groupStr = (__bridge NSString *)group;
                NSString *nameStr = (__bridge NSString *)name;

                if (([groupStr isEqualToString:@"CPU Stats"] ||
                     [groupStr isEqualToString:@"GPU Stats"]) &&
                    format == kIOReportFormatSimple) {
                    int64_t val = IOReportSimpleGetIntegerValue(channel, 0);
                    if (val != 0) {
                        printf("  [%s] %-25s %lld\n",
                               [groupStr UTF8String], [nameStr UTF8String], (long long)val);
                        cpuCount++;
                    }
                }
                return 0;
            });

            if (cpuCount == 0) {
                printf("  (no CPU/GPU stats available)\n");
            }

            // List all available groups
            if (s == 0) {
                printf("\n═══ Available IOReport Groups ═══\n");
                NSMutableSet *groups = [NSMutableSet set];
                IOReportIterate(delta, ^(CFDictionaryRef channel) {
                    CFStringRef group = IOReportChannelGetGroup(channel);
                    if (group) [groups addObject:(__bridge NSString *)group];
                    return 0;
                });
                for (NSString *g in [groups.allObjects sortedArrayUsingSelector:@selector(compare:)]) {
                    printf("  • %s\n", [g UTF8String]);
                }
            }

            CFRelease(delta);
        }

        CFRelease(prevSamples);
        printf("\nDone.\n");
    }
    return 0;
}
