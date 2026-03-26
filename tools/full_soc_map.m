// full_soc_map.m — Complete Apple Silicon SoC mapping
// Discovers ALL IOReport channels, thermal sensors, performance states,
// frequency domains, and interconnect stats
//
// Compile: clang -o full_soc_map full_soc_map.m \
//          -framework Foundation -framework IOKit -lobjc -ldl -fobjc-arc
//
// Usage: ./full_soc_map

#import <Foundation/Foundation.h>
#import <IOKit/IOKitLib.h>
#import <dlfcn.h>

// IOReport function types
typedef struct __IOReportSubscriptionCF *IOReportSubscriptionRef;
typedef IOReportSubscriptionRef (*fn_CreateSubscription)(void*, CFMutableDictionaryRef, CFMutableDictionaryRef*, uint64_t, CFTypeRef);
typedef CFDictionaryRef (*fn_CreateSamples)(IOReportSubscriptionRef, CFMutableDictionaryRef, CFTypeRef);
typedef CFDictionaryRef (*fn_CreateSamplesDelta)(CFDictionaryRef, CFDictionaryRef, CFTypeRef);
typedef void (*fn_Iterate)(CFDictionaryRef, int(^)(CFDictionaryRef));
typedef CFMutableDictionaryRef (*fn_CopyChannelsInGroup)(CFStringRef, CFStringRef, uint64_t, uint64_t, uint64_t);
typedef void (*fn_MergeChannels)(CFMutableDictionaryRef, CFMutableDictionaryRef, CFTypeRef);
typedef int32_t (*fn_GetFormat)(CFDictionaryRef);
typedef int64_t (*fn_SimpleGetIntegerValue)(CFDictionaryRef, int32_t);
typedef CFStringRef (*fn_GetChannelName)(CFDictionaryRef);
typedef CFStringRef (*fn_GetGroup)(CFDictionaryRef);
typedef CFStringRef (*fn_GetSubGroup)(CFDictionaryRef);
typedef CFStringRef (*fn_GetUnitLabel)(CFDictionaryRef);
typedef int (*fn_StateGetCount)(CFDictionaryRef);
typedef CFStringRef (*fn_StateGetNameForIndex)(CFDictionaryRef, int);
typedef int64_t (*fn_StateGetResidency)(CFDictionaryRef, int);

static fn_CreateSubscription pCreateSub;
static fn_CreateSamples pCreateSamples;
static fn_CreateSamplesDelta pCreateDelta;
static fn_Iterate pIterate;
static fn_CopyChannelsInGroup pCopyChannels;
static fn_MergeChannels pMergeChannels;
static fn_GetFormat pGetFormat;
static fn_SimpleGetIntegerValue pSimpleGetInt;
static fn_GetChannelName pGetName;
static fn_GetGroup pGetGroup;
static fn_GetSubGroup pGetSubGroup;
static fn_GetUnitLabel pGetUnit;
static fn_StateGetCount pStateGetCount;
static fn_StateGetNameForIndex pStateGetName;
static fn_StateGetResidency pStateGetResidency;

int main() {
    @autoreleasepool {
        setbuf(stdout, NULL);

        void *h = dlopen("/usr/lib/libIOReport.dylib", RTLD_NOW);
        if (!h) { printf("Cannot load libIOReport\n"); return 1; }

        #define LOAD(var, name) var = (typeof(var))dlsym(h, name)
        LOAD(pCreateSub, "IOReportCreateSubscription");
        LOAD(pCreateSamples, "IOReportCreateSamples");
        LOAD(pCreateDelta, "IOReportCreateSamplesDelta");
        LOAD(pIterate, "IOReportIterate");
        LOAD(pCopyChannels, "IOReportCopyChannelsInGroup");
        LOAD(pMergeChannels, "IOReportMergeChannels");
        LOAD(pGetFormat, "IOReportChannelGetFormat");
        LOAD(pSimpleGetInt, "IOReportSimpleGetIntegerValue");
        LOAD(pGetName, "IOReportChannelGetChannelName");
        LOAD(pGetGroup, "IOReportChannelGetGroup");
        LOAD(pGetSubGroup, "IOReportChannelGetSubGroup");
        LOAD(pGetUnit, "IOReportChannelGetUnitLabel");
        LOAD(pStateGetCount, "IOReportStateGetCount");
        LOAD(pStateGetName, "IOReportStateGetNameForIndex");
        LOAD(pStateGetResidency, "IOReportStateGetResidency");

        printf("╔══════════════════════════════════════════════════════════╗\n");
        printf("║  Complete Apple Silicon SoC Map                        ║\n");
        printf("╚══════════════════════════════════════════════════════════╝\n\n");

        // ALL known IOReport groups to try
        const char *groups[] = {
            "Energy Model",
            "CPU Stats",
            "GPU Stats",
            "CLPC Stats",
            "AMC Stats",
            "PMP",
            "GPU Performance Statistics",
            "Interconnect Stats",
            "IOAccelerator",
            "AppleARMIODevice",
            "DCS Stats",
            "SoC Thermal Sensors",
            "CPU Complex Performance States",
            "CPU Core Performance States",
            "GPU Performance States",
            "ANE Performance States",
            "ISP Performance States",
            "Display",
            "Thunderbolt",
            "NVMe",
            "USB",
            "PCIe",
            "Network",
            "Memory",
            "Fabric Stats",
            "DRAM Stats",
            "Power States",
            "Idle States",
            "Thermal",
            "Voltage",
            "Frequency",
            "CPU Cycles",
            "Instructions",
            "Cache",
            "TLB",
            "Branch",
            NULL
        };

        // Merge all channels
        CFMutableDictionaryRef allCh = NULL;
        int groupsFound = 0;

        for (int i = 0; groups[i]; i++) {
            CFStringRef groupName = CFStringCreateWithCString(NULL, groups[i], kCFStringEncodingUTF8);
            CFMutableDictionaryRef ch = pCopyChannels(groupName, NULL, 0, 0, 0);
            CFRelease(groupName);
            if (ch) {
                if (!allCh) {
                    allCh = ch;
                } else {
                    pMergeChannels(allCh, ch, NULL);
                    CFRelease(ch);
                }
                groupsFound++;
            }
        }

        // Also try NULL group to get ALL channels
        CFMutableDictionaryRef allGroupsCh = pCopyChannels(NULL, NULL, 0, 0, 0);
        if (allGroupsCh) {
            if (!allCh) {
                allCh = allGroupsCh;
            } else {
                pMergeChannels(allCh, allGroupsCh, NULL);
                CFRelease(allGroupsCh);
            }
        }

        if (!allCh) { printf("No channels found\n"); return 1; }

        // Subscribe
        CFMutableDictionaryRef subbedCh = NULL;
        IOReportSubscriptionRef sub = pCreateSub(NULL, allCh, &subbedCh, 0, NULL);
        if (!sub) { printf("Subscription failed\n"); return 1; }

        // Sample twice to get delta
        CFDictionaryRef s1 = pCreateSamples(sub, subbedCh, NULL);
        usleep(500000); // 500ms
        CFDictionaryRef s2 = pCreateSamples(sub, subbedCh, NULL);
        CFDictionaryRef delta = pCreateDelta(s1, s2, NULL);

        if (!delta) { printf("Delta failed\n"); return 1; }

        // Iterate ALL channels and categorize
        NSMutableDictionary *groupMap = [NSMutableDictionary dictionary];
        __block int totalChannels = 0;

        pIterate(delta, ^(CFDictionaryRef ch) {
            CFStringRef group = pGetGroup(ch);
            CFStringRef name = pGetName(ch);
            CFStringRef subgroup = pGetSubGroup ? pGetSubGroup(ch) : NULL;
            CFStringRef unit = pGetUnit ? pGetUnit(ch) : NULL;
            int32_t format = pGetFormat(ch);

            if (!group || !name) return 0;
            totalChannels++;

            NSString *g = (__bridge NSString *)group;
            NSString *n = (__bridge NSString *)name;
            NSString *sg = subgroup ? (__bridge NSString *)subgroup : @"";
            NSString *u = unit ? (__bridge NSString *)unit : @"";

            if (!groupMap[g]) groupMap[g] = [NSMutableArray array];

            NSString *formatStr = @"?";
            if (format == 1) formatStr = @"simple";
            else if (format == 2) formatStr = @"state";
            else if (format == 3) formatStr = @"histogram";

            NSMutableString *info = [NSMutableString stringWithFormat:@"  %-35s", [n UTF8String]];
            [info appendFormat:@" [%@]", formatStr];
            if (sg.length > 0) [info appendFormat:@" sub=%@", sg];
            if (u.length > 0) [info appendFormat:@" unit=%@", u];

            if (format == 1) { // Simple
                int64_t val = pSimpleGetInt(ch, 0);
                if (val != 0) [info appendFormat:@" val=%lld", (long long)val];
            }
            else if (format == 2 && pStateGetCount) { // State
                int count = pStateGetCount(ch);
                if (count > 0 && count < 20) {
                    [info appendString:@" states={"];
                    for (int i = 0; i < count; i++) {
                        CFStringRef sname = pStateGetName(ch, i);
                        int64_t res = pStateGetResidency ? pStateGetResidency(ch, i) : 0;
                        if (sname) {
                            if (i > 0) [info appendString:@", "];
                            [info appendFormat:@"%@:%lld", (__bridge NSString*)sname, (long long)res];
                        }
                    }
                    [info appendString:@"}"];
                }
            }

            [groupMap[g] addObject:info];
            return 0;
        });

        // Print everything organized by group
        NSArray *sortedGroups = [[groupMap allKeys] sortedArrayUsingSelector:@selector(compare:)];

        for (NSString *group in sortedGroups) {
            NSArray *channels = groupMap[group];
            printf("\n═══ %s (%lu channels) ═══\n", [group UTF8String], (unsigned long)[channels count]);
            for (NSString *ch in channels) {
                printf("%s\n", [ch UTF8String]);
            }
        }

        printf("\n════════════════════════════════════════════\n");
        printf("TOTAL: %d channels across %lu groups\n",
               totalChannels, (unsigned long)[groupMap count]);

        CFRelease(delta);
        CFRelease(s1);
        CFRelease(s2);
    }
    return 0;
}
