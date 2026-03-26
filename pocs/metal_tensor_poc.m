// metal_tensor_poc.m — Create and manipulate GPU tensors via private Metal API
// Discovers and uses MTLTensorDescriptor + device.newTensorWithDescriptor:
//
// This is the undocumented Metal Tensor API that Apple uses internally
// for ML workloads on Apple Silicon GPUs.
//
// Compile: clang -o metal_tensor_poc metal_tensor_poc.m \
//          -framework Foundation -framework Metal -framework IOKit \
//          -lobjc -ldl -fobjc-arc
//
// Usage: ./metal_tensor_poc

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <mach/mach_time.h>

static mach_timebase_info_data_t g_tb;
static double ticksToMs(uint64_t t) { return (double)t * g_tb.numer / g_tb.denom / 1e6; }

static void dumpClassMethods(Class cls, const char *label) {
    unsigned int count = 0;
    Method *methods = class_copyMethodList(cls, &count);
    printf("  %s instance methods (%u):\n", label, count);
    for (unsigned int i = 0; i < count; i++) {
        printf("    - %s  [%s]\n", sel_getName(method_getName(methods[i])),
               method_getTypeEncoding(methods[i]) ?: "?");
    }
    free(methods);
    methods = class_copyMethodList(object_getClass(cls), &count);
    if (count > 0) {
        printf("  %s class methods (%u):\n", label, count);
        for (unsigned int i = 0; i < count; i++) {
            printf("    + %s  [%s]\n", sel_getName(method_getName(methods[i])),
                   method_getTypeEncoding(methods[i]) ?: "?");
        }
    }
    free(methods);
}

int main() {
    @autoreleasepool {
        mach_timebase_info(&g_tb);
        dlopen("/System/Library/PrivateFrameworks/IOGPU.framework/IOGPU", RTLD_NOW);

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) { printf("No Metal device\n"); return 1; }

        setbuf(stdout, NULL); // Disable buffering for crash debugging

        printf("╔══════════════════════════════════════════════════════════╗\n");
        printf("║  Metal Tensor API PoC — %s\n", [[device name] UTF8String]);
        printf("╚══════════════════════════════════════════════════════════╝\n\n");

        // === Check support ===
        SEL supportsSel = @selector(supportsTensors);
        SEL supportsMLSel = @selector(supportsMachineLearningCommandEncoders);
        if ([(id)device respondsToSelector:supportsSel]) {
            BOOL sup = ((BOOL(*)(id,SEL))objc_msgSend)((id)device, supportsSel);
            printf("supportsTensors: %s\n", sup ? "YES" : "NO");
            if (!sup) { printf("Device doesn't support tensors\n"); return 1; }
        }
        if ([(id)device respondsToSelector:supportsMLSel]) {
            BOOL sup = ((BOOL(*)(id,SEL))objc_msgSend)((id)device, supportsMLSel);
            printf("supportsMachineLearningCommandEncoders: %s\n", sup ? "YES" : "NO");
        }

        // === Discover MTLTensorDescriptor ===
        Class TensorDesc = NSClassFromString(@"MTLTensorDescriptor");
        if (!TensorDesc) {
            printf("MTLTensorDescriptor not found!\n");
            return 1;
        }
        printf("\n═══ MTLTensorDescriptor ═══\n");
        dumpClassMethods(TensorDesc, "MTLTensorDescriptor");

        // Properties
        unsigned int propCount = 0;
        objc_property_t *props = class_copyPropertyList(TensorDesc, &propCount);
        printf("  Properties (%u):\n", propCount);
        for (unsigned int i = 0; i < propCount; i++) {
            printf("    @property %s  [%s]\n",
                   property_getName(props[i]),
                   property_getAttributes(props[i]) ?: "?");
        }
        free(props);

        // === Discover MTLTensorExtents ===
        Class TensorExtents = NSClassFromString(@"MTLTensorExtents");
        if (TensorExtents) {
            printf("\n═══ MTLTensorExtents ═══\n");
            dumpClassMethods(TensorExtents, "MTLTensorExtents");
        }

        // === Try to create a tensor descriptor ===
        printf("\n═══ Creating Tensor Descriptor ═══\n");

        // Try alloc+init
        id desc = nil;
        @try {
            desc = [[TensorDesc alloc] init];
            if (desc) {
                printf("  Created via alloc/init: %s\n", [[desc description] UTF8String]);
            }
        } @catch (NSException *ex) {
            printf("  alloc/init failed: %s\n", [[ex reason] UTF8String]);
        }

        // Look for factory methods
        SEL factorySels[] = {
            @selector(tensorDescriptorWithDataType:dimensions:),
            @selector(tensorDescriptorWithDataType:shape:),
            @selector(descriptorWithDataType:dimensions:),
            @selector(descriptorWithDataType:shape:),
        };
        const char *factoryNames[] = {
            "tensorDescriptorWithDataType:dimensions:",
            "tensorDescriptorWithDataType:shape:",
            "descriptorWithDataType:dimensions:",
            "descriptorWithDataType:shape:",
        };

        for (int i = 0; i < 4; i++) {
            if ([TensorDesc respondsToSelector:factorySels[i]]) {
                printf("  Factory method found: +%s\n", factoryNames[i]);
            }
        }

        // Try setting properties on desc if we got one
        if (desc) {
            // Try to set dataType (MTLDataTypeFloat16 = 25, Float32 = 3)
            SEL setDataType = @selector(setDataType:);
            SEL setDims = @selector(setDimensions:);
            if ([desc respondsToSelector:setDataType]) {
                @try {
                    ((void(*)(id,SEL,NSUInteger))objc_msgSend)(desc, setDataType, 3); // float32
                    printf("  Set dataType = Float32 (3)\n");
                } @catch (NSException *ex) {
                    printf("  setDataType failed: %s\n", [[ex reason] UTF8String]);
                }
            }

            // Try to read back
            SEL getDataType = @selector(dataType);
            if ([desc respondsToSelector:getDataType]) {
                @try {
                    NSUInteger dt = ((NSUInteger(*)(id,SEL))objc_msgSend)(desc, getDataType);
                    printf("  dataType = %lu\n", (unsigned long)dt);
                } @catch (NSException *ex) {}
            }

            // Explore what dimensions looks like
            SEL getDims = @selector(dimensions);
            if ([desc respondsToSelector:getDims]) {
                @try {
                    id dims = ((id(*)(id,SEL))objc_msgSend)(desc, getDims);
                    if (dims) {
                        printf("  dimensions class: %s\n", class_getName([dims class]));
                        printf("  dimensions: %s\n", [[dims description] UTF8String]);
                    } else {
                        printf("  dimensions: nil\n");
                    }
                } @catch (NSException *ex) {}
            }

            // Set dimensions via MTLTensorExtents (must use the real class, not NSArray)
            if (TensorExtents && [desc respondsToSelector:setDims]) {
                @try {
                    // MTLTensorExtents.initWithRank:extents: takes (uint64, const int64_t*)
                    int64_t extents[] = {4, 256, 256};
                    id tensorExtents = ((id(*)(id,SEL,uint64_t,const int64_t*))objc_msgSend)(
                        [TensorExtents alloc],
                        @selector(initWithRank:values:),
                        (uint64_t)3, extents);
                    if (tensorExtents) {
                        printf("  Created MTLTensorExtents: %s\n", [[tensorExtents description] UTF8String]);
                        ((void(*)(id,SEL,id))objc_msgSend)(desc, setDims, tensorExtents);
                        printf("  Set dimensions = [4, 256, 256]\n");

                        // Read back
                        id dims = ((id(*)(id,SEL))objc_msgSend)(desc, getDims);
                        if (dims) printf("  dimensions readback: %s\n", [[dims description] UTF8String]);
                    } else {
                        printf("  MTLTensorExtents creation returned nil\n");
                    }
                } @catch (NSException *ex) {
                    printf("  setDimensions failed: %s\n", [[ex reason] UTF8String]);
                }
            }
        }

        // === Try to create tensor from device ===
        printf("\n═══ Creating Tensor from Device ═══\n");

        SEL newTensorSel = @selector(newTensorWithDescriptor:error:);
        if ([(id)device respondsToSelector:newTensorSel]) {
            printf("  Device supports newTensorWithDescriptor:error:\n");

            if (desc) {
                @try {
                    NSError *err = nil;
                    id tensor = ((id(*)(id,SEL,id,NSError**))objc_msgSend)(
                        (id)device, newTensorSel, desc, &err);
                    if (tensor) {
                        printf("  TENSOR CREATED!\n");
                        printf("    Class: %s\n", class_getName([tensor class]));
                        printf("    Description: %s\n", [[tensor description] UTF8String]);

                        // Read scalar properties safely
                        NSUInteger allocSize = ((NSUInteger(*)(id,SEL))objc_msgSend)(tensor, @selector(allocatedSize));
                        printf("    allocatedSize = %lu bytes\n", (unsigned long)allocSize);

                        NSInteger dt = ((NSInteger(*)(id,SEL))objc_msgSend)(tensor, @selector(dataType));
                        printf("    dataType = %ld\n", (long)dt);

                        NSUInteger usage = ((NSUInteger(*)(id,SEL))objc_msgSend)(tensor, @selector(usage));
                        printf("    usage = %lu\n", (unsigned long)usage);

                        NSUInteger offset = ((NSUInteger(*)(id,SEL))objc_msgSend)(tensor, @selector(offset));
                        printf("    offset = %lu\n", (unsigned long)offset);

                        // Object properties
                        id dims = ((id(*)(id,SEL))objc_msgSend)(tensor, @selector(dimensions));
                        if (dims) printf("    dimensions = %s\n", [[dims description] UTF8String]);

                        id tensorStrides = ((id(*)(id,SEL))objc_msgSend)(tensor, @selector(strides));
                        if (tensorStrides) printf("    strides = %s\n", [[tensorStrides description] UTF8String]);

                        // === Write data to tensor ===
                        printf("\n═══ Writing Data to Tensor ═══\n");
                        SEL replaceSel = @selector(replaceSlice:withBytes:strides:);
                        SEL getBytesSel = @selector(getBytes:strides:fromSlice:);

                        if ([tensor respondsToSelector:replaceSel]) {
                            printf("  Tensor supports replaceSlice:withBytes:strides:\n");
                        }
                        if ([tensor respondsToSelector:getBytesSel]) {
                            printf("  Tensor supports getBytes:strides:fromSlice:\n");
                        }

                        // Try buffer access
                        SEL bufferSel = @selector(buffer);
                        if ([tensor respondsToSelector:bufferSel]) {
                            id buffer = ((id(*)(id,SEL))objc_msgSend)(tensor, bufferSel);
                            if (buffer) {
                                printf("  Buffer: %s\n", [[buffer description] UTF8String]);
                                SEL contentsSel = @selector(contents);
                                if ([buffer respondsToSelector:contentsSel]) {
                                    void *ptr = ((void*(*)(id,SEL))objc_msgSend)(buffer, contentsSel);
                                    if (ptr) {
                                        printf("  Buffer contents ptr: %p\n", ptr);
                                        // Write test data
                                        float *data = (float *)ptr;
                                        for (int i = 0; i < 16; i++) {
                                            data[i] = (float)i * 0.1f;
                                        }
                                        printf("  Wrote 16 floats to tensor\n");
                                        printf("  Data: [%.1f, %.1f, %.1f, %.1f, ...]\n",
                                               data[0], data[1], data[2], data[3]);

                                        // Read back
                                        printf("  Readback: [%.1f, %.1f, %.1f, %.1f, ...]\n",
                                               data[0], data[1], data[2], data[3]);
                                        printf("  TENSOR READ/WRITE WORKS!\n");
                                    }
                                }
                            }
                        }

                        // === Try tensor view/reshape ===
                        printf("\n═══ Tensor View/Reshape ═══\n");
                        SEL reshapeSel = @selector(newTensorViewWithReshapedDescriptor:error:);
                        SEL sliceSel = @selector(newTensorViewWithSlice:error:);
                        SEL canReshapeSel = @selector(isTensorViewableWithReshapedDescriptor:);

                        if ([tensor respondsToSelector:reshapeSel])
                            printf("  Supports reshape: YES\n");
                        if ([tensor respondsToSelector:sliceSel])
                            printf("  Supports slice: YES\n");
                        if ([tensor respondsToSelector:canReshapeSel])
                            printf("  Supports reshape query: YES\n");

                    } else {
                        printf("  Tensor creation returned nil\n");
                        if (err) printf("  Error: %s\n", [[err description] UTF8String]);
                    }
                } @catch (NSException *ex) {
                    printf("  Exception: %s\n", [[ex reason] UTF8String]);
                }
            }
        }

        // === Try creating tensor from buffer ===
        printf("\n═══ Creating Tensor from Buffer ═══\n");
        SEL newTensorBufSel = @selector(newTensorWithBuffer:descriptor:offset:strides:error:);
        if ([(id)device respondsToSelector:newTensorBufSel]) {
            printf("  Device supports newTensorWithBuffer:descriptor:offset:strides:\n");

            // Create a Metal buffer with test data
            float testData[1024];
            for (int i = 0; i < 1024; i++) testData[i] = (float)i / 1024.0f;

            id<MTLBuffer> buffer = [device newBufferWithBytes:testData
                                                      length:sizeof(testData)
                                                     options:MTLResourceStorageModeShared];
            printf("  Buffer: %lu bytes\n", (unsigned long)[buffer length]);

            if (desc && TensorExtents) {
                @try {
                    NSError *err = nil;
                    // Strides must be MTLTensorExtents too
                    // For [4, 256, 256] float32: strides = [256*256, 256, 1] (element counts)
                    int64_t strideVals[] = {256*256, 256, 1};
                    id strides = ((id(*)(id,SEL,uint64_t,const int64_t*))objc_msgSend)(
                        [TensorExtents alloc], @selector(initWithRank:values:), (uint64_t)3, strideVals);
                    id tensor = ((id(*)(id,SEL,id,id,NSUInteger,id,NSError**))objc_msgSend)(
                        (id)device, newTensorBufSel, buffer, desc, 0, strides, &err);
                    if (tensor) {
                        printf("  TENSOR FROM BUFFER CREATED!\n");
                        printf("    Class: %s\n", class_getName([tensor class]));

                        // Verify data
                        SEL bufSel = @selector(buffer);
                        if ([tensor respondsToSelector:bufSel]) {
                            id tbuf = ((id(*)(id,SEL))objc_msgSend)(tensor, bufSel);
                            if (tbuf) {
                                SEL contentsSel = @selector(contents);
                                if ([tbuf respondsToSelector:contentsSel]) {
                                    float *data = ((float*(*)(id,SEL))objc_msgSend)(tbuf, contentsSel);
                                    if (data) {
                                        printf("    Data[0..3]: [%.4f, %.4f, %.4f, %.4f]\n",
                                               data[0], data[1], data[2], data[3]);
                                        printf("    Data matches input: %s\n",
                                               (data[0] == 0.0f && fabsf(data[1] - 1.0f/1024.0f) < 1e-6) ?
                                               "YES" : "NO");
                                    }
                                }
                            }
                        }
                    } else {
                        printf("  Creation returned nil\n");
                        if (err) printf("  Error: %s\n", [[err description] UTF8String]);
                    }
                } @catch (NSException *ex) {
                    printf("  Exception: %s\n", [[ex reason] UTF8String]);
                }
            }
        }

        // === Size/align query ===
        printf("\n═══ Tensor Size & Alignment ═══\n");
        SEL sizeAlignSel = @selector(tensorSizeAndAlignWithDescriptor:);
        if (desc && [(id)device respondsToSelector:sizeAlignSel]) {
            @try {
                // Returns a struct {size_t size, size_t align}
                typedef struct { NSUInteger size; NSUInteger align; } SizeAlign;
                SizeAlign sa = ((SizeAlign(*)(id,SEL,id))objc_msgSend)(
                    (id)device, sizeAlignSel, desc);
                printf("  Size: %lu bytes, Alignment: %lu bytes\n",
                       (unsigned long)sa.size, (unsigned long)sa.align);
            } @catch (NSException *ex) {
                printf("  Exception: %s\n", [[ex reason] UTF8String]);
            }
        }

        printf("\n═══ Done ═══\n");
    }
    return 0;
}
