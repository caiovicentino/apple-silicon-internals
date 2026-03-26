// capacity_test.m — Measure real capacity for model training & inference on M4
// Tests memory limits, throughput at scale, and estimates max model sizes
//
// Compile: clang -o capacity_test capacity_test.m \
//          -framework Foundation -framework Metal -framework Accelerate \
//          -framework IOKit -lobjc -ldl -fobjc-arc -O2 -DACCELERATE_NEW_LAPACK
//
// Usage: ./capacity_test

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <Accelerate/Accelerate.h>
#import <mach/mach_time.h>
#import <mach/mach.h>
#import <sys/sysctl.h>
#import <objc/runtime.h>
#import <objc/message.h>

static mach_timebase_info_data_t g_tb;
static double ticksToMs(uint64_t t) { return (double)t*g_tb.numer/g_tb.denom/1e6; }

// Get current memory usage
static size_t getMemoryUsage(void) {
    struct task_basic_info info;
    mach_msg_type_number_t count = TASK_BASIC_INFO_COUNT;
    task_info(mach_task_self(), TASK_BASIC_INFO, (task_info_t)&info, &count);
    return info.resident_size;
}

// Simulate transformer layer forward pass (CPU)
// Ops per layer: 4 matmuls (QKV proj + output proj) + attention + 2 FFN matmuls
static double transformer_layer_forward_cpu(int batch, int seq, int dim, int heads, int ffn_dim) {
    // Q,K,V projection: [batch*seq, dim] @ [dim, dim] × 3
    // Attention: [batch*heads, seq, dim/heads] @ [batch*heads, dim/heads, seq] (Q@K^T)
    //            [batch*heads, seq, seq] @ [batch*heads, seq, dim/heads] (attn@V)
    // Output proj: [batch*seq, dim] @ [dim, dim]
    // FFN: [batch*seq, dim] @ [dim, ffn_dim] + [batch*seq, ffn_dim] @ [ffn_dim, dim]

    int total = batch * seq;
    int head_dim = dim / heads;

    // Allocate all matrices
    float *X = calloc(total * dim, sizeof(float));
    float *W_qkv = calloc(dim * dim * 3, sizeof(float));
    float *QKV = calloc(total * dim * 3, sizeof(float));
    float *W_out = calloc(dim * dim, sizeof(float));
    float *W_ff1 = calloc(dim * ffn_dim, sizeof(float));
    float *W_ff2 = calloc(ffn_dim * dim, sizeof(float));
    float *H = calloc(total * ffn_dim, sizeof(float));
    float *Out = calloc(total * dim, sizeof(float));

    // Random init
    for (int i=0; i<total*dim; i++) X[i] = ((float)arc4random()/UINT32_MAX-0.5f)*0.1f;
    for (int i=0; i<dim*dim*3; i++) W_qkv[i] = ((float)arc4random()/UINT32_MAX-0.5f)*0.02f;
    for (int i=0; i<dim*dim; i++) W_out[i] = ((float)arc4random()/UINT32_MAX-0.5f)*0.02f;
    for (int i=0; i<dim*ffn_dim; i++) W_ff1[i] = ((float)arc4random()/UINT32_MAX-0.5f)*0.02f;
    for (int i=0; i<ffn_dim*dim; i++) W_ff2[i] = ((float)arc4random()/UINT32_MAX-0.5f)*0.02f;

    // Warmup
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                total, dim*3, dim, 1.0f, X, dim, W_qkv, dim*3, 0.0f, QKV, dim*3);

    // Benchmark
    uint64_t t0 = mach_absolute_time();

    // QKV projection
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                total, dim*3, dim, 1.0f, X, dim, W_qkv, dim*3, 0.0f, QKV, dim*3);

    // Output projection
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                total, dim, dim, 1.0f, QKV, dim*3, W_out, dim, 0.0f, Out, dim);

    // FFN up
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                total, ffn_dim, dim, 1.0f, Out, dim, W_ff1, ffn_dim, 0.0f, H, ffn_dim);

    // ReLU
    for (int i=0; i<total*ffn_dim; i++) if (H[i]<0) H[i]=0;

    // FFN down
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                total, dim, ffn_dim, 1.0f, H, ffn_dim, W_ff2, dim, 0.0f, Out, dim);

    double ms = ticksToMs(mach_absolute_time()-t0);

    free(X); free(W_qkv); free(QKV); free(W_out);
    free(W_ff1); free(W_ff2); free(H); free(Out);

    return ms;
}

// Compute FLOPs for one transformer layer
static double transformer_layer_flops(int batch, int seq, int dim, int ffn_dim) {
    int total = batch * seq;
    double flops = 0;
    flops += 2.0 * total * dim * (dim*3);   // QKV proj
    flops += 2.0 * total * dim * dim;        // Output proj
    flops += 2.0 * total * dim * ffn_dim;    // FFN up
    flops += 2.0 * total * ffn_dim * dim;    // FFN down
    // Attention: Q@K^T + attn@V (simplified)
    flops += 2.0 * batch * seq * seq * dim;  // Q@K^T
    flops += 2.0 * batch * seq * seq * dim;  // attn@V
    return flops;
}

// Compute parameter count for a transformer model
static double transformer_params(int layers, int dim, int ffn_dim, int vocab) {
    double per_layer = 0;
    per_layer += dim * dim * 3;   // QKV weights
    per_layer += dim * dim;       // Output proj
    per_layer += dim * ffn_dim;   // FFN up
    per_layer += ffn_dim * dim;   // FFN down
    per_layer += dim * 4;         // LayerNorm params (approx)

    double total = per_layer * layers;
    total += vocab * dim;  // Embedding
    total += vocab * dim;  // LM head
    return total;
}

// Compute memory for training
static double training_memory_bytes(double params, const char *precision) {
    if (strcmp(precision, "fp32") == 0) {
        // weights + gradients + Adam(m,v) = 4 + 4 + 8 = 16 bytes/param
        return params * 16.0;
    } else if (strcmp(precision, "fp16") == 0) {
        // mixed precision: fp16 weights + fp32 master + fp16 grads + fp32 Adam
        // = 2 + 4 + 2 + 8 = 16 bytes/param
        return params * 16.0;
    } else if (strcmp(precision, "bf16") == 0) {
        return params * 16.0;
    } else if (strcmp(precision, "int8") == 0) {
        // QLoRA: base model int8 + LoRA adapters fp16
        // ~1 byte/param for frozen + LoRA ~1-5% trainable
        return params * 1.2;
    } else if (strcmp(precision, "int4") == 0) {
        return params * 0.7; // 4-bit + scale factors
    }
    return params * 4.0;
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        setbuf(stdout, NULL);
        mach_timebase_info(&g_tb);

        printf("╔══════════════════════════════════════════════════════════╗\n");
        printf("║  Apple M4 Model Capacity Analysis                      ║\n");
        printf("╚══════════════════════════════════════════════════════════╝\n\n");

        // === System Info ===
        uint64_t memsize = 0;
        size_t sz = sizeof(memsize);
        sysctlbyname("hw.memsize", &memsize, &sz, NULL, 0);
        double totalGB = memsize / (1024.0*1024*1024);

        uint64_t usable = 0;
        sz = sizeof(usable);
        sysctlbyname("hw.memsize_usable", &usable, &sz, NULL, 0);
        double usableGB = usable / (1024.0*1024*1024);

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        double gpuMaxGB = [device recommendedMaxWorkingSetSize] / (1024.0*1024*1024);

        printf("═══ Memory ═══\n");
        printf("  Total RAM:        %.1f GB\n", totalGB);
        printf("  Usable:           %.1f GB\n", usableGB);
        printf("  GPU max working:  %.1f GB\n", gpuMaxGB);
        printf("  Current usage:    %.1f MB\n", getMemoryUsage()/(1024.0*1024));

        // === CPU Peak FLOPS ===
        printf("\n═══ Peak Compute (measured) ═══\n");
        double peak_ms = 0;
        {
            int S = 2048;
            float *A = malloc(S*S*sizeof(float));
            float *B = malloc(S*S*sizeof(float));
            float *C = malloc(S*S*sizeof(float));
            for (int i=0; i<S*S; i++) { A[i]=0.01f; B[i]=0.01f; }
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        S,S,S,1.0f,A,S,B,S,0.0f,C,S);
            int iters = 5;
            uint64_t t0 = mach_absolute_time();
            for (int i=0; i<iters; i++)
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            S,S,S,1.0f,A,S,B,S,0.0f,C,S);
            peak_ms = ticksToMs(mach_absolute_time()-t0)/iters;
            free(A); free(B); free(C);
        }
        double cpu_tflops_fp32 = 2.0*2048*2048*2048 / (peak_ms*1e9);
        // FP16 is ~2x FP32 on SME2
        double cpu_tflops_fp16 = cpu_tflops_fp32 * 2.0;
        printf("  CPU FP32: %.2f TFLOPS (measured, Accelerate/SME2)\n", cpu_tflops_fp32);
        printf("  CPU FP16: ~%.2f TFLOPS (estimated, 2x FP32)\n", cpu_tflops_fp16);
        printf("  CPU INT8: ~%.2f TOPS (estimated, 4x FP32)\n", cpu_tflops_fp32 * 4);
        printf("  GPU FP32: ~3.5 TFLOPS (Apple spec)\n");
        printf("  GPU FP16: ~7 TFLOPS (Apple spec)\n");
        printf("  ANE:      ~38 TOPS INT8 (Apple spec)\n");

        // === Memory capacity table ===
        printf("\n═══ Memory Capacity: What fits in %.0f GB ═══\n", usableGB);
        printf("  ┌─────────────────┬───────────┬───────────┬───────────┬───────────┐\n");
        printf("  │ Precision       │ Inference │ LoRA      │ Full Train│ Full+Adam │\n");
        printf("  │                 │ (weights) │ (~1.2B/p) │ (w+grad)  │ (w+g+opt) │\n");
        printf("  ├─────────────────┼───────────┼───────────┼───────────┼───────────┤\n");

        struct { const char *name; double bytes_per_param_inf; double bytes_per_param_train; } precs[] = {
            {"FP32",  4.0, 16.0},
            {"FP16",  2.0, 16.0},
            {"BF16",  2.0, 16.0},
            {"INT8",  1.0,  1.2},
            {"INT4",  0.5,  0.7},
            {NULL, 0, 0}
        };

        double avail = usableGB * 1e9 * 0.85; // 85% usable (OS overhead)
        for (int i=0; precs[i].name; i++) {
            double inf_params = avail / precs[i].bytes_per_param_inf;
            double lora_params = avail / precs[i].bytes_per_param_train; // Frozen base
            double train_params = avail / (precs[i].bytes_per_param_inf * 2); // w+grad
            double full_train = avail / precs[i].bytes_per_param_train;

            printf("  │ %-15s │ %6.1fB   │ %6.1fB   │ %6.1fB   │ %6.1fB   │\n",
                   precs[i].name,
                   inf_params/1e9, lora_params/1e9,
                   train_params/1e9, full_train/1e9);
        }
        printf("  └─────────────────┴───────────┴───────────┴───────────┴───────────┘\n");
        printf("  * Inference = weights only. Training includes activations overhead.\n");
        printf("  * LoRA/QLoRA: freeze base model, train ~1-5%% of params.\n");

        // === Known models that fit ===
        printf("\n═══ Model Compatibility ═══\n");
        struct {
            const char *name;
            double params;
            int layers, dim, ffn, vocab;
        } models[] = {
            {"GPT-2 Small",          124e6,  12,  768, 3072, 50257},
            {"GPT-2 Medium",         355e6,  24, 1024, 4096, 50257},
            {"GPT-2 Large",          774e6,  36, 1280, 5120, 50257},
            {"Llama-3.2 1B",           1.2e9, 16, 2048, 8192, 128256},
            {"Phi-3 Mini 3.8B",        3.8e9, 32, 3072,  8192, 32064},
            {"Llama-3.2 3B",           3.2e9, 28, 3072, 8192, 128256},
            {"Qwen-2.5 7B",            7.6e9, 28, 3584,18944, 152064},
            {"Llama-3.1 8B",           8.0e9, 32, 4096,14336, 128256},
            {"Llama-3.1 70B",         70.0e9, 80, 8192,28672, 128256},
            {NULL, 0, 0, 0, 0, 0}
        };

        printf("  ┌──────────────────┬─────────┬────────┬────────┬────────┬────────┬────────┐\n");
        printf("  │ Model            │ Params  │ FP16   │ INT8   │ INT4   │ Train  │ LoRA   │\n");
        printf("  │                  │         │ (inf)  │ (inf)  │ (inf)  │ FP16   │ INT4   │\n");
        printf("  ├──────────────────┼─────────┼────────┼────────┼────────┼────────┼────────┤\n");

        for (int i=0; models[i].name; i++) {
            double p = models[i].params;
            double fp16_gb = p * 2.0 / 1e9;
            double int8_gb = p * 1.0 / 1e9;
            double int4_gb = p * 0.5 / 1e9;
            double train_gb = p * 16.0 / 1e9;
            double lora_gb = p * 0.5 / 1e9 + p * 0.02 * 16.0 / 1e9; // INT4 base + 2% LoRA FP16

            const char *fp16_ok = fp16_gb < usableGB*0.85 ? " OK " : " -- ";
            const char *int8_ok = int8_gb < usableGB*0.85 ? " OK " : " -- ";
            const char *int4_ok = int4_gb < usableGB*0.85 ? " OK " : " -- ";
            const char *train_ok = train_gb < usableGB*0.85 ? " OK " : " -- ";
            const char *lora_ok = lora_gb < usableGB*0.85 ? " OK " : " -- ";

            printf("  │ %-16s │ %5.1fB  │%4.1fGB%s│%4.1fGB%s│%4.1fGB%s│%5.1fGB%s│%4.1fGB%s│\n",
                   models[i].name,
                   p/1e9,
                   fp16_gb, fp16_ok,
                   int8_gb, int8_ok,
                   int4_gb, int4_ok,
                   train_gb, train_ok,
                   lora_gb, lora_ok);
        }
        printf("  └──────────────────┴─────────┴────────┴────────┴────────┴────────┴────────┘\n");

        // === Real throughput benchmarks ===
        printf("\n═══ Transformer Layer Throughput (CPU/SME2) ═══\n");
        printf("  ┌──────────────────┬─────────┬──────────┬──────────┬──────────────────┐\n");
        printf("  │ Config           │ Seq len │ Time/lyr │ TFLOPS   │ tok/sec (1 lyr)  │\n");
        printf("  ├──────────────────┼─────────┼──────────┼──────────┼──────────────────┤\n");

        struct { const char *name; int dim; int ffn; int heads; } cfgs[] = {
            {"GPT-2 Small",  768, 3072, 12},
            {"1B class",    2048, 8192, 16},
            {"3B class",    3072, 8192, 24},
            {"7B class",    4096,14336, 32},
            {NULL, 0, 0, 0}
        };

        for (int ci=0; cfgs[ci].name; ci++) {
            int batch=1, seq=128;
            double ms = transformer_layer_forward_cpu(batch, seq, cfgs[ci].dim,
                                                      cfgs[ci].heads, cfgs[ci].ffn);
            double flops = transformer_layer_flops(batch, seq, cfgs[ci].dim, cfgs[ci].ffn);
            double tflops = flops / (ms * 1e9);
            double tok_per_sec = (batch * seq) / (ms / 1000.0);

            printf("  │ %-16s │ %5d   │ %6.2f ms│ %6.2f   │ %14.0f   │\n",
                   cfgs[ci].name, seq, ms, tflops, tok_per_sec);
        }
        printf("  └──────────────────┴─────────┴──────────┴──────────┴──────────────────┘\n");

        // === Estimate full model throughput ===
        printf("\n═══ Estimated Full Model Performance ═══\n");
        printf("  (Inference, batch=1, seq=128, CPU/SME2 FP32)\n\n");

        struct { const char *name; int layers; int dim; int ffn; int heads; double params; } full[] = {
            {"GPT-2 Small",   12,  768, 3072, 12,  124e6},
            {"GPT-2 Medium",  24, 1024, 4096, 16,  355e6},
            {"Llama-3.2 1B",  16, 2048, 8192, 16,  1.2e9},
            {"Phi-3 Mini",    32, 3072, 8192, 24,  3.8e9},
            {"Llama-3.1 8B",  32, 4096,14336, 32,  8.0e9},
            {NULL, 0, 0, 0, 0, 0}
        };

        for (int i=0; full[i].name; i++) {
            double layer_ms = transformer_layer_forward_cpu(1, 128, full[i].dim,
                                                            full[i].heads, full[i].ffn);
            double total_ms = layer_ms * full[i].layers;
            double tok_per_sec = 128.0 / (total_ms / 1000.0);

            // Training: ~3x forward (forward + backward + optimizer)
            double train_ms_per_step = total_ms * 3.0;
            double train_tok_per_sec = 128.0 / (train_ms_per_step / 1000.0);

            double mem_inf_fp16 = full[i].params * 2.0 / 1e9;
            double mem_train = full[i].params * 16.0 / 1e9;

            printf("  %s (%.1fB params):\n", full[i].name, full[i].params/1e9);
            printf("    Inference: %.1f ms/step → %.0f tok/s  (%.1f GB FP16)\n",
                   total_ms, tok_per_sec, mem_inf_fp16);
            printf("    Training:  ~%.0f ms/step → ~%.0f tok/s  (%.1f GB needed)\n\n",
                   train_ms_per_step, train_tok_per_sec, mem_train);
        }

        // === Final summary ===
        printf("═══ SUMMARY: Maximum Model Sizes on M4 (%.0f GB) ═══\n\n", totalGB);
        printf("  INFERENCE:\n");
        printf("    FP16:  up to ~%.0fB params (e.g., Llama-3.2 3B easily)\n", avail/2.0/1e9);
        printf("    INT8:  up to ~%.0fB params (e.g., Llama-3.1 8B)\n", avail/1.0/1e9);
        printf("    INT4:  up to ~%.0fB params (e.g., Qwen-2.5 7B, Llama 8B)\n", avail/0.5/1e9);
        printf("\n");
        printf("  FULL TRAINING (FP32, w+grad+Adam):\n");
        printf("    up to ~%.1fB params\n", avail/16.0/1e9);
        printf("\n");
        printf("  LoRA/QLoRA TRAINING:\n");
        printf("    Base INT4 + LoRA adapters: up to ~%.0fB base model\n", avail/0.7/1e9);
        printf("    (trains ~1-5%% of params, rest frozen in INT4)\n");

        printf("\n═══ Done ═══\n");
    }
    return 0;
}
