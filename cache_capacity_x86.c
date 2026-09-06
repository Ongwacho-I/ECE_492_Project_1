/* Cache Capacity / Memory Latency Microbenchmark
* ECE 592 Project 1
*
* Purpose:
* This program measures dependent memory-access latency across different
* working-set sizes using pointer chasing. The goal is to observe changes
* in access latency as the working set moves through different levels of
* the memory hierarchy.
*
* The benchmark supports two access patterns:
*
*   random      - nodes are connected in a randomized pointer-chasing
*                 cycle to reduce the effectiveness of hardware prefetching.
*
*   sequential  - nodes are connected sequentially to provide a
*                 prefetcher-friendly comparison.
*
* Timing:
* Memory accesses are timed using the x86 Time Stamp Counter (TSC) with
* RDTSC/RDTSCP and LFENCE for serialization. Multiple dependent pointer
* accesses are performed per timed sample, and latency is reported as
* TSC ticks per access.
*
* Measurements:
* The program can test a single working-set size or perform a
* power-of-two sweep over multiple working-set sizes. For each point,
* it records the raw timing samples and calculates statistics including
* mean, median, standard deviation, quartiles, percentiles, minimum,
* maximum, and outlier count.
*
* Output:
*   <prefix>_summary.csv  - summary statistics for each working-set size
*   <prefix>_metadata.txt - experiment configuration and system information
*   <prefix>_*_raw.csv    - individual timing samples
*
* The program is intended for x86-64 systems. On Linux, CPU affinity
* information is recorded so experiments can be run on a single logical
* CPU using taskset.
*/

#define _GNU_SOURCE

#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <x86intrin.h>

#ifdef __linux__
#include <sched.h>
#include <unistd.h>
#endif

#ifdef _WIN32
#include <malloc.h>
#endif

#if !defined(__x86_64__) && !defined(_M_X64)
#error "This source file is for x86-64 only. Use a separate AArch64 implementation on Arm."
#endif

#define ALIGNMENT_BYTES 4096u
#define MIN_ASSIGNMENT_SAMPLES 1000000ULL
#define TIMER_OVERHEAD_SAMPLES 10000ULL

/* Pointer-chase structure. */
struct node {
    struct node *next;
};

/* Prevents the final pointer-chase result from becoming unused. */
static volatile struct node *sink_node;

typedef struct {
    double mean, median, stddev;
    double q1, q3, p5, p95;
    double min, max;
    uint64_t outliers;
    uint64_t n;
} stats_t;

typedef struct {
    uint64_t min_ticks;
    double mean_ticks;
    double median_ticks;
} overhead_stats_t;

/* Compiler barrier only; this is not a CPU fence. */
static inline void compiler_barrier(void) {
#if defined(__GNUC__) || defined(__clang__)
    __asm__ __volatile__("" ::: "memory");
#endif
}

/* Serialized x86 TSC timing primitives. */
static inline uint64_t x86_tsc_start(void) {
    _mm_lfence();
    uint64_t t = __rdtsc();
    _mm_lfence();
    return t;
}

static inline uint64_t x86_tsc_stop(void) {
    unsigned aux;
    uint64_t t = __rdtscp(&aux);
    _mm_lfence();
    return t;
}

/* Deterministic PRNG used for reproducible shuffling. */
static uint32_t xorshift32(uint32_t *state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

/* Randomize sweep order to reduce systematic drift. */
static void shuffle_size_t(size_t *values, size_t n, uint32_t seed) {
    if (n < 2) return;

    uint32_t state = seed ? seed : 1u;
    for (size_t i = n - 1; i > 0; i--) {
        size_t j = (size_t)(xorshift32(&state) % (uint32_t)(i + 1));
        size_t temp = values[i];
        values[i] = values[j];
        values[j] = temp;
    }
}

/* Create one randomized closed cycle containing every node exactly once. */
static void make_random_cycle(struct node *nodes, size_t n, uint32_t seed) {
    if (n < 2) {
        fprintf(stderr, "Need at least two nodes.\n");
        exit(EXIT_FAILURE);
    }

    size_t *order = malloc(n * sizeof(*order));
    if (!order) {
        perror("malloc order");
        exit(EXIT_FAILURE);
    }

    for (size_t i = 0; i < n; i++) order[i] = i;

    uint32_t state = seed ? seed : 1u;
    for (size_t i = n - 1; i > 0; i--) {
        size_t j = (size_t)(xorshift32(&state) % (uint32_t)(i + 1));
        size_t temp = order[i];
        order[i] = order[j];
        order[j] = temp;
    }

    for (size_t i = 0; i < n; i++) {
        size_t current = order[i];
        size_t next = order[(i + 1) % n];
        nodes[current].next = &nodes[next];
    }

    free(order);
}

/* Sequential, prefetcher-friendly comparison cycle. */
static void make_sequential_cycle(struct node *nodes, size_t n) {
    if (n < 2) {
        fprintf(stderr, "Need at least two nodes.\n");
        exit(EXIT_FAILURE);
    }

    for (size_t i = 0; i < n - 1; i++) nodes[i].next = &nodes[i + 1];
    nodes[n - 1].next = &nodes[0];
}

/* Untimed warm-up. The cursor is preserved for the next traversal. */
static void warm_chain(struct node **cursor, uint64_t steps) {
    struct node *p = *cursor;
    for (uint64_t i = 0; i < steps; i++) p = p->next;
    sink_node = p;
    *cursor = p;
}

/*
 * One call = one timed sample.
 * A sample times a batch of dependent loads and returns TSC ticks/access.
 */
static double chase_batch(struct node **cursor, uint64_t steps) {
    struct node *p = *cursor;

    compiler_barrier();
    uint64_t t0 = x86_tsc_start();

    for (uint64_t i = 0; i < steps; i++) {
        p = p->next;
    }

    uint64_t t1 = x86_tsc_stop();
    compiler_barrier();

    sink_node = p;
    *cursor = p;
    return (double)(t1 - t0) / (double)steps;
}

/* Empty timed region for timer/barrier-overhead characterization. */
static uint64_t timer_once(void) {
    compiler_barrier();
    uint64_t t0 = x86_tsc_start();
    uint64_t t1 = x86_tsc_stop();
    compiler_barrier();
    return t1 - t0;
}

static int cmp_double(const void *a, const void *b) {
    const double da = *(const double *)a;
    const double db = *(const double *)b;
    return (da > db) - (da < db);
}

/* Linear-interpolated quantile of an already sorted array. */
static double quantile_linear(const double *sorted, size_t n, double q) {
    if (n == 0) return NAN;
    if (n == 1) return sorted[0];

    double position = q * (double)(n - 1);
    size_t lower = (size_t)floor(position);
    size_t upper = (size_t)ceil(position);
    double fraction = position - (double)lower;

    if (lower == upper) return sorted[lower];
    return sorted[lower] * (1.0 - fraction) + sorted[upper] * fraction;
}

/* Statistics over the complete timed-sample distribution. */
static stats_t compute_stats(const double *samples, size_t n) {
    stats_t s;
    memset(&s, 0, sizeof(s));
    s.n = (uint64_t)n;
    if (n == 0) return s;

    /* Welford mean/stddev plus extrema. */
    double mean = 0.0, m2 = 0.0;
    double min_value = samples[0], max_value = samples[0];

    for (size_t i = 0; i < n; i++) {
        double x = samples[i];
        if (x < min_value) min_value = x;
        if (x > max_value) max_value = x;

        double delta = x - mean;
        mean += delta / (double)(i + 1);
        double delta2 = x - mean;
        m2 += delta * delta2;
    }

    /* Sort a copy so raw measurement order stays intact. */
    double *sorted = malloc(n * sizeof(*sorted));
    if (!sorted) {
        perror("malloc sorted samples");
        exit(EXIT_FAILURE);
    }

    memcpy(sorted, samples, n * sizeof(*sorted));
    qsort(sorted, n, sizeof(*sorted), cmp_double);

    s.mean = mean;
    s.stddev = (n > 1) ? sqrt(m2 / (double)(n - 1)) : 0.0;
    s.median = quantile_linear(sorted, n, 0.50);
    s.q1 = quantile_linear(sorted, n, 0.25);
    s.q3 = quantile_linear(sorted, n, 0.75);
    s.p5 = quantile_linear(sorted, n, 0.05);
    s.p95 = quantile_linear(sorted, n, 0.95);
    s.min = min_value;
    s.max = max_value;

    /* Tukey 1.5*IQR outlier count. */
    double iqr = s.q3 - s.q1;
    double lower_limit = s.q1 - 1.5 * iqr;
    double upper_limit = s.q3 + 1.5 * iqr;

    for (size_t i = 0; i < n; i++) {
        if (samples[i] < lower_limit || samples[i] > upper_limit) s.outliers++;
    }

    free(sorted);
    return s;
}

static overhead_stats_t measure_timer_overhead(void) {
    double *values = malloc((size_t)TIMER_OVERHEAD_SAMPLES * sizeof(*values));
    if (!values) {
        perror("malloc timer overhead");
        exit(EXIT_FAILURE);
    }

    uint64_t minimum = UINT64_MAX;
    long double sum = 0.0L;

    for (uint64_t i = 0; i < TIMER_OVERHEAD_SAMPLES; i++) {
        uint64_t value = timer_once();
        values[i] = (double)value;
        if (value < minimum) minimum = value;
        sum += (long double)value;
    }

    qsort(values, (size_t)TIMER_OVERHEAD_SAMPLES, sizeof(*values), cmp_double);

    overhead_stats_t result;
    result.min_ticks = minimum;
    result.mean_ticks = (double)(sum / (long double)TIMER_OVERHEAD_SAMPLES);
    result.median_ticks = quantile_linear(values, (size_t)TIMER_OVERHEAD_SAMPLES, 0.50);

    free(values);
    return result;
}

static uint64_t parse_u64(const char *text, const char *name) {
    char *end = NULL;
    errno = 0;
    unsigned long long value = strtoull(text, &end, 10);

    if (errno != 0 || end == text || *end != '\0') {
        fprintf(stderr, "Invalid %s: %s\n", name, text);
        exit(EXIT_FAILURE);
    }

    return (uint64_t)value;
}

/* Same aligned allocation behavior as the long version. */
static struct node *allocate_nodes(size_t bytes) {
    struct node *nodes = NULL;

#ifdef _WIN32
    nodes = (struct node *)_aligned_malloc(bytes, ALIGNMENT_BYTES);
    if (!nodes) {
        fprintf(stderr, "Aligned allocation failed for %zu bytes.\n", bytes);
        exit(EXIT_FAILURE);
    }
#else
    int result = posix_memalign((void **)&nodes, ALIGNMENT_BYTES, bytes);
    if (result != 0 || !nodes) {
        fprintf(stderr, "posix_memalign failed for %zu bytes (error=%d).\n", bytes, result);
        exit(EXIT_FAILURE);
    }
#endif

    return nodes;
}

static void free_nodes(struct node *nodes) {
#ifdef _WIN32
    _aligned_free(nodes);
#else
    free(nodes);
#endif
}

/* Verify that a Linux run is pinned to exactly one logical CPU. */
static void print_affinity_information(FILE *metadata) {
#ifdef __linux__
    cpu_set_t set;
    CPU_ZERO(&set);

    if (sched_getaffinity(0, sizeof(set), &set) != 0) {
        fprintf(metadata, "affinity_check=failed\n");
        fprintf(stderr, "WARNING: sched_getaffinity failed. Verify taskset manually.\n");
        return;
    }

    int count = CPU_COUNT(&set);
    fprintf(metadata, "allowed_logical_cpu_count=%d\n", count);
    fprintf(metadata, "allowed_logical_cpus=");

    int first = 1;
    for (int cpu = 0; cpu < CPU_SETSIZE; cpu++) {
        if (CPU_ISSET(cpu, &set)) {
            fprintf(metadata, "%s%d", first ? "" : ",", cpu);
            first = 0;
        }
    }
    fprintf(metadata, "\n");

    if (count != 1) {
        fprintf(stderr,
                "\nWARNING:\n"
                "This process is currently allowed to run on %d logical CPUs.\n"
                "For final assignment measurements, launch with:\n\n"
                "    taskset -c <chosen_cpu> ./cache_capacity_x86 ...\n\n",
                count);
    }
#else
    fprintf(metadata, "affinity_check=not_available_on_this_platform\n");
#endif
}

/* Record run-level metadata used for reproducibility. */
static void write_global_metadata(FILE *metadata, int argc, char **argv,
                                  uint64_t sample_count, uint64_t steps,
                                  uint32_t seed, const overhead_stats_t *overhead) {
    time_t now = time(NULL);
    struct tm *utc = gmtime(&now);
    char timestamp[64] = "unknown";

    if (utc) {
        strftime(timestamp, sizeof(timestamp), "%Y-%m-%dT%H:%M:%SZ", utc);
    }

    fprintf(metadata, "timestamp_utc=%s\n", timestamp);
    fprintf(metadata, "architecture=x86-64\n");
    fprintf(metadata, "timer=serialized_rdtsc_rdtscp\n");
    fprintf(metadata, "latency_unit=tsc_ticks_per_access\n");
    fprintf(metadata, "alignment_bytes=%u\n", ALIGNMENT_BYTES);
    fprintf(metadata, "timed_samples_per_point=%" PRIu64 "\n", sample_count);
    fprintf(metadata, "dependent_accesses_per_timed_sample=%" PRIu64 "\n", steps);
    fprintf(metadata, "seed=%" PRIu32 "\n", seed);
    fprintf(metadata, "timer_overhead_samples=%llu\n",
            (unsigned long long)TIMER_OVERHEAD_SAMPLES);
    fprintf(metadata, "timer_overhead_min_tsc_ticks=%" PRIu64 "\n", overhead->min_ticks);
    fprintf(metadata, "timer_overhead_mean_tsc_ticks=%.6f\n", overhead->mean_ticks);
    fprintf(metadata, "timer_overhead_median_tsc_ticks=%.6f\n", overhead->median_ticks);

#ifdef __linux__
    char hostname[256];
    if (gethostname(hostname, sizeof(hostname)) == 0) {
        hostname[sizeof(hostname) - 1] = '\0';
        fprintf(metadata, "hostname=%s\n", hostname);
    }

    long page_size = sysconf(_SC_PAGESIZE);
    if (page_size > 0) fprintf(metadata, "page_size_bytes=%ld\n", page_size);
#endif

    fprintf(metadata, "command_line=");
    for (int i = 0; i < argc; i++) {
        fprintf(metadata, "%s%s", i == 0 ? "" : " ", argv[i]);
    }
    fprintf(metadata, "\n");

    print_affinity_information(metadata);

    fprintf(metadata, "assignment_sample_count_status=%s\n",
            sample_count < MIN_ASSIGNMENT_SAMPLES
                ? "TEST_ONLY_BELOW_1000000"
                : "MEETS_1000000_MINIMUM");
}

/* Save every timed sample so the full distribution can be reconstructed. */
static void write_raw_csv(const char *path, const double *samples, uint64_t sample_count) {
    FILE *file = fopen(path, "w");
    if (!file) {
        perror(path);
        exit(EXIT_FAILURE);
    }

    setvbuf(file, NULL, _IOFBF, 1024 * 1024);
    fprintf(file, "sample,latency_tsc_ticks_per_access\n");

    for (uint64_t i = 0; i < sample_count; i++) {
        fprintf(file, "%" PRIu64 ",%.9f\n", i, samples[i]);
    }

    fclose(file);
}

/* Run one traversal type at one working-set size. */
static stats_t run_one_mode(struct node *nodes, size_t node_count,
                            size_t requested_bytes, size_t actual_bytes,
                            uint64_t sample_count, uint64_t steps,
                            const char *mode, uint32_t seed,
                            const char *prefix, FILE *summary) {
    if (strcmp(mode, "random") == 0) {
        make_random_cycle(nodes, node_count, seed);
    } else if (strcmp(mode, "sequential") == 0) {
        make_sequential_cycle(nodes, node_count);
    } else {
        fprintf(stderr, "Internal error: invalid mode %s\n", mode);
        exit(EXIT_FAILURE);
    }

    struct node *cursor = &nodes[0];

    /* Four complete untimed warm-up passes. */
    uint64_t warm_steps = node_count > UINT64_MAX / 4ULL
                              ? UINT64_MAX
                              : (uint64_t)node_count * 4ULL;
    warm_chain(&cursor, warm_steps);

    if (sample_count > SIZE_MAX / sizeof(double)) {
        fprintf(stderr, "Sample buffer is too large.\n");
        exit(EXIT_FAILURE);
    }

    double *samples = malloc((size_t)sample_count * sizeof(*samples));
    if (!samples) {
        fprintf(stderr, "Could not allocate sample buffer for %" PRIu64 " samples.\n",
                sample_count);
        exit(EXIT_FAILURE);
    }

    /* Critical timing loop: intentionally no file I/O here. */
    for (uint64_t sample = 0; sample < sample_count; sample++) {
        samples[sample] = chase_batch(&cursor, steps);
    }

    stats_t statistics = compute_stats(samples, (size_t)sample_count);

    char raw_path[1024];
    int written = snprintf(raw_path, sizeof(raw_path),
                           "%s_%zu_%s_raw.csv", prefix, actual_bytes, mode);

    if (written < 0 || (size_t)written >= sizeof(raw_path)) {
        fprintf(stderr, "Output path is too long.\n");
        free(samples);
        exit(EXIT_FAILURE);
    }

    write_raw_csv(raw_path, samples, sample_count);

    fprintf(summary,
            "%s,%zu,%zu,%zu,"
            "%" PRIu64 ",%" PRIu64 ",%" PRIu64 ","
            "%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,"
            "%" PRIu64 ",%" PRIu32 ",%s\n",
            mode, requested_bytes, actual_bytes, node_count,
            sample_count, steps, sample_count * steps,
            statistics.mean, statistics.median, statistics.stddev,
            statistics.q1, statistics.q3, statistics.p5, statistics.p95,
            statistics.min, statistics.max, statistics.outliers,
            seed, raw_path);

    fflush(summary);

    fprintf(stderr,
            "bytes=%zu mode=%s mean=%.4f median=%.4f "
            "p5=%.4f p95=%.4f raw=%s\n",
            actual_bytes, mode, statistics.mean, statistics.median,
            statistics.p5, statistics.p95, raw_path);

    free(samples);
    return statistics;
}

/* Run one working-set point in random, sequential, or both modes. */
static void run_one_point(size_t requested_bytes, uint64_t sample_count,
                          uint64_t steps, const char *mode, uint32_t seed,
                          const char *prefix, FILE *summary) {
    size_t node_count = requested_bytes / sizeof(struct node);

    if (node_count < 2) {
        fprintf(stderr, "Working set %zu bytes is too small. Need at least %zu bytes.\n",
                requested_bytes, 2 * sizeof(struct node));
        exit(EXIT_FAILURE);
    }

    if (node_count > SIZE_MAX / sizeof(struct node)) {
        fprintf(stderr, "Working set is too large.\n");
        exit(EXIT_FAILURE);
    }

    size_t actual_bytes = node_count * sizeof(struct node);
    struct node *nodes = allocate_nodes(actual_bytes);

    /* First-touch allocation after the process has been externally pinned. */
    memset(nodes, 0, actual_bytes);

    if (strcmp(mode, "random") == 0) {
        run_one_mode(nodes, node_count, requested_bytes, actual_bytes,
                     sample_count, steps, "random", seed, prefix, summary);
    } else if (strcmp(mode, "sequential") == 0) {
        run_one_mode(nodes, node_count, requested_bytes, actual_bytes,
                     sample_count, steps, "sequential", seed, prefix, summary);
    } else if (strcmp(mode, "both") == 0) {
        run_one_mode(nodes, node_count, requested_bytes, actual_bytes,
                     sample_count, steps, "random", seed, prefix, summary);

        memset(nodes, 0, actual_bytes);

        run_one_mode(nodes, node_count, requested_bytes, actual_bytes,
                     sample_count, steps, "sequential", seed, prefix, summary);
    } else {
        fprintf(stderr, "MODE must be random, sequential, or both.\n");
        free_nodes(nodes);
        exit(EXIT_FAILURE);
    }

    free_nodes(nodes);
}

static void usage(const char *program) {
    fprintf(stderr,
            "\nUSAGE:\n\n"
            "Single working-set size:\n"
            "  %s single BYTES SAMPLES STEPS MODE SEED PREFIX\n\n"
            "Coarse power-of-two sweep:\n"
            "  %s sweep MIN_BYTES MAX_BYTES SAMPLES STEPS MODE SEED PREFIX\n\n"
            "MODE: random | sequential | both\n\n"
            "Test:\n"
            "  %s single 32768 1000 64 random 12345 test/capacity\n\n"
            "Assignment-scale point:\n"
            "  %s single 32768 1000000 128 random 12345 data/capacity\n\n"
            "Coarse sweep:\n"
            "  %s sweep 4096 67108864 1000000 128 random 12345 data/capacity\n\n",
            program, program, program, program, program);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        usage(argv[0]);
        return EXIT_FAILURE;
    }

    int single_mode = strcmp(argv[1], "single") == 0;
    int sweep_mode = strcmp(argv[1], "sweep") == 0;

    if ((!single_mode && !sweep_mode) ||
        (single_mode && argc != 8) ||
        (sweep_mode && argc != 9)) {
        usage(argv[0]);
        return EXIT_FAILURE;
    }

    uint64_t sample_count, steps;
    uint32_t seed;
    const char *mode, *prefix;
    size_t single_bytes = 0, min_bytes = 0, max_bytes = 0;

    if (single_mode) {
        single_bytes = (size_t)parse_u64(argv[2], "BYTES");
        sample_count = parse_u64(argv[3], "SAMPLES");
        steps = parse_u64(argv[4], "STEPS");
        mode = argv[5];
        seed = (uint32_t)parse_u64(argv[6], "SEED");
        prefix = argv[7];
    } else {
        min_bytes = (size_t)parse_u64(argv[2], "MIN_BYTES");
        max_bytes = (size_t)parse_u64(argv[3], "MAX_BYTES");
        sample_count = parse_u64(argv[4], "SAMPLES");
        steps = parse_u64(argv[5], "STEPS");
        mode = argv[6];
        seed = (uint32_t)parse_u64(argv[7], "SEED");
        prefix = argv[8];
    }

    if (sample_count == 0 || steps == 0) {
        fprintf(stderr, "SAMPLES and STEPS must both be greater than zero.\n");
        return EXIT_FAILURE;
    }

    if (strcmp(mode, "random") != 0 &&
        strcmp(mode, "sequential") != 0 &&
        strcmp(mode, "both") != 0) {
        fprintf(stderr, "MODE must be random, sequential, or both.\n");
        return EXIT_FAILURE;
    }

    if (sample_count < MIN_ASSIGNMENT_SAMPLES) {
        fprintf(stderr,
                "\nWARNING:\n"
                "SAMPLES=%" PRIu64 " is below the assignment minimum of %llu.\n"
                "Use this only as a functionality/debugging run, not as final project data.\n\n",
                sample_count, (unsigned long long)MIN_ASSIGNMENT_SAMPLES);
    }

    overhead_stats_t overhead = measure_timer_overhead();

    char metadata_path[1024], summary_path[1024];
    int meta_len = snprintf(metadata_path, sizeof(metadata_path), "%s_metadata.txt", prefix);
    int sum_len = snprintf(summary_path, sizeof(summary_path), "%s_summary.csv", prefix);

    if (meta_len < 0 || sum_len < 0 ||
        (size_t)meta_len >= sizeof(metadata_path) ||
        (size_t)sum_len >= sizeof(summary_path)) {
        fprintf(stderr, "Output prefix is too long.\n");
        return EXIT_FAILURE;
    }

    FILE *metadata = fopen(metadata_path, "w");
    if (!metadata) {
        perror(metadata_path);
        return EXIT_FAILURE;
    }

    write_global_metadata(metadata, argc, argv, sample_count, steps, seed, &overhead);
    fclose(metadata);

    FILE *summary = fopen(summary_path, "w");
    if (!summary) {
        perror(summary_path);
        return EXIT_FAILURE;
    }

    fprintf(summary,
            "mode,requested_bytes,actual_bytes,nodes,timed_samples,"
            "dependent_accesses_per_sample,total_dependent_accesses,"
            "mean_tsc_ticks_per_access,median_tsc_ticks_per_access,stddev,"
            "q1,q3,p5,p95,min,max,outliers,seed,raw_file\n");

    if (single_mode) {
        run_one_point(single_bytes, sample_count, steps, mode, seed, prefix, summary);
    } else {
        if (min_bytes == 0 || max_bytes == 0 || min_bytes > max_bytes) {
            fprintf(stderr, "Require 0 < MIN_BYTES <= MAX_BYTES.\n");
            fclose(summary);
            return EXIT_FAILURE;
        }

        size_t capacity = 64;
        size_t *sizes = malloc(capacity * sizeof(*sizes));
        if (!sizes) {
            perror("malloc sweep sizes");
            fclose(summary);
            return EXIT_FAILURE;
        }

        size_t count = 0;
        size_t current = min_bytes;

        while (current <= max_bytes) {
            if (count == capacity) {
                capacity *= 2;
                size_t *temporary = realloc(sizes, capacity * sizeof(*sizes));
                if (!temporary) {
                    perror("realloc sweep sizes");
                    free(sizes);
                    fclose(summary);
                    return EXIT_FAILURE;
                }
                sizes = temporary;
            }

            sizes[count++] = current;

            if (current > max_bytes / 2 || current > SIZE_MAX / 2) break;
            current *= 2;
        }

        shuffle_size_t(sizes, count, seed ^ 0x9E3779B9u);

        for (size_t i = 0; i < count; i++) {
            fprintf(stderr, "[%zu/%zu] working_set_bytes=%zu\n",
                    i + 1, count, sizes[i]);

            run_one_point(sizes[i], sample_count, steps,
                          mode, seed, prefix, summary);
        }

        free(sizes);
    }

    fclose(summary);

    fprintf(stderr,
            "\nExperiment finished.\n"
            "Summary file:  %s\n"
            "Metadata file: %s\n",
            summary_path, metadata_path);

    return EXIT_SUCCESS;
}
