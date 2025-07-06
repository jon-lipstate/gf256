package tests

import "../"
import "core:fmt"
import "core:mem"
import "core:testing"
import "core:time"

// Performance benchmarks for GF(256) library
// Tests all SIMD implementations and operation types

main :: proc() {
	fmt.println("=== GF(256) Performance Benchmarks ===")
	fmt.println("Comparing scalar vs SIMD performance across all operations\n")

	run_multiply_region_benchmarks()
	run_operation_type_benchmarks()
	run_cache_vs_memory_benchmarks()
}

run_multiply_region_benchmarks :: proc() {
	fmt.println("=== Multiply Region Performance ===")

	// Test parameters
	test_size := 1024 * 1024 // 1MB
	warmup_iterations := 10
	test_iterations := 1000
	coeff := gf256.GF256(123)

	// Allocate buffers
	src := make([]u8, test_size)
	dst := make([]u8, test_size)
	defer delete(src)
	defer delete(dst)

	// Initialize source data
	for i in 0 ..< test_size {
		src[i] = u8(i & 0xFF)
	}

	// Test each lane width
	lane_configs := []struct {
		width: gf256.Lane_Width,
		name:  string,
	} {
		{.None, "Scalar"},
		{.x16, "SSE/SSSE3 (128-bit)"},
		{.x32, "AVX2 (256-bit)"},
		{.x64, "AVX512 (512-bit)"},
	}

	fmt.printf(
		"Operation: multiply_region (%d MB data, coefficient=%d, %d iterations)\n\n",
		test_size / (1024 * 1024),
		coeff,
		test_iterations,
	)

	best_throughput := 0.0
	best_config := ""

	for config in lane_configs {
		// Create context with specific lane width
		ctx := gf256.context_create(gf256.AES_POLYNOMIAL, config.width)
		defer gf256.context_destroy(ctx)

		if ctx == nil {
			fmt.printf("%-20s: Failed to create context\n", config.name)
			continue
		}

		// Warm up
		for _ in 0 ..< warmup_iterations {
			gf256.ctx_multiply_region(ctx, dst, src, coeff)
		}

		// Benchmark
		start := time.now()
		for _ in 0 ..< test_iterations {
			gf256.ctx_multiply_region(ctx, dst, src, coeff)
		}
		duration := time.since(start)

		throughput :=
			f64(test_size * test_iterations) / (1024.0 * 1024.0 * 1024.0) / time.duration_seconds(duration)

		if throughput > best_throughput {
			best_throughput = throughput
			best_config = config.name
		}

		fmt.printf("%-20s: %4.1f GB/s\n", config.name, throughput)
	}

	fmt.printf("\nBest performance: %s at %.1f GB/s\n", best_config, best_throughput)
}

run_operation_type_benchmarks :: proc() {
	fmt.println("\n=== Operation Type Comparison (x16 only) ===")

	// Use optimal x16 for operation comparison
	ctx := gf256.context_create(gf256.AES_POLYNOMIAL, .x16)
	defer gf256.context_destroy(ctx)

	if ctx == nil {
		fmt.println("Failed to create context")
		return
	}

	test_size := 1024 * 1024
	iterations := 1000

	src := make([]u8, test_size)
	dst := make([]u8, test_size)
	defer delete(src)
	defer delete(dst)

	// Initialize data
	for i in 0 ..< test_size {
		src[i] = u8(i & 0xFF)
		dst[i] = u8((i * 2) & 0xFF)
	}

	fmt.printf("Test size: %d MB, Iterations: %d\n\n", test_size / (1024 * 1024), iterations)

	// Test multiply region
	start := time.now()
	for _ in 0 ..< iterations {
		gf256.ctx_multiply_region(ctx, dst, src, 123)
	}
	duration := time.since(start)
	throughput := f64(test_size * iterations) / (1024.0 * 1024.0 * 1024.0) / time.duration_seconds(duration)
	fmt.printf("%-20s: %4.1f GB/s\n", "Multiply Region", throughput)

	// Test multiply-add region
	start = time.now()
	for _ in 0 ..< iterations {
		gf256.ctx_multiply_add_region(ctx, dst, src, 123)
	}
	duration = time.since(start)
	throughput = f64(test_size * iterations) / (1024.0 * 1024.0 * 1024.0) / time.duration_seconds(duration)
	fmt.printf("%-20s: %4.1f GB/s\n", "Multiply-Add Region", throughput)

	// Test XOR/Add region
	start = time.now()
	for _ in 0 ..< iterations {
		gf256.ctx_add_region(ctx, dst, src)
	}
	duration = time.since(start)
	throughput = f64(test_size * iterations) / (1024.0 * 1024.0 * 1024.0) / time.duration_seconds(duration)
	fmt.printf("%-20s: %4.1f GB/s\n", "XOR/Add Region", throughput)

	// Test XOR standalone
	start = time.now()
	for _ in 0 ..< iterations {
		gf256.xor_region(dst, dst, src)
	}
	duration = time.since(start)
	throughput = f64(test_size * iterations) / (1024.0 * 1024.0 * 1024.0) / time.duration_seconds(duration)
	fmt.printf("%-20s: %4.1f GB/s\n", "XOR (standalone)", throughput)
}


// Test procedure for Odin test framework
@(test)
test_performance_regression :: proc(t: ^testing.T) {
	// Basic performance regression test
	// Ensures the library meets minimum performance thresholds

	ctx := gf256.context_create()
	defer gf256.context_destroy(ctx)

	testing.expect(t, ctx != nil, "Context creation should succeed")

	// Test with 1MB data
	test_size := 1024 * 1024
	iterations := 100

	src := make([]u8, test_size)
	dst := make([]u8, test_size)
	defer delete(src)
	defer delete(dst)

	for i in 0 ..< test_size {
		src[i] = u8(i & 0xFF)
	}

	// Benchmark multiply_region
	start := time.now()
	for _ in 0 ..< iterations {
		gf256.ctx_multiply_region(ctx, dst, src, 123)
	}
	duration := time.since(start)

	throughput := f64(test_size * iterations) / (1024.0 * 1024.0 * 1024.0) / time.duration_seconds(duration)

	// Should achieve at least 1 GB/s on any modern system
	testing.expect(
		t,
		throughput >= 1.0,
		fmt.tprintf("Performance regression: only %.1f GB/s (expected >= 1.0 GB/s)", throughput),
	)

	fmt.printf("Performance test: %.1f GB/s\n", throughput)
}

run_cache_vs_memory_benchmarks :: proc() {
	fmt.println("\n=== Cache vs Memory Bandwidth Performance ===")
	fmt.println("Demonstrates performance difference between in-cache and memory-bound operations")

	ctx := gf256.context_create()
	defer gf256.context_destroy(ctx)

	if ctx == nil {
		fmt.println("Failed to create context")
		return
	}

	coeff := gf256.GF256(123)

	Test_Config :: struct {
		size:        int,
		name:        string,
		description: string,
		iterations:  int,
	}

	// Test sizes designed to show cache hierarchy effects
	test_configs := []Test_Config {
		{32 * 1024, "L1 Cache", "32KB (fits in L1 data cache)", 5000},
		{256 * 1024, "L2 Cache", "256KB (fits in L2 cache)", 2000},
		{8 * 1024 * 1024, "L3 Cache", "8MB (fits in L3 cache)", 500},
		{64 * 1024 * 1024, "Main Memory", "64MB (exceeds L3, hits main memory)", 100},
		{512 * 1024 * 1024, "Memory Bandwidth", "512MB (memory bandwidth limited)", 25},
	}

	fmt.println("\nTesting with optimal x16 SIMD:")
	fmt.printf(
		"%-15s %10s %12s %12s %s\n",
		"Size",
		"Throughput",
		"vs L1",
		"vs Memory",
		"Description",
	)
	fmt.println("-------------------------------------------------------------------------------")

	l1_throughput := 0.0
	memory_throughput := 0.0

	for config, i in test_configs {
		src := make([]u8, config.size)
		dst := make([]u8, config.size)
		defer delete(src)
		defer delete(dst)

		// Initialize with random-like data to prevent optimization
		for j in 0 ..< config.size {
			src[j] = u8((j * 17 + 42) & 0xFF)
		}

		// Warmup - prime the caches
		warmup_iterations := min(config.iterations / 10, 50)
		for _ in 0 ..< warmup_iterations {
			gf256.ctx_multiply_region(ctx, dst, src, coeff)
		}

		// Benchmark
		start := time.now()
		for _ in 0 ..< config.iterations {
			gf256.ctx_multiply_region(ctx, dst, src, coeff)
		}
		duration := time.since(start)

		throughput :=
			f64(config.size * config.iterations) /
			(1024.0 * 1024.0 * 1024.0) /
			time.duration_seconds(duration)

		// Store reference throughputs
		if i == 0 {
			l1_throughput = throughput
		}
		if config.name == "Memory Bandwidth" {
			memory_throughput = throughput
		}

		// Calculate relative performance
		vs_l1 := throughput / l1_throughput
		vs_memory := throughput / memory_throughput if memory_throughput > 0 else 1.0

		fmt.printf(
			"%-15s %4.1f GB/s %4.2fx %4.2fx   %s\n",
			config.name,
			throughput,
			vs_l1,
			vs_memory,
			config.description,
		)
	}
}
