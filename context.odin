package gf256

import "core:fmt"
import "core:mem"

// Create a new GF(256) context with specified polynomial & SIMD target level
context_create :: proc(
	polynomial: u16 = AES_POLYNOMIAL,
	requested := Lane_Width.x16,  // x16 provides best performance for GF256 table lookups
	allocator := context.allocator,
) -> ^Context {
	ctx := new(Context, allocator)
	ctx.polynomial = polynomial

	if !context_init(ctx, requested) {
		free(ctx, allocator)
		return nil
	}

	return ctx
}

context_destroy :: proc(ctx: ^Context, allocator := context.allocator) {
	if ctx != nil {
		free(ctx, allocator)
	}
}

// Initialize tables for a GF(256) context
@(private)
context_init :: proc(ctx: ^Context, requested: Lane_Width) -> bool {
	if ctx == nil {
		return false
	}

	// Clear existing tables
	mem.zero_slice(ctx.exp_table[:])
	mem.zero_slice(ctx.log_table[:])
	mem.zero_slice(ctx.inv_table[:])

	// Generate exponential and logarithm tables
	if !generate_context_exp_log_tables(ctx) {
		return false
	}

	// Generate inverse table
	generate_context_inverse_table(ctx)

	// Build direct lookup tables and set up SIMD
	if !setup_context_tables(ctx, requested) {
		return false
	}

	return true
}


// Verify context tables are correct
ctx_verify :: proc(ctx: ^Context) -> bool {
	if ctx == nil {
		return false
	}

	// Test exp[log[x]] = x for all non-zero x
	for x in 1 ..= 255 {
		if ctx.exp_table[ctx.log_table[x]] != u8(x) {
			return false
		}
	}

	// Test log[exp[x]] = x for x in 0..254
	for x in 0 ..= 254 {
		exp_val := ctx.exp_table[x]
		if exp_val != 0 && ctx.log_table[exp_val] != u8(x) {
			return false
		}
	}

	// Test multiplicative inverses
	for x in 1 ..= 255 {
		inv := ctx.inv_table[x]
		product := ctx_multiply(ctx, GF256(x), GF256(inv))
		if product != GF256_ONE {
			return false
		}
	}

	return true
}

// Print context information
ctx_print_info :: proc(ctx: ^Context) {
	if ctx == nil {
		fmt.println("Context is nil")
		return
	}

	// Derive SIMD name from lane width
	simd_name := ""
	switch ctx.simd_width {
	case .None:
		simd_name = "scalar"
	case .x16:
		when ODIN_ARCH == .amd64 || ODIN_ARCH == .i386 {
			simd_name = "SSE/SSSE3"
		} else {
			simd_name = "NEON"
		}
	case .x32:
		simd_name = "AVX2"
	case .x64:
		simd_name = "AVX512"
	}

	fmt.printf("GF(256) Context Information:\n")
	fmt.printf("  Polynomial: 0x%X\n", ctx.polynomial)
	fmt.printf("  Primitive element: %d\n", ctx.primitive_element)
	fmt.printf("  SIMD enabled: %t\n", ctx.simd_width != .None)
	fmt.printf("  SIMD width: %d bytes (%s)\n", int(ctx.simd_width), simd_name)
}