package gf256

import "core:mem"

// Scalar region operations - fallback when SIMD is not available

// TODO: unused; decide if shoudl keep
@(private)
multiply_region_direct :: proc(ctx: ^Context, dst: []u8, src: []u8, coefficient: GF256) {
	assert(len(dst) == len(src), "Destination and source slices must have equal length")

	// Handle special cases first (catid's pattern)
	if coefficient == GF256_ZERO {
		mem.zero_slice(dst)
		return
	}
	if coefficient == GF256_ONE {
		copy(dst, src)
		return
	}

	// Branchless hot loop using direct table
	for i in 0 ..< len(src) {
		dst[i] = ctx.direct_mul_table[src[i]][coefficient]
	}
}

// Unrolled version for maximum performance
@(private)
multiply_region_direct_unrolled :: proc(ctx: ^Context, dst: []u8, src: []u8, coefficient: GF256) {
	assert(len(dst) == len(src), "Destination and source slices must have equal length")

	// Handle special cases first
	if coefficient == GF256_ZERO {
		mem.zero_slice(dst)
		return
	}
	if coefficient == GF256_ONE {
		copy(dst, src)
		return
	}

	i := 0

	// Process 8 bytes at a time (catid's pattern)
	for i < len(src) - 7 {
		dst[i + 0] = ctx.direct_mul_table[src[i + 0]][coefficient]
		dst[i + 1] = ctx.direct_mul_table[src[i + 1]][coefficient]
		dst[i + 2] = ctx.direct_mul_table[src[i + 2]][coefficient]
		dst[i + 3] = ctx.direct_mul_table[src[i + 3]][coefficient]
		dst[i + 4] = ctx.direct_mul_table[src[i + 4]][coefficient]
		dst[i + 5] = ctx.direct_mul_table[src[i + 5]][coefficient]
		dst[i + 6] = ctx.direct_mul_table[src[i + 6]][coefficient]
		dst[i + 7] = ctx.direct_mul_table[src[i + 7]][coefficient]
		i += 8
	}

	// Handle remaining bytes
	for i < len(src) {
		dst[i] = ctx.direct_mul_table[src[i]][coefficient]
		i += 1
	}
}

// Multiply-add operations with branch elimination
@(private)
multiply_add_region_direct :: proc(ctx: ^Context, dst: []u8, src: []u8, coefficient: GF256) {
	assert(len(dst) == len(src), "Destination and source slices must have equal length")

	// Handle special cases first
	if coefficient == GF256_ZERO {
		return // Adding zero changes nothing
	}
	if coefficient == GF256_ONE {
		// GF256 addition is just XOR
		for i in 0 ..< len(src) {
			dst[i] ~= src[i]
		}
		return
	}

	// Branchless hot loop
	for i in 0 ..< len(src) {
		dst[i] ~= ctx.direct_mul_table[src[i]][coefficient]
	}
}