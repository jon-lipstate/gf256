package gf256

import "base:intrinsics"
import "core:fmt"
import "core:strings"

// GF256 element type - single byte representing polynomial coefficients
GF256 :: distinct u8

// Zero and one constants
GF256_ZERO :: GF256(0)
GF256_ONE :: GF256(1)

// AES polynomial: x^8 + x^4 + x^3 + x + 1 = 0x11D
// This is the default irreducible polynomial used by AES and most implementations
AES_POLYNOMIAL :: 0x11D

// SIMD lane width configuration
Lane_Width :: enum {
	None = 1, // 8 bit
	x16  = 16, // 128 bit
	x32  = 32, // 256 bit
	x64  = 64, // 512 bit
}

// GF256_Context encapsulates all tables and configuration for a GF(256) instance
// This allows multiple contexts with different polynomials to coexist
// Based on catid's GF256 class design with speed-optimized direct lookup tables
Context :: struct {
	// Configuration
	polynomial:        u16,
	primitive_element: u8,
	simd_width:        Lane_Width,

	// Always present: compact exp/log tables for fallback and table generation
	exp_table:         [512]u8, // Exponential table with wrap-around
	log_table:         [256]u8, // Logarithm table  
	inv_table:         [256]u8, // Multiplicative inverse table

	// Direct lookup tables (catid's approach)
	direct_mul_table:  [256][256]u8, // 64KB: direct multiplication table
	direct_div_table:  [256][256]u8, // 64KB: direct division table

	// SIMD table storage (32KB) - cache-friendly column-major layout
	simd_tables:       struct #raw_union {
		simd_16: struct {
			mul_lo_table: [256][16]u8, // 4KB: Low nibble table (coefficient-major)
			mul_hi_table: [256][16]u8, // 4KB: High nibble table (coefficient-major)
		},
		simd_32: struct {
			mul_lo_scaled_32: [256][32]u8, // 8KB: 32-byte scaled table (coefficient-major)
			mul_hi_scaled_32: [256][32]u8, // 8KB: 32-byte scaled table (coefficient-major)
		},
		simd_64: struct {
			mul_lo_scaled_64: [256][64]u8, // 16KB: 64-byte scaled table (coefficient-major)
			mul_hi_scaled_64: [256][64]u8, // 16KB: 64-byte scaled table (coefficient-major)
		},
	},
}

// Note: Coefficient caching was removed as benchmarks showed no performance benefit
// The cache-friendly table layout provides sufficient optimization

// Reduce polynomial modulo irreducible polynomial
@(private)
reduce_polynomial :: proc(value: u16, poly: u16) -> u8 {
	result := value

	// Find the highest bit position in the polynomial
	poly_degree := 15 - intrinsics.count_leading_zeros(poly)

	// Reduce until result fits in 8 bits
	for bit := 15; bit >= 8; bit -= 1 {
		if (result & (1 << uint(bit))) != 0 {
			// XOR with polynomial shifted to align with current bit
			shift := bit - int(poly_degree)
			result ~= poly << uint(shift)
		}
	}

	return u8(result)
}

// Convert element to polynomial string representation
to_poly_string :: proc(a: GF256, allocator := context.allocator) -> string {
	if a == 0 {
		return "0"
	}

	terms := make([dynamic]string, allocator)
	defer delete(terms)

	val := u8(a)
	for i in 0 ..= 7 {
		if (val & (1 << uint(i))) != 0 {
			switch i {
			case 0:
				append(&terms, "1")
			case 1:
				append(&terms, "x")
			case:
				term := fmt.aprintf("x^%d", i, allocator = allocator)
				append(&terms, term)
			}
		}
	}

	if len(terms) == 0 {
		return "0"
	}

	// Join terms with " + "
	return strings.join(terms[:], " + ", allocator)
}

// Context-based arithmetic operations

// Addition in GF(256) - same for all contexts (XOR)
ctx_add :: proc(ctx: ^Context, a, b: GF256) -> GF256 {
	return GF256(u8(a) ~ u8(b))
}

// Subtraction in GF(256) - same as addition (XOR)
ctx_subtract :: proc(ctx: ^Context, a, b: GF256) -> GF256 {
	return GF256(u8(a) ~ u8(b))
}

// Convenience predicates
is_zero :: proc(a: GF256) -> bool {
	return a == GF256_ZERO
}

is_one :: proc(a: GF256) -> bool {
	return a == GF256_ONE
}

// Context-specific multiplication - O(1) direct table lookup
ctx_multiply :: proc(ctx: ^Context, a, b: GF256) -> GF256 {
	assert(ctx != nil, "Context must not be nil")
	return GF256(ctx.direct_mul_table[a][b])
}

// Context-specific division - O(1) direct table lookup
ctx_divide :: proc(ctx: ^Context, a, b: GF256) -> GF256 {
	assert(ctx != nil, "Context must not be nil")
	assert(b != GF256_ZERO, "Division by zero")
	return GF256(ctx.direct_div_table[a][b])
}

// Context-specific inverse - O(1) table lookup
ctx_inverse :: proc(ctx: ^Context, a: GF256) -> GF256 {
	assert(ctx != nil, "Context must not be nil")
	assert(a != GF256_ZERO, "Cannot invert zero")

	return GF256(ctx.inv_table[a])
}

// Context-specific power operation (logarithmic complexity)
ctx_power :: proc(ctx: ^Context, a: GF256, n: int) -> GF256 {
	assert(ctx != nil, "Context must not be nil")

	if n == 0 { return GF256_ONE }
	if a == 0 { return GF256_ZERO }
	if n == 1 { return a }

	// Use logarithm property: log(a^n) = n * log(a)
	log_result := (int(ctx.log_table[a]) * n) % 255
	return GF256(ctx.exp_table[log_result])
}

// Context-based bulk operations

// Add (XOR) source region to destination: dst[i] ^= src[i]  
// Now uses context-based SIMD approach for optimal performance
ctx_add_region :: proc(ctx: ^Context, dst: []u8, src: []u8) {
    assert(len(dst) == len(src), "Destination and source slices must have equal length")
    
    lane_width := Lane_Width.None
    if ctx != nil {
        lane_width = ctx.simd_width
    }
    _add_region(dst, src, lane_width)
}

// Subtract region - alias for add_region
ctx_subtract_region :: proc(ctx: ^Context, dst: []u8, src: []u8) {
	ctx_add_region(ctx, dst, src)
}

// Multiply entire region
ctx_multiply_region :: proc(ctx: ^Context, dst: []u8, src: []u8, coefficient: GF256) {
	assert(ctx != nil, "Context must not be nil")
	if ctx.simd_width != .None {
		multiply_region(ctx, dst, src, coefficient)
	} else {
		multiply_region_direct_unrolled(ctx, dst, src, coefficient)
	}
}


// Scale region - alias for multiply_region
ctx_scale_region :: proc(ctx: ^Context, dst: []u8, src: []u8, scale: GF256) {
	ctx_multiply_region(ctx, dst, src, scale)
}

// Multiply-add region
ctx_multiply_add_region :: proc(ctx: ^Context, dst: []u8, src: []u8, coefficient: GF256) {
	assert(ctx != nil, "Context must not be nil")
	if ctx.simd_width != .None {
		multiply_add_region(ctx, dst, src, coefficient)
	} else {
		multiply_add_region_direct(ctx, dst, src, coefficient)
	}
}

// Divide region using context
ctx_divide_region :: proc(ctx: ^Context, dst: []u8, src: []u8, coefficient: GF256) {
	assert(ctx != nil, "Context must not be nil")
	assert(coefficient != GF256_ZERO, "Cannot divide by zero")

	if coefficient == GF256_ONE {
		copy(dst, src)
		return
	}

	// Divide by multiplying with inverse
	inverse := GF256(ctx.inv_table[coefficient])
	ctx_multiply_region(ctx, dst, src, inverse)
}

////////
