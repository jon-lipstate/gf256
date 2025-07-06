package gf256

import "core:mem"
import "core:sys/info"

// SIMD capability detection
@(private)
detect_simd_capabilities :: proc(requested: Lane_Width = .x64) -> Lane_Width {
	features, ok := info.cpu.features.?
	if !ok {return .None}

	max_available := Lane_Width.None

	when ODIN_ARCH == .amd64 || ODIN_ARCH == .i386 {
		if .avx512f in features {
			max_available = .x64
		} else if .avx2 in features {
			max_available = .x32
		} else if .sse3 in features {
			max_available = .x16
		}
	} else when ODIN_ARCH == .arm64 || ODIN_ARCH == .arm32 {
        if .asimd in features{
            max_available = .x16
        }
	}

    return min(max_available, requested)
}

// Setup direct lookup tables and SIMD optimization
@(private)
setup_context_tables :: proc(ctx: ^Context, requested: Lane_Width) -> bool {
	// Detect SIMD capabilities first (but don't build split tables yet)
	detect_and_configure_simd(ctx, requested)

	// Build direct lookup tables for maximum speed (catid's approach)
	if !build_direct_tables(ctx) {
		return false
	}

	// Note: Function pointer setup eliminated - dispatch happens at call site

	// Build SIMD split tables now that multiplication tables are ready
	if ctx.simd_width != .None && !build_simd_split_tables(ctx) {
		// If split table generation fails, disable SIMD
		ctx.simd_width = .None
	}

	return true
}

// Detect and configure SIMD capabilities (without building split tables yet)
@(private)
detect_and_configure_simd :: proc(ctx: ^Context, requested: Lane_Width) {
	// Detect SIMD capabilities for this context
	detected_width := detect_simd_capabilities(requested)
	
	// Update context with detected SIMD capabilities
	ctx.simd_width = detected_width

	// Note: Split tables will be built after multiplication tables are ready
}

// Generate exponential and logarithm tables for context
@(private)
generate_context_exp_log_tables :: proc(ctx: ^Context) -> bool {
	// Find primitive element for this polynomial
	primitive := find_primitive_element_for_poly(ctx.polynomial)
	if primitive == 0 {
		return false
	}

	ctx.primitive_element = primitive

	// Generate tables: exp[i] = primitive^i, log[primitive^i] = i
	current := u8(1)

	for i in 0 ..= 254 {
		ctx.exp_table[i] = current
		if current != 0 {
			ctx.log_table[current] = u8(i)
		}

		// Multiply by primitive element
		current = multiply_polynomial_mod(current, primitive, ctx.polynomial)
	}

	// Fill wrap-around portion (for efficiency in multiplication)
	for i in 255 ..= 511 {
		ctx.exp_table[i] = ctx.exp_table[i - 255]
	}

	// Verify that we got back to 1
	return current == 1
}

// Generate multiplicative inverse table for context
@(private)
generate_context_inverse_table :: proc(ctx: ^Context) {
	ctx.inv_table[0] = 0 // 0 has no inverse
	ctx.inv_table[1] = 1 // 1 is its own inverse

	for i in 2 ..= 255 {
		// Find inverse using: a * a^(-1) = 1
		// So a^(-1) = a^(254) since a^255 = 1 in GF(256)
		inverse_exp := (255 - int(ctx.log_table[i])) % 255
		ctx.inv_table[i] = ctx.exp_table[inverse_exp]
	}
}

// Build direct multiplication and division tables for maximum speed
@(private)
build_direct_tables :: proc(ctx: ^Context) -> bool {
	// Build multiplication table first using exp/log tables
	for a in 0 ..= 255 {
		for b in 0 ..= 255 {
			if a == 0 || b == 0 {
				ctx.direct_mul_table[a][b] = 0
			} else {
				// Multiplication: use exp/log properties
				log_sum := int(ctx.log_table[a]) + int(ctx.log_table[b])
				if log_sum >= 255 {
					log_sum -= 255
				}
				ctx.direct_mul_table[a][b] = ctx.exp_table[log_sum]
			}
		}
	}

	// Build division table second (after multiplication table is complete)
	for a in 0 ..= 255 {
		for b in 0 ..= 255 {
			if b == 0 {
				ctx.direct_div_table[a][b] = 0 // a/0 = 0 by convention
			} else if a == 0 {
				ctx.direct_div_table[a][b] = 0 // 0/b = 0
			} else {
				// Division: a/b = a * b^(-1)
				ctx.direct_div_table[a][b] = ctx.direct_mul_table[a][ctx.inv_table[b]]
			}
		}
	}

	return true
}

// Build split tables for SIMD optimization (catid's 4-bit split technique)
@(private)
build_simd_split_tables :: proc(ctx: ^Context) -> bool {
	if ctx == nil {
		return false
	}

	// Build split tables for all possible coefficients using direct multiplication table
	for coeff in 0 ..= 255 {
		coeff_table := &ctx.direct_mul_table[coeff]

		// Build low nibble table (0-15)
		for i in 0 ..< 16 {
			ctx.simd_tables.simd_16.mul_lo_table[i][coeff] = coeff_table[i]
		}

		// Build high nibble table (0-15 shifted left by 4)
		for i in 0 ..< 16 {
			ctx.simd_tables.simd_16.mul_hi_table[i][coeff] = coeff_table[i << 4]
		}
	}

	// Pre-compute scaled tables for wider SIMD operations
	// This avoids rebuilding the replicated tables on every operation
	if int(ctx.simd_width) >= 32 {
		// Build 32-byte scaled tables (AVX2)
		for coeff in 0 ..= 255 {
			for i in 0 ..< 16 {
				// Replicate the 16-byte pattern twice for 32-byte vectors
				ctx.simd_tables.simd_32.mul_lo_scaled_32[i][coeff] = ctx.simd_tables.simd_16.mul_lo_table[i][coeff]
				ctx.simd_tables.simd_32.mul_lo_scaled_32[i+16][coeff] = ctx.simd_tables.simd_16.mul_lo_table[i][coeff]
				ctx.simd_tables.simd_32.mul_hi_scaled_32[i][coeff] = ctx.simd_tables.simd_16.mul_hi_table[i][coeff]
				ctx.simd_tables.simd_32.mul_hi_scaled_32[i+16][coeff] = ctx.simd_tables.simd_16.mul_hi_table[i][coeff]
			}
		}
	}

	if int(ctx.simd_width) >= 64 {
		// Build 64-byte scaled tables (AVX512)
		for coeff in 0 ..= 255 {
			for i in 0 ..< 16 {
				// Replicate the 16-byte pattern four times for 64-byte vectors
				for j in 0 ..< 4 {
					ctx.simd_tables.simd_64.mul_lo_scaled_64[i + j*16][coeff] = ctx.simd_tables.simd_16.mul_lo_table[i][coeff]
					ctx.simd_tables.simd_64.mul_hi_scaled_64[i + j*16][coeff] = ctx.simd_tables.simd_16.mul_hi_table[i][coeff]
				}
			}
		}
	}

	return true
}

// Helper procedures for polynomial operations

@(private)
find_primitive_element_for_poly :: proc(polynomial: u16) -> u8 {
	for candidate in 2 ..= 255 {
		if is_primitive_element_for_poly(u8(candidate), polynomial) {
			return u8(candidate)
		}
	}
	return 0
}

@(private)
is_primitive_element_for_poly :: proc(element: u8, polynomial: u16) -> bool {
	if element == 0 || element == 1 {
		return false
	}

	// Check if element^255 = 1 and element^k != 1 for proper divisors k of 255
	divisors := []int{3, 5, 17, 15, 51, 85}

	current := element
	for power in 1 ..= 254 {
		if current == 1 {
			// Check if this power is a proper divisor of 255
			for divisor in divisors {
				if power == divisor {
					return false
				}
			}
		}
		current = multiply_polynomial_mod(current, element, polynomial)
	}

	return current == 1
}

@(private)
multiply_polynomial_mod :: proc(a, b: u8, polynomial: u16) -> u8 {
	if a == 0 || b == 0 {
		return 0
	}

	result := u16(0)
	a_val := u16(a)
	b_val := u16(b)

	// Polynomial multiplication
	for i in 0 ..= 7 {
		if (b_val & (1 << uint(i))) != 0 {
			result ~= a_val << uint(i)
		}
	}

	// Reduce modulo polynomial
	return reduce_polynomial(result, polynomial)
}