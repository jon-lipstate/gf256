package gf256

import "base:intrinsics"
import "core:simd"


// Main SIMD multiply region dispatcher
@(private)
multiply_region :: proc(ctx: ^Context, dst, src: []u8, coeff: GF256) {
    
    // Handle special cases first
    if coeff == GF256_ZERO {
        for i in 0..<len(dst) {
            dst[i] = 0
        }
        return
    }
    if coeff == GF256_ONE {
        copy(dst, src)
        return
    }
    
    // Use scalar fallback if no SIMD available
    if ctx.simd_width == .None {
        multiply_region_direct_unrolled(ctx, dst, src, coeff)
        return
    }
    
    // Use the pre-computed split tables from context - always use direct tables
    coeff_table := &ctx.direct_mul_table[coeff]
    
    // Process in chunks of optimal vector width
    i := 0
    simd_end := (len(src) / int(ctx.simd_width)) * int(ctx.simd_width)
    
    for i < simd_end {
        #partial switch ctx.simd_width {
        case .x16:
            process_chunk(dst[i:i+16], src[i:i+16], ctx, coeff, simd.u8x16)
        case .x32:
            process_chunk(dst[i:i+32], src[i:i+32], ctx, coeff, simd.u8x32)
        case .x64:
            process_chunk(dst[i:i+64], src[i:i+64], ctx, coeff, simd.u8x64)
        case:
            // Fallback to scalar for unknown widths
            multiply_region_scalar(dst[i:], src[i:], coeff, &ctx.direct_mul_table)
            return
        }
        i += int(ctx.simd_width)
    }
    
    // Handle remaining bytes with scalar
    for i < len(src) {
        dst[i] = coeff_table[src[i]]
        i += 1
    }
}

// Unified multiply-add region
@(private)
multiply_add_region :: proc(ctx: ^Context, dst, src: []u8, coeff: GF256) {
    
    // Handle special cases
    if coeff == GF256_ZERO {
        return  // Adding zero changes nothing
    }
    if coeff == GF256_ONE {
        // Add source to destination
        for i in 0..<len(src) {
            dst[i] ~= src[i]
        }
        return
    }
    
    // Use scalar fallback if no SIMD available
    if ctx.simd_width == .None {
        multiply_add_region_scalar(dst, src, coeff, &ctx.direct_mul_table)
        return
    }
    
    // Use the pre-computed split tables from context
    coeff_table := &ctx.direct_mul_table[coeff]
    
    // Process in chunks of optimal vector width
    i := 0
    simd_end := (len(src) / int(ctx.simd_width)) * int(ctx.simd_width)
    
    for i < simd_end {
        #partial switch ctx.simd_width {
        case .x16:
            multiply_add_chunk(dst[i:i+16], src[i:i+16], ctx, coeff, simd.u8x16)
        case .x32:
            multiply_add_chunk(dst[i:i+32], src[i:i+32], ctx, coeff, simd.u8x32)
        case .x64:
            multiply_add_chunk(dst[i:i+64], src[i:i+64], ctx, coeff, simd.u8x64)
        case:
            multiply_add_region_scalar(dst[i:], src[i:], coeff, &ctx.direct_mul_table)
            return
        }
        i += int(ctx.simd_width)
    }
    
    // Handle remaining bytes
    for i < len(src) {
        dst[i] ~= coeff_table[src[i]]
        i += 1
    }
}
// Utility function to report SIMD capabilities
@(private)
get_simd_info :: proc(ctx: ^Context) -> (enabled: bool, width: int, features: string) {
    if ctx == nil {
        return false, 0, "no context"
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
    
    return ctx.simd_width != .None, int(ctx.simd_width), simd_name
}


// Scalar fallback functions and generic SIMD utilities
@(private)
multiply_region_scalar :: proc(dst, src: []u8, coeff: GF256, tables: ^[256][256]u8) {
	if coeff == GF256_ZERO {
		for i in 0 ..< len(dst) {
			dst[i] = 0
		}
		return
	}
	if coeff == GF256_ONE {
		copy(dst, src)
		return
	}

	coeff_table := &tables[coeff]
	for i in 0 ..< len(dst) {
		dst[i] = coeff_table[src[i]]
	}
}

@(private)
multiply_add_region_scalar :: proc(dst, src: []u8, coeff: GF256, tables: ^[256][256]u8) {
	if coeff == GF256_ZERO {
		return // Adding zero changes nothing
	}
	if coeff == GF256_ONE {
		// Add source to destination
		for i in 0 ..< len(src) {
			dst[i] ~= src[i]
		}
		return
	}

	coeff_table := &tables[coeff]
	for i in 0 ..< len(src) {
		dst[i] ~= coeff_table[src[i]]
	}
}

// Region addition (XOR) with Lane_Width enum dispatch
@(private)
_add_region :: proc(dst, src: []u8, lane_width: Lane_Width) {
	lanes := int(lane_width)
	simd_end := (len(src) / lanes) * lanes
	i := 0

	switch lane_width {
	case .None:
		// Scalar implementation
		for i in 0 ..< len(src) {
			dst[i] ~= src[i]
		}
		return
	case .x16:
		// Process in chunks of 16 bytes
		for ; i < simd_end; i += lanes {
			add_chunk(dst[i:i + lanes], src[i:i + lanes], simd.u8x16)
		}
	case .x32:
		for ; i < simd_end; i += lanes {
			add_chunk(dst[i:i + lanes], src[i:i + lanes], simd.u8x32)
		}
	case .x64:
		for ; i < simd_end; i += lanes {
			add_chunk(dst[i:i + lanes], src[i:i + lanes], simd.u8x64)
		}
	}
	// Handle remaining bytes
	for i < len(src) {
		dst[i] ~= src[i]
		i += 1
	}
}

// Generic add chunk that works with any SIMD vector type
@(private)
add_chunk :: proc(dst, src: []u8, $T: typeid) where intrinsics.type_is_simd_vector(T) {
	dst_vec := simd.from_slice(T, dst)
	src_vec := simd.from_slice(T, src)
	result_vec := simd.bit_xor(dst_vec, src_vec)
	result_array := simd.to_array(result_vec)
	copy(dst, result_array[:])
}

// Generic SIMD chunk processing for any vector type using pre-computed tables
@(private)
process_chunk :: proc(dst, src: []u8, ctx: ^Context, coeff: GF256, $T: typeid) 
    where intrinsics.type_is_simd_vector(T) {
    
    src_vec := simd.from_slice(T, src)
    
    // catid's 4-bit split technique
    mask_0f := T(0x0f)
    lo_nibbles := simd.bit_and(src_vec, mask_0f)
    hi_nibbles := simd.bit_and(simd.shr_masked(src_vec, T(4)), mask_0f)
    
    // Extract coefficient column from pre-computed tables (temporary approach)
    when T == simd.u8x16 {
        lo_column: [16]u8
        hi_column: [16]u8
        for i in 0..<16 {
            lo_column[i] = ctx.simd_tables.simd_16.mul_lo_table[i][coeff]
            hi_column[i] = ctx.simd_tables.simd_16.mul_hi_table[i][coeff]
        }
        table_lo_vec := simd.from_array(lo_column)
        table_hi_vec := simd.from_array(hi_column)
    } else when T == simd.u8x32 {
        lo_column: [32]u8
        hi_column: [32]u8
        for i in 0..<32 {
            lo_column[i] = ctx.simd_tables.simd_32.mul_lo_scaled_32[i][coeff]
            hi_column[i] = ctx.simd_tables.simd_32.mul_hi_scaled_32[i][coeff]
        }
        table_lo_vec := simd.from_array(lo_column)
        table_hi_vec := simd.from_array(hi_column)
    } else when T == simd.u8x64 {
        lo_column: [64]u8
        hi_column: [64]u8
        for i in 0..<64 {
            lo_column[i] = ctx.simd_tables.simd_64.mul_lo_scaled_64[i][coeff]
            hi_column[i] = ctx.simd_tables.simd_64.mul_hi_scaled_64[i][coeff]
        }
        table_lo_vec := simd.from_array(lo_column)
        table_hi_vec := simd.from_array(hi_column)
    } else {
        #panic("Unsupported SIMD vector size")
    }
    
    // Hardware-accelerated table lookups
    lo_results := simd.table_lookup(table_lo_vec, lo_nibbles)
    hi_results := simd.table_lookup(table_hi_vec, hi_nibbles)
    
    // GF256 addition (XOR) - minimize copies
    result_vec := simd.bit_xor(lo_results, hi_results)
    // Use to_array which should be optimized by the compiler
    result_array := simd.to_array(result_vec)
    copy(dst, result_array[:])
}

// Generic SIMD multiply-add chunk processing for any vector type using pre-computed tables
@(private)
multiply_add_chunk :: proc(dst, src: []u8, ctx: ^Context, coeff: GF256, $T: typeid) 
    where intrinsics.type_is_simd_vector(T) {
    
    src_vec := simd.from_slice(T, src)
    dst_vec := simd.from_slice(T, dst)
    
    // catid's 4-bit split technique
    mask_0f := T(0x0f)
    lo_nibbles := simd.bit_and(src_vec, mask_0f)
    hi_nibbles := simd.bit_and(simd.shr_masked(src_vec, T(4)), mask_0f)
    
    // Extract coefficient column from pre-computed tables (temporary approach)
    when T == simd.u8x16 {
        lo_column: [16]u8
        hi_column: [16]u8
        for i in 0..<16 {
            lo_column[i] = ctx.simd_tables.simd_16.mul_lo_table[i][coeff]
            hi_column[i] = ctx.simd_tables.simd_16.mul_hi_table[i][coeff]
        }
        table_lo_vec := simd.from_array(lo_column)
        table_hi_vec := simd.from_array(hi_column)
    } else when T == simd.u8x32 {
        lo_column: [32]u8
        hi_column: [32]u8
        for i in 0..<32 {
            lo_column[i] = ctx.simd_tables.simd_32.mul_lo_scaled_32[i][coeff]
            hi_column[i] = ctx.simd_tables.simd_32.mul_hi_scaled_32[i][coeff]
        }
        table_lo_vec := simd.from_array(lo_column)
        table_hi_vec := simd.from_array(hi_column)
    } else when T == simd.u8x64 {
        lo_column: [64]u8
        hi_column: [64]u8
        for i in 0..<64 {
            lo_column[i] = ctx.simd_tables.simd_64.mul_lo_scaled_64[i][coeff]
            hi_column[i] = ctx.simd_tables.simd_64.mul_hi_scaled_64[i][coeff]
        }
        table_lo_vec := simd.from_array(lo_column)
        table_hi_vec := simd.from_array(hi_column)
    } else {
        #panic("Unsupported SIMD vector size")
    }
    
    // Hardware-accelerated table lookups
    lo_results := simd.table_lookup(table_lo_vec, lo_nibbles)
    hi_results := simd.table_lookup(table_hi_vec, hi_nibbles)
    
    // Multiply result
    multiply_result := simd.bit_xor(lo_results, hi_results)
    // Add to destination (GF256 multiply-add) - minimize copies
    final_result := simd.bit_xor(dst_vec, multiply_result)
    result_array := simd.to_array(final_result)
    copy(dst, result_array[:])
}

// Cached SIMD multiply region - use for repetitive operations with same coefficient
@(private)
multiply_region_cached :: proc(ctx: ^Context, dst, src: []u8, coeff: GF256, cache: ^Coeff_Cache) {
    
    // Handle special cases first
    if coeff == GF256_ZERO {
        for i in 0..<len(dst) {
            dst[i] = 0
        }
        return
    }
    if coeff == GF256_ONE {
        copy(dst, src)
        return
    }
    
    // Process in chunks of optimal vector width
    i := 0
    simd_end := (len(src) / int(ctx.simd_width)) * int(ctx.simd_width)
    
    for i < simd_end {
        #partial switch ctx.simd_width {
        case .x16:
            process_chunk_cached(dst[i:i+16], src[i:i+16], ctx, coeff, cache, simd.u8x16)
        case .x32:
            process_chunk_cached(dst[i:i+32], src[i:i+32], ctx, coeff, cache, simd.u8x32)
        case .x64:
            process_chunk_cached(dst[i:i+64], src[i:i+64], ctx, coeff, cache, simd.u8x64)
        case:
            multiply_region_scalar(dst[i:], src[i:], coeff, &ctx.direct_mul_table)
            return
        }
        i += int(ctx.simd_width)
    }
    
    // Handle remaining bytes with scalar
    coeff_table := &ctx.direct_mul_table[coeff]
    for i < len(src) {
        dst[i] = coeff_table[src[i]]
        i += 1
    }
}

// Cached SIMD multiply-add region
@(private)  
multiply_add_region_cached :: proc(ctx: ^Context, dst, src: []u8, coeff: GF256, cache: ^Coeff_Cache) {
    
    // Handle special cases
    if coeff == GF256_ZERO {
        return  // Adding zero changes nothing
    }
    if coeff == GF256_ONE {
        for i in 0..<len(src) {
            dst[i] ~= src[i]
        }
        return
    }
    
    // Process in chunks of optimal vector width
    i := 0
    simd_end := (len(src) / int(ctx.simd_width)) * int(ctx.simd_width)
    
    for i < simd_end {
        #partial switch ctx.simd_width {
        case .x16:
            multiply_add_chunk_cached(dst[i:i+16], src[i:i+16], ctx, coeff, cache, simd.u8x16)
        case .x32:
            multiply_add_chunk_cached(dst[i:i+32], src[i:i+32], ctx, coeff, cache, simd.u8x32)
        case .x64:
            multiply_add_chunk_cached(dst[i:i+64], src[i:i+64], ctx, coeff, cache, simd.u8x64)
        case:
            multiply_add_region_scalar(dst[i:], src[i:], coeff, &ctx.direct_mul_table)
            return
        }
        i += int(ctx.simd_width)
    }
    
    // Handle remaining bytes
    coeff_table := &ctx.direct_mul_table[coeff]
    for i < len(src) {
        dst[i] ~= coeff_table[src[i]]
        i += 1
    }
}

// Cached SIMD chunk processing - eliminates coefficient table rebuilding
@(private)
process_chunk_cached :: proc(dst, src: []u8, ctx: ^Context, coeff: GF256, cache: ^Coeff_Cache, $T: typeid) 
    where intrinsics.type_is_simd_vector(T) {
    
    src_vec := simd.from_slice(T, src)
    
    // catid's 4-bit split technique
    mask_0f := T(0x0f)
    lo_nibbles := simd.bit_and(src_vec, mask_0f)
    hi_nibbles := simd.bit_and(simd.shr_masked(src_vec, T(4)), mask_0f)
    
    // Use coefficient caching for optimal performance
    when T == simd.u8x16 {
        if cache.cached_coeff != coeff {
            // Cache miss - extract coefficient column once
            for i in 0..<16 {
                cache.cached_lo_16[i] = ctx.simd_tables.simd_16.mul_lo_table[i][coeff]
                cache.cached_hi_16[i] = ctx.simd_tables.simd_16.mul_hi_table[i][coeff]
            }
            cache.cached_coeff = coeff  // Update cache
        }
        table_lo_vec := simd.from_array(cache.cached_lo_16)
        table_hi_vec := simd.from_array(cache.cached_hi_16)
    } else when T == simd.u8x32 {
        if cache.cached_coeff != coeff {
            for i in 0..<32 {
                cache.cached_lo_32[i] = ctx.simd_tables.simd_32.mul_lo_scaled_32[i][coeff]
                cache.cached_hi_32[i] = ctx.simd_tables.simd_32.mul_hi_scaled_32[i][coeff]
            }
            cache.cached_coeff = coeff
        }
        table_lo_vec := simd.from_array(cache.cached_lo_32)
        table_hi_vec := simd.from_array(cache.cached_hi_32)
    } else when T == simd.u8x64 {
        if cache.cached_coeff != coeff {
            for i in 0..<64 {
                cache.cached_lo_64[i] = ctx.simd_tables.simd_64.mul_lo_scaled_64[i][coeff]
                cache.cached_hi_64[i] = ctx.simd_tables.simd_64.mul_hi_scaled_64[i][coeff]
            }
            cache.cached_coeff = coeff
        }
        table_lo_vec := simd.from_array(cache.cached_lo_64)
        table_hi_vec := simd.from_array(cache.cached_hi_64)
    } else {
        #panic("Unsupported SIMD vector size")
    }
    
    // Hardware-accelerated table lookups
    lo_results := simd.table_lookup(table_lo_vec, lo_nibbles)
    hi_results := simd.table_lookup(table_hi_vec, hi_nibbles)
    
    // GF256 addition (XOR)
    result_vec := simd.bit_xor(lo_results, hi_results)
    result_array := simd.to_array(result_vec)
    copy(dst, result_array[:])
}

// Cached SIMD multiply-add chunk processing
@(private)
multiply_add_chunk_cached :: proc(dst, src: []u8, ctx: ^Context, coeff: GF256, cache: ^Coeff_Cache, $T: typeid) 
    where intrinsics.type_is_simd_vector(T) {
    
    src_vec := simd.from_slice(T, src)
    dst_vec := simd.from_slice(T, dst)
    
    // catid's 4-bit split technique
    mask_0f := T(0x0f)
    lo_nibbles := simd.bit_and(src_vec, mask_0f)
    hi_nibbles := simd.bit_and(simd.shr_masked(src_vec, T(4)), mask_0f)
    
    // Use coefficient caching for optimal performance
    when T == simd.u8x16 {
        if cache.cached_coeff != coeff {
            for i in 0..<16 {
                cache.cached_lo_16[i] = ctx.simd_tables.simd_16.mul_lo_table[i][coeff]
                cache.cached_hi_16[i] = ctx.simd_tables.simd_16.mul_hi_table[i][coeff]
            }
            cache.cached_coeff = coeff
        }
        table_lo_vec := simd.from_array(cache.cached_lo_16)
        table_hi_vec := simd.from_array(cache.cached_hi_16)
    } else when T == simd.u8x32 {
        if cache.cached_coeff != coeff {
            for i in 0..<32 {
                cache.cached_lo_32[i] = ctx.simd_tables.simd_32.mul_lo_scaled_32[i][coeff]
                cache.cached_hi_32[i] = ctx.simd_tables.simd_32.mul_hi_scaled_32[i][coeff]
            }
            cache.cached_coeff = coeff
        }
        table_lo_vec := simd.from_array(cache.cached_lo_32)
        table_hi_vec := simd.from_array(cache.cached_hi_32)
    } else when T == simd.u8x64 {
        if cache.cached_coeff != coeff {
            for i in 0..<64 {
                cache.cached_lo_64[i] = ctx.simd_tables.simd_64.mul_lo_scaled_64[i][coeff]
                cache.cached_hi_64[i] = ctx.simd_tables.simd_64.mul_hi_scaled_64[i][coeff]
            }
            cache.cached_coeff = coeff
        }
        table_lo_vec := simd.from_array(cache.cached_lo_64)
        table_hi_vec := simd.from_array(cache.cached_hi_64)
    } else {
        #panic("Unsupported SIMD vector size")
    }
    
    // Hardware-accelerated table lookups
    lo_results := simd.table_lookup(table_lo_vec, lo_nibbles)
    hi_results := simd.table_lookup(table_hi_vec, hi_nibbles)
    
    // Multiply result
    multiply_result := simd.bit_xor(lo_results, hi_results)
    // Add to destination (GF256 multiply-add)
    final_result := simd.bit_xor(dst_vec, multiply_result)
    result_array := simd.to_array(final_result)
    copy(dst, result_array[:])
}

