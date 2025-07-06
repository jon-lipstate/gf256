package gf256

import "core:mem"
import "core:slice"

// Core GF(256) utility operations

// XOR two regions: dst[i] = a[i] ^ b[i]
xor_region :: proc(dst: []u8, a: []u8, b: []u8) {
    assert(len(dst) == len(a) && len(a) == len(b), "All slices must have equal length")
    
    for i in 0..<len(dst) {
        dst[i] = a[i] ~ b[i]
    }
}

// Dot product of two regions using context
ctx_dot_product :: proc(ctx: ^Context, a: []u8, b: []u8) -> GF256 {
    assert(len(a) == len(b), "Input slices must have equal length")
    
    result := GF256_ZERO
    for i in 0..<len(a) {
        product := ctx_multiply(ctx, GF256(a[i]), GF256(b[i]))
        result = ctx_add(ctx, result, product)
    }
    
    return result
}

// Polynomial evaluation using Horner's method
ctx_poly_eval :: proc(ctx: ^Context, coeffs: []GF256, x: GF256) -> GF256 {
    if len(coeffs) == 0 {
        return GF256_ZERO
    }
    
    result := coeffs[len(coeffs)-1]
    for i := len(coeffs)-2; i >= 0; i -= 1 {
        result = ctx_add(ctx, ctx_multiply(ctx, result, x), coeffs[i])
    }
    return result
}

// Fast polynomial evaluation for multiple points using context
ctx_poly_eval_multiple :: proc(ctx: ^Context, results: []u8, coeffs: []u8, points: []GF256) {
    assert(len(results) == len(points), "Results length must match points length")
    
    for i in 0..<len(points) {
        results[i] = u8(ctx_poly_eval(ctx, slice.reinterpret([]GF256, coeffs), points[i]))
    }
}

// XOR multiple regions together: dst = src1 ^ src2 ^ ... ^ srcN
ctx_xor_regions_batch :: proc(ctx: ^Context, dst: []u8, sources: [][]u8) {
    if len(sources) == 0 {
        mem.zero_slice(dst)
        return
    }
    
    // Start with first source
    copy(dst, sources[0])
    
    // XOR remaining sources
    for i in 1..<len(sources) {
        ctx_add_region(ctx, dst, sources[i])
    }
}