package tests

import "../"
import "core:fmt"
import "core:testing"

// Comprehensive correctness tests for GF(256) library
// Tests all core functionality for mathematical correctness

@(test)
test_context_creation :: proc(t: ^testing.T) {
    // Test context creation with default polynomial
    ctx := gf256.context_create()
    defer gf256.context_destroy(ctx)
    
    testing.expect(t, ctx != nil, "Context creation should succeed")
    testing.expect(t, gf256.ctx_verify(ctx), "Context should be valid")
    
    // Test context creation with custom polynomial
    custom_ctx := gf256.context_create(0x1D7)
    defer gf256.context_destroy(custom_ctx)
    testing.expect(t, custom_ctx != nil, "Custom polynomial context creation should succeed")
    testing.expect(t, gf256.ctx_verify(custom_ctx), "Custom context should be valid")
    
    // Test context creation with different SIMD preferences
    lane_widths := []gf256.Lane_Width{.None, .x16, .x32, .x64}
    for lane_width in lane_widths {
        simd_ctx := gf256.context_create(gf256.AES_POLYNOMIAL, lane_width)
        defer gf256.context_destroy(simd_ctx)
        testing.expect(t, simd_ctx != nil, fmt.tprintf("SIMD context creation should succeed for %v", lane_width))
    }
}

@(test)
test_scalar_operations :: proc(t: ^testing.T) {
    ctx := gf256.context_create()
    defer gf256.context_destroy(ctx)
    
    testing.expect(t, ctx != nil, "Context creation should succeed")
    
    // Test basic arithmetic properties
    test_values := []gf256.GF256{0, 1, 2, 3, 7, 13, 17, 123, 200, 255}
    
    for a in test_values {
        for b in test_values {
            // Test multiplication properties
            mul_result := gf256.ctx_multiply(ctx, a, b)
            
            // Commutativity: a * b = b * a
            testing.expect(t, 
                gf256.ctx_multiply(ctx, a, b) == gf256.ctx_multiply(ctx, b, a),
                fmt.tprintf("Multiplication should be commutative: %d * %d", a, b))
            
            // Zero multiplication
            testing.expect(t,
                gf256.ctx_multiply(ctx, 0, b) == 0,
                fmt.tprintf("Zero multiplication: 0 * %d should be 0", b))
            
            // Identity multiplication
            testing.expect(t,
                gf256.ctx_multiply(ctx, a, 1) == a,
                fmt.tprintf("Identity multiplication: %d * 1 should be %d", a, a))
            
            // Addition/subtraction equivalence (XOR in GF256)
            add_result := gf256.ctx_add(ctx, a, b)
            sub_result := gf256.ctx_subtract(ctx, a, b)
            testing.expect(t, add_result == sub_result,
                fmt.tprintf("Add and subtract should be equal in GF256: %d + %d = %d - %d", a, b, a, b))
            
            // Division (skip division by zero)
            if b != 0 {
                div_result := gf256.ctx_divide(ctx, a, b)
                reconstructed := gf256.ctx_multiply(ctx, div_result, b)
                testing.expect(t, reconstructed == a,
                    fmt.tprintf("Division check: (%d / %d) * %d should equal %d", a, b, b, a))
            }
            
            // Inverse (skip zero)
            if a != 0 {
                inv_a := gf256.ctx_inverse(ctx, a)
                product := gf256.ctx_multiply(ctx, a, inv_a)
                testing.expect(t, product == gf256.GF256_ONE,
                    fmt.tprintf("Inverse check: %d * inv(%d) should equal 1", a, a))
            }
        }
    }
}

@(test)
test_power_operations :: proc(t: ^testing.T) {
    ctx := gf256.context_create()
    defer gf256.context_destroy(ctx)
    
    testing.expect(t, ctx != nil, "Context creation should succeed")
    
    test_values := []gf256.GF256{1, 2, 3, 7, 123}
    
    for a in test_values {
        // Test power properties
        testing.expect(t, gf256.ctx_power(ctx, a, 0) == 1, "a^0 should equal 1")
        testing.expect(t, gf256.ctx_power(ctx, a, 1) == a, "a^1 should equal a")
        
        // Test a^2 = a * a
        power2 := gf256.ctx_power(ctx, a, 2)
        manual2 := gf256.ctx_multiply(ctx, a, a)
        testing.expect(t, power2 == manual2, fmt.tprintf("a^2 should equal a*a for a=%d", a))
        
        // Test a^255 = 1 for non-zero a (Fermat's little theorem for GF256)
        if a != 0 {
            power255 := gf256.ctx_power(ctx, a, 255)
            testing.expect(t, power255 == 1, fmt.tprintf("a^255 should equal 1 for a=%d", a))
        }
    }
}

@(test)
test_region_operations :: proc(t: ^testing.T) {
    ctx := gf256.context_create()
    defer gf256.context_destroy(ctx)
    
    testing.expect(t, ctx != nil, "Context creation should succeed")
    
    // Test various sizes
    sizes := []int{1, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129}
    
    for size in sizes {
        src := make([]u8, size)
        dst := make([]u8, size)
        expected := make([]u8, size)
        defer delete(src)
        defer delete(dst)
        defer delete(expected)
        
        // Initialize test data
        for i in 0..<size {
            src[i] = u8((i * 17 + 3) & 0xFF)
        }
        
        test_coeffs := []gf256.GF256{0, 1, 2, 7, 123, 255}
        
        for coeff in test_coeffs {
            // Test multiply_region
            gf256.ctx_multiply_region(ctx, dst, src, coeff)
            
            // Calculate expected results using scalar operations
            for i in 0..<size {
                expected[i] = u8(gf256.ctx_multiply(ctx, gf256.GF256(src[i]), coeff))
            }
            
            // Verify results
            for i in 0..<size {
                testing.expect(t, dst[i] == expected[i],
                    fmt.tprintf("multiply_region mismatch at size=%d, index=%d, coeff=%d", size, i, coeff))
            }
            
            // Test multiply_add_region
            // Initialize destination with known values
            for i in 0..<size {
                dst[i] = u8((i * 5 + 7) & 0xFF)
                expected[i] = dst[i]
            }
            
            gf256.ctx_multiply_add_region(ctx, dst, src, coeff)
            
            // Calculate expected: dst = original_dst + (src * coeff)
            for i in 0..<size {
                multiply_result := gf256.ctx_multiply(ctx, gf256.GF256(src[i]), coeff)
                expected[i] = u8(gf256.ctx_add(ctx, gf256.GF256(expected[i]), multiply_result))
            }
            
            // Verify results
            for i in 0..<size {
                testing.expect(t, dst[i] == expected[i],
                    fmt.tprintf("multiply_add_region mismatch at size=%d, index=%d, coeff=%d", size, i, coeff))
            }
        }
    }
}


@(test)
test_xor_operations :: proc(t: ^testing.T) {
    ctx := gf256.context_create()
    defer gf256.context_destroy(ctx)
    
    // Test XOR region operations
    size := 32
    src := make([]u8, size)
    dst := make([]u8, size)
    defer delete(src)
    defer delete(dst)
    
    for i in 0..<size {
        src[i] = u8(i)
        dst[i] = u8(i * 2)
    }
    
    original_dst := make([]u8, size)
    copy(original_dst, dst)
    defer delete(original_dst)
    
    // Test ctx_add_region
    gf256.ctx_add_region(ctx, dst, src)
    
    for i in 0..<size {
        expected := original_dst[i] ~ src[i]
        testing.expect(t, dst[i] == expected,
            fmt.tprintf("XOR region mismatch at index %d", i))
    }
    
    // Test standalone xor_region
    dst1 := []u8{1, 2, 3, 4}
    src1 := []u8{5, 6, 7, 8}
    result := make([]u8, 4)
    defer delete(result)
    
    gf256.xor_region(result, dst1, src1)
    
    for i in 0..<4 {
        expected := dst1[i] ~ src1[i]
        testing.expect(t, result[i] == expected,
            fmt.tprintf("Standalone XOR mismatch at index %d", i))
    }
}

@(test)
test_predicates :: proc(t: ^testing.T) {
    ctx := gf256.context_create()
    defer gf256.context_destroy(ctx)
    
    // Test is_zero and is_one
    testing.expect(t, gf256.is_zero(0), "is_zero should return true for 0")
    testing.expect(t, !gf256.is_zero(1), "is_zero should return false for 1")
    testing.expect(t, !gf256.is_zero(255), "is_zero should return false for 255")
    
    testing.expect(t, gf256.is_one(1), "is_one should return true for 1")
    testing.expect(t, !gf256.is_one(0), "is_one should return false for 0")
    testing.expect(t, !gf256.is_one(2), "is_one should return false for 2")
}

@(test)
test_polynomial_operations :: proc(t: ^testing.T) {
    ctx := gf256.context_create()
    defer gf256.context_destroy(ctx)
    
    // Test polynomial evaluation
    coeffs := []gf256.GF256{1, 2, 3}  // 3x^2 + 2x + 1
    
    // Test at x = 0: should give constant term
    result := gf256.ctx_poly_eval(ctx, coeffs, 0)
    testing.expect(t, result == 1, "Polynomial at x=0 should give constant term")
    
    // Test at x = 1: should give sum of coefficients
    result = gf256.ctx_poly_eval(ctx, coeffs, 1)
    expected := gf256.ctx_add(ctx, gf256.ctx_add(ctx, 1, 2), 3)  // 1 + 2 + 3 in GF256
    testing.expect(t, result == expected, "Polynomial at x=1 should give sum of coefficients")
    
    // Test dot product
    a := []u8{1, 2, 3, 4}
    b := []u8{5, 6, 7, 8}
    dot := gf256.ctx_dot_product(ctx, a, b)
    
    // Manual calculation: (1*5) + (2*6) + (3*7) + (4*8) in GF256
    expected_dot := gf256.ctx_multiply(ctx, 1, 5)
    expected_dot = gf256.ctx_add(ctx, expected_dot, gf256.ctx_multiply(ctx, 2, 6))
    expected_dot = gf256.ctx_add(ctx, expected_dot, gf256.ctx_multiply(ctx, 3, 7))
    expected_dot = gf256.ctx_add(ctx, expected_dot, gf256.ctx_multiply(ctx, 4, 8))
    
    testing.expect(t, dot == expected_dot, "Dot product should match manual calculation")
}

@(test)
test_edge_cases :: proc(t: ^testing.T) {
    ctx := gf256.context_create()
    defer gf256.context_destroy(ctx)
    
    // Test empty slices
    empty_src := []u8{}
    empty_dst := []u8{}
    gf256.ctx_multiply_region(ctx, empty_dst, empty_src, 7) // Should not crash
    
    // Test single element
    single_src := []u8{123}
    single_dst := make([]u8, 1)
    defer delete(single_dst)
    
    gf256.ctx_multiply_region(ctx, single_dst, single_src, 7)
    expected := gf256.ctx_multiply(ctx, gf256.GF256(single_src[0]), 7)
    testing.expect(t, gf256.GF256(single_dst[0]) == expected, "Single element multiply should work")
    
    // Test maximum values
    max_src := []u8{255}
    max_dst := make([]u8, 1)
    defer delete(max_dst)
    
    gf256.ctx_multiply_region(ctx, max_dst, max_src, 255)
    expected = gf256.ctx_multiply(ctx, 255, 255)
    testing.expect(t, gf256.GF256(max_dst[0]) == expected, "Maximum value multiply should work")
}