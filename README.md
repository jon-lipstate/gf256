# GF(256) - High-Performance Galois Field Library

A fast, SIMD-optimized Galois Field GF(256) implementation in Odin, designed for Reed-Solomon coding, erasure coding, and cryptographic applications. Based on catid/gf256 (BSD-3).

## Features

- **High Performance**: 5+ GB/s throughput with SIMD optimization
- **Multiple SIMD Backends**: SSE/SSSE3, AVX2, AVX512, and ARM NEON support
- **Cache-Friendly Design**: Optimized memory layout for modern CPUs
- **Cross-Platform**: Works on x86-64, ARM64, and ARM32
- **Zero Dependencies**: Pure Odin implementation
- **Flexible API**: Both scalar and bulk region operations
- **Multiple Polynomials**: Support for AES polynomial and custom polynomials

## Performance

```text
gf256\tests>odin run performance_tests.odin -file -o:speed

=== GF(256) Performance Benchmarks ===
Comparing scalar vs SIMD performance across all operations

=== Multiply Region Performance ===
Operation: multiply_region (1 MB data, coefficient=123, 1000 iterations)

Scalar              : 02.2 GB/s
SSE/SSSE3 (128-bit) : 22.1 GB/s
AVX2 (256-bit)      : 01.4 GB/s
AVX512 (512-bit)    : 00.9 GB/s

Best performance: SSE/SSSE3 (128-bit) at 22.1 GB/s

=== Operation Type Comparison (x16 only) ===
Test size: 1 MB, Iterations: 1000

Multiply Region     : 22.2 GB/s
Multiply-Add Region : 21.5 GB/s
XOR/Add Region      : 40.1 GB/s
XOR (standalone)    : 52.6 GB/s

=== Cache vs Memory Bandwidth Performance ===
Demonstrates performance difference between in-cache and memory-bound operations

Testing with optimal x16 SIMD:
Size             Throughput     vs L1    vs Memory   Description
-------------------------------------------------------------------------------
L1 Cache         22.1 GB/s      1.00x    1.00x       32KB (fits in L1 data cache)
L2 Cache         21.8 GB/s      0.99x    1.00x       256KB (fits in L2 cache)
L3 Cache         21.7 GB/s      0.98x    1.00x       8MB (fits in L3 cache)
Main Memory      17.0 GB/s      0.77x    1.00x       64MB (exceeds L3, hits main memory)
Memory Bandwidth 15.3 GB/s      0.69x    1.00x       512MB (memory bandwidth limited)
```
*Benchmarked on AMD Ryzen 9 9950X3D*

## Quick Start

```odin
import gf "gf256"

main :: proc() {
    // Create context (defaults to optimal SIMD width)
    ctx := gf.context_create()
    defer gf.context_destroy(ctx)
    
    // Scalar operations
    a := gf.GF256(123)
    b := gf.GF256(45)
    result := gf.ctx_multiply(ctx, a, b)
    
    // Bulk operations
    src := []u8{1, 2, 3, 4, 5, 6, 7, 8}
    dst := make([]u8, len(src))
    defer delete(dst)
    
    // Multiply entire region by coefficient
    gf.ctx_multiply_region(ctx, dst, src, gf.GF256(7))
    
    // Multiply-add: dst = dst + (src * coeff)
    gf.ctx_multiply_add_region(ctx, dst, src, gf.GF256(13))
}
```

## API Reference

### Context Management

```odin
// Create context with default AES polynomial and optimal SIMD
context_create :: proc() -> ^Context

// Create with custom polynomial and SIMD preference
context_create :: proc(polynomial: u16, requested: Lane_Width) -> ^Context

// Clean up
context_destroy :: proc(ctx: ^Context)

// Verify context integrity
ctx_verify :: proc(ctx: ^Context) -> bool
```

### Scalar Operations

```odin
ctx_multiply :: proc(ctx: ^Context, a, b: GF256) -> GF256
ctx_divide :: proc(ctx: ^Context, a, b: GF256) -> GF256
ctx_add :: proc(ctx: ^Context, a, b: GF256) -> GF256        // XOR
ctx_subtract :: proc(ctx: ^Context, a, b: GF256) -> GF256   // XOR
ctx_inverse :: proc(ctx: ^Context, a: GF256) -> GF256
ctx_power :: proc(ctx: ^Context, a: GF256, n: int) -> GF256

// Predicates
is_zero :: proc(a: GF256) -> bool
is_one :: proc(a: GF256) -> bool
```

### Bulk Operations

```odin
// Multiply region: dst[i] = src[i] * coeff
ctx_multiply_region :: proc(ctx: ^Context, dst, src: []u8, coeff: GF256)

// Multiply-add: dst[i] = dst[i] + (src[i] * coeff)
ctx_multiply_add_region :: proc(ctx: ^Context, dst, src: []u8, coeff: GF256)

// Add regions: dst[i] = dst[i] + src[i] (XOR)
ctx_add_region :: proc(ctx: ^Context, dst, src: []u8)

```

### Utilities

```odin
// XOR two regions (standalone, no context needed)
xor_region :: proc(dst, src1, src2: []u8)

// Polynomial evaluation
ctx_poly_eval :: proc(ctx: ^Context, coeffs: []GF256, x: GF256) -> GF256

// Dot product in GF(256)
ctx_dot_product :: proc(ctx: ^Context, a, b: []u8) -> GF256
```

## SIMD Architecture

The library automatically detects and uses the best available SIMD instruction set:

- **x86-64**: SSE3/SSSE3 (128-bit), AVX2 (256-bit), AVX512 (512-bit)
- **ARM64**: NEON (128-bit)
- **ARM32**: NEON (128-bit)
- **Fallback**: Scalar implementation

**Recommendation**: Use the default settings. The library automatically selects **128-bit SIMD (x16)** which provides optimal performance for GF(256) operations. Wider SIMD (AVX2/AVX512) actually performs worse due to the table lookup method chosed. **x32/x64** should use `PCLMULQDQ` which this library does _not_ implement.

## Implementation Details

### Algorithm

Uses **catid's 4-bit split technique** with hardware-accelerated table lookups:

1. Split each byte into 4-bit nibbles
2. Use SIMD table lookup (`vpshufb`/`tbl`) for multiplication
3. XOR results to get final product

This approach provides excellent performance for GF(256) while maintaining accuracy.

### Table Sizes

- **Direct multiplication tables**: 128KB (64KB × 2)
- **SIMD lookup tables**: 8KB for 128-bit operations
- **Total context size**: ~158KB

## Custom Polynomials

Support for custom irreducible polynomials:

```odin
// Use CRC-8 polynomial instead of AES
ctx := gf256.context_create(0x1D7)  // x^8 + x^7 + x^6 + x^4 + x^2 + 1
```

Common polynomials:
- **AES**: `0x11D` (default) - x^8 + x^4 + x^3 + x + 1
- **CRC-8**: `0x1D7` - x^8 + x^7 + x^6 + x^4 + x^2 + 1

## Performance Tips

1. **Use bulk operations** when possible - much faster than loops of scalar operations
2. **Reuse contexts** - table generation is expensive
3. **Prefer x16 SIMD** - wider vectors don't help for GF(256)
4. **Align data** when possible for optimal SIMD performance
5. **Align data** when possible for optimal SIMD performance

## Limitations

- **Table lookup approach**: Does not scale beyond 128-bit SIMD effectively
- **Memory usage**: Requires 158KB per context
- **AVX2/AVX512**: Slower than SSE due to cross-lane penalties in `vpshufb`

For applications requiring wider SIMD, consider switching to carryless multiplication (PCLMULQDQ) approaches, though these are typically slower for GF(256).

## Acknowledgements

This implementation builds upon techniques from:
- catid's GF256 library (4-bit split optimization)

