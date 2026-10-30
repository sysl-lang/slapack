# slapack

A subset of LAPACK written in pure sysl, over the views and Level 3 routines of
[sblas](https://github.com/sysl-lang/sblas). Module `sh.sysl.slapack`; no C, no allocation in the core,
and generic over `f32` and `f64`. It is a stressor for the language — blocked factorisations put
generics, views, aliasing and float semantics under load — and a reference for how dense linear algebra
reads in sysl. For production work use [lapack](https://github.com/sysl-lang/lapack), the binding to the
real LAPACK (Accelerate on macOS, OpenBLAS elsewhere).

*An independent sysl package for LAPACK. It is not affiliated with or endorsed by LAPACK's authors or
trademark holders.*

## The subset

| family | routines | state |
|---|---|---|
| LU | `getrf`, `getrs` (and `getf2`, `laswp`) | done, f32 and f64 |
| Cholesky | `potrf`, `potrs` (and `potf2`) | done, f32 and f64 |
| QR | `geqrf`, `orgqr` | planned |
| symmetric eigenproblem | `syev` | planned |
| singular value decomposition | `gesvd` | planned |

## Use

```hocon
dependencies {
  slapack { git = "github.com/sysl-lang/slapack", version = "0.1.0" }
}
```

A matrix is sblas's column-major view, so `lda` is the view's and a transpose is `a.t()`. Pivots are
0-based, and a zero pivot is an `Err` rather than an `info`:

```sysl
import sh.sysl.sblas.{matrix, matrix_mut}
import sh.sysl.slapack.{Error, Transpose}
import sh.sysl.slapack.heap.{getrf_alloc, getrs_alloc}

/** x with A x = b for the 3 x 3 A in `a`, column-major; `a` is left holding A's LU factors. */
solve3(a: []f64, b: []f64) -> Result[unit, Error]
    var ipiv: [3]usize
    getrf_alloc(matrix_mut(a, 3, 3, 3), ipiv[..])?                                  // PA = LU in place
    getrs_alloc(Transpose.NoTrans, matrix(a, 3, 3, 3), ipiv[..], matrix_mut(b, 3, 1, 3))
    Ok(())

var a: [9]f64 = [2, 4, -2, 1, -6, 7, 1, 0, 2]          // [2 1 1; 4 -6 0; -2 7 2]
var b: [3]f64 = [5, -2, 9]
print(solve3(a[..], b[..]), b)                         // Ok(()) [1, 1, 2]

var s: [4]f64 = [1, 2, 2, 4]                           // [1 2; 2 4] is singular
var p: [2]usize
print(getrf_alloc(matrix_mut(s[..], 2, 2, 2), p[..]) == Err(Error.Singular(1)))   // true: U(1, 1) = 0
```

A symmetric positive definite system is factored from one triangle, named by sblas's `Uplo`, and the
other triangle is never read or written:

```sysl
import sh.sysl.sblas.{Uplo, matrix, matrix_mut}
import sh.sysl.slapack.Error
import sh.sysl.slapack.heap.{potrf_alloc, potrs_alloc}

var a: [9]f64 = [4, 2, 2, 0, 5, 3, 0, 0, 6]           // the lower triangle of [4 2 2; 2 5 3; 2 3 6]
var b: [3]f64 = [8, 10, 11]
print(potrf_alloc(Uplo.Lower, matrix_mut(a[..], 3, 3, 3)))                     // Ok(()): A = L Lᵀ
potrs_alloc(Uplo.Lower, matrix(a[..], 3, 3, 3), matrix_mut(b[..], 3, 1, 3))
print(b)                                                                       // [1, 1, 1]

var c: [4]f64 = [1, 2, 2, 1]                           // [1 2; 2 1] is indefinite
print(potrf_alloc(Uplo.Upper, matrix_mut(c[..], 2, 2, 2)) == Err(Error.NotPositiveDefinite(1)))  // true
```

The routines themselves take sblas's `GemmWork` packing buffers, sized by `getrf_work_len`,
`getrs_work_len`, `potrf_work_len` and `potrs_work_len`, and allocate nothing; `sh.sysl.slapack.heap`
holds the forms that allocate them per call. The design, and what the language made easy or hard, is in [DESIGN.md](DESIGN.md).

## Tests

`sysl test .` runs the package's tests against the routines' definitions — hand-factored systems, and
LAPACK's own residual measures on random matrices of every order from 1 to 64 and past the panel width,
square, tall and wide, and symmetric positive definite from either triangle — and `oracle/` compares the
factors with the platform's `dgetrf`, `sgetrf`, `dpotrf` and `spotrf` through the `lapack` binding.
