# sysl-lang/slapack — design

A subset of LAPACK written in sysl, the companion to [sblas](https://github.com/sysl-lang/sblas). LAPACK
is written almost entirely in terms of the BLAS, and slapack is written in terms of sblas the same way:
a blocked algorithm hands its panels and trailing blocks to `trsm` and `gemm`, its columns and rows to
`iamax`, `scal`, `swap` and `ger`, and adds only the bookkeeping of its own. The package is a stressor
for the language — generic numeric code, views of views, two views of one storage in a single call,
float edge cases — and a gap it finds is a gap to fix in sysl.

## Scope

| family | routines | state |
|---|---|---|
| LU | `getrf` (blocked), `getf2` (its unblocked panel), `getrs`, `laswp` | done, f32 and f64 |
| Cholesky | `potrf` (blocked), `potf2` (its unblocked kernel), `potrs` | done, f32 and f64 |
| QR | `geqrf` (blocked), `geqr2` (its unblocked panel), `orgqr`, `org2r`, `ormqr`, `larfg` | done, f32 and f64 |
| symmetric eigenproblem | `syev` (`sytrd`, `orgtr`, `steqr`/`sterf`) | planned |
| singular value decomposition | `gesvd` (`gebrd`, `orgbr`, `bdsqr`) | planned |

Complex element types, the banded and packed storage forms, and the expert drivers (`gesvx` and its
kind) are not scheduled.

## The API

**A matrix is sblas's view, and nothing is re-declared here.** `Matrix[T]` and `MatrixMut[T]` hold a
slice, a shape and a stride per dimension; `matrix(data, rows, cols, ld)` is LAPACK's column-major layout
with leading dimension `ld`, and `a.t()` is the transpose over the same storage. sblas's
`a.block(i, j, rows, cols)`, `a.col(j)` and `a.row(i)` take a view apart, so `A(j:m, j:j+jb)` with `lda`
is `a.block(j, j, m - j, jb)`. The element types are sblas's `Real`: f32 and f64, each routine written
once over `[T: Real]`.

```sysl
getrf(a, ipiv, work)                       // Result[unit, Error]: PA = LU in place, ipiv 0-based
getrs(Transpose.NoTrans, lu, ipiv, b, work)  // b := A⁻¹ b, with getrf's factors and pivots
getrs(Transpose.Trans, lu, ipiv, b, work)    // b := A⁻ᵀ b
getf2(a, ipiv)                             // the unblocked factorisation
laswp(a, ipiv, from, to, forward)          // the row interchanges ipiv[from..<to] describe
potrf(Uplo.Lower, a, work)                 // Result[unit, Error]: A = L Lᵀ in A's lower triangle
potrf(Uplo.Upper, a, work)                 // A = Uᵀ U in its upper triangle
potrs(Uplo.Lower, l, b, work)              // b := A⁻¹ b, with potrf's factor
potf2(uplo, a)                             // the unblocked factorisation
geqrf(a, tau, qr_work)                     // A = QR in place: R above the diagonal, reflectors below
geqr2(a, tau, work)                        // the unblocked factorisation
orgqr(q, tau, qr_work)                     // Q's first q.cols columns from tau.len reflectors in q
org2r(q, tau, work)                        // the unblocked form
ormqr(Side.Left, Transpose.Trans, a, tau, c, qr_work)  // c := Qᵀ c; Right gives c Q or c Qᵀ
larfg(alpha, x)                            // (beta, tau): the reflector taking (alpha, x) to (beta, 0)
```

**Workspace is the caller's, as in sblas.** `getrf` and `getrs` take sblas's `GemmWork` packing buffers
for the `trsm` and `gemm` they call, sized by `getrf_work_len[T](m, n)` and `getrs_work_len[T](n,
nrhs)` (and `potrf_work_len[T](n)`, `potrs_work_len[T](n, nrhs)`); the core module is `@no_alloc` and
builds for a board with no heap. The QR routines need a block reflector's factor and product besides
the packing buffers, so they take a `QrWork(block, gemm)`, sized by `geqrf_work_len[T](m, n)`,
`orgqr_work_len[T](m, n, k)` and `ormqr_work_len[T](side, m, n, k)` as a `QrWorkLen(block, gemm)`;
the unblocked `geqr2` and `org2r` take one vector of `a.cols`. `sh.sysl.slapack.heap` holds
`getrf_alloc`, `getrs_alloc`, `potrf_alloc`, `potrs_alloc`, `geqrf_alloc`, `orgqr_alloc` and
`ormqr_alloc`, which allocate the buffers per call.

**The triangle is an `Uplo`, sblas's own enum.** Unlike `getrs`'s `Transpose`, it could have been a
view: the upper triangle of A is the lower triangle of `a.t()`, and the upper factor is the transposed
lower one, so `potrf(Uplo.Upper, a)` *is* `potrf(Uplo.Lower, a.t())`, and that is how it is written.
The flag stays because LAPACK's callers think in it and because it names which triangle is read —
the other is never touched, which a caller holding something else there needs to be told.

**`getrs` takes a `Transpose`, not a transposed view.** A view's `t()` cannot stand in for LAPACK's
`trans` here: the factors of Aᵀ are not the transposed factors of A with the same pivots, so the
solve has to know which system it is solving. Within `getrs` the transposed solve *is* written with
views — `trsm` on `lu.t()`, whose triangles swap — so no routine below it takes a flag.

**Pivots are 0-based `usize`,** as sblas's `iamax` answers them: `ipiv[j]` is the row that row `j` was
interchanged with at step `j`, so `ipiv[j] >= j`. LAPACK's `ipiv` is the same plus one.

### Errors are a `Result`, not an `info`

LAPACK's `info` carries two different things in one integer: a negative value names an argument that
was illegal, and a positive one reports something about the data (a zero pivot, a minor that is not
positive definite, an iteration that did not converge). slapack separates them.

- **What is wrong with the data is an `Err`.** `getrf` answers `Err(Error.Singular(j))` for the first
  `j` with `U(j, j)` exactly zero (`info = j + 1`), and `potrf` answers `Err(Error.NotPositiveDefinite(j))`
  for the first `j` whose leading minor of order `j + 1` is not positive definite. An `info` is easy to ignore — every caller of
  `dgetrf` has to remember to test it — while an ignored `Result` is warned about, and `?` passes it
  on. As in LAPACK the factorisation still finishes: the factors and pivots are written, so a caller
  that wants the determinant (zero) or the rank-deficient factors still has them.
- **What is wrong with the call traps,** as it does in sblas: a pivot array shorter than `min(m, n)`, a
  `B` whose rows are not the order of `A`, a pivot naming a row past the matrix, buffers shorter than
  the `_work_len` says. These are bugs in the caller, not conditions to handle, and reference LAPACK's
  own `xerbla` stops the program for them too.

`Error` is an enum so that the families still to come add their own variants (`NotPositiveDefinite`
for `potrf`, `NoConvergence` for `syev` and `gesvd`) rather than a second error type per routine.

**The QR routines answer nothing.** Every matrix has a QR factorisation, so `dgeqrf`'s `info` is only
ever about the arguments: here those trap — a `tau` shorter than `min(m, n)`, a Q with more columns
than rows or more reflectors than columns, reflectors not as long as C's side, a short workspace —
and `geqrf`, `orgqr` and `ormqr` return `unit`. A zero or dependent column gives a zero on R's
diagonal, which is the caller's to look at, and a NaN spreads through R as it does in LAPACK.

## The LU factorisation

`getrf` is LAPACK's right-looking blocked algorithm with a panel of `lu_block = 64` columns (LAPACK's
default `nb` for `getrf`). A matrix whose shorter side is within one panel is factored by `getf2`
alone. Otherwise, for each panel `j..<j + jb`:

1. `getf2` factors the panel `A(j:m, j:j+jb)` — pivot search by `iamax`, interchange by `swap`, the
   column below the pivot scaled by `scal`, the rest updated by `ger` — writing panel-relative pivots,
   which are then made absolute;
2. `laswp` applies the panel's interchanges to the columns on its left and on its right;
3. `trsm` (left, lower, unit diagonal) forms the panel's block row of U, `U₁₂ = L₁₁⁻¹ A₁₂`;
4. one `gemm` updates the trailing matrix, `A₂₂ -= L₂₁ U₁₂`.

The `trsm` and `gemm` read one block of `a` while writing another: two views of a single slice in one
call, which sblas's own recursive `trsm` already does and sysl permits.

**A pivot too small to invert is divided by, not multiplied by its reciprocal.** Where `|pivot|` is
below the smallest normal number (sblas's `T.safe_min()`, LAPACK's `sfmin`), `1 / pivot` would
overflow, so the column is divided element by element, exactly as `dgetf2` does.

`getrs` applies the interchanges to B and solves with L then U (`NoTrans`), or with Uᵀ then Lᵀ and the
interchanges undone in reverse (`Trans`), each a `trsm` over the factors in place. Like LAPACK's it
does not check U's diagonal: the factors of a singular matrix divide by zero.

## The Cholesky factorisation

`potrf` is LAPACK's left-looking blocked algorithm with a block of `chol_block = 64` columns (LAPACK's
default `nb` for `potrf`); an order within one block is factored by `potf2` alone. It is written for
the lower triangle, and `Upper` runs it on `a.t()`. For each block `j..<j + jb`:

1. the unblocked kernel brings the diagonal block up to date with the `j` columns left of it and
   factors it, `L₁₁ L₁₁ᵀ = A₁₁ − L₁₀ L₁₀ᵀ`;
2. one `gemm` updates the rows below, `A₂₁ −= L₂₀ L₁₀ᵀ`;
3. `trsm` (right, with `L₁₁ᵀ`, an upper view) finishes them, `L₂₁ = A₂₁ L₁₁⁻ᵀ`.

**The diagonal block's update is the kernel's, not a `syrk`.** LAPACK updates `A₁₁` with `dsyrk` and
then calls `dpotf2`; sblas has no `syrk`, and a `gemm` there would write the triangle `potrf` may not
touch. So the unblocked kernel is written left-looking over the whole row of the block: column `c`
subtracts its row's dot product with itself from the diagonal and a `gemv` of the rows below against
it, both reaching back past the block to column 0. With no columns to the left that is exactly
`dpotf2`; inside `potrf` it is `dsyrk` and `dpotf2` fused, at the same flop count (`jb² j` per block,
small beside the `gemm`). A `syrk` in sblas would let the step be split as LAPACK splits it.

**A diagonal that is not positive stops the factorisation.** As in `dpotf2`, a diagonal that comes to
zero, a negative value or a NaN is written back in place and `NotPositiveDefinite(j)` answered; the
columns before `j` hold the factor of the leading minor, which is positive definite, and nothing
right of `j` is touched. Unlike `getrf` there is nothing useful to finish.

`potrs` solves with `L` then `Lᵀ`, two `trsm`s over the factor (`Upper`'s `U` read as `Uᵀ`, through
`t()`). Like LAPACK's it does not check the diagonal.

## The QR factorisation

`geqrf` is LAPACK's blocked Householder algorithm with a block of `qr_block = 32` reflectors (LAPACK's
default `nb` for `geqrf`, `orgqr` and `ormqr`); a matrix whose shorter side is within one block is
factored by `geqr2` alone. LAPACK also keeps a crossover (`nx = 128`) below which it stays unblocked
for speed; slapack blocks from the first column past a block, so that orders 33 to 64 already exercise
the blocked path. For each block `i..<i + ib`:

1. `geqr2` factors the panel `A(i:m, i:i+ib)`: for each column, `larfg` makes the reflector that
   zeroes it below the diagonal (`nrm2`, `scal`), and `larf` applies it to the columns right of it in
   the panel — `w = Cᵀ v` by `gemv`, `C −= τ v wᵀ` by `ger`;
2. `larft` gathers the panel's reflectors into one, H(i) … H(i+ib−1) = I − V T Vᵀ with T upper
   triangular, a `gemv` and a triangular product per column;
3. `larfb` applies its transpose to the trailing columns: W = Cᵀ V T, then C −= V Wᵀ — two `gemm`s
   over the rows below the block, and triangular products with V's unit upper block and with T.

**A reflector's leading one is implicit everywhere.** LAPACK's `dlarf` reads v from A with its first
element temporarily set to one (`dgeqr2` saves `A(i, i)`, writes 1, calls `dlarf`, and restores it).
slapack's `larf` takes the tail of v and treats the one itself, so A is never written as scratch, and
`ormqr` can take its reflectors as a read-only `Matrix`, as it is given them. `larft` and `larfb`
likewise never read the diagonal of the reflectors' block, which holds R.

**sblas has no `trmm`**, which `dlarfb` calls three times. The products are `k` x `k` triangles
(`k ≤ 32`) against a `cols` x `k` W, so slapack's private `trmm_right` forms each column of W A with
one `gemv` over the columns of W it combines, walking away from them so that each is still unwritten
when read. That is `cols k²` flops beside the `gemm`s' `rows cols k`: small, and the step a `trmm`
in sblas would take over.

**One side, as one triangle in Cholesky.** Every reflection is applied from the left. C Q is (Qᵀ Cᵀ)ᵀ,
so `ormqr(Side.Right, trans, …, c)` is the left-hand product on `c.t()` with the transpose flipped;
`Side` stays in the signature because LAPACK's callers think in it.

**k is `tau.len`.** LAPACK's `dorgqr` and `dormqr` take `k` beside `tau`; here the slice carries it, so
`orgqr(q, tau[0..<k], work)` forms Q from the first `k` reflectors. `orgqr`'s `q` is `m` x `n` with
`k ≤ n ≤ m`: `n = k` is the thin Q of a tall A, `n = m` the whole of it. `ormqr`'s `a` may be wider
than `k` — the whole of a wide `geqrf`'s A — since only the first `k` columns below the diagonal are
read. `orgqr` mirrors `dorgqr`: the columns past `k` start as the identity's, then each block, last
first, applies its reflector to the columns right of it and forms its own with `org2r`.

`ormqr` applies the blocks in the order the product needs — Qᵀ C from the first block, Q C from the
last — each through `larft` and `larfb` with `trans`; within one block's width it is `larf` per
reflector, as `dorm2r`.

## Oracles

`sysl test .` checks the routines against their definitions: systems factored and solved by hand,
exactly; LAPACK's own measures on random matrices, ‖PA − LU‖ / (n ‖A‖ ε) and ‖op(A) X − B‖ / (n ‖A‖ ‖X‖
ε) in the 1-norm under LAPACK's test threshold of 30, at every order 1 to 64, past the panel width
(65, 97, 128, 129, 200) and tall and wide (150 x 70, 70 x 150, 200 x 65, 130 x 3, 300 x 100 and every
shape up to 12 x 12); singular matrices whose zero pivot lies in the first panel, a later one, and the
first of two; views of blocks and of a transpose; and a trap for every call that is wrong.

Cholesky is checked the same way: hand-factored systems from either triangle, exactly, with the other
triangle holding a sentinel that must survive; random symmetric positive definite matrices (Bᵀ B +
n I) of every order 1 to 64 and past the block (65, 100, 128, 129, 193, 200), from each triangle, under
‖A − L Lᵀ‖ / (n ‖A‖ ε) < 30 and the solve residual < 30, with the other triangle and the padding rows
NaN so that a read or a write of either shows; a zeroed diagonal inside the first block, at its last
column, at the next block's first, later, at the very last and at column 0, each named exactly and
leaving the columns before it bit for bit as the full factorisation has them; semidefinite, negative
and NaN diagonals; the two triangles' factors against each other; views; and the traps.

QR is checked by LAPACK's `dqrt01`–`dqrt03` measures: ‖A − Q R‖ / (max(m, n) ‖A‖ ε), ‖I − Qᵀ Q‖ /
(m ε) for the whole Q and, for a tall A, the thin one, and each of `ormqr`'s four products against
the whole Q formed explicitly, ‖op(Q) C − ormqr‖ / (max(m, p) ‖C‖ ε) — all under 30, at every square
order 1 to 64, past the block (65, 100, 128, 129, 193, 200), every shape up to 12 x 12, and tall and
wide (150 x 70, 200 x 65, 130 x 3, 300 x 100, 97 x 33, 64 x 33 and their transposes). Every workspace
is filled with NaN and every matrix sits in NaN padding, `tau` with a spare NaN element, and the
reflectors handed to `orgqr` have NaN where R and the columns past `k` were, so a read of anything a
routine should not read shows. Exactly: a 2 x 2 worked by hand (R, τ, Q and both sides of `ormqr`);
`larfg` on a zero tail and by hand; an upper triangular A, whose τ are all zero and which comes back
bit for bit; a zero matrix, whose Q is the identity; zero columns at the first, inside a block, at
and past a block edge, and last, each with τ zero and R's column zero; rank-deficient products B C;
columns small enough that `larfg` scales them and big enough to overflow a sum of squares; a NaN; a
block of a larger matrix, bit for bit with the same matrix alone and nothing around it touched; a
row-major view; `ormqr` reading the whole of a wide A; and a trap for every call that is wrong.

`oracle/`, a project of its own so that the package links nothing, factors the same random matrices
with the platform's `dgetrf` and `sgetrf` through the [lapack](https://github.com/sysl-lang/lapack)
binding (Accelerate on macOS) and requires the same pivots, the same first zero pivot, and every factor
within 64 max(m, n) ε of the platform's. It does the same for `dpotrf` and `spotrf` from each
triangle — the same failing column, and every element of the factor within 64 n ε — and checks that
slapack's left the other triangle as it was. The platform's is not held to that: **Accelerate's
`potrf` writes the triangle LAPACK documents as not referenced**, from order 33 on (observed on macOS,
sysl 0.8.3), so a caller of the platform's that keeps data there loses it.

For QR the oracle runs the platform's `dgeqrf` and `sgeqrf` (the binding's `qr_inplace`) on random
matrices of every square order 1 to 64, past the block (65, 100, 128, 129, 193, 200) and tall and wide
(150 x 70, 200 x 65, 130 x 3, 7 x 3 and their transposes), and on zero columns either side of a block
edge. R is compared row by row up to the sign of its diagonal — the sign `larfg` gives beta is a
convention, and a reflector that takes a column to −beta is as good — and where the signs agree τ and
the reflector are compared too, where they differ τ + τ' = 2; all within 64 max(m, n) ε. Accelerate
and slapack choose the same sign on every row of every one of these matrices (sysl 0.8.3, macOS), so
the oracle requires it: no row is flipped, and R, τ and the reflectors agree outright. A `larfg` with
the sign reversed turns all three oracle tests red.

## Gaps found so far

- **S1: a generic float cannot be interpolated without a second bound** (sysl 0.8.2; a design
  question, not a bug). `Float` requires `Zero + One + Eq + Ord + Neg + Add + Sub + Mul + Div` but not
  `Display`, so a `[T: Float]` (or sblas's `[T: Real]`) body cannot write `s"$x"`:

  ```sysl
  import sysl.math.Float

  show[T: Float](x: T) -> string = s"x = $x"

  print(show(1.5))
  ```

  *"'str' needs 'T: Display'"*. Every type that implements `Float` has a `Display` (reference
  `traits.md`: "every built-in type implements both through an `impl`"), so the bound is spelled out
  at every generic numeric function that wants to report a value — here, a residual in a test's
  failure message. The tests print `${f64(ratio)}` instead, converting through the parameter; the
  question for the user is whether `Float` should require `Display` (and `Debug`).

## Notes

- **The tests' 1-norm answers NaN for a NaN.** `norm1` once kept the largest column sum by `sum >
  most`, which is false for a NaN, so a NaN column read as zero and a ratio built on it passed. The
  QR mutation pass found it: an `orgqr` that left NaN in Q passed every residual test. It now answers
  the NaN, as `dlange` does, and every family's ratios are held to it.

- **An integer reaches a generic float as `T(k)`** (`reference/generics.md § Converting through a
  parameter`). sblas's `tests_trsm.sysl` builds `k` from ones, its comment saying `Real` has no
  conversion from an integer; `T(k)` is that conversion, and the tests here use it.
