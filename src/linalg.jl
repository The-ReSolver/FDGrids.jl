struct DiffMatrixLU{T, WIDTH, F}
    factors::F
    ipiv::Vector{Int}
end

function LinearAlgebra.lu(D::DiffMatrix{T, WIDTH}) where {T, WIDTH}
    WD = WIDTH-1
    factors, ipiv = LinearAlgebra.LAPACK.gbtrf!(WD, WD, size(D, 2), to_banded_format(D))
    return DiffMatrixLU{T, WIDTH, typeof(factors)}(factors, ipiv)
end

function LinearAlgebra.ldiv!(lu::DiffMatrixLU{T, WIDTH}, x::AbstractVector{T}) where {T, WIDTH}
    WD = WIDTH-1
    return LinearAlgebra.LAPACK.gbtrs!('N', WD, WD, size(lu.factors, 2), lu.factors, lu.ipiv, x)
end


# for second order accurate discretizations, we migth use a tridiagonal solver
struct TriDiagDiffMatrixLU{T, F}
    args::F
end

function LinearAlgebra.lu(D::DiffMatrix{T, 3}) where {T}
    N = size(D, 1)
    du  = [D[i, i+1] for i ∈ 1:N-1] 
    d   = [D[i, i]   for i ∈ 1:N] 
    dl  = [D[i, i-1] for i ∈ 2:N] 
    args = LinearAlgebra.LAPACK.gttrf!(dl, d, du)
    return TriDiagDiffMatrixLU{T, typeof(args)}(args)
end

function LinearAlgebra.ldiv!(lu::TriDiagDiffMatrixLU{T}, x::AbstractVector{T}) where {T}
    return LinearAlgebra.LAPACK.gttrs!('N', lu.args[1], lu.args[2], lu.args[3], lu.args[4], lu.args[5], x)
end

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Transform to banded format suitable for the factorisation. This has 
# more rows than it should, and can be made a bit more efficient
function to_banded_format(D::DiffMatrix{T, WIDTH}) where {T, WIDTH}
    # number of rows
    n = size(D, 1)

    # allocate output
    out = zeros(T, 3*WIDTH-2, n)

    for _i = 1:(2*WIDTH-1)
        for _j = 1:n
            # indices in global coordinates
            i, j = _j + (_i - WIDTH), _j
            val = checkbounds(Bool, D, i, j) ? D[i, j] : zero(T)
            out[_i + WIDTH-1, _j] = val
        end
    end

    return out
end