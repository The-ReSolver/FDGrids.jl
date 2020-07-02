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