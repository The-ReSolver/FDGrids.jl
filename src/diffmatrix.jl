struct DiffMatrix{T, WIDTH} <: AbstractMatrix{T}
    coeffs::Matrix{T} # finite difference weights
      buff::Vector{T} # small buffer for the matvec code
    function DiffMatrix(xs::AbstractVector, width::Int, order::Int, ::Type{T}=Float64) where {T}
        # checks
        3 ≤ width ≤ MAX_WIDTH || throw(ArgumentError("width must be between 3 and $MAX_WIDTH"))
        width % 2 == 1 || throw(ArgumentError("width must be odd"))
        width ≤ length(xs) || throw(ArgumentError("width must not be greater than number of grid points "))

        # Compute the coefficients of the differentiation matrix. Coefficients
        # are organised in row major format, i.e, the first column contains the
        # weights of the finite difference approximation of the derivative at
        # the first grid point.
        coeffs = get_coeffs(xs, width, order)

        return new{T, width}(T.(coeffs), zeros(T, width))
    end
end

Base.size(d::DiffMatrix) = (size(d.coeffs, 2), size(d.coeffs, 2))
Base.IndexStyle(d::DiffMatrix) = IndexCartesian()

function Base.getindex(d::DiffMatrix{T, WIDTH}, i::Int, j::Int) where {T, WIDTH}
    # global to local indices mapping
    offset = i ≤              WIDTH >> 1 ?          WIDTH>>1 - i + 1 :
             i > size(d, 1) - WIDTH >> 1 ? size(d, 1) - WIDTH>>1 - i : 0
    m, n = WIDTH>>1 + j - i + 1 - offset, i
    
    # return
    return checkbounds(Bool, d.coeffs, m, n) ? d.coeffs[m, n] : zero(T)
end

function Base.setindex!(d::DiffMatrix{T, WIDTH}, v, i::Int, j::Int) where {T, WIDTH}
    # global to local indices mapping
    offset = i ≤              WIDTH >> 1 ?          WIDTH>>1 - i + 1 :
             i > size(d, 1) - WIDTH >> 1 ? size(d, 1) - WIDTH>>1 - i : 0
    m, n = WIDTH>>1 + j - i + 1 - offset, i
    
    # return
    return checkbounds(Bool, d.coeffs, m, n) ? (d.coeffs[m, n] = T(v)) : T(v)
end

function Base.similar(d::DiffMatrix{T, WIDTH}, ::Type{S}=T, _size::Tuple{Vararg{Int64,2}}=size(d)) where {T, S, WIDTH}
    return DiffMatrix(zeros(Float64, _size[1]), WIDTH, 1, S)
end

function Base.copy(d::DiffMatrix{T, WIDTH}) where {T, WIDTH}
    d_ = similar(d)
    d_.coeffs .= d.coeffs
    return d_
end

function full(A::DiffMatrix{T, WIDTH}) where {T, WIDTH}
    N = size(A.coeffs, 2)
    out = zeros(T, N, N)
    @simd for i = 1:N
        # index of the first element of the stencil
        left = clamp(i - (WIDTH>>1), 1, N -WIDTH + 1)

        # expand expressions
        for p = 1:WIDTH
            out[i, left+p-1] = A.coeffs[p, i]
        end
    end
    return out
end

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# broadcasting style
struct DiffMatrixStyle{T, WIDTH} <: Broadcast.BroadcastStyle end
Base.BroadcastStyle(::Type{<:DiffMatrix{T, WIDTH}}) where {T, WIDTH} = DiffMatrixStyle{T, WIDTH}()

# allows broadcasting with numbers
Base.BroadcastStyle( ::Base.Broadcast.DefaultArrayStyle{0}, 
                    s::DiffMatrixStyle{T, WIDTH}) where {T, WIDTH} = s

# allows broadcasting with vectors
Base.BroadcastStyle( ::Base.Broadcast.DefaultArrayStyle{1}, 
                    s::DiffMatrixStyle{T, WIDTH}) where {T, WIDTH} = s

# allow broadcasting with diagonal matrices (but only diagonal * )
Base.BroadcastStyle( ::LinearAlgebra.StructuredMatrixStyle{<:LinearAlgebra.Diagonal},
                    s::DiffMatrixStyle{T, WIDTH}) where {T, WIDTH} = s

# operations with diff matrices of different width but same type
Base.BroadcastStyle(::DiffMatrixStyle{T1, WIDTH1}, ::DiffMatrixStyle{T2, WIDTH2}) where {T1, T2, WIDTH1, WIDTH2} = 
    DiffMatrixStyle{promote_type(T1, T2), max(WIDTH1, WIDTH2)}()

# use broadcasting
function Base.similar(bc::Base.Broadcast.Broadcasted{<:DiffMatrixStyle{T, WIDTH}}, ::Type{S}) where {T, WIDTH, S}
    s = axes(bc)[1][end]
    DiffMatrix(zeros(Float64, s), WIDTH, 1, S)
end