import LinearAlgebra

export DiffMatrix, full

# I do not think someone will ask for derivatives of order > 30, but
# in that case, just open a pull request to increase this value
const MAX_WIDTH = 31

"""
    DiffMatrix(coeffs::A) where {T<:Real, A<:DenseArray{T, 2}}

Construct a differentiation matrix object of order `order` from a set of
arbitrarily spaced grid points `xs` using finite differences with a stencil
of width `width`.
"""
struct DiffMatrix{T<:Real, WIDTH}
    coeffs::Matrix{T} # finite difference weights
      buff::Vector{T} # small buffer for the matvec code
    function DiffMatrix(xs::AbstractVector{T}, width::Int, order::Int) where {T}
        # checks
        3 ≤ width ≤ MAX_WIDTH || throw(ArgumentError("width must be between 3 and $MAX_WIDTH"))
        width % 2 == 1 || throw(ArgumentError("width must be odd"))

        # Compute the coefficients of the differentiation matrix. Coefficients
        # are organised in row major format, i.e, the first column contains the
        # weights of the finite difference approximation of the derivative at
        # the first grid point.
        coeffs = get_coeffs(xs, width, order)

        return new{T, width}(coeffs, zeros(T, width))
    end
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

# generate code for the allowed cases of stencil WIDTH
for WIDTH in 3:2:MAX_WIDTH
    @eval begin
        # demo code for one dimensional data
        function LinearAlgebra.mul!(y::DenseArray{S, 1},
                                    A::DiffMatrix{T, $WIDTH},
                                    x::DenseArray{S, 1}) where {T, S}
            # size of vector
            N = length(y)

            @inbounds @simd for i = 1:N
                # index of the first element of the stencil
                left = clamp(i - $(WIDTH>>1), 1, N - $WIDTH + 1)

                # expand expressions
                y[i] = A.coeffs[1, i]*x[left]
                Base.Cartesian.@nexprs $(WIDTH-1) p -> begin
                    y[i] += A.coeffs[1 + p, i] * x[left + p]
                end
            end

            return y
        end
    end

    @eval begin
        # differentiate x along direction 3
        function LinearAlgebra.mul!(y::DenseArray{S, 3},
                                    A::DiffMatrix{T, $WIDTH},
                                    x::DenseArray{S, 3}, ::Val{3}) where {T, S}
            # check size
            size(x, 3) == size(y, 3) == size(A.coeffs, 2) ||
                throw(ArgumentError("inconsistent inputs size"))

            # size of coeffs
            N1, N2, N3 = size(y)

            # local register
            vals = A.buff

            @inbounds for i = 1:N3
                # store stencil weights into some local register
                vals .= A.coeffs[:, i]

                # index of the first element of the stencil
                left = clamp(i - $(WIDTH>>1), 1, N3 - $WIDTH + 1)

                # initialise
                for k = 1:N2
                    @simd for j = 1:N1
                        y[j, k, i] = vals[1]*x[j, k, left]
                    end
                end

                # this will have to call y[j, k, i] in memory multiple times
                # but at least we do not jump along the third dimension of x
                Base.Cartesian.@nexprs $(WIDTH-1) p -> begin
                    for k = 1:N2
                        @simd for j = 1:N1
                            y[j, k, i] += vals[p+1]*x[j, k, left + p]
                        end
                    end
                end
            end

            return y
        end
    end
end