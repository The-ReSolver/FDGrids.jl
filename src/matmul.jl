# I do not think someone will ask for stencils this large, but
# in that case, just open a pull request to increase this value
const MAX_WIDTH = 31

# generate code for the allowed cases of stencil WIDTH
for WIDTH in 3:2:MAX_WIDTH
    @eval begin
        function LinearAlgebra.mul!(y::DenseArray{S, 1},
                                    A::DiffMatrix{T, $WIDTH},
                                    x::DenseArray{S, 1}) where {T, S}
            # size of vector
            N = length(y)

            # check size
            length(y) == length(x) == size(A, 2) ||
                throw(DimensionMismatch())

            # top
            @inbounds begin
                for i = 1:$(WIDTH>>1)
                    s = A.coeffs[1, i]*x[1]
                    Base.Cartesian.@nexprs $(WIDTH-1) p -> begin
                        s += A.coeffs[1 + p, i] * x[1 + p]
                    end
                    y[i] = s
                end

                # body
                for i = 1+$(WIDTH>>1):N-$(WIDTH>>1)
                    # index of the first element of the stencil
                    left = i - $(WIDTH>>1)

                    # expand expressions
                    s = A.coeffs[1, i]*x[left]
                    Base.Cartesian.@nexprs $(WIDTH-1) p -> begin
                        s += A.coeffs[1 + p, i] * x[left + p]
                    end
                    y[i] = s
                end

                # tail
                for i = N-$(WIDTH>>1)+1:N
                    s = A.coeffs[1, i]*x[(N - $WIDTH + 1)]
                    Base.Cartesian.@nexprs $(WIDTH-1) p -> begin
                        s += A.coeffs[1 + p, i] * x[(N - $WIDTH + 1) + p]
                    end
                    y[i] = s
                end
            end
            return y
        end
    end

    @eval begin
        # Find derivative of `x` at point `i` 
        function LinearAlgebra.mul!(A::DiffMatrix{T, $WIDTH}, x::AbstractVector, i::Int) where {T}
            # size of vector
            N = length(x)

            # check size
            size(A, 2) == N || throw(DimensionMismatch())

            # index of the first element of the stencil
            left = clamp(i - $WIDTH>>1, 1, N - $WIDTH + 1)

            # expand expressions
            val = A.coeffs[1, i]*x[left]
            Base.Cartesian.@nexprs $(WIDTH-1) p -> begin
                val += A.coeffs[1 + p, i] * x[left + p]
            end
            
            return val
        end
    end

    @eval begin
        # differentiate x along direction 1
        function LinearAlgebra.mul!(y::DenseArray{S, 3},
                                    A::DiffMatrix{T, $WIDTH},
                                    x::DenseArray{S, 3}) where {T, S}
            # check size
            size(x, 1) == size(y, 1) == size(A.coeffs, 2) ||
                throw(ArgumentError("inconsistent inputs size"))

            # size of coeffs
            N1, N2, N3 = size(y)

            @inbounds for j = 1:N2, k = 1:N3
                for i = 1:$(WIDTH>>1)
                    s = A.coeffs[1, i]*x[1, j, k]
                    Base.Cartesian.@nexprs $(WIDTH-1) p -> begin
                        s += A.coeffs[1 + p, i] * x[1 + p, j, k]
                    end
                    y[i, j, k] = s
                end

                # body
                for i = 1+$(WIDTH>>1):N1-$(WIDTH>>1)
                    # index of the first element of the stencil
                    left = i - $(WIDTH>>1)

                    # expand expressions
                    s = A.coeffs[1, i]*x[left]
                    Base.Cartesian.@nexprs $(WIDTH-1) p -> begin
                        s += A.coeffs[1 + p, i] * x[left + p, j, k]
                    end
                    y[i, j, k] = s
                end

                # tail
                for i = N1-$(WIDTH>>1)+1:N1
                    s = A.coeffs[1, i]*x[(N1 - $WIDTH + 1), j, k]
                    Base.Cartesian.@nexprs $(WIDTH-1) p -> begin
                        s += A.coeffs[1 + p, i] * x[(N1 - $WIDTH + 1) + p]
                    end
                    y[i, j, k] = s
                end
            end
            return y
        end
    end
end
    # @eval begin
    #     # differentiate x along direction 3
    #     function LinearAlgebra.mul!(y::DenseArray{S, 3},
    #                                 A::DiffMatrix{T, $WIDTH},
    #                                 x::DenseArray{S, 3}, ::Val{3}) where {T, S}
    #         # check size
    #         size(x, 3) == size(y, 3) == size(A.coeffs, 2) ||
    #             throw(ArgumentError("inconsistent inputs size"))

    #         # size of coeffs
    #         N1, N2, N3 = size(y)

    #         # local register
    #         vals = A.buff

    #         @inbounds for i = 1:N3
    #             # store stencil weights into some local register
    #             vals .= A.coeffs[:, i]

    #             # index of the first element of the stencil
    #             left = clamp(i - $(WIDTH>>1), 1, N3 - $WIDTH + 1)

    #             # initialise
    #             for k = 1:N2
    #                 for j = 1:N1
    #                     y[j, k, i] = vals[1]*x[j, k, left]
    #                 end
    #             end

    #             # this will have to call y[j, k, i] in memory multiple times
    #             # but at least we do not jump along the third dimension of x
    #             Base.Cartesian.@nexprs $(WIDTH-1) p -> begin
    #                 for k = 1:N2
    #                     for j = 1:N1
    #                         y[j, k, i] += vals[p+1]*x[j, k, left + p]
    #                     end
    #                 end
    #             end
    #         end

    #         return y
    #     end
    # end

# disallow this, for the moment
# Base.:*(A::DiffMatrix{T}, x::AbstractVector{S}) where {T, S} = 
#     LinearAlgebra.mul!(similar(x), A, x)

# # action of A on a dense matrix
# Base.:*(A::DiffMatrix{T}, x::DenseMatrix{S}) where {T, S} = 
#     LinearAlgebra.mul!(similar(x), A, x)
