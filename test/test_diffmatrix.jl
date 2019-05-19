@testset "test diffmatrix                        " begin
    # the order of accuracy is the width minus one
    for (width, v1_max, v2_center_max, v2_bndr_max) in zip((3,    5,     7), 
                                                           (1.21, 3.22,  9.85), 
                                                           (4.66, 3.61,  3.5),
                                                           (0.33, 0.098, 0.078))
        # number of points on a regular grid
        for M in (30, 40, 50)

            # make grid from -1 to 1 using α = 0.5
            g = FDGrid(M, width, -1, 1, 0.5)

            # arrange to 3D array
            fs          = zeros(2, 2, M)
            fs[1, 1, :] = exp.(1.0.*gridpoints(g))
            fs[1, 2, :] = exp.(1.1.*gridpoints(g))
            fs[2, 1, :] = exp.(1.2.*gridpoints(g))
            fs[2, 2, :] = exp.(1.3.*gridpoints(g))

            # exact first derivative
            d1fs_EX   = copy(fs)
            d1fs_EX[1, 1, :] .*= 1.0
            d1fs_EX[1, 2, :] .*= 1.1
            d1fs_EX[2, 1, :] .*= 1.2
            d1fs_EX[2, 2, :] .*= 1.3

            # exact second derivative
            d2fs_EX   = copy(fs)
            d2fs_EX[1, 1, :] .*= 1.0^2
            d2fs_EX[1, 2, :] .*= 1.1^2
            d2fs_EX[2, 1, :] .*= 1.2^2
            d2fs_EX[2, 2, :] .*= 1.3^2
            
            # compute finite difference approximation along the third direction
            d1fs_FD      = similar(fs)
            d2fs_FD      = similar(fs)
            mul!(d1fs_FD, diffmat(g, 1), fs, Val(3))
            mul!(d2fs_FD, diffmat(g, 2), fs, Val(3))

            # the relative error should scale like M^{-o} where o = width-1
            v1 = maximum(abs.(d1fs_EX - d1fs_FD))/maximum(abs.(d1fs_EX))*M^(width-1)
            @test v1 < v1_max

            # for the second derivative we have the same order in the domain center
            i = M >> 1
            v2 = abs(d2fs_EX[i] - d2fs_FD[i])/abs(d2fs_EX[i])*M^(width-1)
            @test v2 < v2_center_max

            # and one order less at the boundary
            i = 1
            v3 = abs(d2fs_EX[i] - d2fs_FD[i])/abs(d2fs_EX[i])*M^(width-2)
            @test v3 < v2_bndr_max
        end
    end
end