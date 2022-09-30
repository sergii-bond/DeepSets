# =====================================================================
# Generate a dataset
# =====================================================================

## 

using Pkg
packages = ["Plots", "LinearAlgebra", "HDF5"]

for package in packages
    if Pkg.dir(package) == nothing 
        Pkg.add(package)
    end
end

## 

using Plots
using LinearAlgebra
using HDF5
using Printf

##

function rotation_matrix_2d(a)
    [cos(a) -sin(a)
     sin(a)  cos(a)]
end

function task1(A, N, angle_rad; viz=false)
    """
    Generate samples from a 2D Gaussian normal distribution
    Args:
        A (2x2 matrix) Covariance matrix before rotation
        N (int) Number of samples.
        angle_rad (float) Standard normal rotation angle in radians.
    Returns:
        X ([2, N] array) Samples.
        Y (float) Marginal entropy of the first variable.
    """
    # Generate a rotation matrix
    R = rotation_matrix_2d(angle_rad)

    # Eigenvalue decomposition of a covariance matrix:
    # M = R.T * E * R = R.T * A * A * R 
    A_nl = R * A
    M = A_nl * A_nl'
    # print(M, '\n')

    # We can verify the eigenvalues of M
    # print("Eigenvalues: ", LinearAlgebra.eigvals(M), "\n")
    ev = LinearAlgebra.eigvecs(M)

    # Generate N samples
    # We first generate a sample from a standard normal.
    # Then we scale it in each direction by the standard deviation,
    # which is a square root of eigenvalues we generated.
    # Finally we rotate it.
    X = A_nl * randn(Float32, (2, N))

    if viz
        # Let's visualize the generated points
        plot(X[1, :], X[2, :], seriestype = :scatter, aspect_ratio=:equal, 
            markersize=1)
        # Add eigenvectors to the plot
        plot!([0, ev[1, 1]], [0, ev[2, 1]], linewidth=4)
        plot!([0, ev[1, 2]], [0, ev[2, 2]], linewidth=4)
    end

    # The meaning of (x.T * M.inv * x) in Gaussian pdf 
    # is a distance between x and the mean (we took it zero).
    # It equals the Euclidean distance after we change coordinate system
    # to the one with eigenvectors as axes (u = R.T * x),
    # scale points to have variance 1 (w = A.inv * u), 
    # And finally take an L2 norm (w.T * w).


    # Marginal entropy of the first variable
    s = M[1, 1]
    Y = 1/2 * log(2*pi*exp(1)*s^2);
    # print("Variance of the first variable: ", s, "\n")
    # print("Rotation angle: ", angle_rad, "\n")
    # print("Marginal entropy of the first variable: ", Y, '\n')

    return X, Y
end

function save_dataset(A, L, N, filepath)
    X = zeros(N * L, 2)
    Y = zeros(L, 1)
    X_parameter = range(0, pi, L)

    for (i, angle_rad) in enumerate(X_parameter)
        x0, y0 = task1(A, N, angle_rad, viz=false)
        Y[i] = y0
        X[N * (i - 1) + 1:N * i, :] = x0'
    end

    h5open(filepath, "w") do fid
        fid["X"] = X
        fid["Y"] = Y
        fid["N"] = N
        fid["L"] = L
        fid["X_parameter"] = Array(X_parameter)
    end
    print(size(Y), "\n")
    plot(X_parameter, Y)
    gui()
end


outdir = "data/task1"

if !isdir(outdir)
    mkpath(outdir)
end

L = 512  # Number of sets in validation/test
N = 512  # Number of points per set

# Generate a square root of the diagonal eigenvector matrix
# A = diagm(rand(Float32, 2))
# print("== Generated eigenvalues matrix: ", A .^ 2, "\n")

# Generate a random covariance matrix
A = rand(Float32, 2, 2)

# Generate test dataset
save_dataset(A, L, N, joinpath(outdir, "test.mat"))

# Generate validation dataset
save_dataset(A, L, N, joinpath(outdir, "val.mat"))

# Generate truth for plotting
save_dataset(A, 2^14, 1, joinpath(outdir, "truth.mat"))

# Create dataset of different size:
for logL in 7:12  # number of train distributions
    L = 2^logL;
    save_dataset(A, L, N, joinpath(outdir, @sprintf("data_%d.mat", logL)))
end

# gui()
