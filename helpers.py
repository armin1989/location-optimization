import scipy.io
import numpy as np
import matplotlib.pyplot as plt


def get_pdf(spots, params, dx=1, dy=1):
    """
    Return quantized 2D pdf with resolution dx and y, according to spots and params. Quantization is performed on a
    uniform grid


    :param spots: Ns x 2 array of locations of traffic hot-spots
    :param params: dictionary containing distribution parameters
    :param dx: quantization resolution in horizontal direction
    :param dy: quantization resolution in vertical direction
    :return:
    """
    sigma = params["sigma"]
    out_prob = params["out_prob"]
    d_max = params["d_max"]

    def density(x, y):
        """
        Return mixtures of Gaussian density at points x and y
        :param x:
        :param y:
        :return: f(x, y)
        """
        n_s = spots.shape[0]
        return sum([np.exp(-(x - x_s) ** 2 / (2 * sigma ** 2)) * np.exp(-(y - y_s) ** 2 / (2 * sigma ** 2))
                    for x_s, y_s in spots]) / n_s / (2 * np.pi * sigma ** 2)

    x_range = np.arange(-d_max / 2, d_max / 2 + dx, dx)
    y_range = np.arange(-d_max / 2, d_max / 2 + dy, dy)
    f0 = 1 / sum([sum([density(x, y) for x in x_range])for y in y_range])
    pdf = np.array([[f0 * density(x, y) * (1 - out_prob) + out_prob / (d_max ** 2) for x in x_range]for y in y_range],
                   dtype=np.float64)
    X = [[x for x in x_range] for _ in y_range]
    Y = [[y for _ in x_range] for y in y_range]
    return pdf, X, Y


def kernel_density_estimate(samples, kernel, bw, w, h, dx=1, dy=1):
    """
        Return 2D kernel-density-estimated pdf with resolution dx and dy, according to spots and params.

        :param samples: n_s x 2 array of sample locations of UTs
        :param kernel: callable, kernel function to be used for density estimation
        :param bw: band-width of kernel
        :param w: width of cell
        :param h: height of cell
        :param dx: quantization resolution in horizontal direction
        :param dy: quantization resolution in vertical direction
        :return: pdf, X, Y where X and Y are grid of points used for pcolormesh
        """

    def density(x, y):
        """
        Return mixtures of Gaussian density at points x and y
        :param x:
        :param y:
        :return: f(x, y)
        """
        n_s = np.size(samples, 0)
        return sum([kernel((x - x_s) / bw) * kernel((y - y_s) / bw) for x_s, y_s in samples]) / n_s / h ** 2

    x_range = np.arange(-w / 2, w / 2 + dx, dx)
    y_range = np.arange(-h / 2, h / 2 + dy, dy)
    pdf = np.array([[density(x, y) for x in x_range] for y in y_range], dtype=np.float64)
    # normalizing pdf
    f0 = 1 / sum(pdf)
    pdf = f0 * pdf
    X = [[x for x in x_range] for _ in y_range]
    Y = [[y for _ in x_range] for y in y_range]
    return pdf, X, Y


def gauss_kernel(x):
    """
    Return value of Gaussian kernel.

    :param x: input
    :return: e ^ (-x ^ 2 / 2) / (2 * pi)
    """
    return np.exp(- x ** 2) / np.sqrt(2 * np.pi)


def epan_kernel(x):
    """
    Return Epanechnikov kernel.

    :param x: input
    :return:  3 * (1 - x ^ 2) / 4 * 1(|x| <=1 )
    """
    return 3 * (1 - x ** 2) / 4 if abs(x) <= 1 else 0


if __name__ == "__main__":
    # testing area
    input_file = scipy.io.loadmat('ut_positions.mat')
    ut_positions = input_file["ut_positions"]
    # print(spots)
    params = {"sigma": 50, "out_prob": 0.1, "d_max": 1e3}
    cut_data = 5000
    dist_hat, X, Y = kernel_density_estimate(ut_positions[:cut_data, :], gauss_kernel, 10, 1e3, 1e3, 20, 20)

    plt.pcolormesh(X, Y, dist_hat)
    plt.title('Estimated pdf')
    plt.show()