# date created : July 12 : 2016
# date_modified : July 24 2018
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import helpers as h
import pickle
import time


def lloyds(c_0, pdf, w, h, epsilon=1, itr_max=500):
    """
    Quantize given pdf using Lloyds algorithm, used for antenna location optimization

    :param c_0: Initial vector of quantized pdf
    :param pdf: [n_y, n_x] array of pdf values
    :param w: width of cell
    :param h: height of cell
    :param epsilon: stopping threshold for algorithm
    :param itr_max: maximum number of iterations allowed (in case algorithm diverges)
    :return: centers: [N, 2] centers of Voronoi tesselations according to pdf
    """
    def find_voronoi_idx(x, y, points):
        """
        Return index of closest point in points to (x, y). This is the index of Voronoi cell that (x, y) belongs to.

        :param x:
        :param y:
        :param points: [n_p, 1] array of centroids of Voronoi tesselations
        :return: index of Voronoi cell
        """
        return np.argmin([np.linalg.norm([x - x_p, y - y_p]) for x_p, y_p in points])

    n_y, n_x = np.size(pdf, 0), np.size(pdf, 1)
    N = np.size(p0, 0)
    dx = w / (n_x - 1)
    dy = h / (n_y - 1)
    x_range = np.arange(-w / 2 , w / 2 + dx, dx)
    y_range = np.arange(-h / 2, h / 2 + dy, dy)

    c_prev = c_0
    diff = np.inf  # maximum change in centers of tessels in each iteration
    itr = 1
    while itr < itr_max and diff > epsilon:

        cell_mean = np.zeros([N, 1])
        cell_mean_x = np.zeros([N, 1])
        cell_mean_y = np.zeros([N, 1])

        # finding average of points in each Voronoi cell, numerically
        for i in range(len(y_range)):
            for j in range(len(x_range)):

                ut_idx = find_voronoi_idx(x_range[j], y_range[i], c_prev)
                cell_mean[ut_idx] += pdf[i, j] * dx * dy
                cell_mean_x[ut_idx] += x_range[j] * pdf[i, j] * dx * dy
                cell_mean_y[ut_idx] += y_range[i] * pdf[i, j] * dx * dy

        # updating locations of centeroids according to average values
        c_current = np.concatenate((np.divide(cell_mean_x, cell_mean), np.divide(cell_mean_y, cell_mean)), axis=1)

        # finding difference between current and previous iteration
        diff = max([np.linalg.norm([c_prev[i, 0] - c_current[i, 0], c_prev[i, 1] - c_current[i, 1]]) for i in range(N)])
        print("itr = %d, diff = %.2f" % (itr, diff))

        itr += 1
        c_prev = c_current

    return c_current


def hca(c_0, pdf, w, h, alpha=3.8, r0=0.47, epsilon=1, itr_max=500):
    """
    Return result of harmonic means clustering algorithm using alpha and r0 for distance calculation:

    d = (1 / r) ** alpha

    :param c_0: Initial vector of quantized pdf
    :param pdf: [n_y, n_x] array of pdf values
    :param w: width of cell
    :param h: height of cell
    :param alpha : path-loss exponent
    :param r0 : reference distance
    :param epsilon: stopping threshold for algorithm
    :param itr_max: maximum number of iterations allowed (in case algorithm diverges)
    :return: centers: [N, 2] centers of Voronoi tesselations according to pdf
    """
    def get_common_part(x, y, points):
        """
        Return vector of d_n * (sum_n(d_n)) ^ (-2) / r_n ^ 2 where

        d_n = (1 / r_n) ^ alpha
        r_n = distance to antenna n

        :param x:
        :param y:
        :param points: [n_p, 1] array of locations of antenna
        :return: index of Voronoi cell
        """
        r_n = np.array([max(r0, np.linalg.norm([x - x_p, y - y_p])) for x_p, y_p in points])
        d_n = r_n ** -alpha
        return np.reshape(sum(d_n) ** -2 * d_n / r_n ** 2, [r_n.shape[0], 1])

    n_y, n_x = np.size(pdf, 0), np.size(pdf, 1)
    N = np.size(p0, 0)
    dx = w / (n_x - 1)
    dy = h / (n_y - 1)
    x_range = np.arange(-w / 2, w / 2 + dx, dx)
    y_range = np.arange(-h / 2, h / 2 + dy, dy)

    c_prev = c_0
    diff = np.inf  # maximum change in centers of tessels in each iteration
    itr = 1
    while itr < itr_max and diff > epsilon:

        cell_mean = np.zeros([N, 1])
        cell_mean_x = np.zeros([N, 1])
        cell_mean_y = np.zeros([N, 1])

        # finding average of points in each Voronoi cell, numerically
        for i in range(len(y_range)):
            for j in range(len(x_range)):

                common_part = get_common_part(x_range[j], y_range[i], c_prev)
                cell_mean += common_part * pdf[i, j] * dx * dy
                cell_mean_x += x_range[j] * common_part * pdf[i, j] * dx * dy
                cell_mean_y += y_range[i] * common_part * pdf[i, j] * dx * dy

        # updating locations of centeroids according to average values
        c_current = np.concatenate((cell_mean_x / cell_mean, cell_mean_y / cell_mean), axis=1)

        # finding difference between current and previous iteration
        diff = max([np.linalg.norm([c_prev[i, 0] - c_current[i, 0], c_prev[i, 1] - c_current[i, 1]]) for i in range(N)])
        print("itr = %d, diff = %.2f" % (itr, diff))

        itr += 1
        c_prev = c_current

    return c_current


if __name__ == "__main__":
    # main playground
    start_time = time.time()
    params = dict()
    params["N"] = 16          # number of antennas
    params["sigma"] = 50      # std of Gaussian pdf in UT pdf
    params["out_prob"] = 0.1  # probability of UTs being outside
    params["d_max"] = 1000    # side-length of cell
    d_max = 1e3
    N = 16
    dx = 25
    dy = 25
    cut_data = 1000
    # set to false if you want to use saved data
    find_pdf = True
    do_hca = True
    do_lloyds = True
    do_save = False

    # loading data samples
    input_file = scipy.io.loadmat("ut_positions.mat")
    ut_samples = input_file["ut_positions"][:cut_data, :]

    # estimating pdf using data samples
    if find_pdf:
        pdf, X, Y = h.kernel_density_estimate(ut_samples, h.gauss_kernel, 20, 1e3, 1e3, dx, dy)
        if do_save:
            pdf_file = open("pdf_file.p", "wb")
            pickle.dump(pdf, pdf_file)
            pdf_file.close()
    else:
        pdf_file = open("pdf_file.p", "rb")
        pdf = pickle.load(pdf_file)
        x_range = np.arange(-d_max / 2, d_max / 2 + dx, dx)
        y_range = np.arange(-d_max / 2, d_max / 2 + dy, dy)
        X = [[x for x in x_range] for _ in y_range]
        Y = [[y for _ in x_range] for y in y_range]
    # initial points for lloyds or hca algorithms
    p0 = np.random.rand(N, 2) * d_max - d_max / 2

    # calling Lloyds
    if do_lloyds:
        print("Starting Lloyds Algorithm")
        p_lloyds = lloyds(p0, pdf, d_max, d_max)
        if do_save:
            pos_file = open("lloyds_file.p", "wb")
            pickle.dump(p_lloyds, pos_file)
            pos_file.close()
    else:
        pos_file = open("lloyds_file.p", "rb")
        p_lloyds = pickle.load(pos_file)
        pos_file.close()

    # calling hca
    if do_lloyds:
        print("Starting HCA")
        p_hca = hca(p0, pdf, d_max, d_max)
        if do_save:
            pos_file = open("hca_file.p", "wb")
            pickle.dump(p_hca, pos_file)
            pos_file.close()
    else:
        pos_file = open("hca_file.p", "rb")
        p_hca = pickle.load(pos_file)
        pos_file.close()
    end_time = time.time()
    print("Overal exeuction time is ", end_time - start_time)

    # plotting
    plt.subplot(2, 2, 1)
    plt.scatter(ut_samples[:500, 0], ut_samples[:500, 1])
    plt.title('UT sample positions')
    plt.subplot(2, 2, 2)
    plt.pcolormesh(X, Y, pdf)
    plt.title('Estimated distribution')
    plt.subplot(2, 2, 3)
    plt.scatter(p_lloyds[:, 0], p_lloyds[:, 1])
    plt.title('Locations of antennas using Lloyds algorithm')
    plt.subplot(2, 2, 4)
    plt.title('Locations of antennas using HCA algorithm')
    plt.scatter(p_hca[:, 0], p_hca[:, 1])
    plt.show()

