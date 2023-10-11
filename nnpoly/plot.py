import matplotlib.pyplot as plt
import numpy as np
from oned_tests import potential_double_well
from functools import partial


if __name__ == '__main__':

    filename = "double_well.npz"
    filename2 = "double_well_hermite.npz"

    npz = np.load(filename)
    x = npz['arr_0']
    w = npz['arr_1']
    enr = npz['arr_2']
    loss = npz['arr_3']
    
    npz = np.load(filename2)
    nmax_h = npz['arr_0']
    enr_h = npz['arr_1']

    # poten = partial(potential_double_well, k0=0)
    # x0 = np.linspace(-5, 5, 100)
    # # plt.plot(x0, poten(x), 'k-', label='Potential')
    # plt.plot(x, w/np.max(w), 'r-', label='Learned $W(x)$')
    # plt.plot(x0, np.exp(-x0**2), 'k--', label='$\\exp(-x^2)$')
    # # plt.ylim([-20, 150])
    # # plt.xlabel("$x$")
    # # plt.ylabel("Potential")
    # plt.legend()
    # plt.show()
    # exit()

    # plt.plot(nmax_h, [enr[-1,:] for _ in nmax_h], 'r-')
    # plt.plot([], [], 'r-', label="Learned polynomials")
    # plt.plot(nmax_h, enr_h, 'k-', mfc='none', linewidth=1)
    # plt.plot(nmax_h, enr_h, 'ko', mfc='none', markersize=3)
    # plt.plot([], [], 'ko', markersize=3, label="Hermite basis")
    # plt.xlim([28, 85])
    # plt.xlabel("$N_{\\rm max}$")
    # plt.ylabel("Energy")
    # plt.legend()
    # plt.show()
    # exit()

    plt.plot(np.arange(len(loss)), enr)
    plt.xlabel("Training epoch")
    plt.ylabel("Energy")
    # plt.legend()
    plt.show()
    exit()
