import numpy as np
import matplotlib.pyplot as plt


def ax_plot(data):
    fig, ax = plt.subplots()

    convolve_size = 128
    convolve = np.ones(convolve_size) / convolve_size
    x = np.arange(len(data) - convolve_size + 1)

    # for i in range(3, 11):  # rho_powers
    # for i in range(11, 19):  # bandwidth
    # for i in range(19, 27):  # state_vehicle_number
    #     ax.plot(x, np.convolve(data[:, i], convolve, 'same'))
    #     # ax.plot(x, data[:, i])

    # r = np.tanh((r - 9000) * 0.00025)
    ax.plot(x, np.convolve(np.arctanh(data[:, 0])/ 0.00025 + 9000, convolve, 'valid'))  # reward
    # ax.plot(x, np.convolve(data[:, 0], convolve, 'valid'))  # reward
    # ax.plot(x, np.convolve(data[:, 1], convolve, 'valid'))  # uav_z_high
    ax.set_title('ax_plot_title with smooth %d' % convolve_size)
    plt.show()


def hist_plot(data, num_bins=9):
    fig, ax = plt.subplots()

    # data = np.random.randn(512)
    data = np.argmax(data[:, 2:11], axis=1)
    ax.hist(data, num_bins, density=1)
    ax.set_xlabel('x_label')
    ax.set_ylabel('y_label')
    ax.set_title('hist_plot_title')

    fig.tight_layout()
    plt.show()


def run():
    logger_path = 'V4_uav_info.npy'
    # logger_path = 'uav_info.npy'
    # logger_path = 'uav_info_r_m_z_power_bandwidth_vehicle.npy'
    logger = np.load(logger_path)  # ary.shape == (n*logger_gap, 27)
    print(logger.shape)

    '''
    logger:27 == 3 (reward, uav_pos_id, uav_z_high) + 8 vehicles_number + 8 rho_powers + 8 bandwidth
    
    0:1 reward
    1:2 uav_pos_id
    2:3 uav_z_high
    3:11 == 3 + 8 vehicle_number
    11:19 == 11 + 8 rho_powers
    19:27 == 19 + 8 bandwidths
    '''

    # ax_plot(logger[0:4096])
    hist_plot(logger[0:4096])


if __name__ == '__main__':
    # from uav_crossroad import run

    # from DDPG import run

    run()
