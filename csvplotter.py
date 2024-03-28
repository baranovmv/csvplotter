import cProfile
import select

import matplotlib.pyplot as plt

from logplotters import JittPlotter, LatencyPlotter, FreqEstimatorPlotter

if __name__ == '__main__':
    fig = plt.figure()
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)

    # files = [FreqEstimatorPlotter(ax3)]
    files = [FreqEstimatorPlotter(ax3), LatencyPlotter(ax2), JittPlotter(ax1)]
    # files = {}  # fd -> file_name
    # for file_name, processor in file_names:
    #     fd = os.open(file_name, os.O_RDONLY | os.O_NONBLOCK)
    #     os.lseek(fd, 0, os.SEEK_END)
    #     residual = b''
    #     files[fd] = (file_name, residual, processor)

    while True:
        to_read = select.select(files, [], [])[0]
        for fd in to_read:
            fd()
        plt.pause(0.1)

    plt.show()
    for fd in files.keys():
        os.close(fd)
