import cProfile
import select
import sys
import os
import time

import matplotlib.pyplot as plt

from logplotters import JittPlotter, LatencyPlotter, FreqEstimatorPlotter, RTTPlotter

if __name__ == '__main__':
    fig = plt.figure()
    # ax1 = fig.add_subplot(311)
    # ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(111)

    fd = os.open(sys.argv[1], os.O_RDONLY | os.O_NONBLOCK)
    os.lseek(fd, 0, os.SEEK_END)
    text_residual = b''
    line_counter = 0

    # files = [FreqEstimatorPlotter(ax3)]
    plotters = {#'f': [FreqEstimatorPlotter(ax3), 0],
                # 't': [LatencyPlotter(ax2), 0],
                # 'm': [JittPlotter(ax1), 0],
                'r': [RTTPlotter(ax3), 0]}
    time_start = time.time_ns()
    time_print = time_start
    unrecognized_processor_cntr = 0
    lines_s = 0
    # files = {}  # fd -> file_name
    # for file_name, processor in file_names:
    #     fd = os.open(file_name, os.O_RDONLY | os.O_NONBLOCK)
    #     os.lseek(fd, 0, os.SEEK_END)
    #     residual = b''
    #     files[fd] = (file_name, residual, processor)

    while True:
        lines = (text_residual + os.read(fd, 40960)).split(b'\n')
        new_residual = lines.pop()
        if new_residual.endswith(b'\n'):
            lines.append(new_residual)
            new_residual = b''
        text_residual = new_residual
        for line in lines:
            lines_s += 1
            str_line = line.decode('utf-8')
            discriminator = str_line[0:1]
            if discriminator not in plotters:
                unrecognized_processor_cntr += 1
                continue
            else:
                cur_time = time.time_ns()
                plotters[discriminator][0]([str_line])
                plotters[discriminator][1] += time.time_ns() - cur_time

        plt.pause(0.1)
        cur_time = time.time_ns()
        if cur_time - time_print > 1e9:
            for k in plotters.keys():
                print(f"Processor {k}: {plotters[k][1] / (cur_time - time_start) * 100.:.2f}%")
            print(f"Unrecognized processor: {unrecognized_processor_cntr}")
            print(f"Lines per second: {lines_s}")
            print(f"Overtime: {(cur_time - time_print - 1e9) / 1e9}s")
            time_print = time.time_ns()
            lines_s = 0

    plt.show()
    for fd in plotters.keys():
        os.close(fd)
