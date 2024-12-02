import sys

import matplotlib.pyplot as plt
import numpy as np

# grep r, /tmp/recv.csv | cut -d',' -f2-  > /tmp/pr.csv
if __name__ == '__main__':
    csv = np.loadtxt(sys.argv[1], delimiter=',', dtype=np.float128)
    t1 = csv[:, 3]
    t2 = csv[:, 4]
    t3 = csv[:, 5]
    t4 = csv[:, 6]
    t21 = t2 - t1
    t43 = t4 - t3

    rtt = (t4 - t1) - (t2 - t3)
    ts_offset = (t2 - t1) + (t3 - t4)

    t21_idx = np.argsort(t21)
    t21 = t21[t21_idx]
    # t1 = csv[t21_idx, 3]
    # t2 = csv[t21_idx, 4]

    t43_idx = np.argsort(t43)
    t43 = t43[t43_idx]
    # t3 = csv[t43_idx, 5]
    # t4 = csv[t43_idx, 6]

    plt.figure()
    plt.subplot(211)
    # plt.plot(t21*1e-6, label='t2 - t1')
    # plt.plot(t43*1e-6, label='t4 - t3')
    plt.plot(np.abs(t2-t1)*1e-6, label='|t2 - t1|')
    plt.plot(np.abs(t4-t3)*1e-6, label='|t4 - t3|')
    plt.grid(True)
    plt.legend()
    plt.subplot(212)
    plt.plot(rtt*1e-9, label='rtt')
    plt.plot(ts_offset*1e-9, label='ts_offset')
    plt.legend()
    plt.grid(True)
    plt.show()
