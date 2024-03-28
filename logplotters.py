import os
import re
import time
from typing import List, Dict

import numpy as np


class BasePlotter:
    start_time = time.time_ns()

    def __init__(self, filename_, ax_, regexp, x_last_=90):
        self.filename = filename_
        self.ax = ax_
        self.x_last = x_last_
        self.regexp = regexp
        self.measurements = {}
        self.file_path = ''
        self.ts = np.array([])

        self.fd = os.open(filename_, os.O_RDONLY | os.O_NONBLOCK)
        os.lseek(self.fd, 0, os.SEEK_END)
        self.text_residual = b''
        self.line_counter = 0

    def read_lines(self):
        lines = (self.text_residual + os.read(self.fd, 40960)).split(b'\n')
        new_residual = lines.pop()
        if new_residual.endswith(b'\n'):
            lines.append(new_residual)
            new_residual = b''
        self.text_residual = new_residual
        return [line.decode('utf-8') for line in lines]

    def fileno(self):
        return self.fd

    def process_lines(self, decimate=1):
        log_strings = self.read_lines()
        if len(log_strings) == 0:
            return
        d = {}
        for s in log_strings:
            self.line_counter += 1
            if self.line_counter % decimate != 0:
                continue
            m = self.regexp.match(s)
            if m is None:
                continue
            m = m.groupdict()
            d = {k: [v] if k not in d else d[k] + [v] for k, v in m.items()}

        for k, v in d.items():
            v = np.array([float(x) for x in v])
            if decimate > 1:
                v = v[::decimate]
            if k == "ts":
                self.ts = np.append(self.ts, (v - self.start_time) / 1e9)
            else:
                if k in self.measurements:
                    self.measurements[k] = np.append(self.measurements[k], v)
                else:
                    self.measurements[k] = v

        if len(self.ts) == 0:
            return

        x_last = self.ts[-1]
        x_first = max(self.ts[0], x_last - self.x_last)
        idx = np.where((self.ts >= x_first) & (self.ts <= x_last))
        self.ts = self.ts[idx]
        for k, v in self.measurements.items():
            self.measurements[k] = v[idx]

    def plot(self, x: np.array, args: List[Dict[str, np.array or str]], ax=None, clear=True):
        if x.shape[0] < 1:
            return

        x_last = x[-1]
        x_first = max(x[0], x_last - self.x_last)
        if ax is None:
            ax = self.ax
        if clear:
            ax.clear()
        for y in args:
            idxs = np.where((x >= x_first) & (x <= x_last))
            y["x"] = x[idxs]
            label = y["label"] if "label" in y else ""
            fmt = y["fmt"] if "fmt" in y else "-"
            ax.plot(x[idxs], y["y"][idxs], fmt, label=label)
        ax.legend()
        ax.set_xlim([x_first, x_last])


class JittPlotter(BasePlotter):
    def __init__(self, ax_):
        regexp = re.compile(
            '^(?P<ts>\d*),\s(?P<stream_ts>\d*),\s(?P<delta_ms>[\d.]*),\s(?P<jitter_max>[\d.]*),\s(?P<jitter_min>[\d.]*)$')
        self.file_path = "/tmp/jitt.log"
        super().__init__(self.file_path, ax_, regexp)

    def __call__(self):
        # ts, stream ts, delta_ms, jitter_max, jitter_min
        self.process_lines()

        if self.measurements == {}:
            return

        self.plot(self.ts, [
            {"y": self.measurements["delta_ms"], "label": "delta"},
            {"y": self.measurements["jitter_max"] / 1e6, "label": "Jitter max"},
            {"y": self.measurements["jitter_min"] / 1e6, "label": "Jitter min"}])


class LatencyPlotter(BasePlotter):
    def __init__(self, ax_):
        regexp = re.compile(
            '^(?P<ts>\d*),\s(?P<niq>\d*),\s(?P<target>[\d.]*)$')
        self.file_path = "/tmp/tuner.log"
        super().__init__(self.file_path, ax_, regexp)

    def __call__(self):
        # ts, stream ts, delta_ms, jitter_max, jitter_min
        self.process_lines(decimate=8)

        if self.measurements == {}:
            return

        self.plot(self.ts, [
            {"y": self.measurements["niq"] / 44100. * 1e3, "label": "niq ms"},
            {"y": self.measurements["target"] / 44100. * 1e3, "label": "Target ms"}])


class FreqEstimatorPlotter(BasePlotter):
    def __init__(self, ax_):
        regexp = re.compile(
            '^(?P<ts>\d*),\s*(?P<filtered>[\d.]*),\s*(?P<target>[\d.]*),\s*(?P<p>[-e\d.]*),\s*(?P<i>[-e\d.]*)$',
            re.MULTILINE)
        self.file_path = "/tmp/fe.log"
        self.accum_ax = ax_.twinx()
        super().__init__(self.file_path, ax_, regexp)

    def __call__(self):
        # ts, stream ts, delta_ms, jitter_max, jitter_min
        self.process_lines()

        if self.measurements == {}:
            return

        self.plot(self.ts, [{"y": self.measurements["filtered"] / 44100 * 1e3, "label": "Filtered ms"},
                            {"y": self.measurements["target"] / 44100 * 1e3, "label": "Target ms"}], ax=self.ax,
                  clear=True)
        self.plot(self.ts, [{"y": self.measurements["p"], "label": "P", "fmt": "k-"},
                            {"y": self.measurements["i"], "label": "I", "fmt": "r-"}], ax=self.accum_ax)
