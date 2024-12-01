import os
import re
import time
from typing import List, Dict

import numpy as np


class BasePlotter:
    start_time = time.time_ns()

    def __init__(self, ax_, regexp, x_last_=90):
        self.ax = ax_
        self.x_last = x_last_
        self.regexp = regexp
        self.measurements = {}
        self.file_path = ''
        self.ts = np.array([])

        self.text_residual = b''
        self.line_counter = 0

    def process_lines(self, log_strings, decimate=1):
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
            idxs = np.where(x >= x_first)
            y["x"] = x[idxs]
            label = y["label"] if "label" in y else ""
            fmt = y["fmt"] if "fmt" in y else "-"
            ax.plot(x[idxs], y["y"][idxs], fmt, label=label)
        ax.legend()
        ax.set_xlim([x_first, x_last])


class JittPlotter(BasePlotter):
    def __init__(self, ax_):
        regexp = re.compile(
            '^[a-z],(?P<ts>[\d.]*),(?P<stream_ts>[\d.]*),(?P<delta_ms>[\d.]*),(?P<jitter_max>[\d.]*),(?P<jitter_min>[\d.]*)$')
        super().__init__(ax_, regexp)

    def __call__(self, lines):
        # ts, stream ts, delta_ms, jitter_max, jitter_min
        self.process_lines(lines)

        if self.measurements == {}:
            return

        self.plot(self.ts, [
            {"y": self.measurements["delta_ms"], "label": "delta"},
            {"y": self.measurements["jitter_max"] / 1e6, "label": "Jitter max"},
            {"y": self.measurements["jitter_min"] / 1e6, "label": "Jitter min"}])


class LatencyPlotter(BasePlotter):
    def __init__(self, ax_):
        regexp = re.compile(
            '^[a-z],(?P<ts>[\d.]*),(?P<niq>[\d.]*),(?P<target>[\d.]*)$')
        super().__init__(ax_, regexp)

    def __call__(self, lines):
        # ts, stream ts, delta_ms, jitter_max, jitter_min
        self.process_lines(lines)

        if self.measurements == {}:
            return

        self.plot(self.ts, [
            {"y": self.measurements["niq"] / 44100. * 1e3, "label": "niq ms"},
            {"y": self.measurements["target"] / 44100. * 1e3, "label": "Target ms"}])


class FreqEstimatorPlotter(BasePlotter):
    def __init__(self, ax_):
        regexp = re.compile(
            '^[a-z],(?P<ts>[\d.]*),(?P<filtered>[\d.]*),(?P<target>[\d.]*),(?P<p>[-e\d.]*),(?P<i>[-e\d.]*)$',
            re.MULTILINE)
        self.accum_ax = ax_.twinx()
        super().__init__(ax_, regexp)

    def __call__(self, lines):
        # ts, stream ts, delta_ms, jitter_max, jitter_min
        self.process_lines(lines)

        if self.measurements == {}:
            return

        self.plot(self.ts, [{"y": self.measurements["filtered"] / 44100 * 1e3, "label": "Filtered ms"},
                            {"y": self.measurements["target"] / 44100 * 1e3, "label": "Target ms"}], ax=self.ax,
                  clear=True)
        self.plot(self.ts, [{"y": self.measurements["p"], "label": "P", "fmt": "k-"},
                            {"y": self.measurements["i"], "label": "I", "fmt": "r-"}], ax=self.accum_ax)

class RTTPlotter(BasePlotter):
    def __init__(self, ax_):
        regexp = re.compile(
            '^[a-z],(?P<ts>[\d.]*),(?P<rtt>[\d.]*),(?P<ts_offset>-?[\d.]*)$')
        self.ts_ax = ax_.twinx()
        super().__init__(ax_, regexp)

    def __call__(self, lines):
        # ts, stream ts, delta_ms, jitter_max, jitter_min
        self.process_lines(lines)

        if self.measurements == {}:
            return

        self.plot(self.ts, [
            {"y": self.measurements["rtt"] / 1e9 * 1e3, "label": "rtt ms", "fmt": "k-"}
            ], ax=self.ax)
        self.plot(self.ts, [
            {"y": self.measurements["ts_offset"] / 1e9 * 1e3, "label": "ts offset ms", "fmt": "r-"}], ax=self.ts_ax)
