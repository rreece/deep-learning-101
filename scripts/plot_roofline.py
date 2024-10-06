#!/usr/bin/env python3
"""
plot_roofline.py
"""


import argparse
import matplotlib
import matplotlib.pyplot as plt


matplotlib.use("Agg")

A5000_BANDWIDTH = 768.
A5000_PEAK = 27.8

# https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/quadro-product-literature/proviz-print-nvidia-rtx-a6000-datasheet-us-nvidia-1454980-r9-web%20(1).pdf
# https://arxiv.org/abs/2402.16363
A6000_BANDWIDTH = 768.
#A6000_PEAK = 38.7  # fp32
A6000_PEAK = 155  # fp16? From: 2402.16363, sec 3.1.1.
#A6000_PEAK = 310  # int8?

N150_BANDWIDTH = 288.
N150_PEAK = 74.

DEFAULT_BANDWIDTH = A6000_BANDWIDTH
DEFAULT_PEAK = A6000_PEAK


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b", "--bandwidth", default=DEFAULT_BANDWIDTH, type=float,
        help="Memory bandwidth in GB/s.",
    )
    parser.add_argument(
        "-p", "--peak", default=DEFAULT_PEAK, type=float,
        help="Peak compute performance in TFLOPs/s.",
    )
#    parser.add_argument(
#        "-d", "--device", default="Nvidia_A5000", type=str,
#        help="Device key for hardware params",
#    )
    return parser.parse_args()


def plot_roofline(bandwidth, peak, mem_unit=2, log=False):
    # mem_unit is bytes per mop
    # (TFLOPs/s) * (bytes/mop) * (GB/s) = TFLOPs/GMOPs = 1e3 FLOPs/MOPs
#    I_c = peak * mem_unit / bandwidth  # TFLOPs/GMOPs
#    I_c = I_c * 1e3  # FLOPs/MOPs
    I_c = peak / bandwidth  # TFLOPs/GB
    I_c = I_c * 1e3  # FLOPs/B

    if log:
        # for log-log scale
        min_x = 1e-1
        min_y = 1e-1
        max_x = 10.0 * I_c
        max_y = 1.5 * peak
    else:
        # for linear scale
        min_x = 0.0
        min_y = 0.0
        max_x = 2.0 * I_c
        max_y = 1.2 * peak

    x_memory_bound = [0.0, I_c]
    y_memory_bound = [0.0, peak]
    x_compute_bound = [I_c, max_x]
    y_compute_bound = [peak, peak]

    # User should also provide the model and measurements of the rates 
    # TODO: parametrize
    N_params = 8e9
#    R_decode = 20.   # tokens/s
    R_decode = 40.   # tokens/s
    R_prefill = 1.0 # TODO: FIXME overwritten below

    # Decode calculation
    batch_size = 1
    C_f = 2 * N_params # forward pass compute per token generated, FLOPs/token
    I_decode = 2.0  # FLOPs/MOPs
    I_decode = I_decode / mem_unit # FLOPs/B
    P_decode = C_f * batch_size * R_decode  # FLOPs/s
    P_decode = P_decode * 1e-12  # TFLOPs
    print("")
    print("R_decode = %.1f tokens/s" % R_decode)
    print("I_decode = %.1f FLOPs/B" % I_decode)
    print("P_decode = %.1f TFLOPs/s" % P_decode)

    # Prefill calculation
    batch_size = 256  # TODO: this is a placeholder
    efficiency = 0.95  # TODO: this is a placeholder
    R_peak = peak * 1e12 / (C_f*batch_size)
    R_prefill = R_peak * efficiency  # tokens/s  # TODO: this is a placeholder
    I_prefill = I_decode * batch_size  # TODO: this is a placeholder
    P_prefill = C_f * batch_size * R_prefill  # FLOPs/s
    P_prefill = P_prefill * 1e-12  # TFLOPs
    print("")
    print("R_peak = %.1f tokens/s" % R_peak)
    print("R_prefill = %.1f tokens/s" % R_prefill)
    print("I_prefill = %.1f FLOPs/B" % I_prefill)
    print("P_prefill = %.1f TFLOPs/s" % P_prefill)
    print("")

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(
        x_memory_bound,
        y_memory_bound,
        color="black",
        linewidth=2,
        linestyle="dashed",
    )
    ax.plot(
        x_compute_bound,
        y_compute_bound,
        color="black",
        linewidth=2,
        linestyle="dashed",
    )

    ax.plot(
        I_decode,
        P_decode,
        color="red",
        marker="o",
        markersize=8,
        linestyle="None",
        label="Decode (I=%.3g, P=%.3g)" % (I_decode, P_decode),
    )
    ax.plot(
        I_prefill,
        P_prefill,
        color="blue",
        marker="o",
        markersize=8,
        linestyle="None",
        label="Prefill (I=%.3g, P=%.3g)" % (I_prefill, P_prefill),
    )

    ax.set_xlabel("Computational Intensity  [FLOPs/Byte]", fontsize=16)
    ax.set_ylabel("Performance [TFLOPs/s]", fontsize=16)
    if log:
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.set_xlim([min_x, max_x])
    ax.set_ylim([min_y, max_y])

    plt.setp(ax.spines.values(), linewidth=1.6)
    plt.grid()
    ax.tick_params(axis="both", which="major", length=10, width=1, labelsize=14)
    ax.tick_params(axis="both", which="minor", length=5, width=1, labelsize=10)
    ax.xaxis.set_tick_params(direction="in", which="both")
    ax.yaxis.set_tick_params(direction="in", which="both")

    ax.legend(fontsize=16, loc="upper left")
    plt.tight_layout()
    plt.savefig("roofline.png")


def main():
    args = parse_args()
    bandwidth = args.bandwidth
    peak = args.peak
    plot_roofline(bandwidth=bandwidth, peak=peak)


if __name__ == "__main__":
    main()

