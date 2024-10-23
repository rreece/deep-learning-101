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

# https://docs.tenstorrent.com/aibs/wormhole/specifications.html
N150_BANDWIDTH = 288.
N150_PEAK = 74.

DEFAULT_BANDWIDTH = A6000_BANDWIDTH
DEFAULT_PEAK = A6000_PEAK
#DEFAULT_BANDWIDTH = N150_BANDWIDTH
#DEFAULT_PEAK = N150_PEAK


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
        max_y = 1.3 * peak

    x_memory_bound = [0.0, I_c]
    y_memory_bound = [0.0, peak]
    x_compute_bound = [I_c, max_x]
    y_compute_bound = [peak, peak]

    # User should also provide the model and measurements of the rates 
    # TODO: parametrize
    N_params = 8e9
    C_f = 2 * N_params # forward pass compute per token generated, FLOPs/token
    D_f = N_params * mem_unit # model MOPs

    # A6000 measurements
    R_decode = 42. # TODO: put real measurement here  # tokens/s
    R_prefill = 9000. # TODO: put real measurement here  # tokens/s
    # N150 measurements
#    R_decode = 17. # TODO: put real measurement here  # tokens/s
#    R_prefill = 4500. # TODO: put real measurement here  # tokens/s

    # Prefill calculation
    batch_size_prefill = 1 # TODO: placeholder
    context_length = 256 # TODO: placeholder
    R_prefill_peak = peak * 1e12 / C_f
    R_prefill_total = R_prefill * batch_size_prefill  # tokens/s
    efficiency_prefill = R_prefill_total / R_prefill_peak
    # FIXME: This is the main TODO: How to express I_prefill?
    I_prefill = 2.0 * batch_size_prefill * context_length  # TODO: ??? # 2.0 FLOPs/MOPs(fp16) = 1.0 FLOPs/B
    I_prefill = I_prefill / mem_unit # FLOPs/B
    P_prefill = C_f * R_prefill_total # FLOPs/s
    P_prefill = P_prefill * 1e-12  # TFLOPs
    print("Prefill")
    print("batch_size = %i" % batch_size_prefill)
    print("context_length = %i" % context_length)
    print("R = %.1f tokens/s/u" % R_prefill)
    print("R_total = %.1f tokens/s" % R_prefill_total)
    print("R_peak = %.1f tokens/s" % R_prefill_peak)
    print("efficiency = %.2f" % efficiency_prefill)
    print("I = %.1f FLOPs/B" % I_prefill)
    print("P = %.1f TFLOPs/s" % P_prefill)
    print("")

    # Decode calculation
    batch_size_decode = 32  # TODO: placeholder
    R_decode_peak = bandwidth * 1e9 * batch_size_decode / D_f
    R_decode_total = R_decode * batch_size_decode # tokens/s
    efficiency_decode = R_decode_total / R_decode_peak
    I_decode = 2.0 * batch_size_decode  # FLOPs/MOPs
    I_decode = I_decode / mem_unit # FLOPs/B
    P_decode = C_f * R_decode_total  # FLOPs/s
    P_decode = P_decode * 1e-12  # TFLOPs
    print("Decode")
    print("batch_size = %i" % batch_size_decode)
    print("R = %.1f tokens/s/u" % R_decode)
    print("R_total = %.1f tokens/s" % R_decode_total)
    print("R_peak = %.1f tokens/s" % R_decode_peak)
    print("efficiency = %.2f" % efficiency_decode)
    print("I = %.1f FLOPs/B" % I_decode)
    print("P = %.1f TFLOPs/s" % P_decode)
    print("")

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(
        x_memory_bound,
        y_memory_bound,
        color="black",
        linewidth=2,
        linestyle="dashed",
        label="Memory Bound (%.3g GB/s)" % (bandwidth),
    )
    ax.plot(
        x_compute_bound,
        y_compute_bound,
        color="black",
        linewidth=2,
        linestyle="dashed",
        label="Compute Bound (%.3g TFLOPs/s)" % (peak),
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

