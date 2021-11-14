import matplotlib.pyplot as plt
import os

def plot_trace(sample, i, ylabel, file_name):
    fig, ax = plt.subplots(figsize = (8,6))
    x = [1, 10, 100, 1000, 10000, 100000]
    ax.plot(x, sample)
    ax.set_title("Posterior expectation for {i}.daphne".format(i = i))
    ax.set(ylabel = ylabel, xlabel = "Num_iterations")
    fname = os.path.join("figs", file_name)
    plt.savefig(fname)
    print("\nFigure saved as '%s'" % fname)


def plot_histogram(sample, title, xlabel, file_name):
    fig, ax = plt.subplots(figsize = (8, 6))
    ax.hist(sample)
    ax.set_title(title)
    ax.set(ylabel = "Frequency", xlabel = xlabel)
    fname = os.path.join("figs", file_name)
    plt.savefig(fname)
    print("\nFigure saved as '%s'" % fname)