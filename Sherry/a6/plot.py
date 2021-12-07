import matplotlib.pyplot as plt
import os
import math
import numpy as np

def plot_trace(sample, title, ylabel, file_name):
    fig, ax = plt.subplots(figsize = (10, 8))
    xs = [1, 10, 100, 1000, 10000, 100000]
    ax.plot(xs, sample)
    ax.plot(xs, sample, "ro")
    for i in range(len(xs)):
        ax.annotate(sample[i], (xs[i], sample[i]))
    ax.set_title(title)
    ax.set(ylabel = ylabel, xlabel = "Num_iterations")
    fname = os.path.join("figs", file_name)
    plt.savefig(fname)
    print("\nFigure saved as '%s'" % fname)

def plot_histogram_evidence(sample, mean, evidence, title, xlabel, file_name):
    fig, ax = plt.subplots(figsize = (10, 8))
    ax.hist(sample, bins = 10)
    ax.set_title(title + "\n mean is {} \n evidence is {}".format(mean, evidence))
    ax.set(ylabel = "Frequency", xlabel = xlabel)
    fname = os.path.join("figs", file_name)
    plt.savefig(fname)
    print("\nFigure saved as '%s'" % fname)


def plot_histogram_variance_evidence(sample, mean, variance, evidence, title, xlabel, file_name):
    fig, ax = plt.subplots(figsize = (10, 8))
    ax.hist(sample, bins = 10)
    ax.set_title(title + "\n mean is {} \n variance is {} \n evidence is {}".format(mean, variance, evidence))
    ax.set(ylabel = "Frequency", xlabel = xlabel)
    fname = os.path.join("figs", file_name)
    plt.savefig(fname)
    print("\nFigure saved as '%s'" % fname)

def plot_histogram(sample, mean, title, xlabel, file_name):
    fig, ax = plt.subplots(figsize = (10, 8))
    ax.hist(sample, bins = 10)
    ax.set_title(title + "\n mean is {}".format(mean))
    ax.set(ylabel = "Frequency", xlabel = xlabel)
    fname = os.path.join("figs", file_name)
    plt.savefig(fname)
    print("\nFigure saved as '%s'" % fname)


def plot_histogram_variance(sample, mean, variance, title, xlabel, file_name):
    fig, ax = plt.subplots(figsize = (10, 8))
    ax.hist(sample, bins = 10)
    ax.set_title(title + "\n mean is {} \n variance is {}".format(mean, variance))
    ax.set(ylabel = "Frequency", xlabel = xlabel)
    fname = os.path.join("figs", file_name)
    plt.savefig(fname)
    print("\nFigure saved as '%s'" % fname)