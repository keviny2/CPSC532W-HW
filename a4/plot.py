from matplotlib import pyplot as plt
import os
from graph_based_sampling import evaluate, deterministic_eval
import seaborn as sns
import numpy as np


def plot_trace(sample, i, ylabel, file_name):
    fig, ax = plt.subplots(figsize = (8,6))
    ax.plot(sample)
    max = np.max(sample)
    ax.set_title("BBVI loss for {i}.daphne\n Max: {max}".format(i = i, max = max))
    ax.set(ylabel = ylabel, xlabel = "Num_iterations")
    fname = os.path.join("figs", file_name)
    plt.savefig(fname)
    print("\nFigure saved as '%s'" % fname)

def plot_joint_loglik(logPs, title, type, file_name):
    fig, ax = plt.subplots(figsize = (8, 6))
    ax.plot(logPs)
    ax.set_title(title)
    ax.set(ylabel = "Log-Likelihood", xlabel = "Num_iterations")
    fname = os.path.join("figs", type, file_name)
    plt.savefig(fname)
    print("\nFigure saved as '%s'" % fname)

def plot_histogram(sample, title, xlabel, type, file_name):
    fig, ax = plt.subplots(figsize = (8, 6))
    ax.hist(sample)
    ax.set_title(title)
    ax.set(ylabel = "Weights", xlabel = xlabel)
    fname = os.path.join("figs", type, file_name)
    plt.savefig(fname)
    print("\nFigure saved as '%s'" % fname)

def plot_histogram_bbvi(sample, weights, title, xlabel, file_name):
    fig, ax = plt.subplots(figsize = (8,6))
    ax.hist(sample.numpy(), weights = weights.numpy())
    ax.set_title(title)
    ax.set(ylabel="Frequency", xlabel=xlabel)
    fname = os.path.join("figs", file_name)
    plt.savefig(fname)
    print("\nFigure saved as '%s'" % fname)

def joint_log_likelihood(vertices, links, variables_dict):
    logP = 0
    for vertex in vertices:
        logP += deterministic_eval(evaluate(links[vertex][1], variables_dict)).log_prob(variables_dict[vertex])

    return logP


def plot_heatmap(sample, title, file_name):
    fig, ax = plt.subplots(figsize = (8, 6))
    sns.heatmap(sample, annot = True)
    plt.title(title)
    fname = os.path.join("figs", file_name)
    plt.savefig(fname)
    print("\nFigure saved as '%s'" % fname)