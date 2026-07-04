"""Visualization helpers for EEG education notebooks."""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_histogram(values: np.ndarray, title: str, xlabel: str) -> plt.Figure:
    """Create a histogram for one numeric variable.

    Parameters
    ----------
    values:
        One-dimensional numeric array.
    title:
        Figure title.
    xlabel:
        Label for the x-axis.

    Returns
    -------
    plt.Figure
        Matplotlib figure object.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(np.asarray(values, dtype=float), bins=30, edgecolor="black")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    fig.tight_layout()
    return fig


def plot_boxplot(values: np.ndarray, title: str, ylabel: str) -> plt.Figure:
    """Create a boxplot for one numeric variable.

    Parameters
    ----------
    values:
        One-dimensional numeric array.
    title:
        Figure title.
    ylabel:
        Label for the y-axis.

    Returns
    -------
    plt.Figure
        Matplotlib figure object.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot(np.asarray(values, dtype=float), vert=True)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    return fig


def plot_line(time: np.ndarray, values: np.ndarray, title: str, ylabel: str) -> plt.Figure:
    """Create a line plot for a time series.

    Parameters
    ----------
    time:
        Time values for the x-axis.
    values:
        Signal values for the y-axis.
    title:
        Figure title.
    ylabel:
        Label for the y-axis.

    Returns
    -------
    plt.Figure
        Matplotlib figure object.
    """
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(time, values)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    return fig


def plot_scatter(x: np.ndarray, y: np.ndarray, title: str, xlabel: str, ylabel: str) -> plt.Figure:
    """Create a scatter plot for two variables.

    Parameters
    ----------
    x:
        Values for x-axis.
    y:
        Values for y-axis.
    title:
        Figure title.
    xlabel:
        X-axis label.
    ylabel:
        Y-axis label.

    Returns
    -------
    plt.Figure
        Matplotlib figure object.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(x, y, alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(df: pd.DataFrame, title: str = "Correlation Matrix") -> plt.Figure:
    """Create a heatmap of a DataFrame correlation matrix.

    Parameters
    ----------
    df:
        Numeric DataFrame whose columns will be correlated.
    title:
        Figure title.

    Returns
    -------
    plt.Figure
        Matplotlib figure object.
    """
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(corr, annot=True, fmt=".2f", ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    return fig
