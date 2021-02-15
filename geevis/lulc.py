import ee
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import plotly.graph_objects as go

from geevis import utils


def sample_change(start_img, end_img, region, dataset=None, band=None, n=100, scale=None, seed=0, dropna=True):
    """
    Randomly sample values of two images to quantify change over time.

    :param ee.Image start_img: An image representing starting conditions. The image should contain classified values
    representing dictinct classes such as land cover types.
    :param ee.Image end_img: An image representing ending conditions.
    :param geevis.datasets.Dataset dataset: A dataset to which the start and end images belong, which contains a band
    value. If a dataset is not provided, a band name must be explicity provided.
    :param str band: The name of the band of start_img and end_img that contains the class value. Not needed if a
    dataset parameter is provided.
    :param ee.Geometry region: The region to sample.
    :param int n: The number of sample points to extract. More points will take longer to run but generate more
    representative cover statistics.
    :param int scale: The scale to sample point statistics at. Generally, this should be the nominal scale of the
    start and end image.
    :param int seed: Random seed used to generate sample points.
    :param bool dropna: If true, samples with no class data in either image will be dropped. If false, these samples
    will be returned but may cause plotting to fail.
    :return pd.DataFrame: A dataframe in which each row represents as single sample point with the starting class
    in the "start" column and the ending class in the "end" column.
    """
    if all([x is None for x in [dataset, band]]):
        raise ValueError("Provide either a dataset or a band.")
    elif all([x is not None for x in [dataset, band]]):
        raise ValueError("Provide only dataset or band, not both.")
    elif dataset:
        band = dataset.band

    # TODO: Deal with missing scale variable (or see how GEE deals with a null argument)
    start_img = start_img.set({"label": "start"})
    end_img = end_img.set({"label": "end"})
    images = ee.ImageCollection.fromImages([start_img, end_img])

    samples = ee.FeatureCollection.randomPoints(region, n, seed)

    def extract_values(point):
        # Set the location to extract values at
        geom = point.geometry()

        def extract(img, feature):
            # Extract the cover value at the point
            cover = img.reduceRegion(ee.Reducer.first(), geom, scale).get(band)
            # Get the user-defined label that was stored in the image
            label = img.get('label')

            # Set a property where the name is the label and the value is the extracted cover
            return ee.Feature(feature).set(label, cover)

        return ee.Feature(images.iterate(extract, point))

    sample_data = samples.map(extract_values)

    data = pd.DataFrame.from_dict(
        [feat["properties"] for feat in ee.Feature(sample_data).getInfo()["features"]])

    if dropna:
        data = data.dropna()

    return data[["start", "end"]]


def check_plot_params(data, labels, palette):
    """
    Check for values that are present in data and are not present in labels or palette and raise an error if any are
    found.
    """
    missing_labels = utils.get_missing_keys(data["index"], labels)
    missing_palette = utils.get_missing_keys(data["index"], palette)

    if missing_labels:
        raise Exception(
            f"The following values are present in the data and undefined in the labels: {missing_labels}")
    if missing_palette:
        raise Exception(
            f"The following values are present in the data and undefined in the palette: {missing_palette}")


def drop_classes(data, max_classes, metric="area"):
    """
    Remove small classes until a maximum number of classes is reached.

    :param pd.DataFrame data: A dataframe in which each row represents as single sample point and columns represent
    the class of that point at various times.
    :param int max_classes: The maximum number of unique classes to retain. If more classes are present, the smallest
    classes will be removed.
    :return pd.DataFrame: A dataframe with rows that belong to the largest classes.
    """
    class_counts = data.melt().groupby("value").size().reset_index(name="n")
    largest_classes = class_counts.sort_values(
        by="n", ascending=False).value[0:max_classes].tolist()

    dropped_data = data[data.isin(largest_classes)].dropna()

    return dropped_data


def plot_area(data, start_label, end_label, dataset=None, class_labels=None, class_palette=None, max_classes=5, exclude=None, normalize=True):
    """
    Generate a stacked area plot showing how the sampled area of cover changed from a start condition to an end
    condition.

    :param pd.DataFrame data: A dataframe in which each row represents as single sample point with the starting class
    in the "start" column and the ending class in the "end" column.
    :param str start_label: A label to describe the starting conditions, such as "prefire" or "2012".
    :param str end_label: A label to describe the ending conditions, sucsh as "postfire" or "2015".
    :param geevis.dataset.Dataset dataset: A dataset from which the class data was generated, containing labels and
    palettes corresponding to class values. If a dataset is not provided, class labels and a class palette must be
    provided instead.
    :param dict class_labels: A dictionary where keys are the class index values and the values are corresponding
    labels. Every class index in the sample dataset must be included in class_labels.
    :param dict class_palette: A dictionary where keys are the class index values and the values are corresponding
    colors. Every class index in the sample dataset must be included in class_palette.
    :param int max_classes: The maximum number of unique classes to include in the plot. If more classes are present,
    the smallest classes will be omitted from the plot.
    :param list exclude: A list of class values to remove from the plot.
    :param bool normalize: If true, the total area in each group will be normalized to 1. If classes are removed due
    to fit max classes, this will rescale the remaining classes.
    """
    if dataset is None and any([x is None for x in [class_labels, class_palette]]):
        raise ValueError(
            "Provide either a dataset or class labels and a class palette.")
    elif dataset is not None and any([x is not None for x in [class_labels, class_palette]]):
        raise ValueError(
            "Provide only a dataset or class labels and a class palette, not both.")
    elif dataset:
        class_labels = dataset.labels
        class_palette = dataset.palette

    if exclude:
        data = data[~data.isin(exclude)].dropna()

    data = drop_classes(data, max_classes)

    # Count the frequency of each class at the start and end
    freq = data.apply(pd.Series.value_counts).reset_index().melt(
        id_vars="index", var_name="label", value_name="n").fillna(0)

    # Check for missing values in labels or palette
    check_plot_params(freq, class_labels, class_palette)

    if normalize:
        freq = utils.normalize_groups(freq, "label", "n")

    x = ["start", "end"]
    y = list(zip(*[freq[freq.label == label].n for label in x]))

    plot_labels = [class_labels[i] for i in freq["index"]]
    plot_palette = [class_palette[i] for i in freq["index"]]

    fig, ax = plt.subplots()
    ax.stackplot(x, y, labels=plot_labels, colors=plot_palette)

    # Hide plot frame
    for spine in ax.spines.keys():
        ax.spines[spine].set_visible(False)
    # Hide y-axis
    ax.yaxis.set_visible(False)
    # Set tick font size
    plt.setp(ax.get_xticklabels(), fontsize=18)

    # This suppresses a warning about using non-fixed ticks
    tick_loc = ax.get_xticks()
    ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(tick_loc))
    # Set the tick labels
    ax.xaxis.set_ticklabels([start_label, end_label])

    handles, labels = ax.get_legend_handles_labels()
    # Reverse legend order
    plt.legend(handles[::-1], labels[::-1],
               loc='center left', bbox_to_anchor=(1, 0.5),
               fontsize=16,
               frameon=False
               )
    return fig


def plot_sankey(data, start_label, end_label, dataset=None, class_labels=None, class_palette=None, max_classes=5, exclude=None, normalize=True):
    """
    Generate a stacked area plot showing how the sampled area of cover changed from a start condition to an end
    condition.

    :param pd.DataFrame data: A dataframe in which each row represents as single sample point with the starting class
    in the "start" column and the ending class in the "end" column.
    :param str start_label: A label to describe the starting conditions, such as "prefire" or "2012".
    :param str end_label: A label to describe the ending conditions, sucsh as "postfire" or "2015".
    :param geevis.dataset.Dataset dataset: A dataset from which the class data was generated, containing labels and
    palettes corresponding to class values. If a dataset is not provided, class labels and a class palette must be
    provided instead.
    :param dict class_labels: A dictionary where keys are the class index values and the values are corresponding
    labels. Every class index in the sample dataset must be included in class_labels.
    :param dict class_palette: A dictionary where keys are the class index values and the values are corresponding
    colors. Every class index in the sample dataset must be included in class_palette.
    :param int max_classes: The maximum number of unique classes to include in the plot. If more classes are present,
    the unimportant classes will be omitted from the plot.
    :param list exclude: A list of class values to remove from the plot.
    :param bool normalize: If true, the total area in each group will be normalized to 1. If classes are removed due
    to fit max classes, this will rescale the remaining classes.
    """
    if dataset is None and any([x is None for x in [class_labels, class_palette]]):
        raise ValueError(
            "Provide either a dataset or class labels and a class palette.")
    elif dataset is not None and any([x is not None for x in [class_labels, class_palette]]):
        raise ValueError(
            "Provide only a dataset or class labels and a class palette, not both.")
    elif dataset:
        class_labels = dataset.labels
        class_palette = dataset.palette

    if exclude:
        data = data[~data.isin(exclude)].dropna()

    data = drop_classes(data, max_classes)

    # Generate a list of unique labels
    unique_labels = list(np.unique(data.start.to_list() + data.end.to_list()))

    # Transform the data to get counts of each combination of start and end condition
    sankey_data = data.groupby(["start", "end"]).size().reset_index(name="n")

    # Create an index to refer to the start class
    sankey_data["start_index"] = sankey_data.start.apply(
        lambda x: unique_labels.index(x))
    # Create an index to refer to the end class. The end class must have a different value than the same start class,
    # so it is offset by the number of unique class labels
    sankey_data["end_index"] = sankey_data.end.apply(
        lambda x: unique_labels.index(x)) + len(unique_labels)

    # You need to have one label for each class (start and end)
    sankey_labels = [class_labels[i] for i in unique_labels] * 2
    # You only need one color for each source class
    sankey_colors = [class_palette[i] for i in sankey_data.start]

    fig = go.Figure(data=[go.Sankey(
        arrangement="snap",
        node=dict(
            line=dict(color="black", width=0.5),
            label=sankey_labels,
            color=[class_palette[i] for i in unique_labels] * 2
        ),
        link=dict(
            # These values refer to the index of the labels
            source=sankey_data.start_index,
            target=sankey_data.end_index,
            value=sankey_data.n,
            color=sankey_colors,
            hovertemplate='%{value}% changed from %{source.label} to %{target.label}'
        ))])

    return fig
