"""
Utility that handles plotting of images and basic metrics for plots.
"""
import matplotlib.pyplot as plt
from PIL import Image
import random
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score, cohen_kappa_score, precision_score, recall_score
import numpy as np
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import defaultdict
import math


def plot_images_as_grid(imseries, title, n=4, imtitles=None): 
    """
    creates a grid of n*n images by randomply picking n**2 images
    from a series object (imseries) and plots them into the notebook. 

    if there's more axes than images, these will be rendered as white space. 

    Parameters: 
        - imseries (Pandas Series): this should be the fully qualified path to the images on the hard drive.
        - title (string): How to title the plot
        - n (int): how many images per side, (in total n² images will be plotted). 
        - imtitles (Pandas Series)/None: Series of titles for the images. If None, no titles will be shown.

    """
    samplesize = n**2
    if samplesize > len(imseries):
        samplesize = len(imseries)
    sampled_indexes = random.sample(range(len(imseries)), samplesize)
    chosen = imseries.iloc[sampled_indexes].reset_index(drop=True)
    if imtitles is not None:
        chosen_titles = imtitles.iloc[sampled_indexes].reset_index(drop=True)
    fig, axes = plt.subplots(n, n, figsize=(n*2.5, n*2.5))
    axes = axes.flatten()
    i = 0
    for i, ax in enumerate(axes.flat):
        if i >= len(imseries):
            ax.axis('off')
            continue
        image = chosen[i]
        img = Image.open(image)
        ax.imshow(img)
        ax.axis('off')
        if imtitles is not None:
            ax.set_title(chosen_titles[i])
    fig.suptitle(title, fontsize=16)


def cm_delta(pred_one, pred_two, acts, labels): 
    """
        Plots a meta CM comparing two models. It will substract the score of 
        the CM of your second model from the CM of your first model and plot the
        difference on a red-green scale

        red = CM2 performed worse than CM1 for this cell; green is better. 

    ARGUMENTS: 
        pred_one = Predictions of the first model
        pred_two = Prediction of the second model
        acts = actual values (actuals should be of equal length! as pred_one and pred_two
        labels = list of labels (REQUIRED)

    """
    assert(len(pred_one)== len(acts))
    assert(len(pred_two)== len(acts))
    cm_one = confusion_matrix(acts, pred_one, labels=labels)
    cm_two = confusion_matrix(acts, pred_two, labels=labels)
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_one_norm = cm_one.astype('float') / cm_one.sum(axis=1, keepdims=True)
        cm_two_norm = cm_two.astype('float') / cm_two.sum(axis=1, keepdims=True)
        cm_one_norm = np.nan_to_num(cm_one_norm, nan=0.0, posinf=0.0, neginf=0.0)
        cm_two_norm = np.nan_to_num(cm_two_norm, nan=0.0, posinf=0.0, neginf=0.0)

        # Difference in percentage points
        diff_matrix = (cm_one_norm - cm_two_norm) * 100
    fig, ax = plt.subplots(figsize=(20, 20))
    sns.heatmap(diff_matrix, annot=True, fmt=".1f", cmap='RdYlGn', center=0,
                xticklabels=labels, yticklabels=labels, cbar=True, cbar_kws={'shrink': 0.75}, square=True, ax=ax)
    
    ax.set_title("Difference Matrix (Model1 - Model2) in %", fontsize=16)
    fig.text(0.12, 0.05, "Model 1 is better than model2 if the TP-diagonal lights up green\n reds outside the diagonal mean that model1 is making less errors there.", ha='center', fontsize=12)
    ax.set_xlabel("Predicted Labels")
    ax.set_ylabel("Actual Labels")
    ax.tick_params(axis='x', rotation=90)
    ax.tick_params(axis='y', rotation=0)
    plt.tight_layout()
    plt.close(fig)
    return fig

def make_cm(act, pred, labels, embed=False, err_only=False, colormap='Blues'): 
    cm = confusion_matrix(act, pred)
    if err_only:
        cm[np.eye(len(cm), dtype=bool)] = 0
    cm_relative = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_relative = np.round(cm_relative, 3)
    if embed:
        disp = ConfusionMatrixDisplay(confusion_matrix=cm_relative)
        disp.display_labels = labels
        return disp
    fig, ax = plt.subplots(figsize=(20, 20))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_relative)
    disp.plot(ax=ax, cmap=colormap, colorbar=False)
    im = ax.images[-1]
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.9)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=14, rotation=90)
    ax.set_yticklabels(labels, fontsize=14)
    ax.set_xlabel("Predicted Labels", fontsize=16)
    ax.set_ylabel("Actual Labels", fontsize=16)
    return plt

def quickplot(df, key):
    value_counts = df[key].value_counts().reset_index()
    value_counts.columns = [key, 'count']
    fig = px.bar(
        value_counts,
        x=key,
        y='count',
        template="custom"
    )
    return fig



def quick_metrics(act, pred): 
    """
        Calculates eight standard metrics to use in reporting; 
        Accurac, macroF1, weighedF1, cohens kappa, macroprecission
        macrorecall, weighedprecision, weighedrecall
    
    ARGUMENTS: 
        act = list = list of actual
        pred = list = list of model predictions

    RETURNS: 
        dictionary with scoring name metrics and scores. 
    """
    results = {
        'Accuracy': accuracy_score(act, pred),
        'Macro F1 Score': f1_score(act, pred, average='macro'),
        'Weighted F1 Score': f1_score(act, pred, average='weighted'),
        'Cohen\'s Kappa': cohen_kappa_score(act, pred),
        'Macro Precision': precision_score(act, pred, average='macro', zero_division=0),
        'Macro Recall': recall_score(act, pred, average='macro', zero_division=0),
        'Weighted Precision': precision_score(act, pred, average='weighted', zero_division=0),
        'Weighted Recall': recall_score(act, pred, average='weighted', zero_division=0),
    }    
    return results


def plot_quick_metrics(multidict): 
    x_labels = list(multidict.keys())
    metrics_to_plot = list(multidict[x_labels[0]].keys())

    fig = make_subplots(rows=2, cols=4, subplot_titles=metrics_to_plot)

    for i, metric in enumerate(metrics_to_plot):
        row = i // 4 + 1
        col = i % 4 + 1
        values = [multidict[k][metric] for k in x_labels]

        fig.add_trace(
            go.Bar(
                x=x_labels,
                y=values,
                text=[f'{v:.3f}' for v in values],
                textposition='inside',
                textfont=dict(size=18),
                name=metric
            ),
            row=row, col=col
        )
    return fig


def get_correct_counts(preds, actuals):
    correct = defaultdict(int)
    counts = defaultdict(int)
    for pred, actual in zip(preds, actuals):
        counts[actual] += 1
        if pred == actual:
            correct[actual] += 1
    return correct, counts

def compare_model_accuracies(model1_preds, model1_actuals, model2_preds, model2_actuals, labels, stringdict):
    correct1, counts1 = get_correct_counts(model1_preds, model1_actuals)
    correct2, counts2 = get_correct_counts(model2_preds, model2_actuals)
    #For adding text to plot!!
    required_strings = ['title', 'xaxis_title', 'yaxis_title']
    for s in required_strings:
        assert s in stringdict.keys()
    diff_raw = []
    for label in labels:
        total1 = counts1[label]
        total2 = counts2[label]
        acc1 = correct1[label] / total1 if total1 > 0 else 0
        acc2 = correct2[label] / total2 if total2 > 0 else 0
        diff = (acc1 - acc2) * 100  # percent points
        diff_raw.append((label, diff))
    diff_sorted = sorted(diff_raw, key=lambda x: x[1], reverse=False)
    sorted_labels, sorted_diffs = zip(*diff_sorted)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=sorted_labels,
        x=sorted_diffs,
        orientation='h',
        text=[f"{d:+.2f}" for d in sorted_diffs],
        textposition='outside',
        marker_color=['green' if d >= 0 else 'red' for d in sorted_diffs]
    ))
    fig.update_layout(
        title=stringdict['title'],
        xaxis_title=stringdict['xaxis_title'],
        yaxis_title=stringdict['yaxis_title'],
        height=1000,
        width=750, 
        margin=dict(l=150, r=50, t=50, b=50),  # Adjust margins for padding

        yaxis=dict(ticklen=1535),
        template='custom'
    )
    return fig


def side_by_side_cm(cm1, cm1title, cm2, cm2title, innersize = 6, labelsize = 12):
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    axes = axes.flatten()
    for ax, cm_display, title in zip(
        axes,
        [cm1, cm2],
        [cm1title, cm2title]
    ):
        cm_display.plot(ax=ax, cmap='Blues', colorbar=True)
        ax.set_title(title, fontsize=18)
        ax.set_xlabel("Predicted Labels", fontsize=labelsize+2)
        ax.set_ylabel("Actual Labels", fontsize=labelsize+2)
        ax.tick_params(axis='x', labelrotation=90)
        ax.xaxis.label.set_size(labelsize+2)
        ax.yaxis.label.set_size(labelsize+2)
        for text in ax.texts:
            text.set_fontsize(innersize)
        ax.set_xticklabels(cm_display.display_labels, fontsize=labelsize)
        ax.set_yticklabels(cm_display.display_labels, fontsize=labelsize)

    plt.tight_layout()


def brand_accuracy_plot(actuals, predicts):
    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    for true, pred in zip(actuals, predicts):
        class_total[true] += 1
        if true == pred:
            class_correct[true] += 1

    # Compute accuracy and sort by accuracy ascending
    class_accuracy = [
        (c, class_correct[c] / class_total[c]) for c in class_total.keys()
    ]
    class_accuracy.sort(key=lambda x: x[1])  # Sort by accuracy

    sorted_classes, sorted_accuracies = zip(*class_accuracy)

    fig = go.Figure(data=[
        go.Bar(
            x=sorted_classes,
            y=sorted_accuracies,
            text=[f"{a*100:.1f}%" for a in sorted_accuracies],  # Format as percent
            textposition='inside',
            textfont=dict(color='white', size=12)
        )
    ])

    fig.update_layout(
        xaxis_title="Brand",
        yaxis_title="Accuracy",
        yaxis=dict(range=[0, 1]),
        xaxis_tickangle=-45,
        template='custom',
        height=500
    )

    return fig


def brand_missclassification_plot(actuals, predicts):
    misclass_counts = defaultdict(int)

    for true, pred in zip(actuals, predicts):
        if true != pred:
            misclass_counts[pred] += 1  # Count incorrect predictions toward the predicted class

    # Sort by frequency descending
    sorted_items = sorted(misclass_counts.items(), key=lambda x: x[1], reverse=True)
    if not sorted_items:
        print("No misclassifications found.")
        return

    classes, counts = zip(*sorted_items)

    fig = go.Figure(data=[
        go.Bar(
            x=classes,
            y=counts,
            marker_color='red',
            text=counts,
            textfont=dict(color='black', size=16)
        )
    ])

    fig.update_layout(
        xaxis_title="Wrong label predicted",
        yaxis_title="Amount of misspredictions",
        template='custom',
        height=500
    )

    return fig


def plot_boxplots_with_titles(data_dict, title):
    fig = go.Figure()
    for idx, (key, entry) in enumerate(data_dict.items()):
        fig.add_trace(go.Box(
            y=entry['values'],
            name=key,
            boxpoints='outliers',
            line=dict(width=1.5)
        ))

    fig.update_layout(
        title = title, 
        yaxis_title="Score",
        xaxis_title="Key",
        yaxis=dict(range=[0, 1]),
        template="custom",
        height=500
    )

    fig.show()


def plot_histograms_subplots(data_dict, num_bins=100, rows=None, cols=None, clean=False):
    num_items = len(data_dict)
    bin_size = 1.0 / num_bins

    # Auto-calculate grid shape if not provided
    if rows is None or cols is None:
        cols = math.ceil(math.sqrt(num_items))
        rows = math.ceil(num_items / cols)

    h_spacing = 0.03 if clean else 0.6
    v_spacing = 0.07 if clean else 0.15

    fig = make_subplots(
        rows=rows,
        cols=cols,
        shared_xaxes=False,
        shared_yaxes=False,
        horizontal_spacing=h_spacing,
        vertical_spacing=v_spacing,
        subplot_titles=[
            f"{key}: {entry['title']}" for key, entry in data_dict.items()
        ]
    )

    for i, (key, entry) in enumerate(data_dict.items()):
        row = (i // cols) + 1
        col = (i % cols) + 1

        fig.add_trace(
            go.Histogram(
                x=entry['values'],
                xbins=dict(start=0, end=1, size=bin_size),
                name=key,
                showlegend=False,
                marker=dict(color='skyblue', line=dict(width=1))
            ),
            row=row,
            col=col
        )

    fig.update_layout(
        title="Histograms of Certainties",
        height=250 * rows,
        template="custom"
    )

    # Set x/y axis labels (or hide if clean=True)
    for r in range(1, rows + 1):
        for c in range(1, cols + 1):
            fig.update_xaxes(
                title_text=None if clean else "Score",
                range=[0, 1],
                row=r, col=c,
                showticklabels=not clean
            )
            fig.update_yaxes(
                title_text=None if clean else "Count",
                row=r, col=c,
                showticklabels=not clean
            )

    return fig