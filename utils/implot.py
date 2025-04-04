"""
Utility that handles plotting of images
"""
import matplotlib.pyplot as plt
from PIL import Image
import random
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

def plot_images_as_grid(imseries, title, n=4, imtitles=None): 
    """
    creates a grid of n*n images by randomply picking n**2 images
    from a series object (imseries) and plots them into the notebook. 

    Parameters: 
        - imseries (Pandas Series): this should be the fully qualified path to the images on the hard drive.
        - title (string): How to title the plot
        - n (int): how many images per side, (in total n² images will be plotted). 
        - imtitles (Pandas Series)/None: Series of titles for the images. If None, no titles will be shown.

    """
    samplesize = n**2
    sampled_indexes = random.sample(range(len(imseries)), samplesize)
    chosen = imseries.iloc[sampled_indexes].reset_index(drop=True)
    if imtitles is not None:
        chosen_titles = imtitles.iloc[sampled_indexes].reset_index(drop=True)
    fig, axes = plt.subplots(n, n, figsize=(n*2.5, n*2.5))
    axes = axes.flatten()
    i = 0
    for i, ax in enumerate(axes.flat):
        image = chosen[i]
        img = Image.open(image)
        ax.imshow(img)
        ax.axis('off')
        if imtitles is not None:
            ax.set_title(chosen_titles[i])
    fig.suptitle(title, fontsize=16)


def plot_discord(data, samplesize=10, rownames = [], imagecolumn = 'image_path'):
    """
        Creates a discordplot with samplesize images. 
        A discordplot is a plot that visualizes the image instance on the left and the discord among models on the right

        Parameters:
            - data (Pandas Dataframe): pandas dataframe of the entire dataset where you want to visualize the discord of.
            - samplesize (Int): How many discord samples to show. 
            - rownames (List): Names of the columns that hold a prediction score)
            - imagecolumn (string): Name of the column that holds the fully qualified path to the image on the harddrive. 
    """
    data = data.sample(samplesize)
    fig, axes = plt.subplots(samplesize, 2, figsize=(15, samplesize * 6))
    for idx, row in data.reset_index().iterrows(): 
        model_results = []
        for rowname in rownames: 
            model_results.append(row[rowname])
        image_path = row[imagecolumn]
        image = Image.open(image_path)
        # image
        ax_img = axes[idx, 0]
        ax_img.imshow(image)
        # bar chart next to image!!
        ax_bar = axes[idx, 1]
        ax_bar.bar(rownames, model_results)
        ax_bar.set_title(f'Model Predictions for Row {idx+1}')
        ax_bar.set_xlabel('Model')
        ax_bar.set_ylabel('Prediction Value')
    plt.tight_layout()



def make_cm(act, pred, labels): 
    cm = confusion_matrix(act, pred)
    cm_relative = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] 
    cm_relative = np.round(cm_relative, 3)
    # Display confusion matrix using matplotlib
    fig, ax = plt.subplots(figsize=(16, 16))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_relative)
    disp.plot(ax=ax, cmap='Blues', colorbar=True)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.tick_params(axis='x', rotation=90)
    plt.xlabel("Predicted Labels")
    plt.ylabel("Actual Labels")
    return plt