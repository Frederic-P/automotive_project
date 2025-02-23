"""
Utility that handles plotting of images
"""
import matplotlib.pyplot as plt
from PIL import Image

def plot_images_as_grid(imseries, title, n=4): 
    """
    creates a grid of n*n images by randomply picking n**2 images
    from a series object (imseries) and plots them into the notebook. 

    Parameters: 
        - imseries (Pandas Series): this should be the fully qualified path to the images on the hard drive.
        - title (string): How to title the plot
        - n (int): how many images per side, (in total n² images will be plotted). 

    """
    samplesize = n**2
    chosen = imseries.sample(samplesize)
    fig, axes = plt.subplots(n, n, figsize=(n*2.5, n*2.5))
    axes = axes.flatten()
    i = 0
    for image in chosen: 
        img = Image.open(image)
        axes[i].imshow(img)
        axes[i].axis('off')
        i+=1
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

