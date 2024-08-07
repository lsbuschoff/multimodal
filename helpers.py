import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


def plot_params():

    mpl.rcParams["axes.spines.right"] = False
    mpl.rcParams["axes.spines.top"] = False
    
    FIGWIDTH = 7.00697
    SMALL_SIZE = 8
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 12
    
    #  gray for fuyu, purple for adapter, teal for otter, green for gpt-4v, orange for claude, red for humans
    model_colors = [
        "#8c92ac",
        "#624fe8",
        "#008080",
        "#50b990",
        "#da7756",
        "#9b443e",  
    ]
    
    cmap = [
        "#53665c",
        "#ccb3a0",
        "#f4b368",
        "#a08158",
        "#c87858",
        "#94b1d2",
        "#c5d9d8",
        "#e6b951",
        "#666666",
    ]
    sns.set_palette(cmap)
    
    plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
    plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=SMALL_SIZE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title
    
    mpl.rcParams["figure.dpi"] = 300
    mpl.rcParams["savefig.dpi"] = 300

    return model_colors