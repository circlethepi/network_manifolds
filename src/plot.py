"""General plotting functions and settings"""
import torch
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm # for settings
from mpl_toolkits import axes_grid1 # for colorbar
from typing import Union, Optional

import os

from src.utils import check_if_null, display_message

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#                     General Plotting Functions and Settings   
#                          Merrick Ohata 2025, JHU AMS         
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# 
#  
#

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#                               Global Variables
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
GLOBAL_FONT_SIZE = 16           # my default: 16
GLOBAL_PLOT_DIR = 'plots/'
DEFAULT_FIG_SIZE = (10, 8)      # default figure dimensions (w, h) inches

GLOBAL_PLOT_FILETYPE = ".png"
GLOBAL_DPI = 600

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#                          matplotlib restyling settings
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
mpl.rcdefaults()

# add the font I like :-)
weight = "normal"
if "mnzk" in str(os.getcwd()):
    fm.fontManager.addfont("/Users/mnzk/Library/Fonts/LibreFranklin-VariableFont_wght.ttf")
    weight = "medium"

mpl.rcParams.update({
    "text.usetex": False,

    "font.family": "sans-serif",
    # "font.sans-serif": "Comic Sans MS", # default DejaVu Sans per documentation
    "font.sans-serif": "Libre Franklin",


    # Base font settings (applies to many text elements but not all)
    "font.size": GLOBAL_FONT_SIZE,                      # default is 10
    "font.weight": weight, 

    # Overrides for all specific plot elements
    "axes.titlesize": GLOBAL_FONT_SIZE,
    "axes.titleweight": weight,

    "axes.labelsize": GLOBAL_FONT_SIZE,
    "axes.labelweight": weight,

    "xtick.labelsize": GLOBAL_FONT_SIZE,
    "ytick.labelsize": GLOBAL_FONT_SIZE,

    "legend.fontsize": GLOBAL_FONT_SIZE,
    "legend.title_fontsize": GLOBAL_FONT_SIZE,

    "figure.titlesize": GLOBAL_FONT_SIZE,
    "figure.titleweight": weight,
})

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#                        Main Plotting Utility Functions
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

def start_fig(figsize=DEFAULT_FIG_SIZE, xscale="linear", yscale="linear",):
    """Creates simple figure and sets axis scales"""
    plt.figure(figsize=figsize)
    
    if isinstance(xscale, str):
        xscale = (xscale, {})
    plt.xscale(xscale[0], **xscale[1])
    if isinstance(yscale, str):
        yscale = (yscale, {})
    plt.yscale(yscale[0], **yscale[1])


def plot_labels(title=None, 
                xticks=None, xticklabs=None, xlab=None,
                yticks=None, yticklabs=None, ylab=None):
    """Add axis and sup titles; axis ticks"""

    # x axis ticks
    xticks, xticklabs = get_ticks_and_labs(xticks, xticklabs)
    plt.xticks(ticks=xticks, labels=xticklabs)

    # x axis label
    if check_if_null(xlab, False, True):
        plt.xlabel(xlab)

    # y axis ticks
    yticks, yticklabs = get_ticks_and_labs(yticks, yticklabs)
    plt.yticks(ticks=yticks, labels=yticklabs)

    # y axis label
    if check_if_null(ylab, False, True):
        plt.ylabel(ylab)

    # title
    if check_if_null(title, False, True):
        plt.title(title)

    
def plot_legend():
    # TODO add legend functionality
    return


# TODO add colorbar functionality
DEFAULT_CBAR_AX_W = 0.5 #(inches)
DEFAULT_CBAR_PAD = 0.3 #(inches)
def plot_colorbar(mappable=None,
                  target_ax=None, cvals=None, colormap=None, colornorm=None,
                  cbar_thickness:Optional[float]=None, 
                  pad_in:Optional[float]=None,
                  horizontal:bool=False, location_override:Optional[str]=None,

                  cbar_label:Optional[str]=None, labelpad:int=20,
                  cbar_ticks=None, cbar_ticklabs=None,
                  ):
    """Adds a colorbar to the plot

    Either takes the output of some plotting (eg. imshow, scatter) or values 
    and a colormap. 
    
    If not using output of plotting, typically also include target_ax (to 
    attach the colorbar to). Otherwise, this produces a standalone colorbar
    

    :param mappable:    output of imshow or scatter (or a ScalarMappable)

    :param target_ax:   Axes for the colorbar to be associated with (optional)
    :param cvals:       Values for the colormap to take on (optional, Default: 
                        None)
    :param colormap     the colormap to use

    :param cbar_thickness:      width of the colorbar (heigh when horizontal)
                                in inches. Default: DEFAULT_CBAR_AX_W (0.5in)
    :param pad_in:      padding between colorbar and plot in inches. Default:
                        DEFAULT_CBAR_PAD (0.3in)
    :param horizontal:bool  whether the bar should be horizontal
    :param location_override: str|None  override location to place the colorbar
                                        wrt plot axes. Default: None means 
                                        horizontal bars go on bottom and 
                                        vertical bars on the right

    :param cbar_label: str|None     title for the colorbar. Default None
    
    """
    # get orientation
    cbar_orient = "horizontal" if horizontal else "vertical"

    # if not output of plot
    if check_if_null(mappable, True, False):
        if check_if_null(cvals, True, False):
            message = """Must provide `cvals` if `mappable` is None"""
            raise ValueError(display_message(message))

        cvals = format_data(cvals)
        cbar_ticks = check_if_null(cbar_ticks, cvals) # set values as ticks
        
        # colormap validation
        colormap = check_if_null(colormap, mpl.cm.viridis)
        if isinstance(colormap, str):
            colormap = plt.get_cmap(colormap)
        
        # norm validation
        colornorm = check_if_null(colornorm, 
                                  mpl.colors.Normalize(vmin=np.min(cvals), 
                                                       vmax=np.max(cvals)))
        colornorm = format_data(colornorm)
        if isinstance(colornorm, np.ndarray):
            message = """`colornorm` contains more than 2 values. Using only
                      the first two for normalization"""
            if colornorm.shape[0] > 2:
                print(display_message(message))
            colornorm = mpl.colors.Normalize(vmin=colornorm[0], 
                                             vmax=colornorm[1])
        
        # create the mappable
        mappable = mpl.cm.ScalarMappable(norm=colornorm, cmap=colormap)
        mappable.set_array(cvals)

        # STANDALONE
        # if no target axes, then standalone colorbar
        if check_if_null(target_ax, True, False):
            default_standlone_figsizes = { # TODO add alpha support
                (True, "horizontal"): (8,8), 
                (True, "vertical"): (8,8),
                (False, "horizontal"): (8,1.5),
                (False, "vertical"): (1.5,8)
            }
            figsize = default_standlone_figsizes[True, "horizontal" if \
                                                 horizontal else "vertical"]
            # make the figure and colorbar axes
            fig, cax = plt.subplots(figsize=figsize, layout="constrained")
    
    else:
        canvas = axes_grid1.make_axes_locatable(mappable.axes)
        fig = mappable.axes.figure
        current = plt.gca()

        # get size and padding
        size_convert = check_if_null(cbar_thickness, DEFAULT_CBAR_AX_W)
        size = axes_grid1.axes_size.Fixed(size_convert) 
        pad_convert = check_if_null(pad_in, DEFAULT_CBAR_PAD)
        pad = axes_grid1.axes_size.Fixed(pad_convert) 

        
        cbar_loc = check_if_null(location_override,     # get location
                                "bottom" if horizontal else "right")
        cax = canvas.append_axes(cbar_loc, size=size, pad=pad)

        plt.sca(current)
    
    # ticks and labels
    cbar_ticks, cbar_ticklabs = get_ticks_and_labs(cbar_ticks, cbar_ticklabs)

    # add in colorbar
    cbar = fig.colorbar(mappable, cax=cax, orientation=cbar_orient,
                        ticks=cbar_ticks)
    
    # add colorbar ticks
    label_add = cbar.ax.set_xticklabels if horizontal else \
                cbar.ax.set_yticklabels
    label_add(cbar_ticklabs)

    # add colorbar title
    if check_if_null(cbar_label, False, True):
        if horizontal:
            cbar.ax.set_xlabel(cbar_label)
        else:
            cbar.ax.get_yaxis().labelpad = labelpad
            cbar.ax.set_ylabel(cbar_label, rotation=-90)
    
    return cbar


def end_fig(show=True, savename=None, dir=None):
    """Shows and saves fig as indicated"""

    if show:
        plt.show()
    
    if check_if_null(savename, False):
        assert isinstance(savename, str), "Invalid filename"
        if not savename.endswith(GLOBAL_PLOT_FILETYPE):
            savename += GLOBAL_PLOT_FILETYPE

        dir = check_if_null(dir, GLOBAL_PLOT_DIR)
        
        savepath = os.path.join(dir, savename)

        plt.savefig(savepath, 
                    bbox_inches="tight", pad_inches=0,
                    dpi=GLOBAL_DPI)
        

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#                               Format Helpers
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

def format_ticklabs(x):
    """Formats tick (labels) for axes"""
    if check_if_null(x, True, False): # replace None with blank
        y = ""
    elif isinstance(x, str):    # strings unmodified
        y = x
    elif int(x) == x and (abs(x) <= 1e2):    # small enough int
        y = int(x)
    elif 0.01 <= abs(x) <= 1e2:      # small enough float 
        y = f'{x:.2f}'
    elif abs(x) < 1e-10:        # too small 
        y = "0"
    elif is_int_x_pwr10(x):     # integer times power of 10 
        y = f'{x:.0e}'
    else:
        y = f'{x:.2e}'          # scientific otherwise
    return y

def is_int_x_pwr10(x):
    """Check if x is an integer times some power of 10"""
    exp = np.floor(np.log10(abs(x)))
    man = abs(x) / (10**exp)
    return float(man).is_integer()


def get_ticks_and_labs(ticks, ticklabs):
    """check if ticks and/or labels exist and format"""
    if check_if_null(ticks, True, False):
        ticks = []
        ticklabs = []
    else:
        ticklabs = [format_ticklabs(x) \
                     for x in check_if_null(ticklabs, ticks)]
    
    return ticks, ticklabs

def format_data(x):
    """Converts tensors/arrays/lists to numpy for plotting"""
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    elif isinstance(x, list):
        return np.array(x)
    elif isinstance(x, dict):
        return {k: format_data(v) for k, v in x.items()}
    elif isinstance(x, tuple):
        return tuple(format_data(t) for t in x)
    else:
        return x

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#                              Specific Plot Types
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


