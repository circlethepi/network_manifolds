"""General plotting functions and settings"""
import torch
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm # for settings
from mpl_toolkits import axes_grid1 # for colorbar
from matplotlib.ticker import ScalarFormatter # for (linear) tick format

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
GLOBAL_FONT_SIZE = 24           # my default: 16
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

    # Overrides for all plot elements not included in base setting
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


## Format Definitions / Helper
## Linear axis formatting
lin_formatter = ScalarFormatter(useMathText=True)
lin_formatter.set_scientific(True)
lin_formatter.set_powerlimits((-3, 3)) # default 


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#                        Main Plotting Utility Functions
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

def with_ax(func):
    """wrapper to allow plotting on specific axes if provided (for subplots or
    subfigures)"""

    def wrapper(*args, ax=None, figsize=DEFAULT_FIG_SIZE, 
                **kwargs):
        standalone = False

        if check_if_null(ax, True, False):
            fig, ax = plt.subplots(figsize=figsize)
            standalone = True
            # print(f"[with_ax] Created figure {fig.number} - {func.__name__}")
        else:
            fig = ax.figure
            # plt.sca(ax)
            # print(f"[with_ax] Using existing figure {fig.number} - {func.__name__}")
        
        result = func(*args, ax=ax, **kwargs)

        return result
    return wrapper


@with_ax
def scale_ax(ax=None, xscale="linear", yscale="linear", pwr_format=True):
    """Sets axis scales and optional tick formatting"""

    if isinstance(xscale, str):
        xscale = (xscale, {})
    ax.set_xscale(xscale[0], **xscale[1])
    # set formatter
    if pwr_format and xscale[0] == "linear":
        ax.xaxis.set_major_formatter(lin_formatter)

    if isinstance(yscale, str):
        yscale = (yscale, {})
    ax.set_yscale(yscale[0], **yscale[1])
    # set formatter
    if pwr_format and yscale[0] == "linear":
        ax.yaxis.set_major_formatter(lin_formatter)

    return ax


@with_ax
def plot_labels(ax=None, title=None, 
                xticks=None, xticklabs=None, xrot=0, xlab=None, xlims=None,
                yticks=None, yticklabs=None, yrot=0, ylab=None, ylims=None):
    """Add axis and sup titles; axis ticks, axis limits"""

    # x axis ticks
    xticks, xticklabs = get_ticks_and_labs(xticks, xticklabs)
    ax.set_xticks(ticks=xticks, labels=xticklabs, rotation=xrot)

    # x axis label
    if check_if_null(xlab, False, True):
        ax.set_xlabel(xlab)

    # x axis limits
    lim_message = """Axis limits must be exactly two values: (min, max)"""
    if check_if_null(xlims, False, True):
        assert len(xlims) == 2, display_message(lim_message)
        ax.set_xlim(xlims[0], xlims[1])

    # y axis ticks
    yticks, yticklabs = get_ticks_and_labs(yticks, yticklabs)
    ax.set_yticks(ticks=yticks, labels=yticklabs, rotation=yrot)

    # y axis label
    if check_if_null(ylab, False, True):
        ax.set_ylabel(ylab)

    # y axis limits
    if check_if_null(ylims, False, True):
        assert len(ylims) == 2, display_message(lim_message)
        ax.set_ylim(ylims[0], ylims[1])

    # title
    if check_if_null(title, False, True):
        ax.set_title(title)
    
    return ax
    
@with_ax
def format_ticks(ax=None, dx_x=75, dy_x=0, dx_y=-100, dy_y=15, color="#777777"):
    """Formats the ticks and axes for scientific notation
    (overrides default custom tick formatting)"""

     # REDRAW LABELS (for scientific notation vibes)
    if ax.get_yscale() == "linear":
        ax.yaxis.set_major_formatter(lin_formatter)
    if ax.get_xscale() == "linear":
        ax.xaxis.set_major_formatter(lin_formatter)

    move_sci_offset_text(axis='x', dx=dx_x, dy=dy_x, color=color)
    move_sci_offset_text(axis='y', dx=dx_y, dy=dy_y, color=color)

    # redraw the labels
    ax.figure.canvas.draw()

    return ax

    
def plot_legend():
    # TODO add legend functionality
    return


# TODO add colorbar functionality
DEFAULT_CBAR_AX_W = 0.5 #(inches)
DEFAULT_CBAR_PAD = 0.3 #(inches)
def plot_colorbar(mappable=None,
                  target_ax=None, 
                  cvals=None, colormap=None, colornorm=None, figsize=None,
                  cbar_thickness:Optional[float]=None, 
                  pad_in:Optional[float]=None,
                  horizontal:bool=False, location_override:Optional[str]=None,

                  cbar_label:Optional[str]=None, labelpad:int=20,
                  cbar_ticks=None, cbar_ticklabs=None,

                  **kwargs
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

    :param **kwargs:    additional arguments for end_fig() if standalone cbar
    
    """
    standalone = False

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
            standalone = True
            figsize = check_if_null(figsize, 
                                    default_standlone_figsizes[False, \
                                    "horizontal" if horizontal else "vertical"]
                                    )
            # make the figure and colorbar axes
            fig, cax = plt.subplots(figsize=figsize, layout="constrained")
        else:
            fig, cax = target_ax.figure, target_ax
           
    
    else:
        canvas = axes_grid1.make_axes_locatable(mappable.axes)
        fig = mappable.axes.figure
        current = fig.gca()

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
    
    if standalone:
        end_fig(fig, show=True, tight=False, **kwargs)
    
    return fig


def end_fig(fig_or_ax=None, show=True, savename=None, dir=None, 
            filetype=GLOBAL_PLOT_FILETYPE, tight=True):
    """Shows and saves fig as indicated"""

    if check_if_null(fig_or_ax, True, False):
        fig = plt.gcf() # select current if not provided
    elif hasattr(fig_or_ax, "figure"): # ax
        fig = fig_or_ax.figure
    else: # is fig
        fig = fig_or_ax 

    # print(f"[end_fig] Tightening layout and closing figure {fig.number}")

    if tight:
        fig.tight_layout(pad=0.1)

    if check_if_null(savename, False):
        assert isinstance(savename, str), "Invalid filename"
        if not savename.endswith(filetype):
            savename += filetype

        dir = check_if_null(dir, GLOBAL_PLOT_DIR)
        
        savepath = os.path.join(dir, savename)

        fig.savefig(savepath, 
                    bbox_inches="tight", pad_inches=0.15,
                    dpi=GLOBAL_DPI, transparent=True)

    if show:
        plt.show()
    
    plt.close(fig)
    

        

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#                               Format Helpers 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

def space_ticks(limits, n_ticks, keep0=False):
    """Generate tick values of the form
                K*10^{P} 
    for a fixed P (linear scale) from a given set of axis limits

    n_ticks is approximate - actual number of ticks may vary
    """

    lim_lo, lim_hi = limits[0], limits[1]

    pwr = min(np.floor(np.log10(abs(lim_lo))), np.floor(np.log10(abs(lim_hi))))

    K_lo = np.sign(lim_lo) * ( abs(lim_lo) / (10**pwr) ) # mantissa for lo val
    K_hi = np.sign(lim_hi) * ( abs(lim_hi) / (10**pwr) ) # mantissa for hi val

    start_int, end_int = int(np.ceil(K_lo)), int(np.floor(K_hi))
    total_ticks = end_int + np.sign(start_int)*start_int
    space = int(np.floor(total_ticks / n_ticks))

    if total_ticks < n_ticks:
        space = 1

    Ks = list(range(start_int, end_int+space, space))

    # option to always include 0
    if (np.sign(start_int) == -1* np.sign(end_int)) and keep0:
        Ks.append(0)

    ticks = [K * 10**pwr for K in Ks]

    return(ticks)


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
    

def move_sci_offset_text(axis='x', dx=0, dy=0, color="#999999", ax=None):

    ax = check_if_null(ax, plt.gca())

    message= """axis must be 'x' or 'y'"""
    assert axis in ('x', 'y'), display_message(message)

    offset = {'x': ax.xaxis, 'y': ax.yaxis}[axis].get_offset_text()

    ax.figure.canvas.draw() # Force a draw so the text exists
    # Get current position in display coords (pixels)
    x_disp, y_disp = offset.get_transform().transform(offset.get_position())

    # Shift by dx, dy in pixels
    x_disp += dx
    y_disp += dy

    # Convert back to axis coordinates and set new position
    new_x, new_y = ax.transAxes.inverted().transform((x_disp, y_disp))
    offset.set_position((new_x, new_y))
    offset.set_color(color)




#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#                              Specific Plot Types
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

@with_ax
def plot_similarity_matrix(sims, ax=None, title=None,
                           figsize=(10, 10),
                           axis_label=None, ticks=None, ticklabs=None,
                           full_matrix=False, xrot=0, yrot=0,

                           colormap=None, vmin=0, vmax=1,
                           mask_color='#cffcff', nan_color='#ffaffa',

                           do_colorbar=True, cbar_label="similarity",
                           cbar_ticks=None,
                           savename=None, savedir_override=None, show_plot=True,
                           standalone=False
                           # TODO add splits
                           ):
    """Plot similarity matrix heatmap with colorbar
    nan_color option is for if there are nan values in the matrix
    """
    # WISHLIST better arguments for this
    
    ax = scale_ax(figsize=figsize, ax=ax, xscale="linear", yscale="linear", 
             pwr_format=False)

    # Full matrix handling (show only bottom half)
    mask = np.zeros_like(sims, dtype=bool) if full_matrix \
           else np.triu(np.ones_like(sims, dtype=bool), k=0)
    
    colormap = check_if_null(colormap, mpl.cm.binary)
    colormap.set_bad(mask_color)

    outmap = ax.imshow(np.ma.array(sims, mask=mask), cmap=colormap,
                        vmin=vmin, vmax=vmax, interpolation="nearest")
    # (input for the colorbar later)
    
    # NaN value handling
    nan_sims = np.isnan(sims)
    nan_sims = np.ma.masked_where(nan_sims==False, nan_sims)
    nan_sims = np.ma.array(nan_sims, mask=mask)
    colormap_nan = mpl.colors.ListedColormap([nan_color, nan_color])
    
    ax.imshow(nan_sims, aspect="auto", cmap=colormap_nan, vmin=0, vmax=1,
               origin="lower")

    # labels
    ax = plot_labels(ax=ax, title=title, xticks=ticks, xticklabs=ticklabs,
                yticks=ticks, yticklabs=ticklabs,
                xlab=axis_label, ylab=axis_label,
                xrot=xrot, yrot=yrot)

    # colorbar
    if do_colorbar:
        cbar_ticks = check_if_null(cbar_ticks, np.linspace(vmin, vmax, 5))
        plot_colorbar(outmap, cbar_ticks=cbar_ticks, cbar_label=cbar_label, 
                        labelpad=40)
    
    # separation lines
    # TODO implement separation lines in plot for different runs etc

    # force same scales (keeps all cells square)
    ax.set_aspect("equal")
    
    if standalone:
        end_fig(fig_or_ax=ax, show=show_plot, savename=savename, 
                dir=savedir_override)

    return outmap


# Package the mds plotting function more nicely
@with_ax
def plot_MDS(coords, ax=None,
             figsize=DEFAULT_FIG_SIZE,
             color_vals=None, cmap=None, marker="o", msize=400,

             equal=False, 
             title:Optional[str]=None, 
             xticks:Optional[any]=None, yticks:Optional[any]=None,
             xlab:Optional[str]="MDS d1", ylab:Optional[str]="MDS d2",
             xlims=None, ylims=None, xrot=0, yrot=0,
             format=True,

             do_colorbar=True,
             cbar_ticks=None, cbar_h=False, cbar_labpad=30, cbar_label=None,
             savename=None, savedir_override=None, show_plot=True,

             standalone=True,
             ):
    # TODO add option for trajectory lines
    
    cmap = check_if_null(cmap, plt.cm.plasma)
    
    ax = scale_ax(ax=ax, xscale="linear", yscale="linear", pwr_format=True)
    scat = ax.scatter(coords[:, 0], coords[:, 1], c=color_vals, cmap=cmap, 
                       s=msize, marker=marker)

    if equal:
        ax.set_aspect("equal")

    ax = plot_labels(ax=ax, title=title,
                xticks=xticks, yticks=yticks,
                xlab=xlab, ylab=ylab,
                xlims=xlims, ylims=ylims,
                xrot=xrot, yrot=yrot)

    if do_colorbar:
        plot_colorbar(mappable=scat, target_ax=ax,
                      cbar_ticks=cbar_ticks, 
                      horizontal=cbar_h, labelpad=cbar_labpad,
                      cbar_label=cbar_label)
    
    if format:
        ax = format_ticks(ax=ax)

    if standalone:
        end_fig(fig_or_ax=ax, show=show_plot, savename=savename, 
                dir=savedir_override)

    return scat