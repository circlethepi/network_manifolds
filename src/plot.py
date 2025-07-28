#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#                     General Plotting Functions and Settings   
#                          Merrick Ohata 2025, JHU AMS         
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
"""
General plotting functions and settings
"""

import torch
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm # for settings
from mpl_toolkits import axes_grid1 # for colorbar
from matplotlib.ticker import ScalarFormatter # for (linear) tick format
import colorsys

from typing import Union, Optional

import os

from src.utils import check_if_null, display_message

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#                               Global Variables
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#region
GLOBAL_FONT_SIZE = 24           # my default: 16
GLOBAL_PLOT_DIR = 'plots/'
DEFAULT_FIG_SIZE = (10, 8)      # default figure dimensions (w, h) inches

GLOBAL_PLOT_FILETYPE = ".png"
GLOBAL_DPI = 600

DEFAULT_LABEL_PAD_X = 15
DEFAULT_LABEL_PAD_Y = 30


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#                          matplotlib restyling settings
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#region
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
#                           Component/Plot Formatters
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#region

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
                yticks=None, yticklabs=None, yrot=0, ylab=None, ylims=None,
                ylabpad=DEFAULT_LABEL_PAD_Y, xlabelpad=DEFAULT_LABEL_PAD_X,
                tick_positions=["bottom", "left"]):
    """Add axis and sup titles; axis ticks, axis limits"""

    # tick positions
    top = "top" in tick_positions
    right = "right" in tick_positions
    bottom = "bottom" in tick_positions
    left = "left" in tick_positions
    change_tick_position(ax=ax, top=top, right=right,
                         bottom=bottom, left=left)
    change_ax_lab_position(ax=ax, top=top, right=right,
                           bottom=bottom, left=left)
    
    # --------------------
    # x axis ticks
    xticks, xticklabs = get_ticks_and_labs(xticks, xticklabs)
    ax.set_xticks(ticks=xticks, labels=xticklabs, rotation=xrot)

    # x axis label
    if check_if_null(xlab, False, True):
        ax.set_xlabel(xlab, labelpad=xlabelpad)

    # x axis limits
    lim_message = """Axis limits must be exactly two values: (min, max)"""
    if check_if_null(xlims, False, True):
        assert len(xlims) == 2, display_message(lim_message)
        ax.set_xlim(xlims[0], xlims[1])

    # --------------------
    # y axis ticks
    yticks, yticklabs = get_ticks_and_labs(yticks, yticklabs)
    ax.set_yticks(ticks=yticks, labels=yticklabs, rotation=yrot)

    # y axis label
    if check_if_null(ylab, False, True):
        ax.set_ylabel(ylab, rotation=0 if left else -90, labelpad=ylabpad)

    # y axis limits
    if check_if_null(ylims, False, True):
        assert len(ylims) == 2, display_message(lim_message)
        ax.set_ylim(ylims[0], ylims[1])

    # --------------------

    # title
    if check_if_null(title, False, True):
        ax.set_title(title)
    
    return ax

@with_ax
def change_ax_lab_position(ax=None, top=False, right=False, 
                          bottom=True, left=True):
    """Change the position of the axis labels on the axes"""
    # WISHLIST add support for label on both sides for each axis
    
    ax.xaxis.set_label_position("top" if top else "bottom")
    ax.yaxis.set_label_position("right" if right else "left")

    return ax


@with_ax
def change_tick_position(ax=None, top=False, right=False, 
                        bottom=True, left=True):
    """Change the position of the ticks on the axes"""
    
    ax.tick_params(left=left, right=right, top=top, bottom=bottom,
                   labelleft=left, labelright=right, 
                   labeltop=top, labelbottom=bottom)
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

    shift_scinot_fmt_text(axis='x', dx=dx_x, dy=dy_x, color=color)
    shift_scinot_fmt_text(axis='y', dx=dx_y, dy=dy_y, color=color)

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
                                        wrt plot axes. Default: None -- means 
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
        colormap = get_colormap(colormap)
        
        # norm validation
        colornorm = get_colornorm(check_if_null(colornorm, 
                                  mpl.colors.Normalize(vmin=np.min(cvals), 
                                                       vmax=np.max(cvals))))
        
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
#                          Plotting Specific Utilities 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#region

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
    

def shift_scinot_fmt_text(axis='x', dx=0, dy=0, color="#999999", ax=None):
    """Adjusts position of 10^K in power/scientific notation format. Typically
    set using format_ticks() or scale_ax()"""

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


def translate(points, origin=(0,0)):
    """
    :param points:  array of shape (N, 2)
    :param origin:  array or tuple; coordinates of new origin

    :return translated:     array of shape (N, 2) of translated points
    """
    translation = np.asarray(origin)
    if np.max(np.abs(translation)) == 0:
        return points

    # validation for points
    message = """Points should be of shape (N, 2)"""
    if points.shape[1] != 2:
        raise ValueError(display_message(message))
    
    translated = points - translation

    return translated


def get_colormap(cmap):
    """Resolves cmap input into an mpl.colors.Colormap"""
    cmap = check_if_null(cmap, "viridis")

    if isinstance(cmap, mpl.colors.Colormap):
        return cmap
    elif isinstance(cmap, str):
        return mpl.colormaps[cmap]
    else:
        message = f"""Expected a string or matplotlib.colors.Colormap, got 
        {type(cmap)}"""
        raise TypeError(display_message(message))


def get_colornorm(cnorm):
    """Resolves cnorm input into an mpl.colors.Normalize"""
    cnorm = check_if_null(cnorm, [0, 1])

    if isinstance(cnorm, mpl.colors.Normalize):
        return cnorm
    
    else:
        try:
            cnorm = np.asarray(cnorm)
        except Exception:
            message = """colornorm must be None, a Normalize, or a (vmin, vmax)
            value."""
            raise TypeError(display_message(message))

        if len(cnorm) == 1 and cnorm > 0:
            message = """WARNING: Only 1 value provided for colornorm. Using
            this as vmax value."""
            return mpl.colors.Normalize(vmin=0, vmax=cnorm)
        elif len(cnorm) >= 2:
            if len(cnorm) > 2:
                message = """WARNING: More than 2 values provided for colornorm. 
                Using only the first two."""
                print(display_message(message))
            return mpl.colors.Normalize(vmin=cnorm[0], vmax=cnorm[1])
        
    

    
    

def cmap_rgba(vals, cmap:Optional[Union[str, mpl.colors.Colormap]]=None, 
              cnorm=None, new_a:float=1.):
    """Gets RGBA value for a given value in a colormap with a norm

    :param vals: float|array-like   shape (N,) val(s) to map into cmap
    :param cmap: str|Colormap|None  colormap to use (default: None -> viridis)
    :param cnorm: array-like|Normalize|None     How to normalize the values
    :param new_a: float     new alpha value. Float in [0, 1]

    :return rgbas: list(tuple)      RGBA tuples corresponding to the values in
                                    the colormap
    """
    cmap = get_colormap(cmap)
    cnorm = get_colornorm(cnorm)

    vals = np.atleast_1d(vals)
    normed = cnorm(vals)
    rgbas = cmap(normed)   # (N, 4)

    if check_if_null(new_a, False):
        message = f"""`new_a`={new_a} not between (0, 1). Clipping to range."""
        if new_a < 0 or new_a > 1:
            print(display_message(message))
        new_a = np.clip(new_a, 0., 1.)

        rgbas[:, 3] = new_a
    
    return rgbas


def lighten_color(color, amount=0.5):
    """Lightens the given color by multiplying (1-luminosity) by the given 
    amount. Input can be matplotlib color string, hex string, or RGB tuple.
    Darkens if amount > 1. 

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    
    Directly from https://stackoverflow.com/questions/37765197/darken-or-lighten-a-color-in-matplotlib

    :return lightened color as RGB tuple
    """
    try:
        c = mpl.colors.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mpl.colors.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount*(1-c[1]))), c[2])


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#                            Basic Plotting Functions
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#region

@with_ax
def plot_hlines(ax=None, yvals=None, xmin=None, xmax=None,
                color=None, linestyle='--', linewidth=1.5,
                label=None, legend_dict=None, **kwargs):
    """Plot horizontal lines at specified y-values with optional labels
    **kwargs additional hline arguments (eg. alpha, zorder, etc.)
    """    
    xmin = check_if_null(xmin, 0)
    xmax = check_if_null(xmax, 1)

    yvals = format_data(yvals)

    lines = ax.hlines(yvals, xmin=xmin, xmax=xmax,
              color=color, linestyles=linestyle, linewidth=linewidth,
              label=label)
    
    # Add to legend if label is provided
    if check_if_null(label, False, True):
        legend_dict = check_if_null(legend_dict, {})
        legend_dict.update({label: lines})

    return ax, legend_dict

@with_ax
def plot_vlines(ax=None, xvals=None, ymin=None, ymax=None,
                color=None, linestyle='--', linewidth=1.5,
                label=None, legend_dict=None, **kwargs):
    """Plot vertical lines at specified x-values with optional labels
    
    **kwargs additional vline arguments (eg. alpha, zorder, etc.)

    :return ax:     the matplotlib axes the lines were on
    :return legend_dict:    dict [label -> lines] for creating a legend
    """
    ymin = check_if_null(ymin, 0)
    ymax = check_if_null(ymax, 1)

    xvals = format_data(xvals)

    lines = ax.vlines(xvals, ymin=ymin, ymax=ymax,
              color=color, linestyles=linestyle, linewidth=linewidth,
              label=label)
    
    # Add to legend if label is provided
    if check_if_null(label, False, True):
        legend_dict = check_if_null(legend_dict, {})
        legend_dict.update({label: lines})

    return ax, legend_dict


@with_ax
def plot_scatter(xs, ys, ax=None, color=None, marker="o", size=20, label=None,
                 legend_dict=None, **kwargs):
    """Plot a scatter plot
    :param xs:      size (N,)
    :param ys:      size (N,)
    :param color:   matplotlib color options
    :param size:    radii of points (default: 20)
    :param label:   str or sequence of str. If sequence, should be the same 
                    length number of markers.
    :param kwargs:  additional scatter arguments, see matplotlib docs
    """

    def pointwise(x, default, n=len(xs), multi=False):
        """Converts colors, markers, and sizes so each point has its own"""
        out_shape = (n,) if not multi else (n,) + np.shape(np.asarray(default))
        if check_if_null(x, True, False):
            out = np.full(out_shape, default)
        elif isinstance(x, tuple) or isinstance(x, list) \
            or isinstance(x, np.ndarray):
            out = np.asarray(x)
            if multi and (out.ndim == 2) and (out.shape[0] == 1):
                out = np.repeat(out, n, axis=0)
        else:
            out = np.full(out_shape, x)
        return out
    
    xs = pointwise(xs, None)
    ys = pointwise(ys, None)
    colors = pointwise(color, "C0", multi=True)
    sizes = pointwise(size, 20) ** 2
    markers = pointwise(marker, "o")
    u_marks = np.unique(markers)

    # validation for labels
    multilab = False
    if check_if_null(label, False, True):
        legend_dict = check_if_null(legend_dict, {})
        message = """Number of labels should be equal to the number of 
        different markers if more than one"""
        if not isinstance(label, str):
            if len(label) < len(u_marks):
                raise ValueError(display_message(message))
            elif len(label) > len(u_marks):
                message = f"""{len(label)} labels provided but only 
                {len(u_marks)} markers. Only using the first {len(u_marks)}
                labels"""
                print(display_message(message))
            multilab = True

    for k, mark in enumerate(u_marks):
        mask = (markers == mark) 
        lab = label[k] if multilab else label
        points = ax.scatter(xs[mask],ys[mask], c=colors[mask], s=sizes[mask], 
                            marker=mark, label=lab, **kwargs)
        if check_if_null(lab, False):
            legend_dict.update({lab: points})
        
    return ax, legend_dict


@with_ax
def plot_SD(mean, lengths, directions, ax=None, n_stds:list=[1], edgecolor='k',
            ellipse:bool=True, label=None, legend_dict=None, **kwargs):
    """Plots arrows or ellipses to represent spread of data
    :param mean:
    :param lengths:
    :param directions:
    :param ax:
    :param n_stds:
    :param edgecolor:
    :param ellipse:
    :param kwargs:
    
    :return ax:
    :return var_plot:
    """
    mean = format_data(mean)


    # ellipse plotting
    if ellipse: 
        # angle of first axis
        angle = np.degrees(np.arctan2(directions[0][1], directions[0][0]))

        do_label = check_if_null(label, False, True)
        for s in n_stds:
            label = label if do_label else None
            width = 2 * s * lengths[0]
            height = 2 * s * lengths[1]

            patch = mpl.patches.Ellipse(xy=mean, width=width, height=height,
                                        angle=angle, edgecolor=edgecolor,
                                        facecolor='none', label=label, 
                                        **kwargs)
            
            ax.add_patch(patch)

            if do_label:
                legend_dict = check_if_null(legend_dict, {})
                legend_dict.update({label: patch})
                do_label = False

    # TODO add plotting arrows (when ellipse=False)

    return ax, legend_dict





#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#                              Specific Plot Types
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#region

bool_default_tick_positions = { # (UPPER, RIGHT) -> tick position strings
    (True, True): ["top", "right"],     # origin upper right
    (True, False): ["top", "left"],     # origin upper left
    (False, True): ["bottom", "right"], # origin lower right
    (False, False): ["bottom", "left"]  # origin lower left
}

@with_ax
def plot_similarity_matrix(sims, full_matrix:bool=False, 
                           lower_triangular:bool=False,
                           
                           ax=None, title=None, figsize=(10, 10),

                           axis_label=None, ticks=None, ticklabs=None,
                           xrot=0, yrot=0, tick_positions=None,

                           colormap=None, vmin=0, vmax=1,
                           mask_color='#cffcff', nan_color='#ffaffa',

                           do_colorbar=True, cbar_label="similarity",
                           cbar_ticks=None, cbar_ticklabs=None,
                           cbar_location=None, 
                           cbar_horizontal=True, cbar_pad_in=None,

                           savename=None, savedir_override=None, 
                           show_plot=True, standalone=False,

                           splits:Optional[Union[list, tuple]]=None, 
                           split_colors:Union[list, str]="red",

                           origin_upper:bool=True, origin_right:bool=True
                           ):
    """Plot similarity matrix heatmap with colorbar
    nan_color option is for if there are nan values in the matrix
    """
    # print("origin: ", ("upper" if origin_upper else "lower"), 
    #       ("right" if origin_right else "left"))
    # TODO better documentation
    # WISHLIST better arguments for this
    tick_positions = check_if_null(tick_positions, 
                    bool_default_tick_positions[origin_upper, origin_right])
    
    ax = scale_ax(figsize=figsize, ax=ax, xscale="linear", yscale="linear", 
             pwr_format=False)

    tri = np.tril if lower_triangular else np.triu  # choose triangular mask

    # Full matrix handling (show only bottom half)
    mask = np.zeros_like(sims, dtype=bool) if full_matrix \
           else tri(np.ones_like(sims, dtype=bool), k=0)
    
    colormap = get_colormap(check_if_null(colormap, mpl.cm.binary))
    colormap.set_bad(mask_color)

    # plotting the matrix
    outmap = ax.imshow(np.ma.array(sims, mask=mask), cmap=colormap,
                        vmin=vmin, vmax=vmax, interpolation="nearest")
    # (input for the colorbar later)
    
    # NaN value handling
    nan_sims = np.isnan(sims)
    nan_sims = np.ma.masked_where(nan_sims==False, nan_sims)
    nan_sims = np.ma.array(nan_sims, mask=mask)
    colormap_nan = mpl.colors.ListedColormap([nan_color, nan_color])
    
    # add the NaN values
    ax.imshow(nan_sims, aspect="auto", cmap=colormap_nan, vmin=0, vmax=1,
               origin="upper" if origin_upper else "lower",)
    
    # setting the origin settings + associated colorbar padding settings
    if origin_right:
        ax.invert_xaxis()
        fallback_cbar_pad = DEFAULT_CBAR_PAD*3.5 if not cbar_horizontal \
                                                 else DEFAULT_CBAR_PAD
        cbar_pad_in = check_if_null(cbar_pad_in, fallback_cbar_pad)

    # labels
    ax = plot_labels(ax=ax, title=title, xticks=ticks, xticklabs=ticklabs,
                yticks=ticks, yticklabs=ticklabs,
                xlab=axis_label, ylab=axis_label,
                xrot=xrot, yrot=yrot, tick_positions=tick_positions)


    # colorbar
    if do_colorbar:
        cbar_ticks = check_if_null(cbar_ticks, np.linspace(vmin, vmax, 5))
        plot_colorbar(outmap, cbar_ticks=cbar_ticks, 
                      cbar_ticklabs=cbar_ticklabs, cbar_label=cbar_label, 
                        labelpad=40, location_override=cbar_location,
                        horizontal=cbar_horizontal, 
                        pad_in=check_if_null(cbar_pad_in, DEFAULT_CBAR_PAD))
    
    # separation lines
    # TODO implement separation lines in plot for different runs etc
    if check_if_null(splits, False, True):
        # adjust all values to get correct placement
        split_coord_vals = format_data([k -0.5 for k in splits])
        minval, maxval = -0.5, sims.shape[0] - 0.5 

        if isinstance(split_colors, list): # one color per split
            message = """If `split_colors` is a list, it must match the length 
            of `splits`"""
            assert len(split_colors) == len(splits), display_message(message)
        # plot the split lines
        plot_vlines(ax=ax, xvals=split_coord_vals, ymin=minval, ymax=maxval,
                    color=split_colors, linestyle="--", linewidth=1.5)
        plot_hlines(ax=ax, yvals=split_coord_vals, xmin=minval, xmax=maxval,
                    color=split_colors, linestyle="--", linewidth=1.5)


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
    """"""
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


@with_ax
def plot_scatter_var(coords:np.ndarray, ax=None, origin=None,
            color_val=None, cmap=None, cnorm=None,
            marker="o", msize=20,

            do_mean:bool=True, do_var:bool=True, n_stds=[1,],

            var_a=0.5, mean_a=1, data_a=0.2,
            
            label=None, legend_dict=None,
            add_var_to_legend=False,
            add_mean_to_legend=True,
            add_data_to_legend=False
            
            ):
    """Plot a set of coordinates. Intended for use with generated MDS 
    coordinates to show mean and SD of different experiments. Coordinates 
    should be of shape (N, 2). Generally, one grouping of points

    :param coords: np.ndarray       shape (N, 2) the coordinates to plot.  
    :param ax: matplotlib axes      axes to place plot on
    :param origin:          new origin to translate the coordinates 
                            (default: None)
    :param color_val:       color for the points to be used with colormap
    :param cmap: matplotlib colormap    colormap used in conjunction with 
                                        color_val to color the points
    :param cnorm: matplotlib color Normalization      
    :param marker: str      marker shape for the points
    :param msize: int       n_pixels radius of the points

    :param do_mean: bool    whether to calculate and plot the mean of the 
                            coordinates (default: True)
    :param do_var: bool     whether to calculate and plot the variance/SD of 
                            the coordinates (default: True). `do_var=True` 
                            forces `do_mean=True`
    
    :param mean_a: float    float in [0,1] - alpha value of the mean point 
                            (default: 1)
    :param data_a: float    float in [0,1] - alpha value of the data points
                            (default: 0.2 if do_mean, otherwise 1)
    
    """
    data_a = check_if_null(data_a, 0.2) if do_mean else 1.
    # check the color
    if isinstance(color_val, float):
        color_val = cmap_rgba(color_val, cmap, cnorm, new_a=data_a)

        if do_mean:
            color_mean = color_val.copy()
            color_mean[:, 3] = check_if_null(mean_a, 1)
        if do_var:
            color_var = color_val.copy()
            color_var[:, :3] = lighten_color(color_var[:, :3], 0.3)
            color_var[:, 3] = check_if_null(var_a, 0.6)

    # TODO more type handling for the color val

    coords = translate(coords, origin=check_if_null(origin, (0,0))) # (N, 2)
 
    # plot the scatter
    ax, legend_dict = plot_scatter(coords[:,0], coords[:, 1], ax=ax, 
                                color=color_val, marker=marker, size=msize, 
                                label=label, legend_dict=legend_dict)
    
    # get mean and CIs
    def get_coord_axes(c):
        mean = np.mean(c, axis=0)  # get mean  (2,)

        centered = c - mean
        cov = (1/c.shape[0])*centered.T @ centered    # get centered cov (2, 2)

        vals, vecs = np.linalg.eigh(cov)
        vals, vecs = np.flip(vals), np.flip(vecs)

        lengths = [np.sqrt(k) for k in vals[:2]]    # get SD lengths 
        directions = [vecs[:, k] for k in (0,1)]    # get SD directions

        return mean, lengths, directions # lengths and directions len = 2
    
    if do_var:
        mean, lengths, directions = get_coord_axes(coords)

    elif do_mean:
        mean = np.mean(coords.T, axis=0)
        lengths, directions = None, None
    else:
        mean, lengths, directions = None, None, None
    
    # plot the mean
    if check_if_null(mean, False, True):
        print(mean)
        ax, legend_dict = plot_scatter([mean[0]],[mean[1]], ax=ax, 
                                               color=color_mean, marker=marker,
                                               size=msize, label=label,
                                               legend_dict=legend_dict)
    
    
    # add the arrows
    if check_if_null(lengths, False, True):
        # first axis
        var_label = check_if_null(label, None, label + " SD") if \
                    add_var_to_legend else None

        ax, legend_dict = plot_SD(mean=mean, lengths=lengths, 
                                  directions=directions, ax=ax, n_stds=n_stds,
                                  edgecolor=color_var, ellipse=True,
                                  label=var_label, legend_dict=legend_dict,
                                  linestyle="--")



    # TODO add variance label to dict
    
    

    
    return


