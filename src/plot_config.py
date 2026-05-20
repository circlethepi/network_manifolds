#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#                               Plotting Settings   
#                          Merrick Ohata 2026, JHU AMS         
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm # for settings

import os

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#                          matplotlib restyling settings
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#region
mpl.rcdefaults()

weight = "normal"
# add the font I like :-)
if "mnzk" in os.getcwd():
    fm.fontManager.addfont("/Users/mnzk/Library/Fonts/LibreFranklin-VariableFont_wght.ttf")
    weight = "medium"
if "weka" in os.getcwd():
    fm.fontManager.addfont("/weka/home/mohata1/scratchcpriebe1/MO/fonts/LibreFranklin[wght].ttf")

def set_paper_style():
    """Set matplotlib/seaborn defaults for camera-ready figures.

    Sized for ICLR/NeurIPS: textwidth ≈ 5.5 in.
    All font sizes are set so that they remain legible at 1:1 scale
    when included via \\includegraphics[width=\\textwidth].
    """
    sns.set_context("paper", font_scale=1.0)
    sns.set_style("whitegrid")

    plt.rcParams.update({
        "figure.figsize": (5.5, 2.0),
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "serif",
        "text.usetex": False,
        "font.size": 7,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "legend.fontsize": 4.5,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "lines.linewidth": 1.0,
        "lines.markersize": 3,
        "axes.linewidth": 0.5,
        "grid.linewidth": 0.3,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2,
        "ytick.major.size": 2,
        "legend.handlelength": 1.5,
        "legend.handletextpad": 0.4,
        "legend.columnspacing": 0.8,
        "legend.borderpad": 0.3,
        "legend.labelspacing": 0.2,
        "legend.framealpha": 0.8,
        "pdf.fonttype": 42,  # TrueType fonts in PDF
        "ps.fonttype": 42,
    })

def set_mnzk_style():
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



