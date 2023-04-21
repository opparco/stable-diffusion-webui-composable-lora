from dataclasses import dataclass
from typing import Tuple, List, Union
import io
import matplotlib
matplotlib.use('Agg')
import pandas as pd
from pandas.plotting._matplotlib.style import get_standard_colors
from PIL import Image

@dataclass
class YAxis:
    name: str
    columns: List[str]

@dataclass
class PlotDefinition:
    title: str
    x_axis: str
    y_axis: List[YAxis]

def plot_lora_weight(lora_weights, lora_names):
    data = pd.DataFrame(lora_weights, columns=lora_names)
    ax = data.plot()
    ax.set_xlabel("Steps")
    ax.set_ylabel("LoRA weight")
    ax.set_title("LoRA weight in all steps")
    ax.legend(loc=0)
    result_image = fig2img(ax)
    matplotlib.pyplot.close(ax.figure)
    del ax # RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`). Consider using `matplotlib.pyplot.close()`.
    return result_image

def fig2img(fig):
    buf = io.BytesIO()
    fig.figure.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img


def plot_graph(
        data: pd.DataFrame,
        plot_definition: PlotDefinition,
        spacing: float = 0.1,
):
    colors = get_standard_colors(num_colors=(len(plot_definition.y_axis) + 7))
    loss_color = colors[0]
    avg_colors = colors[1:]
    for i, yi in enumerate(plot_definition.y_axis):
        if i == 0:
            ax = data.plot(
                x=plot_definition.x_axis,
                y=yi.columns,
                title=plot_definition.title,
                color=[loss_color] * len(yi.columns)
            )
            ax.set_ylabel(ylabel=yi.name)

        else:
            # Multiple y-axes
            ax_new = ax.twinx()
            ax_new.spines["right"].set_position(("axes", 1 + spacing * (i - 1)))
            data.plot(
                ax=ax_new,
                x=plot_definition.x_axis,
                y=yi.columns,
                color=[avg_colors[yl] for yl in range(len(yi.columns))]
            )
            ax_new.set_ylabel(ylabel=yi.name)

    ax.legend(loc=0)

    return ax