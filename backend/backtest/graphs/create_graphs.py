import plotly.graph_objs as go
from typing import Dict


def create_figure(data: [], title: str, axis_labels: []) -> Dict[str, any]:
    """
    Create a Plotly figure object.

    Parameters:
    - data: The data for the figure.
    - title: The title of the figure.
    - axis_labels: The labels of the figure.

    Returns:
    - A go.Scatter object ready to be used in a Plotly figure.
    """
    figure = {
        "data": data,
        "layout": go.Layout(
            title=title,
            xaxis={"title": axis_labels[0]},
            yaxis={"title": axis_labels[1]},
            hovermode="closest",
        ),
    }

    return figure


def create_scatter_line(x_data, y_data, trace_name, mode="lines"):
    """
    Create a Plotly Scatter trace object.

    Parameters:
    - x_data: The data for the x-axis.
    - y_data: The data for the y-axis.
    - trace_name: The name of the trace, to appear in the legend.
    - mode: The mode for the plot, e.g., 'lines', 'markers', 'lines+markers'.

    Returns:
    - A go.Scatter object ready to be used in a Plotly figure.
    """
    trace = go.Scatter(x=x_data, y=y_data, mode=mode, name=trace_name)
    return trace


def create_pie_chart(labels, values, chart_name):
    """
    Create a Plotly Pie chart object.

    Parameters:
    - labels: The labels for the pie chart sectors.
    - values: The values corresponding to the labels.
    - chart_name: The name of the chart, to appear in the title.

    Returns:
    - A go.Figure object representing a pie chart.
    """
    trace = go.Pie(labels=labels, values=values, name=chart_name)
    layout = go.Layout(title=chart_name)
    fig = go.Figure(data=[trace], layout=layout)
    return fig
