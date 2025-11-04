import argparse
import datetime
import os
import pickle as pkl
import re
from datetime import date

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
from scipy.stats import gaussian_kde

from preprocess import preprocess

# Define custom BuGy (blue → gray)
colorscale = [
    "#053061",  # dark navy
    "#2166ac",  # medium blue
    "#4393c3",  # light blue
    "#92c5de",  # very light blue
    # "#d1e5f0",  # pale blue
    # "#f7f7f7",  # very pale blue
    "#bababa",  # mediumgray
    "#878787",  # dark gray
    "#4d4d4d",  # very dark gray
    "#1c1c1c",  # near black
]

text_font = dict(family="Roboto", size=14, color="#333")
title_font = dict(family="Roboto Black", size=22, color="#333")

PLOT_DIR = "plots/"
os.makedirs(PLOT_DIR, exist_ok=True)


def analyze(info):
    df = info["df"]

    # overall analyses
    total_messages = df.height
    total_length = df["message_length"].sum()
    total_days = (df["datetime"].max().date() - df["datetime"].min().date()).days + 1
    author_unique = sorted(df["author"].unique().to_list())

    info["total_messages"] = total_messages
    info["total_length"] = total_length
    info["total_days"] = total_days
    info["author_unique"] = author_unique
    info["min_date"] = df["datetime"].min().date()
    info["max_date"] = df["datetime"].max().date()

    # per author analyses
    author_stats = (
        df.with_columns(time_diff=pl.col("datetime").diff().over("author"))
        .group_by("author")
        .agg(
            pl.col("message").len().alias("message_count"),
            (pl.col("message").len() / df.height * 100).alias("message_share"),
            pl.col("message_length").mean().alias("avg_message_length"),
            pl.col("message_length").sum().alias("total_message_length"),
            # total time in chat: sum of diffs between messages < 2 minutes
            pl.when(pl.col("time_diff") < datetime.timedelta(minutes=2))
            .then(pl.col("time_diff"))
            .sum()
            .alias("time_in_chat"),
            # number of sessions: count of diffs between own messages > 15 minutes
            pl.when(pl.col("time_diff") > datetime.timedelta(minutes=15))
            .then(1)
            .sum()
            .alias("sessions"),
            # total active days per author
            (pl.col("datetime").dt.date().n_unique()).alias("active_days"),
        )
        .with_columns((pl.col("sessions") / total_days).alias("avg_sessions_per_day"))
    )
    info["author_stats"] = author_stats
    return info


def plot_charts(info):
    df = info["df"]
    chat_name = info["chat_name"]

    author_stats = info["author_stats"]
    author_unique = info["author_unique"]

    # colorscale = px.colors.sequential.Blues
    colors = px.colors.sample_colorscale(
        colorscale, np.linspace(0, 1, len(author_unique))
    )
    author_color_map = {author: color for author, color in zip(author_unique, colors)}

    # pie chart of shares in chat
    author_stats_tmp = author_stats.sort("message_share", descending=True)
    pie_chart(
        author_stats_tmp["message_count"],
        author_stats_tmp["author"],
        "Share of Messages per Author",
        color_discrete_map=author_color_map,
        color=author_stats_tmp["author"],
        save_path=os.path.join(PLOT_DIR, f"{chat_name}_message_share_pie.png"),
    )

    # bar chart of avg msg length per author
    author_stats_tmp = author_stats.sort("avg_message_length", descending=True)
    bar_chart(
        author_stats_tmp["author"],
        author_stats_tmp["avg_message_length"],
        "Durchschnittliche Nachrichtenlänge pro Autor",
        "Autor",
        "Zeichen",
        save_path=os.path.join(PLOT_DIR, f"{chat_name}_avg_message_length_bar.png"),
        color=author_stats_tmp["author"],
        color_discrete_map=author_color_map,
        range_y=[
            author_stats_tmp["avg_message_length"].max() * 0.8,
            author_stats_tmp["avg_message_length"].max() * 1.05,
        ],
    )

    # heatmap of messages per day
    heatmap_chart(
        df,
        time_col="datetime",
        title="Messages per Day",
        save_path=os.path.join(PLOT_DIR, f"{chat_name}_messages_per_day_heatmap.png"),
        days_back=365,
        width=1500,
        height=350,
    )

    # hourly kde chart
    hourly_kde_chart(
        df,
        time_col="datetime",
        author_col="author",
        title="Distribution of Messages over the Day",
        save_path=os.path.join(
            PLOT_DIR, f"{chat_name}_hourly_message_distribution.png"
        ),
        author_color_map=author_color_map,
        width=1500,
        height=350,
    )


def pie_chart(values, labels, title, save_path=None, **kwargs):
    fig = px.pie(
        names=labels,
        values=values,
        title=title,
        width=512,
        height=512,
        hole=0.3,
        **(kwargs),
    )
    # display total amount in center
    total = sum(values)
    fig.add_annotation(
        dict(
            text=f"Total<br><b>{total}</b>", x=0.5, y=0.5, font_size=18, showarrow=False
        )
    )
    fig.update_traces(
        textinfo="label+percent+value"
    )  # , insidetextorientation='horizontal', textposition='inside')
    fig.update_layout(
        showlegend=False,
        plot_bgcolor="white",
        autosize=True,
        margin=dict(t=65, b=30, l=30, r=30),
        title_x=0.5,
        title_y=0.95,
        font=text_font,
        title_font=title_font,
    )
    if save_path:
        fig.write_image(save_path, scale=2)
    else:
        fig.show()


def bar_chart(x, y, title, x_title, y_title, save_path=None, **kwargs):
    fig = px.bar(x=x, y=y, title=title, width=512, height=512, **(kwargs))
    fig.update_layout(
        plot_bgcolor="white",
        xaxis_title=x_title,
        yaxis_title=y_title,
        showlegend=False,
        autosize=True,
        margin=dict(t=50, b=60, l=65, r=20),
        title_x=0.5,
        title_y=0.95,
        font=text_font,
        title_font=title_font,
    )
    # display count on top of bars in x.x format
    fig.update_traces(texttemplate="%{y:.1f}", textposition="outside")
    if save_path:
        fig.write_image(save_path, scale=2)
    else:
        fig.show()


def heatmap_chart(
    df: pl.DataFrame,
    time_col: str,
    count_col: str | None = None,
    title: str = "Heatmap",
    xaxis_name: str = "",
    yaxis_name: str = "",
    save_path: str | None = None,
    days_back: int = 365,
    color_scale: str = "Blues",
    width: int = 1500,
    height: int = 350,
    text_font: dict | None = None,
    title_font: dict | None = None,
):
    """
    Creates a GitHub-style calendar heatmap over the last N days.

    Parameters
    ----------
    df : pl.DataFrame
        Input dataframe.
    time_col : str
        Name of the datetime column.
    count_col : str | None
        Optional name for the column to count occurrences of. If None, counts rows.
    title : str
        Plot title.
    xaxis_name, yaxis_name : str
        Axis labels.
    save_path : str | None
        Path to save the figure (optional).
    days_back : int
        Number of days to include (default 365).
    color_scale : str
        Continuous color map name.
    width, height : int
        Figure dimensions.
    text_font, title_font : dict | None
        Font dictionaries for customization.
    """

    cutoff_date = df.select(pl.col(time_col).max()).to_series()[0] - datetime.timedelta(
        days=days_back
    )
    df_filtered = df.filter(pl.col(time_col) >= cutoff_date)

    # Aggregate message counts per day
    messages_per_day = (
        df_filtered.with_columns(pl.col(time_col).dt.date().alias("date"))
        .group_by("date")
        .agg(
            pl.len().alias("message_count")
            if count_col is None
            else pl.count(count_col).alias("message_count")
        )
        .with_columns(
            pl.col("date").dt.weekday().alias("day_of_week"),
            pl.col("date").dt.strftime("%G-%V").alias("year_week"),
        )
    )

    # Convert week identifier to Monday of ISO week for continuous axis
    messages_per_day = messages_per_day.with_columns(
        pl.col("year_week")
        .map_elements(
            lambda s: date.fromisocalendar(int(s[:4]), int(s[5:]), 1),
            return_dtype=pl.Date,
        )
        .alias("week_start")
    )

    # Pivot for heatmap
    heatmap_data = messages_per_day.select(
        ["day_of_week", "week_start", "message_count"]
    ).to_pandas()
    heatmap_pivot = heatmap_data.pivot(
        index="day_of_week", columns="week_start", values="message_count"
    ).fillna(0)

    # Plot
    fig = px.imshow(
        heatmap_pivot,
        labels=dict(x=xaxis_name, y=yaxis_name, color="Anzahl"),
        x=heatmap_pivot.columns,
        y=["Mon  ", "Tue  ", "Wed  ", "Thu  ", "Fri  ", "Sat  ", "Sun  "],
        title=title,
        width=width,
        height=height,
        color_continuous_scale=color_scale,
    )
    fig.update_layout(
        plot_bgcolor="white",
        autosize=True,
        margin=dict(t=60, b=50, l=0, r=120),
        title_x=0.5,
        title_y=0.95,
        font=text_font or dict(family="Roboto", size=14, color="#333"),
        title_font=title_font or dict(family="Roboto Black", size=22, color="#333"),
        xaxis_title="",
        yaxis_title="",
    )
    fig.update_coloraxes(colorbar_title="")
    fig.update_xaxes(dtick="M1", tickformat="%b\n%Y")

    if save_path:
        fig.write_image(save_path, scale=2)
    else:
        fig.show()


def hourly_kde_chart(
    df: pl.DataFrame,
    time_col: str,
    author_col: str = "author",
    title: str = "Distribution of Messages over the Day",
    save_path: str | None = None,
    text_font: dict | None = None,
    title_font: dict | None = None,
    author_color_map: dict | None = None,
    width: int = 1500,
    height: int = 350,
):
    """
    Plot KDE distributions of message frequency per hour of day for each author.

    Args:
        df: Polars DataFrame containing at least time_col and author_col.
        time_col: Column name containing datetime values.
        author_col: Column name containing author identifiers.
        title: Plot title.
        save_path: Optional path to save the figure (e.g. 'plots/hourly_dist.svg').
        text_font: Font dictionary for axis and legend text.
        title_font: Font dictionary for the title.
        author_color_map: Optional dict mapping authors to colors (rgb or hex).
        width, height: Figure dimensions in pixels.
    """
    # Compute hour-of-day as continuous value
    hourly_df = df.with_columns(
        (pl.col(time_col).dt.hour() + pl.col(time_col).dt.minute() / 60.0).alias(
            "hour_of_day"
        )
    )

    # Collect unique authors
    authors = hourly_df.select(pl.col(author_col).unique()).to_series().to_list()

    # Prepare plot
    fig = go.Figure()
    x = np.linspace(0, 24, 200)

    # KDE per author
    for author in authors:
        hours = hourly_df.filter(pl.col(author_col) == author)["hour_of_day"].to_numpy()
        if len(hours) < 2:
            continue
        kde = gaussian_kde(hours, bw_method="scott")
        y = kde(x)

        color = (
            author_color_map.get(author, "rgb(100,100,100)")
            if author_color_map
            else "rgb(100,100,100)"
        )
        fill_color = color.replace("rgb", "rgba").replace(")", ",0.2)")

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name=author,
                line=dict(width=2, color=color),
                fill="tozeroy",
                fillcolor=fill_color,
            )
        )

    # Layout
    fig.update_layout(
        title=title,
        xaxis_title="Hour of the Day",
        yaxis_title="Density",
        template="plotly_white",
        width=width,
        height=height,
        plot_bgcolor="white",
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            tick0=0,
            dtick=2,
        ),
        yaxis=dict(visible=False),
        margin=dict(t=70, b=60, l=35, r=35),
        font=text_font or dict(family="Roboto", size=14, color="#333"),
        title_font=title_font or dict(family="Roboto Black", size=22, color="#333"),
        title_x=0.5,
        legend=dict(
            bgcolor="rgba(255,255,255,0.8)",
            x=0,
            y=1,
        ),
    )

    # Save or show
    if save_path:
        fig.write_image(save_path, scale=2)
    else:
        fig.show()

    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess chat data")
    parser.add_argument("input_file", help="Path to the input chat file")
    parser.add_argument(
        "--include_metaAI", action="store_true", help="Include messages from Meta AI"
    )
    args = parser.parse_args()

    info = preprocess(args.input_file, args.include_metaAI)
    info = analyze(info)
    plot_charts(info)

    pkl.dump(info, open(f"{args.input_file.replace('.txt', '')}_analyzed.pkl", "wb"))
