import argparse
import datetime
import os
import pickle as pkl
import regex as re
from datetime import date
from typing import Counter

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
from numpy.linalg import LinAlgError
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


def analyze(info, top_n_authors: int = 7):
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

    # Determine top N authors by message count and group the rest
    if author_stats.height > top_n_authors:
        author_stats_sorted = author_stats.sort("message_count", descending=True)
        top_authors = author_stats_sorted.head(top_n_authors)
        other_authors = author_stats_sorted.tail(author_stats.height - top_n_authors)

        # Aggregate "Other" category using Polars aggregation to preserve types
        other_row = other_authors.select(
            [
                pl.lit("Other").alias("author"),
                pl.col("message_count").sum().alias("message_count"),
                pl.col("message_share").sum().alias("message_share"),
                (
                    pl.col("total_message_length").sum() / pl.col("message_count").sum()
                ).alias("avg_message_length"),
                pl.col("total_message_length").sum().alias("total_message_length"),
                pl.col("time_in_chat").sum().alias("time_in_chat"),
                pl.col("sessions").sum().alias("sessions"),
                pl.col("active_days").sum().alias("active_days"),
                pl.col("avg_sessions_per_day").mean().alias("avg_sessions_per_day"),
            ]
        )

        author_stats_grouped = pl.concat([top_authors, other_row])
    else:
        author_stats_grouped = author_stats

    info["author_stats"] = author_stats  # Full stats for all authors
    info["author_stats_grouped"] = author_stats_grouped  # Top N + "Other"
    return info


def plot_charts(info):
    df = info["df"]
    chat_name = info["chat_name"]

    author_stats_grouped = info["author_stats_grouped"]
    author_unique = sorted(author_stats_grouped["author"].to_list())

    # colorscale = px.colors.sequential.Blues
    colors = px.colors.sample_colorscale(
        colorscale, np.linspace(0, 1, len(author_unique))
    )
    # map authors to colors for top n authors + other
    author_color_map = {author: color for author, color in zip(author_unique, colors)}
    author_color_map["Other"] = "#999999"  # Gray for "Other" category

    # Pie chart: sorted by message share
    author_stats_sorted = author_stats_grouped.sort("message_share", descending=True)
    pie_chart(
        values=author_stats_sorted["message_count"].to_list(),
        labels=author_stats_sorted["author"].to_list(),
        title="Share of Messages",
        color_discrete_map=author_color_map,
        color=author_stats_sorted["author"].to_list(),
        save_path=os.path.join(PLOT_DIR, f"{chat_name}_message_share_pie.png"),
    )

    # Bar chart: sorted by avg message length
    author_stats_sorted = author_stats_grouped.sort(
        "avg_message_length", descending=True
    )
    bar_chart(
        author_stats_sorted["author"].to_list(),
        author_stats_sorted["avg_message_length"].to_list(),
        title="Average Message Length",
        x_title=None,
        y_title="Characters",
        save_path=os.path.join(PLOT_DIR, f"{chat_name}_avg_message_length_bar.png"),
        color=author_stats_sorted["author"].to_list(),
        color_discrete_map=author_color_map,
        range_y=[
            author_stats_sorted["avg_message_length"].min() * 0.95,
            author_stats_sorted["avg_message_length"].max() * 1.05,
        ],
    )

    # bar chart of emoji usage
    whole_text = " ".join(df["message"].to_list())
    emojis = re.findall(r"\p{Extended_Pictographic}", whole_text)
    # count emojis
    emojis_counter = Counter(emojis)
    common_emojis = emojis_counter.most_common(15)
    e, ec = zip(*common_emojis)
    emoji_color_map = px.colors.sample_colorscale(colorscale, np.linspace(0, 1, len(e)))
    emoji_color_map = {e: color for e, color in zip(e, emoji_color_map)}
    bar_chart(
        e,
        ec,
        "Most Common Emojis in Chat",
        x_title=None,
        y_title="Count",
        save_path=os.path.join(PLOT_DIR, f"{chat_name}_common_emojis_bar.png"),
        color=e,
        color_discrete_map=emoji_color_map,
        width=800,
        height=500,
        number_format="d",  # Integer format for emoji counts
    )

    # heatmap of messages per day
    heatmap_chart(
        df,
        time_col="datetime",
        title="Messages per Day",
        save_path=os.path.join(PLOT_DIR, f"{chat_name}_messages_per_day_heatmap.png"),
        days_back=365,
        width=1400,
        height=300,
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
        width=800,
        height=300,
    )


def pie_chart(values, labels, title, save_path=None, **kwargs):
    """
    Create a pie chart with conditional labeling.
    Slices >10% show labels inside, smaller slices appear in legend only.

    Args:
        values: List/array of values to plot.
        labels: List/array of labels corresponding to values.
        title: Chart title.
        save_path: Optional path to save the figure.
        **kwargs: Additional arguments passed to px.pie (e.g., color_discrete_map).
    """
    # Calculate percentages to determine which labels to show inside
    total = sum(values)
    percentages = [v / total * 100 for v in values]

    # Create custom text based on percentage threshold
    text_labels = []
    show_legend_flags = []
    for i, (label, value, pct) in enumerate(zip(labels, values, percentages)):
        if pct > 10:
            # Show full info for large slices
            text_labels.append(f"{label}<br>{value}<br>{pct:.1f}%")
            show_legend_flags.append(False)
        else:
            # Show only percentage for small slices
            text_labels.append(f"{pct:.1f}% {label}")
            show_legend_flags.append(True)

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
    fig.add_annotation(
        dict(
            text=f"Total<br><b>{total}</b>", x=0.5, y=0.5, font_size=18, showarrow=False
        )
    )

    # Update text display
    fig.update_traces(
        text=text_labels,
        textinfo="text",
        textposition="inside",
    )

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


def bar_chart(
    labels,
    values,
    title,
    x_title,
    y_title,
    width=512,
    height=512,
    save_path=None,
    number_format=".0f",
    **kwargs,
):
    """
    Create a bar chart.

    Args:
        labels: List/array of x-axis labels (categories).
        values: List/array of y-axis values (numeric).
        title: Chart title.
        x_title: X-axis label.
        y_title: Y-axis label.
        width: Chart width in pixels.
        height: Chart height in pixels.
        save_path: Optional path to save the figure.
        number_format: Format string for numbers on bars (default ".0f" for integers).
                       Use ".1f" for one decimal, "d" for integers, etc.
        **kwargs: Additional arguments passed to px.bar (e.g., color_discrete_map).
    """
    fig = px.bar(
        x=labels, y=values, title=title, width=width, height=height, **(kwargs)
    )
    fig.update_layout(
        plot_bgcolor="white",
        xaxis_title=x_title,
        yaxis_title=y_title,
        showlegend=False,
        autosize=True,
        margin=dict(t=60, b=50, l=65, r=20),
        title_x=0.5,
        title_y=0.95,
        font=text_font,
        title_font=title_font,
    )
    # display count on top of bars with specified format
    fig.update_traces(texttemplate=f"%{{y:{number_format}}}", textposition="outside")
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

    # Reindex to ensure all 7 days are present (Polars weekday: 1=Mon, 7=Sun)
    heatmap_pivot = heatmap_pivot.reindex(range(1, 8), fill_value=0)

    # Day labels
    day_labels = ["Mon  ", "Tue  ", "Wed  ", "Thu  ", "Fri  ", "Sat  ", "Sun  "]

    # Plot
    fig = px.imshow(
        heatmap_pivot,
        labels=dict(x=xaxis_name, y=yaxis_name, color="Anzahl"),
        x=heatmap_pivot.columns,
        y=day_labels,
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


import numpy as np
import plotly.graph_objects as go
import polars as pl
from numpy.linalg import LinAlgError
from scipy.stats import gaussian_kde


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
    min_var: float = 1e-3,
):
    """
    Plot KDE distributions of message frequency per hour of day per author.
    If more than 3 authors exist, only the top 2 and the overall average are plotted.

    Args:
        df: Polars DataFrame containing time_col and author_col.
        time_col: Column name containing datetime values.
        author_col: Column name containing author identifiers.
        title: Plot title.
        save_path: Optional output path (e.g. 'plots/hourly_dist.svg').
        text_font: Font dict for axis/legend text.
        title_font: Font dict for the title.
        author_color_map: Optional dict mapping authors to colors (rgb or hex).
        width, height: Figure dimensions in pixels.
        min_var: Minimum variance threshold for KDE stability.
    """
    # Compute hour-of-day as float
    hourly_df = df.with_columns(
        (pl.col(time_col).dt.hour() + pl.col(time_col).dt.minute() / 60.0).alias(
            "hour_of_day"
        )
    )

    # Count messages per author and sort
    author_counts = (
        hourly_df.group_by(author_col)
        .len()
        .sort("len", descending=True)
        .to_dict(as_series=False)
    )
    authors = author_counts[author_col]
    counts = author_counts["len"]

    # Select top 4 + optionally average
    if len(authors) > 5:
        authors = authors[:4]
        plot_average = True
    else:
        plot_average = False

    fig = go.Figure()
    x = np.linspace(0, 24, 200)

    # Plot per-author KDEs
    for author in authors:
        hours = hourly_df.filter(pl.col(author_col) == author)["hour_of_day"].to_numpy()
        if len(hours) < 2 or np.var(hours) < min_var:
            continue

        try:
            kde = gaussian_kde(hours, bw_method="scott")
            y = kde(x)
        except LinAlgError:
            continue

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
                name=f"{author}",
                line=dict(width=2, color=color),
                fill="tozeroy",
                fillcolor=fill_color,
            )
        )

    # Add overall average KDE
    if plot_average:
        all_hours = hourly_df["hour_of_day"].to_numpy()
        if len(all_hours) >= 2 and np.var(all_hours) >= min_var:
            try:
                kde_avg = gaussian_kde(all_hours, bw_method="scott")
                y_avg = kde_avg(x)
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=y_avg,
                        mode="lines",
                        name="Average",
                        line=dict(width=3, color="black", dash="dot"),
                    )
                )
            except LinAlgError:
                pass

    # Layout styling
    fig.update_layout(
        title=title,
        xaxis_title="Hour of the Day",
        yaxis_title="Density",
        template="plotly_white",
        width=width,
        height=height,
        plot_bgcolor="white",
        xaxis=dict(showgrid=False, zeroline=False, tick0=0, dtick=2),
        yaxis=dict(visible=False),
        margin=dict(t=70, b=60, l=25, r=25),
        font=text_font or dict(family="Roboto", size=14, color="#333"),
        title_font=title_font or dict(family="Roboto Black", size=22, color="#333"),
        title_x=0.5,
        legend=dict(bgcolor="rgba(255,255,255,0.8)", x=0, y=1),
    )

    # Save or show
    if save_path:
        fig.write_image(save_path)
    else:
        fig.show()

    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess chat data")
    parser.add_argument("input_file", help="Path to the input chat file")
    parser.add_argument(
        "--include_metaAI", action="store_true", help="Include messages from Meta AI"
    )
    parser.add_argument(
        "--top_n_authors",
        type=int,
        default=10,
        help="Number of top authors to show individually (default: 10)",
    )
    args = parser.parse_args()

    info = preprocess(args.input_file, args.include_metaAI)
    info = analyze(info, top_n_authors=args.top_n_authors)
    plot_charts(info)

    pkl.dump(info, open(f"{args.input_file.replace('.txt', '')}_analyzed.pkl", "wb"))
