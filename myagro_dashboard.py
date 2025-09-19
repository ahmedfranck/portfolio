"""
myagro_dashboard.py
====================

This script provides a self‑contained Dash application that illustrates how
interactive dashboards can be used to empower myAgro’s agriculture field teams.
The dashboard demonstrates a few key concepts relevant to myAgro’s
operations: a sales pipeline funnel, payment trends over time and payment
behaviour by day of week.  Although the data in this example is randomly
generated, it follows the same structure described in myAgro’s FY22 Q1
report, where real‑time dashboards track farmer engagement from the initial
lead to the final enrolment stage【449023599194199†L103-L114】.  Field teams can
filter the view by region and date range to focus on the segments of
interest.  The code is written so it can easily be adapted to live data
sources (e.g., a database or API).

To run the dashboard locally:
    1. Install the dependencies listed in ``requirements.txt`` via pip or
       another package manager.
    2. Execute ``python myagro_dashboard.py`` in a terminal.
    3. Navigate to ``http://localhost:8050`` in your web browser.

Note: When adapting this dashboard for production, replace the synthetic
datasets with real data queries and remove the debugging flags.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px

import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output


def generate_sample_data(seed: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate synthetic pipeline and payment datasets.

    The pipeline dataset tracks the number of farmers at each stage of the
    sales funnel (Leads → Prospects → Registered → Enrolled) across dates
    and regions.  The payment dataset records the number of payments made
    each day per region and labels each record with the day of the week.

    Args:
        seed: Optional random seed for reproducibility.

    Returns:
        A tuple containing the pipeline DataFrame and the payment DataFrame.
    """
    rng = np.random.default_rng(seed)
    regions = ["Senegal", "Mali", "Tanzania"]
    stages = ["Leads", "Prospects", "Registered", "Enrolled"]

    # Create weekly dates for pipeline tracking from the start of the year
    # to today's date.  This mirrors how myAgro reviews weekly trends
    # across its network.【449023599194199†L103-L114】
    date_range_pipeline = pd.date_range(start="2025-01-01", end="2025-09-19", freq="W")
    pipeline_records: list[dict[str, object]] = []
    for dt in date_range_pipeline:
        for region in regions:
            # Start with a baseline number of leads and apply successive drop‑off
            # rates to simulate conversion through the funnel.
            base = rng.integers(low=1000, high=5000)
            conversions = {
                "Leads": base,
                # 40–80% convert from leads to prospects
                "Prospects": int(base * rng.uniform(0.4, 0.8)),
                # 20–60% convert from leads to registered
                "Registered": int(base * rng.uniform(0.2, 0.6)),
                # 10–40% convert from leads to enrolled
                "Enrolled": int(base * rng.uniform(0.1, 0.4)),
            }
            for stage, count in conversions.items():
                pipeline_records.append({
                    "date": dt,
                    "region": region,
                    "stage": stage,
                    "count": count,
                })
    pipeline_df = pd.DataFrame(pipeline_records)

    # Create daily dates for payment tracking.  Payments spike around key
    # agricultural income periods, similar to the pattern described by
    # myAgro’s heat‑map analysis【236227889843008†L26-L42】.
    date_range_payments = pd.date_range(start="2025-01-01", end="2025-09-19", freq="D")
    payment_records: list[dict[str, object]] = []
    for dt in date_range_payments:
        weekday = dt.day_name()
        for region in regions:
            # Base payment volume – weekday and seasonality effects included.
            # Higher payments in March and mid‑week to mimic real patterns.
            month_factor = 1.5 if dt.month == 3 else 1.0
            weekday_factor = 1.3 if weekday in ["Thursday", "Friday"] else 1.0
            base_payments = rng.poisson(lam=120)
            payments = int(base_payments * month_factor * weekday_factor)
            payment_records.append({
                "date": dt,
                "region": region,
                "payments": payments,
                "weekday": weekday,
            })
    payment_df = pd.DataFrame(payment_records)
    return pipeline_df, payment_df


def create_dashboard(app: dash.Dash, pipeline_df: pd.DataFrame, payment_df: pd.DataFrame) -> None:
    """Configure the Dash app layout and callbacks.

    The layout includes dropdown filters for region and date range, along with
    three main graphs: a funnel chart summarising the sales pipeline by stage,
    a line chart showing payment trends over time, and a bar chart comparing
    payments by day of the week.  Each graph updates interactively when the
    user changes the filters.

    Args:
        app: The Dash application instance.
        pipeline_df: The pipeline dataset.
        payment_df: The payment dataset.
    """
    regions = sorted(pipeline_df["region"].unique())
    min_date = pipeline_df["date"].min()
    max_date = pipeline_df["date"].max()

    app.layout = html.Div([
        html.H1(
            "myAgro Field Performance Dashboard",
            style={"textAlign": "center", "marginBottom": "20px"},
        ),
        html.Div([
            html.Div([
                html.Label("Select region:"),
                dcc.Dropdown(
                    options=[{"label": "All", "value": "All"}] + [
                        {"label": r, "value": r} for r in regions
                    ],
                    value="All",
                    id="region-dropdown",
                    clearable=False,
                ),
            ], style={"width": "32%", "display": "inline-block", "verticalAlign": "top"}),
            html.Div([
                html.Label("Select date range:"),
                dcc.DatePickerRange(
                    id="date-picker",
                    min_date_allowed=min_date,
                    max_date_allowed=max_date,
                    start_date=min_date,
                    end_date=max_date,
                    display_format="Y-MM-DD",
                ),
            ], style={"width": "48%", "display": "inline-block", "marginLeft": "20px"}),
        ], style={"marginBottom": "30px"}),
        html.Div([
            dcc.Graph(id="pipeline-funnel", style={"height": "400px"}),
        ], style={"marginBottom": "40px"}),
        html.Div([
            dcc.Graph(id="payments-line", style={"height": "350px"}),
            dcc.Graph(id="payments-weekday", style={"height": "350px"}),
        ], style={"display": "flex", "flexWrap": "wrap"}),
        html.Div([
            html.P(
                [
                    "This dashboard uses synthetic data that follows the same structure described in myAgro’s FY22 Q1 report, where field teams track farmers through a sales pipeline and monitor payments over time. Learn more about how myAgro uses data to drive decisions in their", 
                    html.A("report", href="https://www.myagro.org/wp-content/uploads/2021/11/fy22-q1-report_myagro.pdf", target="_blank"),
                    "."
                ],
                style={"fontSize": "12px", "color": "#666", "marginTop": "40px"},
            ),
        ]),
    ], style={"maxWidth": "1200px", "margin": "auto", "padding": "20px"})

    # Callback to update graphs based on region and date range filters
    @app.callback(
        [
            Output("pipeline-funnel", "figure"),
            Output("payments-line", "figure"),
            Output("payments-weekday", "figure"),
        ],
        [
            Input("region-dropdown", "value"),
            Input("date-picker", "start_date"),
            Input("date-picker", "end_date"),
        ],
    )
    def update_graphs(selected_region: str, start_date: str, end_date: str):
        # Filter pipeline data
        if selected_region and selected_region != "All":
            pipeline_filtered = pipeline_df[
                (pipeline_df["region"] == selected_region)
                & (pipeline_df["date"] >= pd.to_datetime(start_date))
                & (pipeline_df["date"] <= pd.to_datetime(end_date))
            ]
        else:
            pipeline_filtered = pipeline_df[
                (pipeline_df["date"] >= pd.to_datetime(start_date))
                & (pipeline_df["date"] <= pd.to_datetime(end_date))
            ]
        # Aggregate by stage
        pipeline_summary = (
            pipeline_filtered.groupby("stage")
            ["count"]
            .sum()
            .reindex(["Leads", "Prospects", "Registered", "Enrolled"])
        )
        funnel_fig = px.bar(
            x=pipeline_summary.values,
            y=pipeline_summary.index,
            orientation="h",
            labels={"x": "Number of farmers", "y": "Stage"},
            title="Sales Funnel Counts",
            color=pipeline_summary.index,
            color_discrete_sequence=px.colors.qualitative.Plotly,
        )
        funnel_fig.update_layout(
            yaxis=dict(categoryorder="total ascending"),
            showlegend=False,
            margin=dict(l=100, r=20, t=40, b=40),
        )

        # Filter payment data
        if selected_region and selected_region != "All":
            payment_filtered = payment_df[
                (payment_df["region"] == selected_region)
                & (payment_df["date"] >= pd.to_datetime(start_date))
                & (payment_df["date"] <= pd.to_datetime(end_date))
            ]
        else:
            payment_filtered = payment_df[
                (payment_df["date"] >= pd.to_datetime(start_date))
                & (payment_df["date"] <= pd.to_datetime(end_date))
            ]
        # Aggregate payments by date
        payments_trend = payment_filtered.groupby("date")["payments"].sum().reset_index()
        line_fig = px.line(
            payments_trend,
            x="date",
            y="payments",
            labels={"date": "Date", "payments": "Number of payments"},
            title="Payments Over Time",
        )
        line_fig.update_layout(margin=dict(l=40, r=20, t=40, b=40))

        # Aggregate payments by weekday
        payments_by_weekday = (
            payment_filtered.groupby("weekday")["payments"].sum().reindex(
                ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            )
        )
        weekday_fig = px.bar(
            x=payments_by_weekday.index,
            y=payments_by_weekday.values,
            labels={"x": "Day of Week", "y": "Number of payments"},
            title="Payments by Day of Week",
            color=payments_by_weekday.index,
            color_discrete_sequence=px.colors.qualitative.Plotly,
        )
        weekday_fig.update_layout(showlegend=False, margin=dict(l=40, r=20, t=40, b=40))

        return funnel_fig, line_fig, weekday_fig


def main() -> None:
    """Entry point for running the dashboard application."""
    pipeline_df, payment_df = generate_sample_data(seed=42)
    app = dash.Dash(__name__)
    app.title = "myAgro Field Dashboard"
    create_dashboard(app, pipeline_df, payment_df)
    # Expose the server for gunicorn or other WSGI servers
    server = app.server
    app.run_server(debug=False, host="0.0.0.0", port=8050)


if __name__ == "__main__":
    main()