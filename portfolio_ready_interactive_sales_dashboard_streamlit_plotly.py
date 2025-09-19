# app.py
# -------------------------
# Portfolio-Ready Interactive Sales Dashboard
# Primary UI: Streamlit + Plotly
# Fallback (no Streamlit): CLI that generates a standalone HTML report
# Author: Replace with your name & links
# -------------------------

import argparse
import io
import math
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

# =============================================================
# Core data utilities (UI-agnostic)
# =============================================================

REQUIRED_COLS = {
    "date": ["date", "order_date", "transaction_date"],
    "region": ["region", "area", "territory"],
    "product": ["product", "sku", "item"],
    "sales_rep": ["sales_rep", "salesperson", "rep", "agent"],
    "units": ["units", "qty", "quantity"],
    "revenue": ["revenue", "sales", "amount", "total"],
}

OPTIONAL_COLS = {
    "unit_price": ["unit_price", "price", "unitprice"],
}


@dataclass
class Filters:
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    regions: Optional[List[str]] = None
    products: Optional[List[str]] = None
    reps: Optional[List[str]] = None
    agg: str = "Month"  # Day | Week | Month
    top_n: int = 10
    breakdown_dim: str = "product"  # product | sales_rep | region


def generate_demo_data(seed: int = 7, days: int = 365) -> pd.DataFrame:
    """Generate a realistic demo dataset with trend and seasonality.
    Columns: date, region, product, sales_rep, units, unit_price, revenue
    """
    np.random.seed(seed)
    today = pd.Timestamp.today().normalize()
    dates = pd.date_range(today - pd.Timedelta(days=days - 1), today, freq="D")

    regions = ["North", "South", "East", "West"]
    products = ["Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot",
                "Golf", "Hotel", "India", "Juliet", "Kilo", "Lima"]
    reps = [f"Rep {i}" for i in range(1, 9)]

    rows = []
    for d in dates:
        base = 1000 + 200 * math.sin(2 * math.pi * (d.dayofyear / 365)) + 0.8 * (d - dates[0]).days
        day_multiplier = 0.8 if d.weekday() in (5, 6) else 1.0  # weekends softer
        day_total_orders = max(50, int(np.random.normal(base * day_multiplier / 30, 12)))
        for _ in range(day_total_orders):
            region = np.random.choice(regions, p=[0.28, 0.25, 0.25, 0.22])
            product = np.random.choice(products)
            rep = np.random.choice(reps)
            units = max(1, int(np.random.exponential(2)))
            unit_price = np.random.choice([9, 15, 25, 39, 59, 99], p=[0.08, 0.25, 0.28, 0.2, 0.14, 0.05])
            revenue = units * unit_price * np.random.uniform(0.95, 1.05)
            rows.append((d, region, product, rep, units, unit_price, round(revenue, 2)))

    df = pd.DataFrame(rows, columns=[
        "date", "region", "product", "sales_rep", "units", "unit_price", "revenue"
    ])
    return df


def auto_map_columns(df: pd.DataFrame) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    lower_cols = {c.lower(): c for c in df.columns}
    for key, cands in REQUIRED_COLS.items():
        for cand in cands:
            if cand in lower_cols:
                mapping[key] = lower_cols[cand]
                break
    for key, cands in OPTIONAL_COLS.items():
        for cand in cands:
            if cand in lower_cols:
                mapping[key] = lower_cols[cand]
                break
    return mapping


def coerce_schema(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    df = df.copy()
    missing = [k for k in REQUIRED_COLS.keys() if k not in mapping]
    if missing:
        raise ValueError(
            "Missing required columns: " + ", ".join(missing) +
            ". Map your columns or rename them in your CSV."
        )
    df = df.rename(columns={v: k for k, v in mapping.items()})

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.tz_localize(None)
    if df["date"].isna().any():
        raise ValueError("Some dates could not be parsed. Expected YYYY-MM-DD or similar.")

    for col in ["units", "revenue"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[col].isna().any():
            raise ValueError(f"Column '{col}' contains non-numeric values.")

    if "unit_price" in df.columns:
        df["unit_price"] = pd.to_numeric(df["unit_price"], errors="coerce")

    for col in ["region", "product", "sales_rep"]:
        df[col] = df[col].astype(str).str.strip()

    return df


def apply_filters(df: pd.DataFrame, flt: Filters) -> pd.DataFrame:
    mask = (
        (df["date"].dt.date >= flt.start_date.date()) &
        (df["date"].dt.date <= flt.end_date.date())
    )
    if flt.regions:
        mask &= df["region"].isin(flt.regions)
    if flt.products:
        mask &= df["product"].isin(flt.products)
    if flt.reps:
        mask &= df["sales_rep"].isin(flt.reps)

    out = df.loc[mask].copy()
    if out.empty:
        return out

    if flt.agg == "Day":
        out["period"] = out["date"].dt.to_period("D").dt.to_timestamp()
    elif flt.agg == "Week":
        out["period"] = out["date"].dt.to_period("W-MON").dt.start_time
    else:
        out["period"] = out["date"].dt.to_period("M").dt.to_timestamp()
    return out


def compute_kpis(df: pd.DataFrame, base_df: pd.DataFrame, flt: Filters) -> Tuple[float, float, float, float]:
    total_rev = float(df["revenue"].sum())
    total_units = float(df["units"].sum())
    aov = total_rev / max(1, len(df))

    span = (flt.end_date - flt.start_date).days + 1
    prev_start = flt.start_date - pd.Timedelta(days=span)
    prev_end = flt.start_date - pd.Timedelta(days=1)
    prev_flt = Filters(
        start_date=prev_start,
        end_date=prev_end,
        regions=flt.regions,
        products=flt.products,
        reps=flt.reps,
        agg=flt.agg,
        top_n=flt.top_n,
        breakdown_dim=flt.breakdown_dim,
    )
    prev_df = apply_filters(base_df, prev_flt)
    prev_rev = float(prev_df["revenue"].sum()) if not prev_df.empty else float("nan")
    growth = ((total_rev - prev_rev) / prev_rev * 100.0) if (prev_rev and not np.isnan(prev_rev) and prev_rev != 0) else float("nan")
    return total_rev, total_units, aov, growth


# =============================================================
# Chart builders (UI-agnostic)
# =============================================================

def fig_revenue_trend(filtered: pd.DataFrame) -> go.Figure:
    trend = filtered.groupby("period", as_index=False).agg(revenue=("revenue", "sum"), units=("units", "sum"))
    fig = px.line(trend, x="period", y="revenue", markers=True, title="Revenue Trend")
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), xaxis_title="", yaxis_title="Revenue")
    return fig


def fig_top_products(filtered: pd.DataFrame, top_n: int) -> go.Figure:
    by_product = (
        filtered.groupby("product", as_index=False)["revenue"].sum()
        .sort_values("revenue", ascending=False).head(top_n)
    )
    fig = px.bar(by_product, x="product", y="revenue", title=f"Top {len(by_product)} Products by Revenue")
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), xaxis_title="Product", yaxis_title="Revenue")
    return fig


def fig_breakdown(filtered: pd.DataFrame, dim: str) -> go.Figure:
    if dim not in {"product", "sales_rep", "region"}:
        dim = "product"
    by_dim = (
        filtered.groupby(dim, as_index=False)["revenue"].sum()
        .sort_values("revenue", ascending=False)
    )
    fig = px.treemap(by_dim, path=[dim], values="revenue", title=f"Revenue Breakdown by {dim.title()}")
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
    return fig


def fig_region_bar(filtered: pd.DataFrame) -> go.Figure:
    reg = (
        filtered.groupby(["region"], as_index=False)
        .agg(revenue=("revenue", "sum"), units=("units", "sum"))
        .sort_values("revenue", ascending=False)
    )
    fig = px.bar(reg, x="region", y="revenue", hover_data=["units"], title="Performance by Region")
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), xaxis_title="Region", yaxis_title="Revenue")
    return fig


def fig_heatmap(filtered: pd.DataFrame) -> go.Figure:
    pivot = filtered.copy()
    pivot["month"] = pivot["date"].dt.to_period("M").dt.to_timestamp()
    heat = pivot.pivot_table(index="region", columns="month", values="revenue", aggfunc="sum", fill_value=0)
    fig = px.imshow(heat, title="Revenue Heatmap by Region × Month", aspect="auto")
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
    return fig


# =============================================================
# Fallback HTML report (no Streamlit needed)
# =============================================================

def build_html_report(title: str, kpis: Tuple[float, float, float, float], figures: List[go.Figure]) -> str:
    total_rev, units, aov, growth = kpis
    snips = [pio.to_html(fig, full_html=False, include_plotlyjs="cdn") for fig in figures]
    growth_txt = f"{growth:+.1f}% vs prev period" if not (growth is None or np.isnan(growth)) else "n/a"
    html = f"""
    <!DOCTYPE html>
    <html lang='en'>
    <head>
      <meta charset='utf-8'>
      <meta name='viewport' content='width=device-width,initial-scale=1'>
      <title>{title}</title>
      <style>
        body {{ font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial; margin: 24px; }}
        .grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; }}
        .card {{ border: 1px solid #e5e7eb; border-radius: 12px; padding: 16px; box-shadow: 0 1px 2px rgba(0,0,0,0.06); }}
        .title {{ font-size: 28px; font-weight: 700; margin-bottom: 6px; }}
        .kpi-title {{ color: #6b7280; font-size: 13px; margin-bottom: 4px; }}
        .kpi-value {{ font-size: 22px; font-weight: 700; }}
        .section {{ margin-top: 24px; }}
      </style>
    </head>
    <body>
      <div class='title'>Sales Intelligence Dashboard (Report)</div>
      <div class='grid'>
        <div class='card'><div class='kpi-title'>Total Revenue</div><div class='kpi-value'>${total_rev:,.0f}</div></div>
        <div class='card'><div class='kpi-title'>Units Sold</div><div class='kpi-value'>{int(units):,}</div></div>
        <div class='card'><div class='kpi-title'>Avg Order Value</div><div class='kpi-value'>${aov:,.2f}</div></div>
        <div class='card'><div class='kpi-title'>Revenue Growth</div><div class='kpi-value'>{growth_txt}</div></div>
      </div>

      <div class='section card'>{snips[0]}</div>
      <div class='section card'>{snips[1]}</div>
      <div class='section card'>{snips[2]}</div>
      <div class='section card'>{snips[3]}</div>
      <div class='section card'>{snips[4]}</div>

      <p style='color:#6b7280;font-size:12px;margin-top:24px'>Generated without Streamlit (fallback mode). Install Streamlit to get the full interactive web app experience.</p>
    </body>
    </html>
    """
    return html


def save_html_report(html: str, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


# =============================================================
# Streamlit UI (only runs when Streamlit is available)
# =============================================================

def run_streamlit_app():
    import streamlit as st  # imported here to avoid ModuleNotFoundError at import-time in non-streamlit envs

    st.set_page_config(
        page_title="Sales Intelligence Dashboard",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Minimal CSS for KPI cards and clean look
    st.markdown(
        """
        <style>
        section[data-testid="stSidebar"] > div {padding-top: 1rem;}
        .kpi-card {background: var(--background-color); border: 1px solid rgba(49,51,63,0.2); border-radius: 16px; padding: 16px; height: 100%; box-shadow: 0 1px 2px rgba(0,0,0,0.06);}
        .kpi-title {font-size: 0.85rem; color: #6b7280; margin-bottom: 6px;}
        .kpi-value {font-size: 1.6rem; font-weight: 700;}
        .caption {color: #6b7280; font-size: 0.85rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    @st.cache_data(show_spinner=False)
    def _cached_demo():
        return generate_demo_data()

    with st.sidebar:
        st.title("📊 Sales Dashboard")
        st.caption("Portfolio-ready demo. Upload your CSV or use the synthetic dataset.")

        src = st.radio("Data Source", ["Demo data", "Upload CSV"], horizontal=True)
        if src == "Upload CSV":
            f = st.file_uploader("Upload CSV", type=["csv"]) 
            if f:
                raw = pd.read_csv(f)
                st.success(f"Loaded {len(raw):,} rows. Auto-detecting columns…")
                auto_map = auto_map_columns(raw)
                with st.expander("Map your columns (auto-detected)"):
                    mapping = {}
                    for key in REQUIRED_COLS.keys():
                        mapping[key] = st.selectbox(
                            f"{key}", options=[None] + list(raw.columns), index=( [None] + list(raw.columns) ).index(auto_map.get(key)) if auto_map.get(key) in raw.columns else 0
                        )
                    for key in OPTIONAL_COLS.keys():
                        mapping[key] = st.selectbox(
                            f"{key} (optional)", options=[None] + list(raw.columns), index=( [None] + list(raw.columns) ).index(auto_map.get(key)) if auto_map.get(key) in raw.columns else 0
                        )
                mapping = {k:v for k,v in mapping.items() if v}
                try:
                    df = coerce_schema(raw, mapping)
                except Exception as e:
                    st.error(str(e))
                    st.stop()
            else:
                st.info("Awaiting CSV upload… or switch to Demo data.")
                df = _cached_demo()
        else:
            df = _cached_demo()

        # Filters
        st.subheader("Filters")
        min_d, max_d = df["date"].min(), df["date"].max()
        date_range = st.date_input("Date range", value=(min_d.date(), max_d.date()), min_value=min_d.date(), max_value=max_d.date())
        if isinstance(date_range, tuple):
            start_date, end_date = date_range
        else:
            start_date, end_date = min_d.date(), date_range

        agg = st.select_slider("Aggregate by", options=["Day", "Week", "Month"], value="Month")
        regions = sorted(df["region"].dropna().unique().tolist())
        products = sorted(df["product"].dropna().unique().tolist())
        reps = sorted(df["sales_rep"].dropna().unique().tolist())
        sel_regions = st.multiselect("Region", regions, default=regions)
        sel_products = st.multiselect("Product", products, default=products)
        sel_reps = st.multiselect("Sales Rep", reps, default=reps)
        top_n = st.slider("Top N products", min_value=3, max_value=20, value=10)
        breakdown_dim = st.selectbox("Breakdown dimension", options=["product", "sales_rep", "region"], index=0)

    flt = Filters(
        start_date=pd.Timestamp(start_date), end_date=pd.Timestamp(end_date),
        regions=sel_regions, products=sel_products, reps=sel_reps,
        agg=agg, top_n=top_n, breakdown_dim=breakdown_dim
    )

    filtered = apply_filters(df, flt)
    if filtered.empty:
        st.warning("No data matches your filters. Try widening the date range or selections.")
        st.stop()

    total_rev, total_units, aov, growth = compute_kpis(filtered, df, flt)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown('<div class="kpi-card"><div class="kpi-title">Total Revenue</div>'
                    f'<div class="kpi-value">${total_rev:,.0f}</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="kpi-card"><div class="kpi-title">Units Sold</div>'
                    f'<div class="kpi-value">{int(total_units):,}</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="kpi-card"><div class="kpi-title">Avg Order Value</div>'
                    f'<div class="kpi-value">${aov:,.2f}</div></div>', unsafe_allow_html=True)
    with c4:
        delta = f"{growth:+.1f}% vs prev period" if not np.isnan(growth) else "n/a"
        st.markdown('<div class="kpi-card"><div class="kpi-title">Revenue Growth</div>'
                    f'<div class="kpi-value">{delta}</div></div>', unsafe_allow_html=True)

    left, right = st.columns((2, 1))

    fig_trend = fig_revenue_trend(filtered)
    left.plotly_chart(fig_trend, use_container_width=True)

    fig_top = fig_top_products(filtered, flt.top_n)
    right.plotly_chart(fig_top, use_container_width=True)

    fig_break = fig_breakdown(filtered, flt.breakdown_dim)
    st.plotly_chart(fig_break, use_container_width=True)

    fig_reg = fig_region_bar(filtered)
    st.plotly_chart(fig_reg, use_container_width=True)

    fig_ht = fig_heatmap(filtered)
    st.plotly_chart(fig_ht, use_container_width=True)

    st.subheader("Detailed Transactions")
    st.dataframe(
        filtered.sort_values("date", ascending=False),
        use_container_width=True,
        hide_index=True,
    )

    csv_bytes = filtered.to_csv(index=False).encode("utf-8")
    st.download_button("Download filtered CSV", data=csv_bytes, file_name="filtered_sales.csv", mime="text/csv")

    with st.expander("Export charts as PNG (optional)"):
        st.caption("This uses the 'kaleido' package. If unavailable, install it and rerun.")
        try:
            import kaleido  # noqa: F401
            buf1, buf2, buf3, buf4, buf5 = io.BytesIO(), io.BytesIO(), io.BytesIO(), io.BytesIO(), io.BytesIO()
            fig_trend.write_image(buf1, format="png", width=1400, height=700)
            fig_top.write_image(buf2, format="png", width=1000, height=700)
            fig_break.write_image(buf3, format="png", width=1000, height=700)
            fig_reg.write_image(buf4, format="png", width=1000, height=700)
            fig_ht.write_image(buf5, format="png", width=1000, height=700)
            st.download_button("Revenue Trend (PNG)", buf1.getvalue(), file_name="trend.png")
            st.download_button("Top Products (PNG)", buf2.getvalue(), file_name="top_products.png")
            st.download_button("Breakdown (PNG)", buf3.getvalue(), file_name="breakdown.png")
            st.download_button("Regions (PNG)", buf4.getvalue(), file_name="regions.png")
            st.download_button("Heatmap (PNG)", buf5.getvalue(), file_name="heatmap.png")
        except Exception:
            st.info("Install kaleido to enable image exports: pip install -U kaleido")

    with st.expander("About this dashboard"):
        st.markdown(
            """
            **Purpose.** A clean, professional sales analytics dashboard demonstrating filtering, KPI design, and interactive visuals.

            **Data model.** Expected columns: `date`, `region`, `product`, `sales_rep`, `units`, `revenue` (optional: `unit_price`). Use the sidebar mapper when uploading your own CSV.

            **Filters.** Date range, aggregation grain (Day/Week/Month), and multiselects for Region, Product, Sales Rep.

            **KPIs.** Total Revenue, Units Sold, Average Order Value (AOV), and Revenue Growth vs the previous equal-length period.

            **Visuals.** Revenue trend line; Top N products; revenue breakdown treemap by product / rep / region; region bar; monthly heatmap.

            **Exports.** Download filtered CSV; optional PNG exports using Kaleido.

            **Tech.** Streamlit + Plotly; Pandas for transforms; realistic synthetic data for demos.
            """
        )

    st.markdown("<div class='caption'>Made with ❤️ using Streamlit & Plotly • Replace with your name & links.</div>", unsafe_allow_html=True)


# =============================================================
# CLI fallback (no Streamlit): generate a standalone HTML report
# =============================================================

def run_cli():
    parser = argparse.ArgumentParser(description="Sales dashboard fallback (HTML report) if Streamlit is unavailable.")
    parser.add_argument("--csv", type=str, default=None, help="Path to CSV (if omitted, uses demo data)")
    parser.add_argument("--start", type=str, default=None, help="Start date YYYY-MM-DD (default: min date)")
    parser.add_argument("--end", type=str, default=None, help="End date YYYY-MM-DD (default: max date)")
    parser.add_argument("--regions", type=str, default=None, help="Comma-separated regions to include")
    parser.add_argument("--products", type=str, default=None, help="Comma-separated products to include")
    parser.add_argument("--reps", type=str, default=None, help="Comma-separated reps to include")
    parser.add_argument("--agg", type=str, default="Month", choices=["Day", "Week", "Month"], help="Aggregation granularity")
    parser.add_argument("--top-n", type=int, default=10, help="Top-N products")
    parser.add_argument("--breakdown", type=str, default="product", choices=["product", "sales_rep", "region"], help="Treemap breakdown dimension")
    parser.add_argument("--out", type=str, default="sales_dashboard_report.html", help="Output HTML report path")
    parser.add_argument("--save-csv", type=str, default=None, help="Optional path to save filtered CSV")
    parser.add_argument("--write-reqs", action="store_true", help="Write a requirements.txt next to app.py and exit")
    parser.add_argument("--run-tests", action="store_true", help="Run built-in tests and exit")

    args = parser.parse_args()

    if args.write_reqs:
        write_requirements("requirements.txt")
        print("[OK] requirements.txt written. Inspect and install with: pip install -r requirements.txt")
        return

    if args.run_tests:
        run_tests()
        return

    if args.csv and os.path.exists(args.csv):
        raw = pd.read_csv(args.csv)
        mapping = auto_map_columns(raw)
        df = coerce_schema(raw, mapping)
    else:
        df = generate_demo_data()

    min_d, max_d = df["date"].min(), df["date"].max()
    start_date = pd.Timestamp(args.start) if args.start else min_d
    end_date = pd.Timestamp(args.end) if args.end else max_d

    regions = args.regions.split(",") if args.regions else None
    products = args.products.split(",") if args.products else None
    reps = args.reps.split(",") if args.reps else None

    flt = Filters(start_date=start_date, end_date=end_date, regions=regions, products=products, reps=reps, agg=args.agg, top_n=args.top_n, breakdown_dim=args.breakdown)
    filtered = apply_filters(df, flt)
    if filtered.empty:
        print("[INFO] No data matches your filters. Try widening the date range or selections.")
        return

    kpis = compute_kpis(filtered, df, flt)
    figs = [
        fig_revenue_trend(filtered),
        fig_top_products(filtered, flt.top_n),
        fig_breakdown(filtered, flt.breakdown_dim),
        fig_region_bar(filtered),
        fig_heatmap(filtered),
    ]
    html = build_html_report("Sales Intelligence Dashboard (Report)", kpis, figs)
    save_html_report(html, args.out)
    if args.save_csv:
        filtered.to_csv(args.save_csv, index=False)
    print(f"[OK] Report written to: {args.out}")
    if args.save_csv:
        print(f"[OK] Filtered CSV saved to: {args.save_csv}")


# =============================================================
# Lightweight tests (do not require Streamlit)
# Run with: python app.py --run-tests
# =============================================================

def run_tests():
    import unittest

    class TestCore(unittest.TestCase):
        def setUp(self):
            self.df = generate_demo_data(days=60)
            self.min_d, self.max_d = self.df["date"].min(), self.df["date"].max()

        def test_columns_exist(self):
            cols = set(self.df.columns)
            for c in ["date", "region", "product", "sales_rep", "units", "revenue"]:
                self.assertIn(c, cols)

        def test_filtering(self):
            flt = Filters(start_date=self.min_d, end_date=self.min_d + pd.Timedelta(days=14))
            f = apply_filters(self.df, flt)
            self.assertFalse(f.empty)
            self.assertTrue((f["date"].dt.date >= flt.start_date.date()).all())
            self.assertTrue((f["date"].dt.date <= flt.end_date.date()).all())

        def test_kpis(self):
            flt = Filters(start_date=self.min_d, end_date=self.max_d, agg="Month")
            f = apply_filters(self.df, flt)
            k = compute_kpis(f, self.df, flt)
            self.assertEqual(len(k), 4)
            self.assertGreaterEqual(k[0], 0.0)
            self.assertGreaterEqual(k[1], 0.0)

        def test_html_report(self):
            flt = Filters(start_date=self.min_d, end_date=self.max_d)
            f = apply_filters(self.df, flt)
            k = compute_kpis(f, self.df, flt)
            figs = [fig_revenue_trend(f), fig_top_products(f, 5), fig_breakdown(f, "product"), fig_region_bar(f), fig_heatmap(f)]
            html = build_html_report("Test Report", k, figs)
            out = "_test_report.html"
            save_html_report(html, out)
            self.assertTrue(os.path.exists(out))
            os.remove(out)

        # --- Additional tests ---
        def test_auto_map_and_coerce(self):
            # Create a small DF with alternate column names
            raw = pd.DataFrame({
                "order_date": ["2025-01-01", "2025-01-02"],
                "area": ["North", "South"],
                "sku": ["A", "B"],
                "salesperson": ["Rep 1", "Rep 2"],
                "qty": [3, 2],
                "amount": [30.5, 20.0],
            })
            mapping = auto_map_columns(raw)
            coerced = coerce_schema(raw, mapping)
            for c in ["date", "region", "product", "sales_rep", "units", "revenue"]:
                self.assertIn(c, coerced.columns)
            self.assertTrue(np.issubdtype(coerced["date"].dtype, np.datetime64))

        def test_week_aggregation(self):
            flt = Filters(start_date=self.min_d, end_date=self.max_d, agg="Week")
            f = apply_filters(self.df, flt)
            self.assertIn("period", f.columns)
            # weekly periods should be Mondays
            self.assertTrue(all(pd.to_datetime(f["period"]).dt.weekday == 0))

        def test_breakdown_default(self):
            # invalid breakdown should default to product (handled inside fig)
            f = apply_filters(self.df, Filters(start_date=self.min_d, end_date=self.max_d))
            _ = fig_breakdown(f, "invalid_dimension")  # should not raise

    suite = unittest.TestLoader().loadTestsFromTestCase(TestCore)
    unittest.TextTestRunner(verbosity=2).run(suite)


# =============================================================
# Entry points
# =============================================================

def _streamlit_available() -> bool:
    try:
        import importlib
        importlib.import_module("streamlit")
        return True
    except ModuleNotFoundError:
        return False


# If Streamlit is available, this file acts as a Streamlit app when launched with
#   streamlit run app.py
# Otherwise, it can be used as a CLI to generate an HTML report:
#   python app.py --out report.html --save-csv filtered.csv

if _streamlit_available():
    # Running in an environment with Streamlit installed
    run_streamlit_app()
else:
    # Fallback CLI mode
    if __name__ == "__main__":
        run_cli()


# -------------------------
# END OF app.py
# -------------------------


# requirements.txt (embedded as a string to avoid SyntaxError when running app.py)
# Write it with:  python app.py --write-reqs
REQUIREMENTS_TXT = """
streamlit>=1.36.0
# Optional for full web app experience
plotly>=5.24.0
pandas>=2.2.2
numpy>=1.26.4
kaleido>=0.2.1
"""


def write_requirements(path: str = "requirements.txt") -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(REQUIREMENTS_TXT)
