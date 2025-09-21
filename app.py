import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from streamlit_dynamic_filters import DynamicFilters

st.set_page_config(page_title="Lebanon's Educational Resouces Distribution", layout="wide")

# Data loading
@st.cache_data
def load_data(path: str):
    return pd.read_csv(path)

df = load_data("data_clean.csv")

# Required column names from the dataset
COL_LAT = "lat"
COL_LON = "lon"
COL_TOWN = "Town"
COL_GOV  = "Governorate"
COL_COVI = "Public school coverage index (number of schools per citizen)"
COL_PUB  = "Type and size of educational resources - public schools"
COL_PRIV = "Type and size of educational resources - private schools"
COL_POP  = "population"
COL_UNI = "Type and size of educational resources - universities"

# Clean population for bubble size (fill NA with 1)
df[COL_POP] = df[COL_POP].fillna(1.0)
df = df.dropna(subset=[COL_LAT, COL_LON]) # remove towns with missing coordinates

# add column for the size of schools
df['Type and size of educational resources - schools'] = df[COL_PUB] + df[COL_PRIV]
COL_SCH = "Type and size of educational resources - schools"

# sidebar interactions
st.sidebar.header("Filters")

# Governorates filters
govs = sorted(df[COL_GOV].dropna().unique())
dynamic_filters = DynamicFilters(df, filters=[COL_GOV])
dynamic_filters.display_filters(location='sidebar')
df_f = dynamic_filters.filter_df()


def safe_slider(label, series, as_int=False, step=1):
    if series.empty:
        return None, None
    lo, hi = float(series.min()), float(series.max())
    # if identical, widen a bit so slider still works
    if lo == hi:
        if as_int:
            lo, hi = int(lo), int(hi + 1)
        else:
            hi = hi + 1.0
    if as_int:
        lo, hi = int(lo), int(hi)
    return st.sidebar.slider(label, min_value=lo, max_value=hi, value=(lo, hi), step=step if as_int else None)

# usage
cov_min, cov_max = safe_slider("Coverage index range (schools per citizen)", df_f[COL_COVI])
sch_min, sch_max = safe_slider("Number of Schools", df_f[COL_SCH], as_int=True, step=1)
uni_min, uni_max = safe_slider("Number of Universities", df_f[COL_UNI], as_int=True, step=1)
pop_min, pop_max = safe_slider("Population Size", df_f[COL_POP])

# --- apply ALL filters to df_f
mask = (
    df_f[COL_COVI].between(cov_min, cov_max)
    & df_f[COL_SCH].between(sch_min, sch_max)
    & df_f[COL_UNI].between(uni_min, uni_max)
    & df_f[COL_POP].between(pop_min, pop_max)
)
df_f = df_f.loc[mask].copy()

# page header
st.title("Lebanon's Educational Resouces Distribution")
st.caption("Use the filters to interact with the visuals. The coverage index is schools per citizen (higher = better coverage).")

# Viz #1: Map
map_fig = px.scatter_mapbox(
    df_f,
    lat=COL_LAT,
    lon=COL_LON,
    size=COL_COVI,
    color=COL_GOV,
    size_max=28,
    hover_name=COL_TOWN,
    hover_data={
        COL_COVI:":.5f",
        COL_GOV: True,
        COL_PUB: True,
        COL_PRIV: True
    },
    zoom=7,
    height=620,
    mapbox_style="carto-positron",
    title="Public school coverage index — towns",
)
map_fig.update_layout(margin=dict(l=0, r=0, t=60, b=0), legend_title_text="Governorate")
st.plotly_chart(map_fig, use_container_width=True)

# Viz #2: Public vs Private (by governorate, using *filtered* towns)

# Keep only needed cols and coerce to numeric
tmp = df_f[[COL_GOV, COL_SCH, COL_UNI, COL_POP]].copy()

# Aggregate by governorate
#    - Schools & Unis: sum across towns
#    - Population: assume a single governorate-level value repeated -> take first/median
grouped = (
    tmp.groupby(COL_GOV, as_index=False)
       .agg({
           COL_SCH: "sum",
           COL_UNI: "sum",
           COL_POP: "median",
       })
)

# Build the chart (trendline only if statsmodels is available)
kwargs = dict(
    data_frame=grouped,
    x=COL_SCH,
    y=COL_UNI,
    size=COL_POP,
    text=COL_GOV,         
    title="Governorates: Schools vs Universities vs Population",
    height=560,
)

scatter_fig = px.scatter(**kwargs, trendline="ols")

scatter_fig.update_traces(textposition="top center")
scatter_fig.update_xaxes(title="Schools (count)")
scatter_fig.update_yaxes(title="Universities (count)")
st.plotly_chart(scatter_fig, use_container_width=True)