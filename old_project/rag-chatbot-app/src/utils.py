"""Utilities: parse markdown tables and render Plotly charts in Streamlit."""
import re
import io
import pandas as pd
import plotly.express as px


MD_TABLE_RE = re.compile(r"^\|.*\|\s*$", re.MULTILINE)


def extract_markdown_table(text: str) -> pd.DataFrame:
    """Extract the first markdown table found in text and return a DataFrame.

    Raises ValueError if no table found.
    """
    # Find lines that look like a markdown table
    lines = text.splitlines()
    table_lines = []
    in_table = False
    for ln in lines:
        if ln.strip().startswith("|") and ln.strip().endswith("|"):
            table_lines.append(ln)
            in_table = True
        else:
            if in_table:
                break

    if not table_lines:
        raise ValueError("No markdown table found")

    buf = io.StringIO("\n".join(table_lines))
    # Use pandas to read the markdown-like table by treating '|' as sep and skipping empty columns
    df = pd.read_csv(buf, sep="|", engine="python")
    # pandas will include empty columns from leading/trailing pipes; drop them
    df = df.loc[:, ~df.columns.str.strip().eq("")]
    # Strip whitespace from column names
    df.columns = [c.strip() for c in df.columns]
    # Drop the separator/header divider row if present (---)
    df = df[~df.iloc[:, 0].astype(str).str.contains('^-{3,}$', na=False)]
    df = df.reset_index(drop=True)
    # Try to convert numeric columns
    for col in df.columns:
        df[col] = pd.to_numeric(df[col].str.strip(), errors='ignore') if df[col].dtype == object else df[col]

    return df


def plot_from_dataframe(df: pd.DataFrame, kind: str = "bar"):
    """Return a Plotly figure from a dataframe. Chooses x as first column and y as numeric columns."""
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    x_col = df.columns[0]
    if not numeric_cols:
        raise ValueError("No numeric columns to plot")

    if kind == "line":
        fig = px.line(df, x=x_col, y=numeric_cols)
    else:
        fig = px.bar(df, x=x_col, y=numeric_cols)
    return fig
