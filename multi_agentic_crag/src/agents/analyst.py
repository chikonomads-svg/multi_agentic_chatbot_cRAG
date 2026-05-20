"""Data Analyst Agent: detects numerical/tabular data and outputs Plotly JSON or code."""
from typing import Dict, Any, List
import logging
import json
import pandas as pd

LOG = logging.getLogger(__name__)


def detect_and_build_plot(data_text: str) -> Dict[str, Any]:
    """Naive detector: attempts to parse CSV-like blocks or simple key:value pairs into a dataframe.

    Returns a dict like {"type":"bar","data":{...}} or empty dict if nothing detected.
    """
    # Try parse as CSV
    try:
        df = pd.read_csv(pd.compat.StringIO(data_text))
        # For simplicity choose bar if dataframe has 2 columns with numeric second
        if df.shape[1] >= 2:
            cols = list(df.columns)
            if pd.api.types.is_numeric_dtype(df[cols[1]]):
                result = {"type": "bar", "data": {"x": df[cols[0]].tolist(), "y": df[cols[1]].tolist(), "columns": cols}}
                return result
    except Exception:
        LOG.debug("CSV parse failed, trying key:value parse")

    # Try key:value per-line
    lines = [l.strip() for l in data_text.splitlines() if l.strip()]
    kv = {}
    for l in lines:
        if ":" in l:
            k, v = l.split(":", 1)
            try:
                kv[k.strip()] = float(v.strip())
            except Exception:
                pass
    if kv:
        return {"type": "bar", "data": {"x": list(kv.keys()), "y": list(kv.values())}}

    return {}
