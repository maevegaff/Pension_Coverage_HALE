"""
01_clean_data.py

Pension Coverage & Healthy Life Expectancy (HALE) Study
Pipeline Steps 1-7: Load raw data -> harmonize -> restrict period -> merge ->
handle missingness -> check implausible values -> construct derived variables.

TWO-VARIABLE PENSION DESIGN: contributory and social (non-contributory)
pension coverage are kept as separate variables throughout, rather than
combined into one "any pension" measure -- see project discussion. This
means: (1) HALE is still required for every retained row, but pension
coverage is NOT required -- each of the two pension variables is allowed to
be missing independently, since 02_analyze_data.py runs two separate models
(contributory-only, social-only) and each model does its own listwise
deletion on just the variable it needs; (2) Step 7 builds two full sets of
derived terms (centered, squared, income-group interactions) instead of one.

Study window: 2000-2019 (narrowed from the original 2000-2022 -- see project
discussion on pension data availability).

INPUT:  Five raw .xlsx files (one per source database), paths set in CONFIG below.
OUTPUT: cleaned_panel.csv  -- one row per country-year, ready for 02_analyze_data.py
"""

import pandas as pd
import numpy as np
import pycountry

pd.set_option("display.width", 120)

# =============================================================================
# CONFIG -- edit this section to match your actual files
# =============================================================================

RAW_DIR = r"C:\Users\maeve\Downloads\FinDiss\data"
OUT_DIR = "cleaned_data"

START_YEAR = 2000
END_YEAR = 2019

SOURCES = {
    "hale": {
        "path": f"{RAW_DIR}/who_hale.xlsx",
        "column_map": {
            "Country": "country_name",
            "ISO3": "iso3",
            "Year": "year",
            "HALE_60": "hale_60",
        },
    },
    "pension": {
        # ASPIRE: contributory and social pension coverage as TWO separate
        # series, not one combined measure.
        "path": f"{RAW_DIR}/pension_coverage.xlsx",
        "column_map": {
            "Country": "country_name",
            "ISO3": "iso3",
            "Year": "year",
            "Coverage_Contributory_pct": "pension_contributory",
            "Coverage_Social_pct": "pension_social",
        },
    },
    "controls": {
        "path": f"{RAW_DIR}/world_bank_wdi.xlsx",
        "column_map": {
            "Country": "country_name",
            "ISO3": "iso3",
            "Year": "year",
            "GDP_per_capita": "gdp_per_capita",
            "Urban_pop_pct": "urbanization_rate",
            "OOP_health_exp_pct": "oop_health_exp",
            "Gov_health_exp_pct_GDP": "gov_health_exp",
        },
    },
    "demographics": {
        "path": f"{RAW_DIR}/un_population.xlsx",
        "column_map": {
            "Country": "country_name",
            "ISO3": "iso3",
            "Year": "year",
            "OldAge_Dependency_Ratio": "dependency_ratio",
            "Female_Share_65plus": "female_share_elderly",
        },
    },
    "income_group": {
        "path": f"{RAW_DIR}/income_group.xlsx",
        "column_map": {
            "Country": "country_name",
            "ISO3": "iso3",
            "Year": "year",
            "Income_Group": "income_group",
        },
    },
}

AGGREGATE_NAME_FRAGMENTS = [
    "world", "income", "region", "union", "area", "ibrd", "ida",
    "euro area", "africa", "asia", "europe", "america", "pacific",
    "caribbean", "small states", "fragile", "oecd", "arab world",
]

PCT_VARS = ["pension_contributory", "pension_social", "urbanization_rate",
            "oop_health_exp", "gov_health_exp", "female_share_elderly"]
HALE_MAX_PLAUSIBLE = 35

MISSING_FLAG_THRESHOLD = 0.15

# =============================================================================
# STEP 1: LOAD RAW DATA
# =============================================================================

def load_source(name, spec):
    print(f"\n[Step 1] Loading '{name}' from {spec['path']}")
    try:
        df = pd.read_excel(spec["path"])
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Could not find {spec['path']}. Update SOURCES['{name}']['path'] "
            f"in the CONFIG section to point at your actual file."
        )
    missing_cols = [c for c in spec["column_map"] if c not in df.columns]
    if missing_cols:
        raise KeyError(
            f"'{name}' file is missing expected columns {missing_cols}. "
            f"Found columns: {list(df.columns)}. "
            f"Update COLUMN_MAP for '{name}' in CONFIG to match your file."
        )
    df = df.rename(columns=spec["column_map"])
    df = df[list(spec["column_map"].values())]
    print(f"  -> {len(df):,} rows loaded")
    return df


# =============================================================================
# STEP 2: HARMONIZE COUNTRY IDENTIFIERS
# =============================================================================

VALID_ISO3 = {c.alpha_3 for c in pycountry.countries}


def harmonize_countries(df, name):
    print(f"[Step 2] Harmonizing country identifiers for '{name}'")
    df = df.copy()
    df["iso3"] = df["iso3"].astype(str).str.strip().str.upper()

    before = len(df)
    name_lower = df["country_name"].astype(str).str.lower()
    is_aggregate = name_lower.apply(
        lambda n: any(frag in n for frag in AGGREGATE_NAME_FRAGMENTS)
    )
    df = df[~is_aggregate]
    df = df[df["iso3"].isin(VALID_ISO3)]

    dropped = before - len(df)
    print(f"  -> dropped {dropped:,} non-country rows ({len(df):,} remain)")
    return df


# =============================================================================
# STEP 3: RESTRICT TO STUDY PERIOD
# =============================================================================

def restrict_period(df, name):
    before = len(df)
    df = df[(df["year"] >= START_YEAR) & (df["year"] <= END_YEAR)]
    print(f"[Step 3] '{name}': restricted to {START_YEAR}-{END_YEAR} "
          f"({before:,} -> {len(df):,} rows)")
    return df


# =============================================================================
# STEP 4: MERGE INTO ONE COUNTRY-YEAR PANEL
# =============================================================================

def merge_panel(frames):
    print("\n[Step 4] Merging sources into one country-year panel")
    panel = frames["hale"][["iso3", "country_name", "year", "hale_60"]].copy()

    merge_specs = [
        ("pension", ["iso3", "year", "pension_contributory", "pension_social"]),
        ("controls", ["iso3", "year", "gdp_per_capita", "urbanization_rate",
                       "oop_health_exp", "gov_health_exp"]),
        ("demographics", ["iso3", "year", "dependency_ratio", "female_share_elderly"]),
        ("income_group", ["iso3", "year", "income_group"]),
    ]

    for name, cols in merge_specs:
        panel = panel.merge(frames[name][cols], on=["iso3", "year"], how="outer")

    panel = panel.sort_values(["iso3", "year"]).reset_index(drop=True)
    n_countries = panel["iso3"].nunique()
    n_years = panel["year"].nunique()
    print(f"  -> panel built: {len(panel):,} country-year rows, "
          f"{n_countries} countries x up to {n_years} years")
    return panel


# =============================================================================
# STEP 5: MISSINGNESS ASSESSMENT AND HANDLING
# =============================================================================

def assess_missingness(panel):
    print("\n[Step 5] Missingness report (% missing per variable)")
    cols = ["hale_60", "pension_contributory", "pension_social", "gdp_per_capita",
            "urbanization_rate", "oop_health_exp", "gov_health_exp",
            "dependency_ratio", "female_share_elderly", "income_group"]
    report = (panel[cols].isna().mean() * 100).round(1).sort_values(ascending=False)
    print(report.to_string())

    flagged = report[report > MISSING_FLAG_THRESHOLD * 100]
    if len(flagged):
        print(f"\n  NOTE: variables above {MISSING_FLAG_THRESHOLD:.0%} missing:\n"
              f"{flagged.to_string()}")
        print("  pension_contributory / pension_social are EXPECTED to be highly "
              "missing -- this is a known feature of ASPIRE's survey-based "
              "collection (see project discussion), not a processing error. "
              "Each is handled by its own model in 02_analyze_data.py, so high "
              "missingness here does not get listwise-deleted at this stage.")
    return report


def handle_missingness(panel):
    """
    HALE is required for every retained row (it's the dependent variable for
    every model). Pension coverage is deliberately NOT required here --
    pension_contributory and pension_social are each missing independently
    in most rows, and 02_analyze_data.py runs two separate models (one per
    pension type), each doing its own listwise deletion on just the variable
    it needs. Dropping rows missing EITHER pension variable at this stage
    would needlessly throw away usable rows for both models.

    Control variables get small within-country gaps filled via interpolation,
    same as before. Pension coverage is NOT interpolated -- it's a sparse,
    central explanatory variable, and fabricating points for it via
    interpolation would risk manufacturing the very relationship being tested.
    """
    print("\n[Step 6 prep] Handling missingness")
    before = len(panel)

    control_cols = ["gdp_per_capita", "urbanization_rate", "oop_health_exp",
                     "gov_health_exp", "dependency_ratio", "female_share_elderly"]
    panel = panel.sort_values(["iso3", "year"])
    for col in control_cols:
        panel[col] = panel.groupby("iso3")[col].transform(
            lambda s: s.interpolate(limit=2, limit_direction="both")
        )

    panel = panel.dropna(subset=["hale_60"])

    print(f"  -> {before:,} -> {len(panel):,} rows after interpolating control "
          f"gaps and dropping rows missing HALE (pension coverage NOT required "
          f"at this stage -- see docstring)")
    return panel


# =============================================================================
# STEP 6: IMPLAUSIBLE VALUE / OUTLIER CHECKS
# =============================================================================

def check_implausible_values(panel):
    print("\n[Step 6] Implausible value checks")

    for col in PCT_VARS:
        out_of_bounds = ~panel[col].between(0, 100) & panel[col].notna()
        n_bad = out_of_bounds.sum()
        if n_bad:
            print(f"  -> {col}: {n_bad} values outside [0, 100], setting to NaN for review")
            panel.loc[out_of_bounds, col] = np.nan

    implausible_hale = panel["hale_60"] > HALE_MAX_PLAUSIBLE
    n_bad = implausible_hale.sum()
    if n_bad:
        print(f"  -> hale_60: {n_bad} rows exceed {HALE_MAX_PLAUSIBLE} years "
              f"-- flagging, not auto-dropping")
        panel["hale_flag_implausible"] = implausible_hale
    else:
        panel["hale_flag_implausible"] = False

    return panel


# =============================================================================
# STEP 7: CONSTRUCT DERIVED VARIABLES (two full sets -- one per pension type)
# =============================================================================

def construct_derived_variables(panel):
    print("\n[Step 7] Constructing derived variables (two-variable pension design)")

    panel["log_gdp_pc"] = np.log(panel["gdp_per_capita"])

    panel["income_group"] = panel["income_group"].astype("category")
    income_dummies = pd.get_dummies(panel["income_group"], prefix="income",
                                     drop_first=True).astype(float)
    panel = pd.concat([panel, income_dummies], axis=1)

    pension_vars = {
        "pension_contributory": "coverage_contrib",
        "pension_social": "coverage_social",
    }

    for raw_col, prefix in pension_vars.items():
        mean_val = panel[raw_col].mean()
        c_col = f"{prefix}_c"
        sq_col = f"{prefix}_c_sq"
        panel[c_col] = panel[raw_col] - mean_val
        panel[sq_col] = panel[c_col] ** 2
        for dummy_col in income_dummies.columns:
            panel[f"{prefix}_x_{dummy_col}"] = panel[c_col] * panel[dummy_col]
        print(f"  -> {raw_col}: centered on mean={mean_val:.2f} (n={panel[raw_col].notna().sum()}), "
              f"added {c_col}, {sq_col}, {len(income_dummies.columns)} interactions")

    return panel


# =============================================================================
# FINAL PANEL CHECK
# =============================================================================

def final_panel_check(panel):
    print("\n[Final check] Cleaned panel summary")
    n_countries = panel["iso3"].nunique()
    n_years = panel["year"].nunique()
    n_obs = len(panel)
    max_possible = n_countries * n_years
    balance_pct = 100 * n_obs / max_possible if max_possible else 0
    print(f"  Countries: {n_countries}")
    print(f"  Years: {n_years} ({panel['year'].min():.0f}-{panel['year'].max():.0f})")
    print(f"  Observations (HALE present): {n_obs:,} (of {max_possible:,} possible "
          f"= {balance_pct:.1f}% balanced)")
    for raw_col, label in [("pension_contributory", "Contributory pension"),
                             ("pension_social", "Social pension")]:
        n_present = panel[raw_col].notna().sum()
        n_countries_with_data = panel.loc[panel[raw_col].notna(), "iso3"].nunique()
        print(f"  {label} coverage present: {n_present:,} rows across "
              f"{n_countries_with_data} countries ({n_present / n_obs * 100:.1f}% of panel)")


# =============================================================================
# MAIN
# =============================================================================

def main():
    import os
    os.makedirs(OUT_DIR, exist_ok=True)

    frames = {}
    for name, spec in SOURCES.items():
        df = load_source(name, spec)
        df = harmonize_countries(df, name)
        df = restrict_period(df, name)
        frames[name] = df

    panel = merge_panel(frames)
    assess_missingness(panel)
    panel = handle_missingness(panel)
    panel = check_implausible_values(panel)
    panel = construct_derived_variables(panel)
    final_panel_check(panel)

    out_path = f"{OUT_DIR}/cleaned_panel.csv"
    panel.to_csv(out_path, index=False)
    print(f"\nSaved cleaned panel to {out_path}")


if __name__ == "__main__":
    main()
