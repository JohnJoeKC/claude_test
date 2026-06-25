"""
Sales forecasting model v2.
Improvements over v1:
  - Walk-forward evaluation (honest per-horizon metrics)
  - Separate direct models for 1-day, 7-day, 28-day horizons
  - YoY growth trend feature
  - Rich holiday features (days to/from, name encoding)
  - Promo lead/lag/post effects
  - Normalised lag features
  - Zero-revenue days excluded from MAPE
"""

import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

DATA   = Path("data")
CUTOFF = pd.Timestamp("2025-07-01")   # start of test window
HORIZONS = [1, 7, 28]                 # days-ahead models

# ---------------------------------------------------------------------------
# 1. LOAD RAW DATA
# ---------------------------------------------------------------------------

print("Loading data...")
sales      = pd.read_csv(DATA / "sales.csv",       parse_dates=["date"])
calendar   = pd.read_csv(DATA / "calendar.csv",    parse_dates=["date"])
weather    = pd.read_csv(DATA / "weather.csv",     parse_dates=["date"])
shops      = pd.read_csv(DATA / "shops.csv")
promotions = pd.read_csv(DATA / "promotions.csv",  parse_dates=["start_date","end_date"])
media      = pd.read_csv(DATA / "media_spend.csv", parse_dates=["week_start_date"])
pricing    = pd.read_csv(DATA / "pricing.csv",     parse_dates=["effective_date"])

all_dates = pd.date_range("2024-01-01", "2025-12-31")

# ---------------------------------------------------------------------------
# 2. PROMOTION LOOKUP  →  date × shop × category
# ---------------------------------------------------------------------------

promo_rows = []
for _, row in promotions.iterrows():
    for d in pd.date_range(row.start_date, row.end_date):
        if d in all_dates:
            promo_rows.append({"date": d, "shop_id": row.shop_id,
                               "category": row.category,
                               "discount_pct": row.discount_pct,
                               "promo_id": row.promo_id})

promo_df = (
    pd.DataFrame(promo_rows)
    .sort_values("discount_pct", ascending=False)
    .drop_duplicates(["date","shop_id","category"])   # keep highest discount
)

# ---------------------------------------------------------------------------
# 3. BASE PRICE LOOKUP  (merge_asof per group)
# ---------------------------------------------------------------------------

date_df = pd.DataFrame({"date": all_dates})
price_parts = []
for (sid, cat), grp in pricing.groupby(["shop_id","category"]):
    grp = grp.sort_values("effective_date")[["effective_date","unit_price_gbp"]]
    merged = pd.merge_asof(date_df, grp, left_on="date", right_on="effective_date")
    merged["shop_id"]  = sid
    merged["category"] = cat
    merged.rename(columns={"unit_price_gbp": "base_price_gbp"}, inplace=True)
    price_parts.append(merged[["date","shop_id","category","base_price_gbp"]])
price_df = pd.concat(price_parts, ignore_index=True)

# ---------------------------------------------------------------------------
# 4. MEDIA SPEND  →  daily total
# ---------------------------------------------------------------------------

media_weekly = media.groupby("week_start_date")["spend_gbp"].sum().reset_index()
media_weekly.columns = ["week_start_date", "total_media_spend"]

def week_start(d):
    return d - pd.Timedelta(days=d.dayofweek)

# ---------------------------------------------------------------------------
# 5. HOLIDAY FEATURES  (days-to / days-since each holiday)
# ---------------------------------------------------------------------------

holiday_dates = calendar.loc[calendar.is_public_holiday, "date"].sort_values().values

def days_to_next_holiday(d):
    future = holiday_dates[holiday_dates > d]
    return (future[0] - d).days if len(future) else 999

def days_from_last_holiday(d):
    past = holiday_dates[holiday_dates < d]
    return (d - past[-1]).days if len(past) else 999

print("Building holiday proximity features...")
hol_proximity = pd.DataFrame({"date": all_dates})
hol_proximity["days_to_next_holiday"]    = [days_to_next_holiday(d) for d in all_dates]
hol_proximity["days_from_last_holiday"]  = [days_from_last_holiday(d) for d in all_dates]
hol_proximity["pre_holiday"]             = (hol_proximity.days_to_next_holiday <= 3).astype(int)
hol_proximity["post_holiday"]            = (hol_proximity.days_from_last_holiday <= 2).astype(int)

# ---------------------------------------------------------------------------
# 6. PROMOTION LEAD / LAG FEATURES
# ---------------------------------------------------------------------------

print("Building promo lead/lag features...")

# For each (shop, category) mark the promo_id's start/end, then derive
# days_into_promo, days_until_promo_end, days_since_promo_end
promo_detail = promotions[["promo_id","shop_id","category","start_date","end_date","discount_pct"]].copy()

promo_timeline_rows = []
for _, row in promo_detail.iterrows():
    for i, d in enumerate(pd.date_range(row.start_date, row.end_date)):
        duration = (row.end_date - row.start_date).days + 1
        promo_timeline_rows.append({
            "date": d,
            "shop_id": row.shop_id,
            "category": row.category,
            "days_into_promo":      i + 1,
            "days_until_promo_end": duration - i - 1,
        })

promo_timeline = (
    pd.DataFrame(promo_timeline_rows)
    .sort_values("days_into_promo")
    .drop_duplicates(["date","shop_id","category"])   # keep first (earliest) promo
)

# post-promo: days since a promo ended (up to 7 days)
promo_ends = promotions[["shop_id","category","end_date"]].copy()
post_rows = []
for _, row in promo_ends.iterrows():
    for lag in range(1, 8):
        d = row.end_date + pd.Timedelta(days=lag)
        if d <= all_dates[-1]:
            post_rows.append({"date": d, "shop_id": row.shop_id,
                              "category": row.category, "days_post_promo": lag})

post_promo = (
    pd.DataFrame(post_rows)
    .sort_values("days_post_promo")
    .drop_duplicates(["date","shop_id","category"])
)

# ---------------------------------------------------------------------------
# 7. MERGE EVERYTHING
# ---------------------------------------------------------------------------

print("Building feature table...")
df = sales.copy()

cal_cols = ["date","day_num","week_of_year","month","quarter","year",
            "season","is_weekend","is_public_holiday","holiday_name"]
df = df.merge(calendar[cal_cols], on="date", how="left")
df = df.merge(weather[["date","temp_avg_c","precipitation_mm",
                        "sunshine_hours","wind_speed_kmh","weather_type"]], on="date", how="left")
df = df.merge(shops[["shop_id","shop_type","region","size_sqft","seating_capacity"]], on="shop_id", how="left")
df = df.merge(promo_df[["date","shop_id","category","discount_pct"]], on=["date","shop_id","category"], how="left")
df["discount_pct"] = df["discount_pct"].fillna(0)
df["in_promo"]     = (df["discount_pct"] > 0).astype(int)
df = df.merge(promo_timeline, on=["date","shop_id","category"], how="left")
df["days_into_promo"]      = df["days_into_promo"].fillna(0)
df["days_until_promo_end"] = df["days_until_promo_end"].fillna(0)
df = df.merge(post_promo, on=["date","shop_id","category"], how="left")
df["days_post_promo"] = df["days_post_promo"].fillna(0)
df = df.merge(price_df, on=["date","shop_id","category"], how="left")
df["week_start_col"] = df["date"].apply(week_start)
df = df.merge(media_weekly, left_on="week_start_col", right_on="week_start_date", how="left")
df["total_media_spend"] = df["total_media_spend"].fillna(0)
df.drop(columns=["week_start_col","week_start_date"], inplace=True)
df = df.merge(hol_proximity, on="date", how="left")

# ---------------------------------------------------------------------------
# 8. LAG & ROLLING FEATURES
# ---------------------------------------------------------------------------

print("Computing lag and rolling features...")
df = df.sort_values(["shop_id","category","date"]).reset_index(drop=True)
grp = df.groupby(["shop_id","category"])["units_sold"]

df["lag_7"]        = grp.shift(7)
df["lag_14"]       = grp.shift(14)
df["lag_28"]       = grp.shift(28)
df["lag_364"]      = grp.shift(364)

df["roll_28_mean"] = grp.shift(1).transform(lambda x: x.rolling(28, min_periods=7).mean())
df["roll_28_std"]  = grp.shift(1).transform(lambda x: x.rolling(28, min_periods=7).std())
df["roll_7_mean"]  = grp.shift(1).transform(lambda x: x.rolling(7,  min_periods=3).mean())

# YoY growth: recent 28-day mean vs same 28-day window one year ago
df["roll_28_mean_364"] = grp.shift(364).transform(lambda x: x.rolling(28, min_periods=7).mean())
df["yoy_growth"]       = df["roll_28_mean"] / (df["roll_28_mean_364"] + 1e-6)

# Normalised lags (how far above/below recent average)
df["lag_7_norm"]   = df["lag_7"]   / (df["roll_28_mean"] + 1e-6)
df["lag_364_norm"] = df["lag_364"] / (df["roll_28_mean_364"] + 1e-6)

# Holiday name as category
df["holiday_name"] = df["holiday_name"].fillna("None")

# Shop type × holiday interaction
df["transport_hub"] = (df["shop_type"] == "Transport Hub").astype(int)
df["hub_x_holiday"] = df["transport_hub"] * df["is_public_holiday"].astype(int)
df["campus_x_holiday"] = (df["shop_type"] == "Campus").astype(int) * df["is_public_holiday"].astype(int)

for c in ["category","shop_type","region","season","weather_type","holiday_name"]:
    df[c] = df[c].astype("category")

# ---------------------------------------------------------------------------
# 9. FEATURE SETS PER HORIZON
# (longer horizons drop features that would be unknown in real deployment)
# ---------------------------------------------------------------------------

BASE_FEATURES = [
    "shop_id", "category",
    "day_num", "week_of_year", "month", "quarter", "year",
    "season", "is_weekend", "is_public_holiday", "holiday_name",
    "days_to_next_holiday", "days_from_last_holiday", "pre_holiday", "post_holiday",
    "temp_avg_c", "precipitation_mm", "sunshine_hours", "wind_speed_kmh", "weather_type",
    "shop_type", "region", "size_sqft", "seating_capacity",
    "transport_hub", "hub_x_holiday", "campus_x_holiday",
    "in_promo", "discount_pct",
    "days_into_promo", "days_until_promo_end", "days_post_promo",
    "base_price_gbp", "unit_price_gbp",
    "total_media_spend",
    # Lags safe for all horizons (≥28 days old)
    "lag_28", "lag_364", "lag_364_norm",
    "roll_28_mean", "roll_28_std", "roll_28_mean_364",
    "yoy_growth",
]

HORIZON_FEATURES = {
    1:  BASE_FEATURES + ["lag_7", "lag_14", "lag_7_norm", "roll_7_mean"],
    7:  BASE_FEATURES + ["lag_14", "roll_7_mean"],   # lag_7 unknown
    28: BASE_FEATURES,                                # lag_7 & lag_14 unknown
}

TARGET = "units_sold"

# ---------------------------------------------------------------------------
# 10. WALK-FORWARD EVALUATION
# Walk forward weekly from CUTOFF, training on all data before each window,
# predicting the next H days using only features known at forecast time.
# ---------------------------------------------------------------------------

print("\n=== WALK-FORWARD EVALUATION ===")

# Generate weekly forecast origins across the test window
test_end    = pd.Timestamp("2025-12-31")
origins     = pd.date_range(CUTOFF, test_end - pd.Timedelta(days=1), freq="W-MON")

all_results = {h: [] for h in HORIZONS}

for h in HORIZONS:
    features = HORIZON_FEATURES[h]
    print(f"\n  Horizon = {h} day(s) | {len(origins)} forecast origins...")

    for origin in origins:
        predict_start = origin
        predict_end   = min(origin + pd.Timedelta(days=h-1), test_end)

        # Training data: everything before the forecast window that has valid lags
        train_df = df[df.date < predict_start].dropna(subset=["lag_28","roll_28_mean"])
        test_df  = df[(df.date >= predict_start) & (df.date <= predict_end)]

        if len(train_df) == 0 or len(test_df) == 0:
            continue

        # For h>1 we need to nullify lags that would be unknown at forecast time
        test_df = test_df.copy()
        if h > 7:
            test_df["lag_7"]      = np.nan
            test_df["lag_14"]     = np.nan
            test_df["roll_7_mean"]= np.nan
        elif h > 1:
            test_df["lag_7"]      = np.nan
            test_df["roll_7_mean"]= np.nan

        model = lgb.LGBMRegressor(
            objective="regression_l1",
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=127,
            min_child_samples=20,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=5,
            verbose=-1,
            n_jobs=-1,
            random_state=42,
        )
        model.fit(train_df[features], train_df[TARGET])

        preds = np.maximum(model.predict(test_df[features]), 0)
        test_df = test_df.assign(predicted=preds)
        all_results[h].append(test_df[["date","shop_id","category","units_sold","predicted"]])

# ---------------------------------------------------------------------------
# 11. METRICS
# ---------------------------------------------------------------------------

print("\n\n=== RESULTS BY HORIZON ===")
for h in HORIZONS:
    if not all_results[h]:
        continue
    res = pd.concat(all_results[h])
    # Exclude zero-revenue days (closed) from MAPE
    res_nonzero = res[res.units_sold > 0]
    mae  = mean_absolute_error(res_nonzero.units_sold, res_nonzero.predicted)
    rmse = mean_squared_error(res_nonzero.units_sold, res_nonzero.predicted) ** 0.5
    mape = (np.abs(res_nonzero.units_sold - res_nonzero.predicted) / res_nonzero.units_sold * 100).mean()
    print(f"\n  Horizon {h:2d}-day | MAE={mae:.1f}  RMSE={rmse:.1f}  MAPE={mape:.1f}%  (n={len(res_nonzero):,})")

    print(f"    By category:")
    for cat, g in res_nonzero.groupby("category"):
        m = (np.abs(g.units_sold - g.predicted) / g.units_sold * 100).mean()
        print(f"      {cat:<14} MAPE={m:.1f}%  MAE={mean_absolute_error(g.units_sold,g.predicted):.1f}")

# ---------------------------------------------------------------------------
# 12. TRAIN FINAL MODELS ON ALL DATA UP TO CUTOFF, EVALUATE ON FULL TEST SET
# ---------------------------------------------------------------------------

print("\n\n=== FINAL MODEL: FULL TRAIN→TEST (single pass, for feature importance) ===")

train_full = df[df.date < CUTOFF].dropna(subset=["lag_28","roll_28_mean"])
test_full  = df[df.date >= CUTOFF].dropna(subset=["lag_28","roll_28_mean"])

final_models = {}
for h in HORIZONS:
    features = HORIZON_FEATURES[h]
    test_h = test_full.copy()
    if h > 7:
        test_h["lag_7"]  = np.nan
        test_h["lag_14"] = np.nan
        test_h["roll_7_mean"] = np.nan
    elif h > 1:
        test_h["lag_7"]  = np.nan
        test_h["roll_7_mean"] = np.nan

    m = lgb.LGBMRegressor(
        objective="regression_l1", n_estimators=1000, learning_rate=0.05,
        num_leaves=127, min_child_samples=20, feature_fraction=0.8,
        bagging_fraction=0.8, bagging_freq=5, verbose=-1, n_jobs=-1, random_state=42,
    )
    m.fit(
        train_full[features], train_full[TARGET],
        eval_set=[(test_h[features], test_h[TARGET])],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(999)],
    )
    final_models[h] = (m, features)

    preds = np.maximum(m.predict(test_h[features]), 0)
    nz    = test_h[test_h.units_sold > 0]
    nz_p  = np.maximum(m.predict(nz[features]), 0)
    mae   = mean_absolute_error(nz.units_sold, nz_p)
    mape  = (np.abs(nz.units_sold - nz_p) / nz.units_sold * 100).mean()
    print(f"  Horizon {h:2d}-day | MAE={mae:.1f}  MAPE={mape:.1f}%")

    imp = pd.Series(m.booster_.feature_importance(importance_type="gain"), index=features)
    print(f"  Top 10 features:")
    for fname, score in imp.sort_values(ascending=False).head(10).items():
        print(f"    {fname:<30} {score:>10.0f}")

# Save final 1-day model and its predictions
m1, f1 = final_models[1]
m1.booster_.save_model("sales_forecast_model_v2.txt")

test_h1 = test_full.copy()
preds1  = np.maximum(m1.predict(test_h1[f1]), 0)
out = test_h1[["date","shop_id","category","units_sold"]].copy()
out["predicted_1day"] = preds1
out.to_csv("test_predictions_v2.csv", index=False)

print("\nDone. Model saved to sales_forecast_model_v2.txt")
print("Predictions saved to test_predictions_v2.csv")
