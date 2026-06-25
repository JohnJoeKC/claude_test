"""
Sales forecasting model: predict daily units_sold per shop per category.
Train: Jan 2024 – Jun 2025 (18 months)
Test:  Jul 2025 – Dec 2025 (6 months)
Model: LightGBM with calendar, weather, promo, pricing, media and lag features.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

DATA = Path("data")
CUTOFF = pd.Timestamp("2025-07-01")

# ---------------------------------------------------------------------------
# 1. LOAD RAW DATA
# ---------------------------------------------------------------------------

print("Loading data...")
sales     = pd.read_csv(DATA / "sales.csv",      parse_dates=["date"])
calendar  = pd.read_csv(DATA / "calendar.csv",   parse_dates=["date"])
weather   = pd.read_csv(DATA / "weather.csv",    parse_dates=["date"])
shops     = pd.read_csv(DATA / "shops.csv")
promotions= pd.read_csv(DATA / "promotions.csv", parse_dates=["start_date","end_date"])
media     = pd.read_csv(DATA / "media_spend.csv",parse_dates=["week_start_date"])
pricing   = pd.read_csv(DATA / "pricing.csv",    parse_dates=["effective_date"])

# ---------------------------------------------------------------------------
# 2. BUILD PROMOTION LOOKUP  (date × shop × category → discount_pct)
# ---------------------------------------------------------------------------

all_dates = pd.date_range("2024-01-01", "2025-12-31")
promo_rows = []
for _, row in promotions.iterrows():
    for d in pd.date_range(row.start_date, row.end_date):
        if d in all_dates:
            promo_rows.append({"date": d, "shop_id": row.shop_id,
                               "category": row.category,
                               "discount_pct": row.discount_pct})

promo_df = (
    pd.DataFrame(promo_rows)
    .groupby(["date","shop_id","category"], as_index=False)["discount_pct"]
    .max()                          # take highest discount on a given day
)

# ---------------------------------------------------------------------------
# 3. BUILD ACTIVE PRICE LOOKUP  (date × shop × category → base_price_gbp)
# Use merge_asof per group to efficiently get the price active on each date.
# ---------------------------------------------------------------------------

date_df = pd.DataFrame({"date": all_dates})
price_parts = []
for (sid, cat), grp in pricing.groupby(["shop_id","category"]):
    grp = grp.sort_values("effective_date")[["effective_date","unit_price_gbp"]]
    merged = pd.merge_asof(date_df, grp, left_on="date", right_on="effective_date")
    merged["shop_id"] = sid
    merged["category"] = cat
    merged.rename(columns={"unit_price_gbp": "base_price_gbp"}, inplace=True)
    price_parts.append(merged[["date","shop_id","category","base_price_gbp"]])

price_df = pd.concat(price_parts, ignore_index=True)

# ---------------------------------------------------------------------------
# 4. MEDIA SPEND  → daily total (sum all channels per week)
# ---------------------------------------------------------------------------

media_weekly = media.groupby("week_start_date")["spend_gbp"].sum().reset_index()
media_weekly.columns = ["week_start_date", "total_media_spend"]

def week_start(d):
    return d - pd.Timedelta(days=d.dayofweek)

# ---------------------------------------------------------------------------
# 5. MERGE EVERYTHING INTO ONE FLAT TABLE
# ---------------------------------------------------------------------------

print("Building feature table...")
df = sales.copy()

# Calendar
cal_cols = ["date","day_num","week_of_year","month","quarter","year",
            "season","is_weekend","is_public_holiday","holiday_name"]
df = df.merge(calendar[cal_cols], on="date", how="left")

# Weather
wx_cols = ["date","temp_avg_c","precipitation_mm","sunshine_hours","wind_speed_kmh","weather_type"]
df = df.merge(weather[wx_cols], on="date", how="left")

# Shop metadata
df = df.merge(shops[["shop_id","shop_type","region","size_sqft","seating_capacity"]],
              on="shop_id", how="left")

# Promotions
df = df.merge(promo_df, on=["date","shop_id","category"], how="left")
df["discount_pct"] = df["discount_pct"].fillna(0)
df["in_promo"] = (df["discount_pct"] > 0).astype(int)

# Pricing (base price from pricing.csv — sales.csv already has effective/discounted price)
df = df.merge(price_df, on=["date","shop_id","category"], how="left")

# Media spend
df["week_start"] = df["date"].apply(week_start)
df = df.merge(media_weekly, left_on="week_start", right_on="week_start_date", how="left")
df["total_media_spend"] = df["total_media_spend"].fillna(0)
df.drop(columns=["week_start","week_start_date"], inplace=True)

# ---------------------------------------------------------------------------
# 6. LAG & ROLLING FEATURES  (built from the full sorted series)
# ---------------------------------------------------------------------------

print("Computing lag features...")
df = df.sort_values(["shop_id","category","date"]).reset_index(drop=True)

group = df.groupby(["shop_id","category"])["units_sold"]

# Lag 7 and 14 (same weekday last/fortnight)
df["lag_7"]  = group.shift(7)
df["lag_14"] = group.shift(14)
# Lag 364: same weekday last year (captures YoY seasonality)
df["lag_364"] = group.shift(364)

# Rolling 4-week average (smoothed trend)
df["roll_28_mean"] = group.shift(1).transform(lambda x: x.rolling(28, min_periods=7).mean())
df["roll_28_std"]  = group.shift(1).transform(lambda x: x.rolling(28, min_periods=7).std())

# ---------------------------------------------------------------------------
# 7. ENCODE CATEGORICALS
# ---------------------------------------------------------------------------

cat_cols = ["category","shop_type","region","season","weather_type","holiday_name"]
for c in cat_cols:
    df[c] = df[c].astype("category")

# ---------------------------------------------------------------------------
# 8. FEATURE LIST
# ---------------------------------------------------------------------------

FEATURES = [
    # Identifiers (treated as categoricals by LGB)
    "shop_id", "category",
    # Calendar
    "day_num", "week_of_year", "month", "quarter", "year",
    "season", "is_weekend", "is_public_holiday", "holiday_name",
    # Weather
    "temp_avg_c", "precipitation_mm", "sunshine_hours", "wind_speed_kmh", "weather_type",
    # Shop
    "shop_type", "region", "size_sqft", "seating_capacity",
    # Promotions & pricing
    "in_promo", "discount_pct", "unit_price_gbp", "base_price_gbp",
    # Media
    "total_media_spend",
    # Lags
    "lag_7", "lag_14", "lag_364",
    "roll_28_mean", "roll_28_std",
]
TARGET = "units_sold"

# ---------------------------------------------------------------------------
# 9. TRAIN / TEST SPLIT
# ---------------------------------------------------------------------------

train = df[df.date < CUTOFF].dropna(subset=["lag_7","lag_14","roll_28_mean"])
test  = df[df.date >= CUTOFF].dropna(subset=["lag_7","lag_14","roll_28_mean"])

print(f"Train: {train.date.min().date()} → {train.date.max().date()}  ({len(train):,} rows)")
print(f"Test:  {test.date.min().date()} → {test.date.max().date()}   ({len(test):,} rows)")

X_train, y_train = train[FEATURES], train[TARGET]
X_test,  y_test  = test[FEATURES],  test[TARGET]

# ---------------------------------------------------------------------------
# 10. TRAIN LIGHTGBM
# ---------------------------------------------------------------------------

print("\nTraining LightGBM...")
params = {
    "objective":       "regression_l1",   # MAE loss — robust to outliers
    "metric":          "mae",
    "n_estimators":    1000,
    "learning_rate":   0.05,
    "num_leaves":      127,
    "min_child_samples": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq":    5,
    "verbose":         -1,
    "n_jobs":          -1,
    "random_state":    42,
}

model = lgb.LGBMRegressor(**params)
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)],
)

# ---------------------------------------------------------------------------
# 11. EVALUATE
# ---------------------------------------------------------------------------

preds = model.predict(X_test)
preds = np.maximum(preds, 0)   # floor at zero

mae  = mean_absolute_error(y_test, preds)
rmse = mean_squared_error(y_test, preds) ** 0.5
mape = np.mean(np.abs((y_test - preds) / (y_test + 1))) * 100  # +1 avoids /0

print("\n=== TEST SET METRICS (Jul–Dec 2025) ===")
print(f"  MAE  : {mae:.1f} units")
print(f"  RMSE : {rmse:.1f} units")
print(f"  MAPE : {mape:.1f}%")

# Breakdown by category
test_results = test[["date","shop_id","category","units_sold"]].copy()
test_results["predicted"] = preds
test_results["abs_error"] = (test_results.units_sold - test_results.predicted).abs()

print("\n=== MAE BY CATEGORY ===")
print(test_results.groupby("category").apply(
    lambda x: pd.Series({
        "actual_mean":    x.units_sold.mean().round(1),
        "pred_mean":      x.predicted.mean().round(1),
        "MAE":            x.abs_error.mean().round(1),
        "MAPE_%":         (x.abs_error / (x.units_sold + 1) * 100).mean().round(1),
    })
).to_string())

print("\n=== MAE BY SHOP ===")
print(test_results.groupby("shop_id").apply(
    lambda x: pd.Series({
        "actual_mean": x.units_sold.mean().round(1),
        "MAE":         x.abs_error.mean().round(1),
        "MAPE_%":      (x.abs_error / (x.units_sold + 1) * 100).mean().round(1),
    })
).to_string())

# ---------------------------------------------------------------------------
# 12. FEATURE IMPORTANCE
# ---------------------------------------------------------------------------

print("\n=== TOP 20 FEATURE IMPORTANCES (gain) ===")
imp = pd.Series(model.booster_.feature_importance(importance_type="gain"),
                index=FEATURES).sort_values(ascending=False)
print(imp.head(20).round(0).to_string())

# ---------------------------------------------------------------------------
# 13. SAVE MODEL & PREDICTIONS
# ---------------------------------------------------------------------------

model.booster_.save_model("sales_forecast_model.txt")
test_results.to_csv("test_predictions.csv", index=False)
print("\nModel saved to sales_forecast_model.txt")
print("Predictions saved to test_predictions.csv")
