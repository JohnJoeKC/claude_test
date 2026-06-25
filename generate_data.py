"""
Dummy Sales Database Generator
Generates 2 years (2024-2025) of daily sales data across 10 shops,
5 categories, with weather, media spend, promotions, pricing, and calendar.
"""

import pandas as pd
import numpy as np
from datetime import date, timedelta
import random
import math
import os

np.random.seed(42)
random.seed(42)

OUTPUT_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

START_DATE = date(2024, 1, 1)
END_DATE = date(2025, 12, 31)
CATEGORIES = ["Coffee", "Cold Drinks", "Sandwiches", "Snacks", "Bakery"]

# ---------------------------------------------------------------------------
# SHOPS
# ---------------------------------------------------------------------------

shops = pd.DataFrame([
    {"shop_id": 1,  "shop_name": "City Central",       "location": "City Centre",   "shop_type": "High Street",     "region": "North",  "size_sqft": 1200, "seating_capacity": 40},
    {"shop_id": 2,  "shop_name": "Westside Corner",    "location": "Suburb West",   "shop_type": "Neighbourhood",   "region": "West",   "size_sqft": 800,  "seating_capacity": 20},
    {"shop_id": 3,  "shop_name": "Station Stop",       "location": "Train Station", "shop_type": "Transport Hub",   "region": "North",  "size_sqft": 600,  "seating_capacity": 10},
    {"shop_id": 4,  "shop_name": "Park Lane Cafe",     "location": "Park Area",     "shop_type": "High Street",     "region": "South",  "size_sqft": 1000, "seating_capacity": 35},
    {"shop_id": 5,  "shop_name": "Eastgate Express",   "location": "Suburb East",   "shop_type": "Drive-Through",   "region": "East",   "size_sqft": 1500, "seating_capacity": 15},
    {"shop_id": 6,  "shop_name": "Riverside Roast",    "location": "Riverside",     "shop_type": "Destination",     "region": "South",  "size_sqft": 1800, "seating_capacity": 60},
    {"shop_id": 7,  "shop_name": "Airport Terminal",   "location": "Airport",       "shop_type": "Transport Hub",   "region": "West",   "size_sqft": 700,  "seating_capacity": 25},
    {"shop_id": 8,  "shop_name": "Uni Quarter",        "location": "University",    "shop_type": "Campus",          "region": "East",   "size_sqft": 900,  "seating_capacity": 30},
    {"shop_id": 9,  "shop_name": "Retail Park Hub",    "location": "Retail Park",   "shop_type": "Retail Park",     "region": "North",  "size_sqft": 1100, "seating_capacity": 20},
    {"shop_id": 10, "shop_name": "Old Town Brew",      "location": "Old Town",      "shop_type": "High Street",     "region": "South",  "size_sqft": 950,  "seating_capacity": 30},
])
shops.to_csv(f"{OUTPUT_DIR}/shops.csv", index=False)
print("shops.csv written")

# ---------------------------------------------------------------------------
# CALENDAR (public holidays: UK-style)
# ---------------------------------------------------------------------------

def easter_date(year):
    """Anonymous Gregorian algorithm for Easter Sunday."""
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)

def get_public_holidays(year):
    holidays = {}
    # New Year's Day
    d = date(year, 1, 1)
    if d.weekday() == 6: d = date(year, 1, 2)
    if d.weekday() == 5: d = date(year, 1, 3)
    holidays[d] = "New Year's Day"

    # Easter
    easter = easter_date(year)
    holidays[easter - timedelta(days=2)] = "Good Friday"
    holidays[easter + timedelta(days=1)] = "Easter Monday"

    # Early May Bank Holiday (first Monday in May)
    d = date(year, 5, 1)
    while d.weekday() != 0: d += timedelta(days=1)
    holidays[d] = "Early May Bank Holiday"

    # Spring Bank Holiday (last Monday in May)
    d = date(year, 5, 31)
    while d.weekday() != 0: d -= timedelta(days=1)
    holidays[d] = "Spring Bank Holiday"

    # Summer Bank Holiday (last Monday in August)
    d = date(year, 8, 31)
    while d.weekday() != 0: d -= timedelta(days=1)
    holidays[d] = "Summer Bank Holiday"

    # Christmas Day
    d = date(year, 12, 25)
    if d.weekday() == 5:   # Saturday -> substitute Tuesday
        holidays[date(year, 12, 27)] = "Christmas Day (substitute)"
    elif d.weekday() == 6: # Sunday -> substitute Monday
        holidays[date(year, 12, 27)] = "Christmas Day (substitute)"
    holidays[d] = "Christmas Day"

    # Boxing Day
    d = date(year, 12, 26)
    if d.weekday() == 5:
        holidays[date(year, 12, 28)] = "Boxing Day (substitute)"
    elif d.weekday() == 6:
        holidays[date(year, 12, 28)] = "Boxing Day (substitute)"
    holidays[d] = "Boxing Day"

    return holidays

all_holidays = {}
for y in [2024, 2025]:
    all_holidays.update(get_public_holidays(y))

dates = [START_DATE + timedelta(days=i) for i in range((END_DATE - START_DATE).days + 1)]

def season(d):
    m = d.month
    if m in [12, 1, 2]:  return "Winter"
    if m in [3, 4, 5]:   return "Spring"
    if m in [6, 7, 8]:   return "Summer"
    return "Autumn"

calendar_rows = []
for d in dates:
    is_holiday = d in all_holidays
    is_weekend = d.weekday() >= 5
    # "trading day" weight used later
    calendar_rows.append({
        "date": d,
        "day_of_week": d.strftime("%A"),
        "day_num": d.weekday(),          # 0=Mon
        "week_of_year": d.isocalendar()[1],
        "month": d.month,
        "month_name": d.strftime("%B"),
        "quarter": (d.month - 1) // 3 + 1,
        "year": d.year,
        "season": season(d),
        "is_weekend": is_weekend,
        "is_public_holiday": is_holiday,
        "holiday_name": all_holidays.get(d, ""),
    })

calendar = pd.DataFrame(calendar_rows)
calendar.to_csv(f"{OUTPUT_DIR}/calendar.csv", index=False)
print("calendar.csv written")

# ---------------------------------------------------------------------------
# WEATHER  (single weather station, daily)
# ---------------------------------------------------------------------------

def sinusoidal(dates, base, amplitude, phase_days=0, noise_std=0):
    """Seasonal sinusoidal curve + Gaussian noise."""
    vals = []
    for d in dates:
        doy = d.timetuple().tm_yday  # 1-365
        angle = 2 * math.pi * (doy - phase_days) / 365
        val = base + amplitude * math.sin(angle)
        val += np.random.normal(0, noise_std)
        vals.append(val)
    return vals

# Temperature: peak ~July (doy 196), trough ~Jan
temp_max = sinusoidal(dates, base=14, amplitude=8, phase_days=-80, noise_std=3)
temp_min = [t - np.random.uniform(5, 9) for t in temp_max]

# Rainfall: higher in Autumn/Winter
precip_chance = sinusoidal(dates, base=0.40, amplitude=-0.15, phase_days=-80, noise_std=0.05)
precip_mm = [max(0, np.random.exponential(8)) if np.random.random() < max(0.1, min(0.9, p)) else 0
             for p in precip_chance]

# Sunshine hours: peak in summer
sunshine = sinusoidal(dates, base=5, amplitude=4, phase_days=-80, noise_std=1)
sunshine = [max(0, min(16, s - (p * 0.3))) for s, p in zip(sunshine, precip_mm)]

# Wind speed (km/h)
wind = [max(0, np.random.gamma(shape=2, scale=10)) for _ in dates]

def weather_type(t_max, precip, sun):
    if precip > 10: return "Rainy"
    if precip > 0 and sun < 4: return "Overcast"
    if t_max < 3 and precip > 0: return "Snowy"
    if sun > 7 and t_max > 18: return "Sunny"
    if sun > 5: return "Partly Cloudy"
    return "Cloudy"

weather = pd.DataFrame({
    "date": dates,
    "temp_max_c": np.round(temp_max, 1),
    "temp_min_c": np.round(temp_min, 1),
    "temp_avg_c": np.round([(a + b) / 2 for a, b in zip(temp_max, temp_min)], 1),
    "precipitation_mm": np.round(precip_mm, 1),
    "sunshine_hours": np.round(sunshine, 1),
    "wind_speed_kmh": np.round(wind, 1),
    "weather_type": [weather_type(t, p, s) for t, p, s in zip(temp_max, precip_mm, sunshine)],
})
weather.to_csv(f"{OUTPUT_DIR}/weather.csv", index=False)
print("weather.csv written")

# ---------------------------------------------------------------------------
# PRICING  (price per unit, can change over time — occasional increases)
# ---------------------------------------------------------------------------

base_prices = {
    "Coffee":       3.50,
    "Cold Drinks":  2.80,
    "Sandwiches":   5.20,
    "Snacks":       1.90,
    "Bakery":       2.40,
}

# Premium shops charge slightly more
shop_price_multiplier = {
    1: 1.00, 2: 0.95, 3: 1.10, 4: 1.00, 5: 0.95,
    6: 1.15, 7: 1.20, 8: 0.90, 9: 0.95, 10: 1.05,
}

pricing_rows = []
for shop_id in range(1, 11):
    for cat in CATEGORIES:
        base = base_prices[cat] * shop_price_multiplier[shop_id]
        # Two possible price increases during the 2 years
        changes = sorted(random.sample(dates[60:], 2))
        effective_dates = [START_DATE] + changes
        price = base
        for i, eff_date in enumerate(effective_dates):
            if i > 0:
                price = round(price * random.uniform(1.02, 1.07), 2)  # 2-7% increase
            pricing_rows.append({
                "shop_id": shop_id,
                "category": cat,
                "effective_date": eff_date,
                "unit_price_gbp": round(price, 2),
            })

pricing = pd.DataFrame(pricing_rows).sort_values(["shop_id", "category", "effective_date"])
pricing.to_csv(f"{OUTPUT_DIR}/pricing.csv", index=False)
print("pricing.csv written")

# ---------------------------------------------------------------------------
# PROMOTIONS
# ---------------------------------------------------------------------------

promo_types = ["BOGOF", "20% Off", "Loyalty Double Points", "Meal Deal", "Happy Hour", "Seasonal Special"]

promo_rows = []
promo_id = 1

# Brand-wide seasonal promotions
seasonal_promos = [
    # (label, start, end, category, discount_pct, type)
    ("Veganuary", date(2024, 1, 2), date(2024, 1, 31), "Snacks",      15, "Seasonal Special"),
    ("Veganuary", date(2025, 1, 2), date(2025, 1, 31), "Snacks",      15, "Seasonal Special"),
    ("Valentine's", date(2024, 2, 10), date(2024, 2, 14), "Bakery",   10, "Seasonal Special"),
    ("Valentine's", date(2025, 2, 10), date(2025, 2, 14), "Bakery",   10, "Seasonal Special"),
    ("Summer Refresh", date(2024, 6, 1), date(2024, 8, 31), "Cold Drinks", 20, "20% Off"),
    ("Summer Refresh", date(2025, 6, 1), date(2025, 8, 31), "Cold Drinks", 20, "20% Off"),
    ("Back to School", date(2024, 9, 2), date(2024, 9, 20), "Sandwiches", 10, "Meal Deal"),
    ("Back to School", date(2025, 9, 1), date(2025, 9, 19), "Sandwiches", 10, "Meal Deal"),
    ("Christmas Bundle", date(2024, 12, 1), date(2024, 12, 24), "Bakery", 15, "Seasonal Special"),
    ("Christmas Bundle", date(2025, 12, 1), date(2025, 12, 24), "Bakery", 15, "Seasonal Special"),
    ("Winter Warmer",  date(2024, 11, 1), date(2024, 11, 30), "Coffee", 10, "Loyalty Double Points"),
    ("Winter Warmer",  date(2025, 11, 1), date(2025, 11, 30), "Coffee", 10, "Loyalty Double Points"),
]

for label, start, end, cat, disc, ptype in seasonal_promos:
    for shop_id in range(1, 11):
        promo_rows.append({
            "promo_id": promo_id,
            "promo_name": label,
            "shop_id": shop_id,
            "category": cat,
            "start_date": start,
            "end_date": end,
            "discount_pct": disc,
            "promo_type": ptype,
            "is_brand_wide": True,
        })
    promo_id += 1

# Local shop-specific promos (random)
for shop_id in range(1, 11):
    n_local = random.randint(4, 8)
    for _ in range(n_local):
        start_idx = random.randint(0, len(dates) - 30)
        duration = random.randint(7, 21)
        start = dates[start_idx]
        end = min(dates[start_idx + duration], END_DATE)
        cat = random.choice(CATEGORIES)
        disc = random.choice([10, 15, 20, 25])
        ptype = random.choice(promo_types)
        promo_rows.append({
            "promo_id": promo_id,
            "promo_name": f"Local Deal - {ptype}",
            "shop_id": shop_id,
            "category": cat,
            "start_date": start,
            "end_date": end,
            "discount_pct": disc,
            "promo_type": ptype,
            "is_brand_wide": False,
        })
        promo_id += 1

promotions = pd.DataFrame(promo_rows)
promotions.to_csv(f"{OUTPUT_DIR}/promotions.csv", index=False)
print("promotions.csv written")

# Build a lookup: (date, shop_id, category) -> (discount_pct, promo_type)
promo_lookup = {}
for _, row in promotions.iterrows():
    d = row["start_date"]
    while d <= row["end_date"] and d <= END_DATE:
        key = (d, row["shop_id"], row["category"])
        if key not in promo_lookup or row["discount_pct"] > promo_lookup[key][0]:
            promo_lookup[key] = (row["discount_pct"], row["promo_type"])
        d += timedelta(days=1)

# ---------------------------------------------------------------------------
# MEDIA SPEND  (weekly, by channel)
# ---------------------------------------------------------------------------

channels = ["TV", "Social Media", "Outdoor", "Radio", "Email"]

media_rows = []
# Generate by ISO week
week_starts = []
d = START_DATE
while d <= END_DATE:
    if d.weekday() == 0:  # Monday
        week_starts.append(d)
    d += timedelta(days=1)

for ws in week_starts:
    year = ws.year
    week = ws.isocalendar()[1]
    month = ws.month

    # Seasonal uplift on spend: more spend in Q4 and summer
    seasonal_factor = 1.0
    if month in [11, 12]:      seasonal_factor = 1.5   # Christmas push
    elif month in [6, 7, 8]:   seasonal_factor = 1.2   # Summer campaign
    elif month in [1]:         seasonal_factor = 0.7   # January cost-cutting

    for ch in channels:
        base_spends = {
            "TV":           8000,
            "Social Media": 4000,
            "Outdoor":      3500,
            "Radio":        2000,
            "Email":        500,
        }
        spend = base_spends[ch] * seasonal_factor * np.random.uniform(0.8, 1.2)
        # Occasional burst campaigns
        if np.random.random() < 0.05:
            spend *= random.uniform(2.0, 3.5)
        media_rows.append({
            "week_start_date": ws,
            "year": year,
            "week_of_year": week,
            "channel": ch,
            "spend_gbp": round(spend, 2),
        })

media_spend = pd.DataFrame(media_rows)
media_spend.to_csv(f"{OUTPUT_DIR}/media_spend.csv", index=False)
print("media_spend.csv written")

# Build weekly media total lookup (date -> total_spend_that_week)
media_weekly_total = {}
for _, row in media_spend.iterrows():
    ws = row["week_start_date"]
    spend = row["spend_gbp"]
    for i in range(7):
        d = ws + timedelta(days=i)
        media_weekly_total[d] = media_weekly_total.get(d, 0) + spend

# ---------------------------------------------------------------------------
# SALES  (daily, per shop, per category)
# ---------------------------------------------------------------------------

# Base daily unit volumes at "average" conditions
base_units = {
    "Coffee":       120,
    "Cold Drinks":   60,
    "Sandwiches":    80,
    "Snacks":        90,
    "Bakery":        70,
}

# Shop size / footfall multiplier
shop_volume_multiplier = {
    1: 1.20,  # City Central - busy high street
    2: 0.75,  # Westside Corner - quiet suburb
    3: 1.50,  # Station Stop - high footfall
    4: 0.90,  # Park Lane Cafe
    5: 1.10,  # Eastgate Drive-Through
    6: 1.00,  # Riverside Roast
    7: 1.30,  # Airport Terminal
    8: 0.85,  # Uni Quarter (term time effect applied separately)
    9: 0.95,  # Retail Park Hub
    10: 0.80, # Old Town Brew
}

# Day of week multipliers (0=Mon ... 6=Sun)
dow_multiplier = {
    0: 0.95,  # Monday
    1: 1.00,  # Tuesday
    2: 1.02,  # Wednesday
    3: 1.05,  # Thursday
    4: 1.15,  # Friday
    5: 0.85,  # Saturday (varies by shop type)
    6: 0.60,  # Sunday
}

# Season x category multipliers
season_cat_mult = {
    "Winter": {"Coffee": 1.35, "Cold Drinks": 0.60, "Sandwiches": 0.95, "Snacks": 1.05, "Bakery": 1.20},
    "Spring": {"Coffee": 1.10, "Cold Drinks": 0.85, "Sandwiches": 1.05, "Snacks": 1.00, "Bakery": 1.00},
    "Summer": {"Coffee": 0.80, "Cold Drinks": 1.60, "Sandwiches": 1.15, "Snacks": 1.10, "Bakery": 0.85},
    "Autumn": {"Coffee": 1.15, "Cold Drinks": 0.75, "Sandwiches": 1.00, "Snacks": 1.05, "Bakery": 1.10},
}

# Weather impact on category units
def weather_multiplier(cat, temp, precip, sun):
    mult = 1.0
    # Temperature effect
    if cat == "Coffee":
        mult *= max(0.6, 1.0 - (temp - 10) * 0.02)
    elif cat == "Cold Drinks":
        mult *= max(0.4, 1.0 + (temp - 15) * 0.04)
    # Rain reduces footfall overall
    if precip > 5:
        mult *= 0.88
    elif precip > 0:
        mult *= 0.95
    # Sunshine boosts outdoor/destination shops (we apply extra in shop logic)
    return mult

# Public holiday footfall factor
holiday_mult = {
    "Christmas Day": 0.0,
    "Boxing Day": 0.5,
    "New Year's Day": 0.3,
    "Good Friday": 0.7,
    "Easter Monday": 0.6,
    "Early May Bank Holiday": 0.65,
    "Spring Bank Holiday": 0.65,
    "Summer Bank Holiday": 0.60,
}

# University term dates (shop 8 affected)
def is_uni_term(d):
    y = d.year
    term_dates = [
        (date(y, 1, 8),  date(y, 3, 22)),   # Spring term
        (date(y, 4, 22), date(y, 6, 14)),   # Summer term
        (date(y, 9, 23), date(y, 12, 14)),  # Autumn term
    ]
    return any(start <= d <= end for start, end in term_dates)

# Build price lookup: (shop_id, category, date) -> unit_price
price_by_shop_cat = {}
for shop_id in range(1, 11):
    for cat in CATEGORIES:
        rows = pricing[(pricing.shop_id == shop_id) & (pricing.category == cat)].sort_values("effective_date")
        price_by_shop_cat[(shop_id, cat)] = list(zip(rows.effective_date, rows.unit_price_gbp))

def get_price(shop_id, cat, d):
    entries = price_by_shop_cat[(shop_id, cat)]
    price = entries[0][1]
    for eff_date, p in entries:
        if eff_date <= d:
            price = p
    return price

# Media spend uplift (lagged effect, log scale)
def media_uplift(d):
    total = media_weekly_total.get(d, 0)
    # Log uplift: doubling spend adds ~5% sales
    return 1.0 + 0.05 * math.log1p(total / 18000)

# Random event generator (one-off local spikes/dips)
random_events = {}
for shop_id in range(1, 11):
    n_events = random.randint(3, 8)
    for _ in range(n_events):
        d = random.choice(dates)
        duration = random.randint(1, 3)
        factor = random.choice([0.5, 0.6, 1.4, 1.6, 1.8, 2.0])  # dip or spike
        for i in range(duration):
            ed = d + timedelta(days=i)
            if ed <= END_DATE:
                random_events[(ed, shop_id)] = factor

weather_dict = {row.date: row for row in weather.itertuples()}
cal_dict = {row.date: row for row in calendar.itertuples()}

sales_rows = []

for d in dates:
    w = weather_dict[d]
    c = cal_dict[d]

    for shop_id in range(1, 11):
        s_mult = shop_volume_multiplier[shop_id]

        # Day of week
        dow = d.weekday()
        d_mult = dow_multiplier[dow]

        # Weekend adjustment by shop type
        shop_type = shops.loc[shops.shop_id == shop_id, "shop_type"].values[0]
        if dow >= 5:  # weekend
            if shop_type in ["High Street", "Destination", "Retail Park"]:
                d_mult *= 1.20  # weekend boost
            elif shop_type == "Transport Hub":
                d_mult *= 0.80  # fewer commuters

        # Public holiday
        h_mult = 1.0
        if c.is_public_holiday:
            h_mult = holiday_mult.get(c.holiday_name, 0.5)

        # Uni term (shop 8)
        uni_mult = 1.0
        if shop_id == 8:
            uni_mult = 1.0 if is_uni_term(d) else 0.45

        # Random event
        rand_mult = random_events.get((d, shop_id), 1.0)

        # Media uplift
        m_mult = media_uplift(d)

        for cat in CATEGORIES:
            base = base_units[cat]

            # Season x category
            seas_mult = season_cat_mult[c.season][cat]

            # Weather effect
            w_mult = weather_multiplier(cat, w.temp_avg_c, w.precipitation_mm, w.sunshine_hours)

            # Promotion effect: discount drives volume up
            promo = promo_lookup.get((d, shop_id, cat))
            promo_vol_mult = 1.0
            promo_price_adj = 1.0
            if promo:
                disc = promo[0] / 100
                promo_vol_mult = 1 + disc * 1.5   # price elasticity ~1.5
                promo_price_adj = 1 - disc

            # Price elasticity vs base price (minor day-to-day effect)
            unit_price = get_price(shop_id, cat, d)
            base_p = base_prices[cat] * shop_price_multiplier[shop_id]
            price_ratio = unit_price / base_p
            price_vol_mult = price_ratio ** -0.8  # elasticity -0.8

            # Compose all multipliers
            units_float = (
                base
                * s_mult
                * d_mult
                * h_mult
                * uni_mult
                * seas_mult
                * w_mult
                * promo_vol_mult
                * price_vol_mult
                * m_mult
                * rand_mult
                * np.random.lognormal(0, 0.08)  # residual noise
            )

            units_sold = max(0, int(round(units_float)))
            effective_price = round(unit_price * promo_price_adj, 2)
            revenue = round(units_sold * effective_price, 2)

            sales_rows.append({
                "date": d,
                "shop_id": shop_id,
                "category": cat,
                "units_sold": units_sold,
                "unit_price_gbp": effective_price,
                "revenue_gbp": revenue,
                "in_promotion": promo is not None,
                "promotion_discount_pct": promo[0] if promo else 0,
            })

sales = pd.DataFrame(sales_rows)
sales.to_csv(f"{OUTPUT_DIR}/sales.csv", index=False)
print(f"sales.csv written  ({len(sales):,} rows)")

# ---------------------------------------------------------------------------
# SUMMARY STATS
# ---------------------------------------------------------------------------

print("\n=== DATASET SUMMARY ===")
print(f"Date range  : {START_DATE} to {END_DATE} ({len(dates)} days)")
print(f"Shops       : {len(shops)}")
print(f"Categories  : {CATEGORIES}")
print(f"\nSales totals by category:")
print(sales.groupby("category")[["units_sold", "revenue_gbp"]].sum().to_string())
print(f"\nSales totals by shop:")
print(sales.groupby("shop_id")[["units_sold", "revenue_gbp"]].sum().to_string())
print(f"\nWeather sample:")
print(weather.groupby("weather_type").size().sort_values(ascending=False).to_string())
print(f"\nPublic holidays in dataset: {sum(1 for v in calendar['is_public_holiday'] if v)}")
print(f"\nOutput files in ./{OUTPUT_DIR}/")
for f in sorted(os.listdir(OUTPUT_DIR)):
    size = os.path.getsize(f"{OUTPUT_DIR}/{f}")
    print(f"  {f:30s}  {size:>10,} bytes")
