"""Generate the toy Airbnb cancellation dataset for the Colab demo.

Setting
-------
Airbnb tests a host-side reliability program meant to reduce cancellations. It is
a market-wide policy, and hosts in a market compete for the same guests, so the
test is randomized by MARKET: 60 markets, 30 treated.

    randomization unit : market    (60)
    observation unit   : booking   (thousands per market)
    metric             : cancel rate = cancelled bookings / bookings

The pre-period is spring, dominated by solo and business travelers. The
experiment runs in July, when families arrive -- and they arrive in some markets
(beach towns) far more than in others (business cities). Families book
multi-bedroom homes, further ahead, and both of those cancel more.

Standard CUPED uses each market's spring cancel rate. That knows each market's
persistent personality, but it describes a crowd that has since left. Booking-level
CUPED learns P(cancel | home type, lead time) in spring and applies it to the
bookings that actually arrived in July, so it sees the new mix.

What has to be true
-------------------
1. The July shift differs across markets. A uniform shift is absorbed by the
   intercept and the method gains nothing (verified: the edge drops to ~0).
2. P(cancel | booking features) is stable across seasons -- only the mix moves.
3. Spring contains some family-style bookings, so the model can learn their rate.
4. The treatment cannot change who books. A guest-invisible host program
   satisfies this; a more flexible guest cancellation policy would not, because
   it would attract families and the July mix would itself be an outcome.

The model only ever sees `home_type` and `lead_time_days`. The guest segment
(family vs solo/business) drives those features but is never in the data, which
is realistic: you observe what people book, not who they are.

Calibration
-----------
Twelve independently generated datasets, each run through exactly the notebook's
procedure (gradient-boosted model on 70% of spring, 500 re-randomizations):

    seed   standard CUPED   booking-level   both    gap
      1        21.8%            58.5%       67.4%   36.7
      2         4.1%            29.4%       40.7%   25.2
      3        35.0%            54.1%       65.9%   19.2
      4        20.8%            32.6%       51.7%   11.8
      5        15.4%            52.3%       61.8%   36.9
      6        24.9%            58.7%       69.8%   33.8
      7        30.0%            47.9%       60.5%   17.9
      8        11.8%            40.2%       50.8%   28.4
      9        16.4%            53.4%       61.5%   37.1
     10        28.1%            50.8%       66.2%   22.6   <- SHIPPED
     11        18.8%            32.5%       49.8%   13.7
     12        23.4%            56.3%       69.6%   32.9
    mean       20.9%            47.2%       59.6%   26.3

Booking-level CUPED beat standard CUPED in 12 of 12. Seed 10 is shipped because its
levels sit near the average and its gap is slightly BELOW average, so the demo
understates the advantage rather than overselling it. Magnitudes swing a lot with
only 60 markets; the ordering is what is stable.

The model's AUC is only about 0.61 -- individual cancellations are mostly noise --
yet it removes roughly half the variance at the market level. Averaging over
thousands of bookings cancels the individual noise and keeps the mix signal.
"""

import numpy as np
import pandas as pd

N_MARKETS = 60
N_SPRING = 1_500  # bookings per market, pre-period (used to train the model)
N_JULY = 2_500  # bookings per market, experiment period
TAU = -0.15  # host program, as a shift in log-odds of cancelling

# Family share of bookings = base + slope * family_appeal, appeal ~ U(0, 1).
# Spring barely varies across markets; July varies a lot. That difference is the design.
FAMILY_SPRING = (0.12, 0.10)
FAMILY_JULY = (0.15, 0.60)

MARKET_SD = 0.12  # persistent market cancel propensity (log-odds); spring rate captures it
SHOCK_SD = 0.08  # period-specific market shock (log-odds); nothing can predict it


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def cancel_logit(multi_br, lead_time_days):
    """P(cancel | booking) on the log-odds scale. Identical in spring and July."""
    lead = np.log1p(lead_time_days / 14.0)
    return -2.45 + 0.50 * multi_br + 0.38 * lead + 0.12 * multi_br * lead


def make_period(rng, n, family_share, market_effect, period):
    """One period of bookings for every market."""
    market = np.repeat(np.arange(N_MARKETS), n)
    total = market.size

    # The guest segment drives what gets booked. It is NOT written to the data.
    family = rng.random(total) < family_share[market]
    multi_br = np.where(family, rng.random(total) < 0.85, rng.random(total) < 0.12).astype(int)
    lead = np.where(family, rng.gamma(2.2, 25.0, total), rng.gamma(1.4, 9.0, total))
    lead = np.clip(lead, 0, 330)

    shock = rng.normal(0, SHOCK_SD, N_MARKETS)
    logit = cancel_logit(multi_br, lead) + market_effect[market] + shock[market]

    return pd.DataFrame(
        {
            "market": market,
            "period": period,
            "home_type": np.where(multi_br == 1, "multi_br", "1br"),
            "lead_time_days": np.round(lead).astype(int),
            "p0": _sigmoid(logit),
            "p1": _sigmoid(logit + TAU),
        }
    )


def build(seed=10):
    """Return the full booking-level toy dataset, both periods."""
    rng = np.random.default_rng(seed)
    appeal = rng.uniform(0, 1, N_MARKETS)  # beach town (1) vs business city (0)
    market_effect = rng.normal(0, MARKET_SD, N_MARKETS)

    spring = make_period(rng, N_SPRING, FAMILY_SPRING[0] + FAMILY_SPRING[1] * appeal,
                         market_effect, "spring")
    july = make_period(rng, N_JULY, FAMILY_JULY[0] + FAMILY_JULY[1] * appeal,
                       market_effect, "july")
    df = pd.concat([spring, july], ignore_index=True)

    # Realized outcome with nobody treated. The notebook re-randomizes treatment
    # itself and redraws July outcomes from p0 / p1.
    df["cancelled"] = (rng.random(len(df)) < df.p0.values).astype("int8")
    return df


def write_csv(path, seed=10):
    df = build(seed)
    df["p0"], df["p1"] = df.p0.round(5), df.p1.round(5)
    df.to_csv(path, index=False)
    return df


if __name__ == "__main__":
    import pathlib

    p = pathlib.Path(__file__).resolve().parents[1] / "data" / "airbnb_cancellations.csv.gz"
    p.parent.mkdir(exist_ok=True)
    d = write_csv(p)
    print(f"{len(d):,} rows -> {p}  ({p.stat().st_size / 1e6:.1f} MB)")
