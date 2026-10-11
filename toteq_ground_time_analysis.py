from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import optimize, stats


BASE = Path(__file__).resolve().parent
OUT = BASE / "analysis_outputs"
OUT.mkdir(exist_ok=True)
FIG = BASE.parent / "figures"
FIG.mkdir(exist_ok=True)
CSV = BASE / "base_unificada.csv"


COLS = [
    "misrel",
    "activation",
    "aircraft",
    "unit",
    "aerial_effort",
    "ground",
    "proc_type",
    "pax",
    "organ",
    "qty",
    "cost",
    "mission_result",
    "organ_status",
    "cancel_reason",
    "region",
    "origin",
    "destination",
    "route",
    "year",
]


def hhmm_to_min(value: str) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if not text:
        return np.nan
    if ":" in text:
        hour, minute = text.split(":", 1)
        return int(hour) * 60 + int(minute)
    return float(text)


def parse_activation(series: pd.Series) -> pd.Series:
    text = series.str.replace("Z", "", regex=False).str.strip()
    dt = pd.to_datetime(text, format="%d/%m/%Y %H:%M", errors="coerce")
    missing = dt.isna()
    dt.loc[missing] = pd.to_datetime(text.loc[missing], format="%d/%m/%Y %H%M", errors="coerce")
    return dt


def organ_profile(values: pd.Series) -> str:
    clean = sorted(v for v in values.dropna().unique() if str(v).strip())
    return " + ".join(clean)


def read_records() -> pd.DataFrame:
    df = pd.read_csv(CSV, sep=";", dtype=str, encoding="utf-8-sig")
    df.columns = COLS
    for col in df.columns:
        df[col] = df[col].astype(str).str.strip().replace({"nan": np.nan})
    df["ground_min"] = df["ground"].map(hhmm_to_min)
    df["activation_dt"] = parse_activation(df["activation"])
    # Repair activation timestamps whose parsed year falls outside the data
    # collection window (Jan 2023 - May 2025) while the operational `year`
    # field is valid. One record (MISREL 3242331) carried an activation year
    # of 2021, whereas its operational year field and mission context indicate
    # 2024; the year component is reconciled to the operational year field.
    year_col = pd.to_numeric(df["year"], errors="coerce")
    out_of_window = df["activation_dt"].notna() & (
        (df["activation_dt"].dt.year < 2023) | (df["activation_dt"].dt.year > 2025)
    )
    fixable = out_of_window & year_col.between(2023, 2025)
    if fixable.any():
        df.loc[fixable, "activation_dt"] = [
            ts.replace(year=int(y))
            for ts, y in zip(df.loc[fixable, "activation_dt"], year_col.loc[fixable])
        ]
    df["mission_key"] = df["misrel"]
    invalid_misrel = ~df["misrel"].str.fullmatch(r"\d+", na=False)
    # Non-numeric MISREL values in the source contain air-unit labels or placeholders.
    # They are kept as distinct missions using activation and route information.
    repaired = df.loc[invalid_misrel].copy()
    df.loc[invalid_misrel, "mission_key"] = [
        f"ROW_{idx}_{row.activation_dt:%Y%m%d%H%M}_{row.route or ''}"
        for idx, row in repaired.iterrows()
    ]
    return df


def mission_level(df: pd.DataFrame) -> pd.DataFrame:
    mission = (
        df.sort_values(["mission_key", "activation_dt"])
        .groupby("mission_key", as_index=False)
        .agg(
            misrel=("misrel", "first"),
            activation_dt=("activation_dt", "first"),
            ground_min=("ground_min", "first"),
            proc_type=("proc_type", "first"),
            organs=("organ", organ_profile),
            organ_count=("organ", "nunique"),
            organ_rows=("organ", "size"),
            transported_organs=("organ_status", lambda s: int((s == "TRANSPORTADO").sum())),
            aircraft=("aircraft", "first"),
            unit=("unit", "first"),
            origin=("origin", "first"),
            destination=("destination", "first"),
            route=("route", "first"),
            year=("year", "first"),
        )
    )
    mission["month"] = mission["activation_dt"].dt.to_period("M").astype(str)
    mission["quarter"] = mission["activation_dt"].dt.to_period("Q").astype(str)
    return mission


def fit_nbinom(k: np.ndarray) -> dict[str, float]:
    k = np.asarray(k, dtype=int)

    def nll(par: np.ndarray) -> float:
        log_mu, log_alpha = par
        mu = math.exp(log_mu)
        alpha = math.exp(log_alpha)
        size = 1.0 / alpha
        prob = size / (size + mu)
        ll = stats.nbinom.logpmf(k, size, prob)
        return -float(np.sum(ll))

    mu0 = max(float(np.mean(k)), 0.1)
    var0 = float(np.var(k, ddof=1))
    alpha0 = max((var0 - mu0) / (mu0**2), 1e-3)
    res = optimize.minimize(nll, np.log([mu0, alpha0]), method="Nelder-Mead")
    mu = math.exp(res.x[0])
    alpha = math.exp(res.x[1])
    size = 1.0 / alpha
    prob = size / (size + mu)
    loglik = -nll(res.x)
    return {"model": "Negative binomial", "mu": mu, "alpha": alpha, "size": size, "prob": prob, "loglik": loglik, "kpar": 2}


def fit_poisson(k: np.ndarray) -> dict[str, float]:
    mu = float(np.mean(k))
    loglik = float(np.sum(stats.poisson.logpmf(k, mu)))
    return {"model": "Poisson", "mu": mu, "loglik": loglik, "kpar": 1}


def grouped_loglik(dist, params: tuple[float, ...], x: np.ndarray) -> float:
    lower = np.maximum(0.0, x - 2.5)
    upper = x + 2.5
    prob = dist.cdf(upper, *params) - dist.cdf(lower, *params)
    prob = np.clip(prob, 1e-300, 1.0)
    return float(np.sum(np.log(prob)))


CONTINUOUS_DISTS = {
    "Gamma": stats.gamma,
    "Lognormal": stats.lognorm,
    "Weibull": stats.weibull_min,
    "Log-logistic": stats.fisk,
}


def fit_continuous(name: str, x: np.ndarray) -> dict[str, object]:
    # Parameters are estimated by maximizing the grouped (interval) log-likelihood
    # in grouped_loglik, so both estimation and AIC account for five-minute rounding.
    dist = CONTINUOUS_DISTS[name]
    x = np.asarray(x, dtype=float)
    shape0, _, scale0 = dist.fit(x, floc=0)
    shape0 = max(float(shape0), 1e-3)
    scale0 = max(float(scale0), 1e-3)

    def nll(theta: np.ndarray) -> float:
        shape, scale = math.exp(theta[0]), math.exp(theta[1])
        return -grouped_loglik(dist, (shape, 0.0, scale), x)

    res = optimize.minimize(nll, np.log([shape0, scale0]), method="Nelder-Mead")
    shape, scale = math.exp(res.x[0]), math.exp(res.x[1])
    params = (shape, 0.0, scale)
    loglik = grouped_loglik(dist, params, x)
    return {"model": name, "params": params, "loglik": loglik, "kpar": 2}


def model_fits(x: np.ndarray) -> pd.DataFrame:
    x = np.asarray(x, dtype=float)
    k = np.rint(x / 5.0).astype(int)
    rows = [fit_poisson(k), fit_nbinom(k)]
    for name in ["Gamma", "Lognormal", "Weibull", "Log-logistic"]:
        rows.append(fit_continuous(name, x))
    out = pd.DataFrame(rows)
    out["aic"] = 2 * out["kpar"] - 2 * out["loglik"]
    out = out.sort_values("aic").reset_index(drop=True)
    out["delta_aic"] = out["aic"] - out.loc[0, "aic"]
    return out


def ppf_from_fit(fit: dict, p: float) -> float:
    model = fit["model"]
    if model == "Negative binomial":
        return 5.0 * stats.nbinom.ppf(p, fit["size"], fit["prob"])
    if model == "Poisson":
        return 5.0 * stats.poisson.ppf(p, fit["mu"])
    dist = {
        "Gamma": stats.gamma,
        "Lognormal": stats.lognorm,
        "Weibull": stats.weibull_min,
        "Log-logistic": stats.fisk,
    }[model]
    return float(dist.ppf(p, *fit["params"]))


def survival_from_fit(fit: dict, threshold: float) -> float:
    model = fit["model"]
    if model == "Negative binomial":
        return float(stats.nbinom.sf(math.floor(threshold / 5.0), fit["size"], fit["prob"]))
    if model == "Poisson":
        return float(stats.poisson.sf(math.floor(threshold / 5.0), fit["mu"]))
    dist = {
        "Gamma": stats.gamma,
        "Lognormal": stats.lognorm,
        "Weibull": stats.weibull_min,
        "Log-logistic": stats.fisk,
    }[model]
    return float(dist.sf(threshold, *fit["params"]))


def cdf_from_fit(fit: dict, x: np.ndarray) -> np.ndarray:
    model = fit["model"]
    x = np.asarray(x, dtype=float)
    if model == "Negative binomial":
        return stats.nbinom.cdf(np.rint(x / 5.0), fit["size"], fit["prob"])
    if model == "Poisson":
        return stats.poisson.cdf(np.rint(x / 5.0), fit["mu"])
    dist = {
        "Gamma": stats.gamma,
        "Lognormal": stats.lognorm,
        "Weibull": stats.weibull_min,
        "Log-logistic": stats.fisk,
    }[model]
    return dist.cdf(x, *fit["params"])


def fit_named_model(model: str, x: np.ndarray) -> dict:
    if model == "Poisson":
        return fit_poisson(np.rint(x / 5.0).astype(int))
    if model == "Negative binomial":
        return fit_nbinom(np.rint(x / 5.0).astype(int))
    return fit_continuous(model, x)


def simulate_from_fit(fit: dict, n: int, rng: np.random.Generator) -> np.ndarray:
    model = fit["model"]
    if model == "Negative binomial":
        return 5.0 * rng.negative_binomial(fit["size"], fit["prob"], size=n)
    if model == "Poisson":
        return 5.0 * rng.poisson(fit["mu"], size=n)
    dist = CONTINUOUS_DISTS[model]
    # Round simulated durations to five-minute blocks to match the recorded-data
    # heaping, consistent with the grouped-likelihood estimation.
    return 5.0 * np.rint(dist.rvs(*fit["params"], size=n, random_state=rng) / 5.0)


def bootstrap_summary(x: np.ndarray, model: str, b: int = 5000) -> dict[str, float]:
    rng = np.random.default_rng(20250622)
    x = np.asarray(x, dtype=float)
    fit = fit_named_model(model, x)
    n = len(x)
    p50, p95, exc180 = [], [], []
    for _ in range(b):
        sample = simulate_from_fit(fit, n, rng)
        try:
            refit = fit_named_model(model, sample)
            p50.append(ppf_from_fit(refit, 0.50))
            p95.append(ppf_from_fit(refit, 0.95))
            exc180.append(survival_from_fit(refit, 180.0))
        except Exception:
            continue
    return {
        "p50": ppf_from_fit(fit, 0.50),
        "p50_l": float(np.percentile(p50, 2.5)),
        "p50_u": float(np.percentile(p50, 97.5)),
        "p95": ppf_from_fit(fit, 0.95),
        "p95_l": float(np.percentile(p95, 2.5)),
        "p95_u": float(np.percentile(p95, 97.5)),
        "p95_upper_95": float(np.percentile(p95, 95.0)),
        "exceed_180": survival_from_fit(fit, 180.0),
        "exceed_180_l": float(np.percentile(exc180, 2.5)),
        "exceed_180_u": float(np.percentile(exc180, 97.5)),
    }


PROC_LABELS = {
    "DESACOMPANHADO": "Unaccompanied",
    "SIMPLES": "Simple",
    "MÚLTIPLA": "Multiple",
}
PROC_COLORS = {
    "Unaccompanied": "#2c6e9c",
    "Simple": "#b5482f",
    "Multiple": "#5a7d2a",
}
ENGLISH_ORGAN = {
    "FÍGADO": "Liver",
    "CORAÇÃO": "Heart",
    "PULMÃO": "Lung",
    "BAÇO": "Spleen",
    "LINFONODOS": "Lymph nodes",
    "RINS": "Kidneys",
    "CÓRNEA": "Cornea",
    "PÂNCREAS": "Pancreas",
}


ENGLISH_RESULT = {"REALIZADA": "Executed", "CANCELADA": "Cancelled"}
ENGLISH_STATUS = {"TRANSPORTADO": "Transported", "NÃO TRANSPORTADO": "Not transported"}
ENGLISH_CANCEL = {"CNT": "CNT", "FAB": "FAB", "METEORO": "Weather", "NA": "Not applicable"}
ENGLISH_REGION = {
    "CO": "Central-West",
    "N": "North",
    "NE": "Northeast",
    "S": "South",
    "SE": "Southeast",
}


def translate_profile(profile: str) -> str:
    parts = [p.strip() for p in profile.split("+")]
    return " + ".join(ENGLISH_ORGAN.get(p, p.title()) for p in parts)


def _remap(series: pd.Series, mapping: dict) -> pd.Series:
    return series.map(lambda v: mapping.get(v, v))


def translate_records(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["proc_type"] = _remap(out["proc_type"], PROC_LABELS)
    out["organ"] = _remap(out["organ"], ENGLISH_ORGAN)
    out["mission_result"] = _remap(out["mission_result"], ENGLISH_RESULT)
    out["organ_status"] = _remap(out["organ_status"], ENGLISH_STATUS)
    out["cancel_reason"] = _remap(out["cancel_reason"], ENGLISH_CANCEL)
    out["region"] = _remap(out["region"], ENGLISH_REGION)
    return out


def translate_mission(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["proc_type"] = _remap(out["proc_type"], PROC_LABELS)
    out["organs"] = out["organs"].map(
        lambda v: translate_profile(v) if isinstance(v, str) and v else v
    )
    return out


def make_temporal_figure(temporal: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for proc in ["DESACOMPANHADO", "SIMPLES", "MÚLTIPLA"]:
        g = temporal[(temporal["proc_type"] == proc) & (temporal["n"] >= 3)].copy()
        g = g.sort_values("quarter")
        label = PROC_LABELS[proc]
        ax.plot(g["quarter"], g["median"], marker="o", label=label, color=PROC_COLORS[label])
    ax.set_xlabel("Quarter")
    ax.set_ylabel("Quarterly median critical ground time (minutes)")
    ax.legend(loc="upper left", frameon=False)
    ax.grid(axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(FIG / "fig_14.pdf")
    plt.close(fig)


MODEL_DISPLAY = {
    "Negative binomial": "Negative Binomial",
    "Poisson": "Poisson",
    "Gamma": "Gamma",
    "Lognormal": "Lognormal",
    "Weibull": "Weibull",
    "Log-logistic": "Log-logistic",
}
FIG_ORDER = ["Unaccompanied", "Simple", "Multiple", "Simple-Heart", "Simple-Liver", "Simple-Lung"]


def clean_label(label: str) -> str:
    return label.replace(" (mission-level)", "").replace("--", "-")


def _ordered(diag: list[dict]) -> list[dict]:
    by_label = {clean_label(d["label"]): d for d in diag}
    return [by_label[k] for k in FIG_ORDER if k in by_label]


def density_curve(fit: dict, x_max: float):
    model = fit["model"]
    if model == "Negative binomial":
        k = np.arange(0, int(x_max / 5) + 2)
        return 5.0 * k, stats.nbinom.pmf(k, fit["size"], fit["prob"]) / 5.0
    if model == "Poisson":
        k = np.arange(0, int(x_max / 5) + 2)
        return 5.0 * k, stats.poisson.pmf(k, fit["mu"]) / 5.0
    dist = CONTINUOUS_DISTS[model]
    x = np.linspace(0.0, x_max, 400)
    return x, dist.pdf(x, *fit["params"])


def make_boxplot_figure(diag: list[dict]) -> None:
    items = _ordered(diag)
    data = [np.sort(it["x"]) for it in items]
    labels = [clean_label(it["label"]) for it in items]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bp = ax.boxplot(
        data,
        patch_artist=True,
        labels=labels,
        widths=0.6,
        flierprops=dict(marker="o", markerfacecolor="#c44e52", markeredgecolor="#c44e52", markersize=4, alpha=0.6),
    )
    for box in bp["boxes"]:
        box.set(facecolor="#cfe0ef", edgecolor="#2c6e9c")
    for med in bp["medians"]:
        med.set(color="black", linewidth=1.3)
    for element in bp["whiskers"] + bp["caps"]:
        element.set(color="#2c6e9c")
    ax.set_ylabel("Critical ground time (minutes)")
    ax.set_xlabel("Mission-level stratum")
    ax.grid(axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(FIG / "fig_3.pdf")
    plt.close(fig)


def make_distribution_figure(diag: list[dict]) -> None:
    items = _ordered(diag)
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.5))
    axes = axes.ravel()
    for ax, it in zip(axes, items):
        x = np.asarray(it["x"], dtype=float)
        x_max = float(x.max()) * 1.05
        bins = np.arange(0, x_max + 25, 25)
        ax.hist(x, bins=bins, density=True, color="#cfe0ef", edgecolor="#2c6e9c", linewidth=0.5)
        gx, gy = density_curve(it["fit"], x_max)
        ax.plot(gx, gy, color="#c44e52", linewidth=1.6)
        ax.set_title(f"{clean_label(it['label'])} ({MODEL_DISPLAY.get(it['selected'], it['selected'])})", fontsize=9)
        ax.set_xlabel("Minutes", fontsize=8)
        ax.set_ylabel("Density", fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in axes[len(items):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIG / "fig_4.pdf")
    plt.close(fig)


def make_diagnostic_figure(diag: list[dict]) -> None:
    n_strata = len(diag)
    fig, axes = plt.subplots(n_strata, 2, figsize=(9, 2.6 * n_strata))
    if n_strata == 1:
        axes = axes.reshape(1, 2)
    for row, item in enumerate(diag):
        x = np.sort(item["x"])
        fit = item["fit"]
        label = item["label"]
        n = len(x)
        pp = (np.arange(1, n + 1) - 0.5) / n
        theo_q = np.array([ppf_from_fit(fit, p) for p in pp])
        theo_cdf = cdf_from_fit(fit, x)

        ax_q = axes[row, 0]
        lim = [min(x.min(), theo_q.min()), max(x.max(), theo_q.max())]
        ax_q.plot(lim, lim, "k--", lw=1)
        ax_q.scatter(x, theo_q, s=12, color="#b5482f", alpha=0.7)
        ax_q.set_xlabel("Empirical quantiles (min)")
        ax_q.set_ylabel("Theoretical quantiles (min)")
        ax_q.set_title(f"{label} - {item['selected']}: Q-Q", fontsize=9)

        ax_p = axes[row, 1]
        ax_p.plot([0, 1], [0, 1], "k--", lw=1)
        ax_p.scatter(pp, np.sort(theo_cdf), s=12, color="#2c6e9c", alpha=0.7)
        ax_p.set_xlabel("Empirical cumulative probability")
        ax_p.set_ylabel("Theoretical cumulative probability")
        ax_p.set_title(f"{label} - {item['selected']}: P-P", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "fig_15.pdf")
    plt.close(fig)


def main() -> None:
    records = read_records()
    mission = mission_level(records)
    translate_records(records).to_csv(OUT / "records_clean.csv", index=False)
    translate_mission(mission).to_csv(OUT / "mission_level.csv", index=False)

    audit = pd.DataFrame(
        {
            "metric": [
                "organ_record_rows",
                "raw_unique_misrel_values",
                "numeric_unique_misrel_values",
                "non_numeric_misrel_rows",
                "mission_level_rows_after_key_repair",
                "missions_with_repeated_organ_rows",
            ],
            "value": [
                len(records),
                records["misrel"].nunique(),
                records.loc[records["misrel"].str.fullmatch(r"\d+", na=False), "misrel"].nunique(),
                int((~records["misrel"].str.fullmatch(r"\d+", na=False)).sum()),
                len(mission),
                int((mission["organ_rows"] > 1).sum()),
            ],
        }
    )
    audit.to_csv(OUT / "audit_counts.csv", index=False)

    desc = (
        mission.groupby("proc_type")["ground_min"]
        .agg(n="count", mean="mean", median="median", sd="std", q25=lambda s: s.quantile(0.25), q75=lambda s: s.quantile(0.75), min="min", max="max")
        .round(2)
        .reset_index()
    )
    desc["proc_type"] = desc["proc_type"].map(PROC_LABELS)
    desc.to_csv(OUT / "mission_descriptive_by_modality.csv", index=False)

    organ_profiles = (
        mission.groupby(["proc_type", "organs"])["ground_min"]
        .agg(n="count", mean="mean", median="median", min="min", max="max")
        .round(2)
        .sort_values(["proc_type", "n"], ascending=[True, False])
        .reset_index()
    )
    organ_profiles["proc_type"] = organ_profiles["proc_type"].map(PROC_LABELS)
    organ_profiles["organs"] = organ_profiles["organs"].map(translate_profile)
    organ_profiles.to_csv(OUT / "mission_descriptive_by_organ_profile.csv", index=False)

    strata: dict[str, pd.Series] = {
        f"{PROC_LABELS[proc]} (mission-level)": group["ground_min"]
        for proc, group in mission.groupby("proc_type")
    }
    simple = mission[mission["proc_type"] == "SIMPLES"]
    for organ, group in simple.groupby("organs"):
        if len(group) >= 30:
            strata[f"Simple--{translate_profile(organ)}"] = group["ground_min"]

    all_fits = []
    summaries = []
    diagnostics = []
    for label, values in strata.items():
        x = values.dropna().to_numpy(dtype=float)
        fits = model_fits(x)
        fits.insert(0, "stratum", label)
        fits.insert(1, "n", len(x))
        all_fits.append(fits[["stratum", "n", "model", "aic", "delta_aic"]])
        selected = fits.loc[0, "model"]
        summary = bootstrap_summary(x, selected)
        summaries.append({"stratum": label, "n": len(x), "selected_model": selected, **summary})
        diagnostics.append(
            {"label": label, "x": x, "selected": selected, "fit": fit_named_model(selected, x)}
        )

    pd.concat(all_fits, ignore_index=True).round(3).to_csv(OUT / "model_comparison_grouped_likelihood.csv", index=False)
    pd.DataFrame(summaries).round(3).to_csv(OUT / "bootstrap_percentile_summary.csv", index=False)

    threshold_rows = []
    for label, values in strata.items():
        if "(mission-level)" not in label:
            continue
        x = values.dropna().to_numpy(dtype=float)
        selected = model_fits(x).loc[0, "model"]
        fit = fit_named_model(selected, x)
        threshold_rows.append(
            {
                "modality": label.replace(" (mission-level)", ""),
                "model": selected,
                "n": len(x),
                "p_gt_120": survival_from_fit(fit, 120.0),
                "p_gt_180": survival_from_fit(fit, 180.0),
                "p_gt_240": survival_from_fit(fit, 240.0),
            }
        )
    pd.DataFrame(threshold_rows).round(4).to_csv(OUT / "threshold_sensitivity.csv", index=False)

    temporal = (
        mission.groupby(["quarter", "proc_type"])["ground_min"]
        .agg(n="count", median="median", mean="mean", p95=lambda s: s.quantile(0.95))
        .round(2)
        .reset_index()
    )
    temporal.assign(proc_type=temporal["proc_type"].map(PROC_LABELS)).to_csv(
        OUT / "quarterly_temporal_summary.csv", index=False
    )

    validation_rows = []
    train = mission[mission["activation_dt"] < "2025-01-01"]
    test = mission[mission["activation_dt"] >= "2025-01-01"]
    for proc in sorted(mission["proc_type"].dropna().unique()):
        x_train = train.loc[train["proc_type"] == proc, "ground_min"].dropna().to_numpy(dtype=float)
        x_test = test.loc[test["proc_type"] == proc, "ground_min"].dropna().to_numpy(dtype=float)
        if len(x_train) < 20 or len(x_test) < 5:
            continue
        selected = model_fits(x_train).loc[0, "model"]
        fit = fit_named_model(selected, x_train)
        p95 = ppf_from_fit(fit, 0.95)
        pred_exc180 = survival_from_fit(fit, 180)
        validation_rows.append(
            {
                "proc_type": PROC_LABELS.get(proc, proc),
                "train_n_2023_2024": len(x_train),
                "test_n_2025": len(x_test),
                "selected_model_train": selected,
                "train_predicted_p95": p95,
                "test_exceed_p95_rate": float(np.mean(x_test > p95)),
                "train_predicted_exceed_180": pred_exc180,
                "test_observed_exceed_180": float(np.mean(x_test > 180)),
                "test_median": float(np.median(x_test)),
            }
        )
    pd.DataFrame(validation_rows).round(3).to_csv(OUT / "chronological_validation_2025.csv", index=False)

    # Simple non-parametric temporal trend screen using quarterly medians.
    trend_rows = []
    for proc, group in temporal.groupby("proc_type"):
        g = group[group["n"] >= 3].copy()
        if len(g) < 3:
            continue
        rho, p = stats.spearmanr(np.arange(len(g)), g["median"])
        trend_rows.append({"proc_type": PROC_LABELS.get(proc, proc), "quarters": len(g), "spearman_rho_quarter_median": rho, "p_value": p})
    pd.DataFrame(trend_rows).round(3).to_csv(OUT / "temporal_trend_screen.csv", index=False)

    make_boxplot_figure(diagnostics)
    make_distribution_figure(diagnostics)
    make_temporal_figure(temporal)
    make_diagnostic_figure(diagnostics)

    print("Wrote analysis outputs to", OUT)
    print("Wrote figures to", FIG)
    print(audit.to_string(index=False))
    print(pd.DataFrame(summaries).round(2).to_string(index=False))


if __name__ == "__main__":
    main()



