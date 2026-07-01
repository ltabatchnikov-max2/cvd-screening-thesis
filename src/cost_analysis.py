# Cost-effectiveness analysis for standard vs. uncertainty-informed screening

from src.config import (
    BD_CURRENCY, BD_EVENT_COST, BD_GP_COST, BD_PREVENTION_COST, BD_SCREENING_COST,
    DE_CURRENCY, DE_EVENT_COST, DE_GP_COST, DE_PREVENTION_COST, DE_SCREENING_COST,
)


def calculate_costs(fn_std, fn_unc, fp_unc, zone0_n, n_total,
                    event_cost, prevention_cost, screening_cost, gp_cost, currency):
    """
    Compute total costs for standard and uncertainty-informed screening.

    Parameters
    ----------
    fn_std          : false negatives under standard screening
    fn_unc          : false negatives under uncertainty-informed screening
    fp_unc          : false positives under uncertainty-informed screening
    zone0_n         : number of Zone 0 (clinical review) referrals
    n_total         : total number of patients
    event_cost      : cost per missed CVD event
    prevention_cost : cost per preventively treated patient/year
    screening_cost  : cost per patient screening
    gp_cost         : cost per GP consultation
    currency        : currency label (e.g. "USD", "EUR")

    Returns
    -------
    dict with all cost components and savings
    """
    events_prevented   = fn_std - fn_unc

    # Standard screening
    cvd_cost_std       = fn_std  * event_cost
    screening_cost_std = n_total * screening_cost
    total_std          = cvd_cost_std + screening_cost_std

    # Uncertainty-informed screening
    cvd_cost_unc       = fn_unc  * event_cost
    prevention_total   = fp_unc  * prevention_cost
    screening_cost_unc = n_total * screening_cost
    gp_cost_total      = zone0_n * gp_cost
    total_unc          = cvd_cost_unc + prevention_total + screening_cost_unc + gp_cost_total

    savings     = total_std - total_unc
    savings_pct = round((savings / total_std) * 100, 1) if total_std > 0 else 0.0

    return {
        "currency":                   currency,
        "False Negatives":            [fn_std, fn_unc],
        "CVD events prevented":       events_prevented,
        "CVD event costs":            [cvd_cost_std, cvd_cost_unc],
        "Preventive treatment costs": [0, prevention_total],
        "GP consultation costs":      [0, round(gp_cost_total, 2)],
        "Screening costs":            [screening_cost_std, screening_cost_unc],
        "Total costs":                [round(total_std, 2), round(total_unc, 2)],
        "Cost savings":               round(savings, 2),
        "Cost savings (%)":           savings_pct,
    }


def run_cost_analysis(fn_standard, fn_uncertainty, fp_uncertainty, zone0_referrals, n_total):
    """
    Run cost analysis for both Bangladesh and Germany scenarios.

    Returns
    -------
    bd : dict with Bangladesh cost results
    de : dict with Germany cost results
    """
    bd = calculate_costs(
        fn_standard, fn_uncertainty, fp_uncertainty, zone0_referrals, n_total,
        BD_EVENT_COST, BD_PREVENTION_COST, BD_SCREENING_COST, BD_GP_COST, BD_CURRENCY,
    )
    de = calculate_costs(
        fn_standard, fn_uncertainty, fp_uncertainty, zone0_referrals, n_total,
        DE_EVENT_COST, DE_PREVENTION_COST, DE_SCREENING_COST, DE_GP_COST, DE_CURRENCY,
    )
    return bd, de