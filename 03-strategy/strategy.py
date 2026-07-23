import numpy as np


N_INST = 51

MAX_POS_ALGO = 100_000
MAX_POS_ELSE = 10_000

BET_POS_ALGO = 1_000
BET_POS_ELSE = 10_000

# Pair-strategy parameters
WINDOW = 40
Z_THRESHOLD = 0

# Individual return mean-reversion parameters
WINDOW_INDIVIDUAL = 40
Z_INDIVIDUAL = 2

currentPos = np.zeros(N_INST, dtype=int)


good_pairs = [
    ["AENO", "NWIG"],
    ["SMAH", "ILVX"],
    ["ACIX", "ITPA"],
    ["MHRM", "EAFC"],
    ["EORC", "NGTE"],
    ["NPCK", "SRTX"],
    ["HUXZ", "ACAC"],
]

individual = ["CUBO"]


# Must match the row order of prcSoFar exactly.
INSTRUMENTS = [
    "ALGO", "AENO", "LSST", "SRNA", "ELLT", "AMRP", "OTCS", "HETT",
    "HUXZ", "DUCT", "SMAH", "NPCK", "MSDP", "EORC", "CUBO", "HRET",
    "ANSO", "DIHO", "RTTH", "SPLZ", "NWIG", "MMBT", "MDGI", "AGVF",
    "RRES", "CTGI", "ALUT", "ACAC", "SRTX", "GARI", "RCRI", "ACIX",
    "CCNS", "MTNS", "IHOZ", "NAYO", "FWWG", "EELT", "HRND", "AETS",
    "ULXY", "BLBT", "BENI", "ITPA", "HTRK", "NGTE", "ILVX", "FCSG",
    "FARS", "MHRM", "EAFC",
]


def position_limits(instrument_name):
    """Return bet size and maximum position for an instrument."""
    if instrument_name == "ALGO":
        return BET_POS_ALGO, MAX_POS_ALGO

    return BET_POS_ELSE, MAX_POS_ELSE


def pairs_strategy(prcSoFar, instrument_index):
    """
    Generate target positions for instruments in good_pairs.

    The strategy fits:

        y = alpha + beta * x + residual

    over WINDOW observations and trades residual mean reversion.
    """
    nins, nt = prcSoFar.shape
    target = np.zeros(nins, dtype=float)

    if nt < WINDOW:
        return target

    for y_name, x_name in good_pairs:
        y_idx = instrument_index[y_name]
        x_idx = instrument_index[x_name]

        y = np.asarray(
            prcSoFar[y_idx, -WINDOW:],
            dtype=float,
        )
        x = np.asarray(
            prcSoFar[x_idx, -WINDOW:],
            dtype=float,
        )

        if (
            not np.isfinite(y).all()
            or not np.isfinite(x).all()
            or (y <= 0).any()
            or (x <= 0).any()
        ):
            continue

        # Rolling OLS: y = alpha + beta*x + residual.
        design = np.column_stack(
            (np.ones(WINDOW), x)
        )
        alpha, beta = np.linalg.lstsq(
            design,
            y,
            rcond=None,
        )[0]

        residuals = y - (alpha + beta * x)
        residual_std = residuals.std(ddof=2)

        if (
            not np.isfinite(residual_std)
            or residual_std <= np.finfo(float).eps
        ):
            continue

        z_score = residuals[-1] / residual_std

        if not np.isfinite(z_score):
            continue

        if abs(z_score) < Z_THRESHOLD:
            continue

        # Positive residual:
        # y is relatively expensive, so short y and long beta*x.
        #
        # Negative residual:
        # y is relatively cheap, so long y and short beta*x.
        y_direction = -np.sign(z_score)

        if "ALGO" in (y_name, x_name):
            bet_size = BET_POS_ALGO
            max_position = MAX_POS_ALGO
        else:
            bet_size = BET_POS_ELSE
            max_position = MAX_POS_ELSE

        y_position = y_direction * bet_size / y[-1]
        x_position = -y_direction * beta * bet_size / x[-1]

        target[y_idx] += np.clip(
            y_position,
            -max_position,
            max_position,
        )
        target[x_idx] += np.clip(
            x_position,
            -max_position,
            max_position,
        )

    return target


def individual_strategy(prcSoFar, instrument_index):
    """
    Generate return mean-reversion positions for `individual`.

    The latest simple return is compared with the mean and standard
    deviation of the preceding WINDOW_INDIVIDUAL returns.

    A positive return z-score produces a short position.
    A negative return z-score produces a long position.
    """
    nins, nt = prcSoFar.shape
    target = np.zeros(nins, dtype=float)

    # Need:
    #   WINDOW_INDIVIDUAL historical returns
    #   1 current return
    # Therefore, we need WINDOW_INDIVIDUAL + 2 prices.
    required_prices = WINDOW_INDIVIDUAL + 2

    if nt < required_prices:
        return target

    for instrument_name in individual:
        idx = instrument_index[instrument_name]

        prices = np.asarray(
            prcSoFar[idx, -required_prices:],
            dtype=float,
        )

        if (
            not np.isfinite(prices).all()
            or (prices <= 0).any()
        ):
            continue

        returns = prices[1:] / prices[:-1] - 1.0

        # Use only returns known before the latest return to estimate
        # its expected value and volatility.
        historical_returns = returns[:-1]
        current_return = returns[-1]

        return_mean = historical_returns.mean()
        return_std = historical_returns.std(ddof=1)

        if (
            not np.isfinite(return_std)
            or return_std <= np.finfo(float).eps
        ):
            continue

        z_score = (
            current_return - return_mean
        ) / return_std

        if (
            not np.isfinite(z_score)
            or abs(z_score) < Z_INDIVIDUAL
        ):
            continue

        # Mean reversion:
        # unusually positive return -> short
        # unusually negative return -> long
        direction = -np.sign(z_score)

        bet_size, max_position = position_limits(
            instrument_name
        )

        position = (
            direction
            * bet_size
            / prices[-1]
        )

        target[idx] += np.clip(
            position,
            -max_position,
            max_position,
        )

    return target


def getMyPosition(prcSoFar):
    """Combine pair and individual strategy target positions."""
    global currentPos

    nins, _ = prcSoFar.shape

    if nins != len(INSTRUMENTS):
        raise ValueError(
            f"Expected {len(INSTRUMENTS)} instruments, "
            f"but prcSoFar contains {nins}"
        )

    instrument_index = {
        name: index
        for index, name in enumerate(INSTRUMENTS)
    }

    pair_targets = pairs_strategy(
        prcSoFar,
        instrument_index,
    )
    individual_targets = individual_strategy(
        prcSoFar,
        instrument_index,
    )

    target = pair_targets + individual_targets

    # Enforce final limits after combining strategies.
    for name, idx in instrument_index.items():
        _, max_position = position_limits(name)

        target[idx] = np.clip(
            target[idx],
            -max_position,
            max_position,
        )

    currentPos = np.rint(target).astype(int)
    return currentPos