"""Task 2 - Commodity storage contract valuation.

The valuation combines model-implied gas prices with physical storage and
logistics constraints.

Run from the repository root with:
    python -m src.task2_storage_contract
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd

try:
    from .task1_gas_forecasting import (
        DATA_PATH,
        fit_gas_price_model,
        load_gas_data,
        predict_price,
    )
except ImportError:  # Allows direct execution from src/
    from task1_gas_forecasting import (
        DATA_PATH,
        fit_gas_price_model,
        load_gas_data,
        predict_price,
    )


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"


@dataclass
class ContractValuation:
    purchase_cost_kusd: float
    sale_revenue_kusd: float
    injection_cost_kusd: float
    withdrawal_cost_kusd: float
    transportation_cost_kusd: float
    storage_cost_kusd: float
    contract_value_kusd: float
    max_inventory_mmbtu: float


def _months_inclusive(start: pd.Timestamp, end: pd.Timestamp) -> int:
    """Number of calendar months from start to end, inclusive."""
    if end < start:
        raise ValueError("The final withdrawal must not precede the first injection.")
    return (end.year - start.year) * 12 + (end.month - start.month) + 1


def _validate_inventory_path(
    injection_dates: Sequence[pd.Timestamp],
    withdrawal_dates: Sequence[pd.Timestamp],
    injection_volume_mmbtu: float,
    withdrawal_volume_mmbtu: float,
    max_volume_mmbtu: float,
) -> float:
    """Check that inventory never becomes negative or exceeds storage capacity."""
    events = []

    for date in injection_dates:
        events.append((date, injection_volume_mmbtu))
    for date in withdrawal_dates:
        events.append((date, -withdrawal_volume_mmbtu))

    # On identical dates, injections are processed before withdrawals.
    events.sort(key=lambda item: (item[0], -item[1]))

    inventory = 0.0
    max_inventory = 0.0

    for date, change in events:
        inventory += change
        if inventory < -1e-9:
            raise ValueError(
                f"Withdrawal schedule creates negative inventory on {date.date()}."
            )
        if inventory > max_volume_mmbtu + 1e-9:
            raise ValueError(
                f"Storage capacity exceeded on {date.date()}: "
                f"{inventory:,.0f} MMBtu > {max_volume_mmbtu:,.0f} MMBtu."
            )
        max_inventory = max(max_inventory, inventory)

    return max_inventory


def price_storage_contract(
    injection_dates: Sequence[str],
    withdrawal_dates: Sequence[str],
    injection_volume_mmbtu: float,
    withdrawal_volume_mmbtu: float,
    injection_cost_kusd_per_million_mmbtu: float,
    withdrawal_cost_kusd_per_million_mmbtu: float,
    transportation_cost_kusd_per_event: float,
    max_volume_mmbtu: float,
    storage_cost_kusd_per_month: float,
    gas_data_path: Path = DATA_PATH,
) -> ContractValuation:
    """Value a simple natural-gas storage contract.

    Assumptions retained from the simulation:
    - zero interest rates;
    - no transport delay;
    - no explicit weekend / holiday treatment;
    - one fixed injection volume per injection date;
    - one fixed withdrawal volume per withdrawal date.
    """
    if not injection_dates or not withdrawal_dates:
        raise ValueError("At least one injection and one withdrawal date are required.")

    inj_dates = [pd.Timestamp(date) for date in injection_dates]
    wd_dates = [pd.Timestamp(date) for date in withdrawal_dates]

    max_inventory = _validate_inventory_path(
        inj_dates,
        wd_dates,
        injection_volume_mmbtu,
        withdrawal_volume_mmbtu,
        max_volume_mmbtu,
    )

    data = load_gas_data(gas_data_path)
    model, base_date = fit_gas_price_model(data)

    injection_prices = [
        predict_price(model, base_date, date) for date in inj_dates
    ]
    withdrawal_prices = [
        predict_price(model, base_date, date) for date in wd_dates
    ]

    total_injected = injection_volume_mmbtu * len(inj_dates)
    total_withdrawn = withdrawal_volume_mmbtu * len(wd_dates)

    if total_withdrawn > total_injected + 1e-9:
        raise ValueError("Total withdrawn volume exceeds total injected volume.")

    # Gas prices are interpreted as USD per MMBtu. Convert resulting cash flows to kUSD.
    purchase_cost_kusd = (
        sum(price * injection_volume_mmbtu for price in injection_prices) / 1_000
    )
    sale_revenue_kusd = (
        sum(price * withdrawal_volume_mmbtu for price in withdrawal_prices) / 1_000
    )

    injection_cost_kusd = (
        total_injected / 1_000_000
    ) * injection_cost_kusd_per_million_mmbtu

    withdrawal_cost_kusd = (
        total_withdrawn / 1_000_000
    ) * withdrawal_cost_kusd_per_million_mmbtu

    transportation_cost_kusd = (
        len(inj_dates) + len(wd_dates)
    ) * transportation_cost_kusd_per_event

    storage_months = _months_inclusive(min(inj_dates), max(wd_dates))
    storage_cost_kusd = storage_months * storage_cost_kusd_per_month

    contract_value_kusd = (
        sale_revenue_kusd
        - purchase_cost_kusd
        - injection_cost_kusd
        - withdrawal_cost_kusd
        - transportation_cost_kusd
        - storage_cost_kusd
    )

    return ContractValuation(
        purchase_cost_kusd=purchase_cost_kusd,
        sale_revenue_kusd=sale_revenue_kusd,
        injection_cost_kusd=injection_cost_kusd,
        withdrawal_cost_kusd=withdrawal_cost_kusd,
        transportation_cost_kusd=transportation_cost_kusd,
        storage_cost_kusd=storage_cost_kusd,
        contract_value_kusd=contract_value_kusd,
        max_inventory_mmbtu=max_inventory,
    )


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    valuation = price_storage_contract(
        injection_dates=["2025-06-30"],
        withdrawal_dates=["2025-10-31"],
        injection_volume_mmbtu=1_000_000,
        withdrawal_volume_mmbtu=1_000_000,
        injection_cost_kusd_per_million_mmbtu=10,
        withdrawal_cost_kusd_per_million_mmbtu=10,
        transportation_cost_kusd_per_event=50,
        max_volume_mmbtu=1_000_000,
        storage_cost_kusd_per_month=100,
    )

    result = pd.DataFrame([asdict(valuation)])
    result.to_csv(OUTPUT_DIR / "storage_contract_valuation.csv", index=False)

    print("Storage contract valuation (kUSD):")
    for key, value in asdict(valuation).items():
        print(f"{key}: {value:,.2f}")


if __name__ == "__main__":
    main()
