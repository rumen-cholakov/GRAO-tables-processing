from typing import Callable, Optional

from grao_tables_processing.common.pipeline import Pipeline
from grao_tables_processing.common.custom_types import SettlementDataTuple

import grao_tables_processing.settlement_disambiguation.settlement_disambiguation as sd

settlement_disambiguation: Callable[[SettlementDataTuple], Optional[SettlementDataTuple]] = Pipeline(
  functions=(
    sd.fetch_raw_settlement_data,
    sd.parse_raw_settlement_data,
    sd.mach_key_with_code
  )
)
