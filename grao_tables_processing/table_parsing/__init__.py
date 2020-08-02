from typing import Callable, Optional

from grao_tables_processing.common.custom_types import DataTuple
from grao_tables_processing.common.pipeline import Pipeline

import grao_tables_processing.table_parsing.table_parsing as tp

table_parser: Callable[[DataTuple], Optional[DataTuple]] = Pipeline(
  functions=(
    tp.fetch_raw_table,
    tp.raw_table_to_lines,
    tp.parse_lines,
    tp.parsed_lines_to_full_info_list,
    tp.full_info_list_to_data_frame
  )
)
