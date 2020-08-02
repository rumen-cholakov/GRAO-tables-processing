from typing import Callable, List

import grao_tables_processing.table_processing.table_processing as tp

from grao_tables_processing.common.custom_types import DataTuple
from grao_tables_processing.common.configuration import Configuration
from grao_tables_processing.common.pipeline import Pipeline


def create_table_processor(config: Configuration) -> Callable[[List[DataTuple]], List[DataTuple]]:
  processing_pipeline = Pipeline(functions=(
    (lambda data: tp.process_data(data, config)),
    (lambda data: tp.disambiguate_data(data, config)),
    (lambda data: tp.store_data_list(data, config)),
    (lambda data: tp.combine_data(data, config)),
    (lambda data: tp.store_combined_data(data, config)),
  ))

  return processing_pipeline
