import pandas as pd

from typing import Dict, Any
from numpy import str as np_str

from grao_tables_processing.common.configuration import Configuration

from grao_tables_processing.wikidata_interaction.common import find_latest_processed_file_info
from grao_tables_processing.wikidata_interaction.common import file_prefix_for_directory
from grao_tables_processing.wikidata_interaction.common import find_date_suffix


def dict_from_csv(csv_path: str, index_name: str) -> Dict[Any, Any]:
  loaded_dict = pd.DataFrame(pd.read_csv(csv_path, dtype=np_str)).set_index(index_name).to_dict(orient='index', into=dict)

  if not isinstance(loaded_dict, dict):
    result: Dict[Any, Any] = {}
  else:
    result = loaded_dict

  return result


def update_matched_data(config: Configuration):
  matched_tables_path = config.matched_tables_path
  matched_data_time, _, matched_data_path = find_latest_processed_file_info(
    matched_tables_path,
    config.data
  )
  grao_data_time, grao_data_url, grao_data_path = find_latest_processed_file_info(
    config.processed_tables_path,
    config.data
  )

  if grao_data_time <= matched_data_time:
    return

  date_suffix = find_date_suffix(grao_data_url)

  matched_data_dict = dict_from_csv(matched_data_path, index_name='ekatte')
  grao_data_dict = dict_from_csv(grao_data_path, index_name='ekatte')

  new_matched_data = {}
  for key, value in matched_data_dict.items():
    new_matched_data[key] = value
    new_matched_data[key]['permanent_population'] = grao_data_dict[key][f'permanent_{date_suffix}']
    new_matched_data[key]['current_population'] = grao_data_dict[key][f'current_{date_suffix}']

  new_matched_df = pd.DataFrame.from_dict(new_matched_data, orient='index', dtype=np_str).reset_index()

  if new_matched_df is None:
    raise Exception('Failed to cerate updated DataFrame')

  new_matched_df.rename(columns={'index': 'ekatte'}, inplace=True)
  file_name = f'{matched_tables_path}/{file_prefix_for_directory(matched_tables_path)}{date_suffix}.csv'
  new_matched_df.to_csv(file_name, index=False)
