import pandas as pd  # type: ignore
import random
import time

from regex import search  # type: ignore
from itertools import chain
from typing import Tuple, Callable, List, Dict, Any, Optional, Generator

from grao_tables_processing.common.custom_types import DataTuple, SettlementDataTuple, HeaderEnum, TableTypeEnum
from grao_tables_processing.common.helper_functions import execute_in_parallel, fix_names, force_unwrap_optional
from grao_tables_processing.common.regex_pattern_wrapper import RegexPatternWrapper
from grao_tables_processing.common.pickle_wrapper import PickleWrapper
from grao_tables_processing.common.configuration import Configuration


def process_data_tuple(input_data: Tuple[Callable[[DataTuple], DataTuple], DataTuple]) -> DataTuple:
  parsing_pipeline, data_tuple = input_data

  if data_tuple.table_type == TableTypeEnum.Quarterly:
    date_group = RegexPatternWrapper().date_group
  else:
    date_group = RegexPatternWrapper().year_group

  date_string: str = search(date_group, data_tuple.data).group(1).replace('-', '_')
  data_frame = parsing_pipeline(data_tuple).data
  data_frame = data_frame.rename(columns={'permanent_residents': f'permanent_{date_string}',
                                          'current_residents': f'current_{date_string}'})

  return DataTuple(data_frame, data_tuple.header_type, data_tuple.table_type)


def process_data(data_source: List[DataTuple], config: Configuration) -> List[DataTuple]:
  parsing_pipeline = config['table_parser']
  wrapped_data_source = ((parsing_pipeline, dt) for dt in data_source)

  data_frame_list = execute_in_parallel(process_data_tuple, wrapped_data_source)

  data_frame_list = force_unwrap_optional(data_frame_list, 'Failed parsing tables!')

  PickleWrapper.pickle_data(data_frame_list, 'data_frames_list')

  return data_frame_list


def load_ekatte_dicts() -> Tuple[Dict[Any, Any], Dict[Any, Any]]:
  processed_sdts = PickleWrapper.load_data('triple_to_ekatte')
  if (processed_sdts is None) or (not isinstance(processed_sdts, dict)):
    processed_sdts = {}

  reverse_dict = PickleWrapper.load_data('ekatte_to_triple')
  if (reverse_dict is None) or (not isinstance(reverse_dict, dict)):
    reverse_dict = {}

  return processed_sdts, reverse_dict


def make_settlements_data_tuple_list(data_frame_list: List[DataTuple]) -> List[Tuple[SettlementDataTuple, str]]:
  sdt_list = list(map(lambda tup: (SettlementDataTuple(tup[0], tup[0][2]), tup[1]),
                      map(lambda name: ((fix_names(name[0].strip()),
                                         fix_names(name[1].strip()),
                                         fix_names(name[2].split('.')[1].strip())),
                                        name),
                          set(chain.from_iterable(
                              map(lambda dt: dt.data.index.values.tolist(),
                                  data_frame_list))))))
  return sdt_list


def sleep_time_generator(random_seed: float) -> Generator[float, None, None]:
  return (st + (st + 1) * random_seed for st in range(round(random_seed), 60, round(5 + 10 * random_seed)))


def try_disambiguation(
  input_data: Tuple[Callable[[SettlementDataTuple], SettlementDataTuple], SettlementDataTuple]
) -> Tuple[SettlementDataTuple, SettlementDataTuple]:
  disambiguation_pipeline, sdt = input_data

  for sleep_time in sleep_time_generator(random.random()):
    time.sleep(sleep_time)
    try:
      result = disambiguation_pipeline(sdt)
    except ValueError:
      print(f'Failed disambiguating {sdt} with {sleep_time:.3f}s sleep')
      continue

    return (result, sdt)

  return (SettlementDataTuple(sdt.key), sdt)


def check_sdt_availability(key: SettlementDataTuple, processed_sdts: Dict[Any, Any], reverse_dict: Dict[Any, Any]) -> bool:
  return key.key in processed_sdts and processed_sdts[key.key] in reverse_dict


def update_data_frame(input_data: Tuple[DataTuple, Dict[Any, Any]]) -> DataTuple:
  dt, processed_sdts = input_data
  df = dt.data
  df.reset_index(inplace=True)
  df['ekatte'] = df['settlement']
  cols = df.columns

  def update_df(x: Tuple[str, str, str, int, int, str]) -> Tuple[str, str, str, int, int, Optional[str]]:
    result = (x[0],
              x[1],
              x[2],
              x[3],
              x[4],
              processed_sdts.get((fix_names(x[0].strip()),
                                  fix_names(x[1].strip()),
                                  fix_names(x[2].split('.')[1].strip())),
                                 None))

    return result

  df = pd.DataFrame([update_df(x) for x in df.to_numpy()])

  df.columns = cols
  df.dropna(inplace=True)
  df.set_index(['ekatte'], drop=True, inplace=True)

  df = df.loc[~df.index.duplicated(keep='first')]

  return DataTuple(df, dt.header_type, dt.table_type)


def filter_disambiguated_sdts(
  sdt_pairs: List[Tuple[SettlementDataTuple, SettlementDataTuple]]
) -> List[Tuple[SettlementDataTuple, SettlementDataTuple]]:
  result = []
  failures = set()

  for new, old in sdt_pairs:
    if new.data is None:
      failures.add(old)
    else:
      result.append((new, old))

  if failures:
    PickleWrapper.pickle_data(failures, 'failures')

  return result


def disambiguate_data(data_frame_list: List[DataTuple], config: Configuration) -> List[DataTuple]:

  settlement_disambiguation_pipeline = config['settlement_disambiguation']

  processed_sdts, reverse_dict = load_ekatte_dicts()

  sdt_list = make_settlements_data_tuple_list(data_frame_list)

  wrapped_data_source = ((settlement_disambiguation_pipeline, sdt[0])
                         for sdt in sdt_list if not check_sdt_availability(sdt[0], processed_sdts, reverse_dict))

  # Higher number of concurrent jobs leads to issues with failing request to NSI's website
  results = execute_in_parallel(try_disambiguation, wrapped_data_source, 2)

  results = force_unwrap_optional(results, 'Settlement disambiguation failed!')

  for value, sdt in filter_disambiguated_sdts(results):
    processed_sdts[value.key] = value.data
    reverse_dict[value.data] = sdt[1]

  PickleWrapper.pickle_data(processed_sdts, 'triple_to_ekatte')
  PickleWrapper.pickle_data(reverse_dict, 'ekatte_to_triple')

  wrapped_data_tuple_source = ((dt, processed_sdts) for dt in data_frame_list)
  disambiguated_data = execute_in_parallel(update_data_frame, wrapped_data_tuple_source)

  disambiguated_data = force_unwrap_optional(disambiguated_data, 'Updating DataFrames failed!')

  PickleWrapper.pickle_data(disambiguated_data, 'data_frames_list_disambiguated')

  return disambiguated_data


def combine_data(processed_data: List[DataTuple], config: Configuration) -> List[DataTuple]:
  combined: Optional[pd.DataFrame] = None
  names = ['region', 'municipality', 'settlement']

  for dt in processed_data:
    if combined is None:
      combined = dt.data.drop(labels=names, axis=1)
    else:
      df: pd.DataFrame = dt.data.drop(labels=names, axis=1)
      combined = combined.merge(df,
                                how='outer',
                                copy=False,
                                left_index=True,
                                right_index=True)

  combined_unwrapped: pd.DataFrame = force_unwrap_optional(combined, 'Failed to combine DataFarmes')

  combined_unwrapped.fillna(value=0, inplace=True)
  for column in combined_unwrapped.columns.to_list():
    combined_unwrapped[column] = combined_unwrapped[column].astype(int)

  PickleWrapper.pickle_data(combined_unwrapped, 'combined_tables')

  return [DataTuple(combined_unwrapped, HeaderEnum(0), TableTypeEnum(0))]


def store_data_list(processed_data: List[DataTuple], config: Configuration) -> List[DataTuple]:
  for dt in processed_data:
    df: pd.DataFrame = dt.data

    name = f'grao_data_{"_".join(df.columns[-1].split("_")[1:])}'
    df.to_csv(f'{config.processed_tables_path}/{name}.csv')

  return processed_data


def store_combined_data(processed_data: List[DataTuple], config: Configuration) -> List[DataTuple]:
  combined_data: pd.DataFrame = processed_data[0].data

  combined_data.to_csv(f'{config.combined_tables_path}/grao_data_combined.csv')

  return processed_data
