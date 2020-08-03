from regex import search  # type: ignore
from bs4 import BeautifulSoup  # type: ignore
from pandas import DataFrame  # type: ignore
from typing import Optional, Tuple, Match

from grao_tables_processing.common.custom_types import SettlementInfo, MunicipalityIdentifier, FullSettlementInfo, ParsedLines
from grao_tables_processing.common.custom_types import DataTuple, TableTypeEnum, HeaderEnum
from grao_tables_processing.common.helper_functions import fix_names, fetch_raw_data
from grao_tables_processing.common.regex_pattern_wrapper import RegexPatternWrapper


def fetch_raw_table(data_tuple: DataTuple) -> DataTuple:
  url = data_tuple.data
  req = fetch_raw_data(url)

  return DataTuple(req, data_tuple.header_type, data_tuple.table_type)


def raw_table_to_lines(data_tuple: DataTuple) -> DataTuple:
  req = data_tuple.data
  soup_str: str = str(BeautifulSoup(req.text, 'lxml').prettify(encoding=None))
  separator = str('\r\n')
  split = soup_str.split(sep=separator)

  return DataTuple(split, data_tuple.header_type, data_tuple.table_type)


def parse_data_line(line: str, table_type: TableTypeEnum) -> Optional[SettlementInfo]:
  settlement_info_re = ''
  permanent_population_position = 2
  current_population_position = -1

  if table_type == TableTypeEnum.Quarterly:
    settlement_info_re = RegexPatternWrapper().settlement_info_quarterly
    current_population_position = 3
  elif table_type == TableTypeEnum.Yearly:
    settlement_info_re = RegexPatternWrapper().settlement_info_yearly
    current_population_position = 6

  settlement_info = None
  if settlement_info_group := search(settlement_info_re, line):
    name: str = settlement_info_group.group(1)
    permanent: int = int(settlement_info_group.group(permanent_population_position))
    current: int = int(settlement_info_group.group(current_population_position))

    name_parts = name.split('.')
    name = '. '.join([name_parts[0], fix_names(name_parts[1])])
    settlement_info = SettlementInfo(name.strip(), permanent, current)

  return settlement_info


def parse_header_line(
  line: str,
  header_type: HeaderEnum,
  old_header_state: Optional[Match]
) -> Tuple[Optional[MunicipalityIdentifier], Optional[Match]]:
  region: str = ''
  municipality: str = ''
  region_name = None

  if header_type == HeaderEnum.New and (region_gr := search(RegexPatternWrapper().region_name_new, line)):
    region = region_gr.group(1)
    municipality = region_gr.group(2)
    region_name = MunicipalityIdentifier(region.strip(), municipality.strip())

  elif header_type == HeaderEnum.Old:
    if old_header_state is None:
      old_header_state = search(RegexPatternWrapper().old_reg, line)
      region_name = None
    elif mun_gr := search(RegexPatternWrapper().old_mun, line):
      region = old_header_state.group(1)
      municipality = mun_gr.group(1)
      region_name = MunicipalityIdentifier(fix_names(region.strip()), fix_names(municipality.strip()))
    else:
      old_header_state = None

  return (region_name, old_header_state)


def parse_lines(data_tuple: DataTuple) -> DataTuple:
  municipality_ids = {}
  settlements_info = {}
  old_header_state = None

  for line_num, line in enumerate(data_tuple.data):
    municipality_id, old_header_state = parse_header_line(line, data_tuple.header_type, old_header_state)
    if municipality_id:
      municipality_ids[line_num] = municipality_id
      continue

    if settlement_info := parse_data_line(line, data_tuple.table_type):
      settlements_info[line_num] = settlement_info

  return DataTuple(ParsedLines(municipality_ids, settlements_info), data_tuple.header_type, data_tuple.table_type)


def parsed_lines_to_full_info_list(data_tuple: DataTuple) -> DataTuple:

  regions = data_tuple.data.municipality_ids
  settlements_info = data_tuple.data.settlements_info

  reg_keys = list(regions.keys())
  settlement_keys = list(settlements_info.keys())

  reg_keys_pairs = zip(reg_keys[:-1], reg_keys[1:])

  sk_index = 0
  full_name_settlement_infos = []

  for current_mun, next_mun in reg_keys_pairs:
    while current_mun < settlement_keys[sk_index] < next_mun:
      reg = regions[current_mun]
      set_info = settlements_info[settlement_keys[sk_index]]
      full_info = FullSettlementInfo(fix_names(reg.region),
                                     fix_names(reg.municipality),
                                     fix_names(set_info.name),
                                     set_info.permanent_residents,
                                     set_info.current_residents)
      full_name_settlement_infos.append(full_info)

      sk_index += 1

  return DataTuple(full_name_settlement_infos, data_tuple.header_type, data_tuple.table_type)


def full_info_list_to_data_frame(data_tuple: DataTuple) -> DataTuple:
  df = DataFrame(data_tuple.data)
  df.set_index(['region', 'municipality', 'settlement'], drop=True, inplace=True)

  return DataTuple(df, data_tuple.header_type, data_tuple.table_type)
