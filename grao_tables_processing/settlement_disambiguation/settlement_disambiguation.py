from collections import defaultdict
from urllib.parse import quote
from datetime import datetime
from bs4 import BeautifulSoup  # type: ignore
from typing import Dict, Tuple, List

from grao_tables_processing.common.custom_types import SettlementDataTuple, SettlementNamesForPeriod
from grao_tables_processing.common.helper_functions import fetch_raw_data


def fetch_raw_settlement_data(settlement: SettlementDataTuple) -> SettlementDataTuple:
  name = settlement.data

  # HACK!!! used to circumvent stripping of non-letter chars from the name
  if name.find('-') != -1:
    name = name.split('-')[1]

  encoded_name = quote(name.encode('windows-1251'))
  data = fetch_raw_data(f'https://www.nsi.bg/nrnm/index.php?ezik=bul&f=6&name={encoded_name}&code=&kind=-1')
  req = data

  if req.status_code != 200:
    raise ValueError

  return SettlementDataTuple(settlement.key, req)


def parse_raw_settlement_data(settlement: SettlementDataTuple) -> SettlementDataTuple:
  req = settlement.data
  soup = BeautifulSoup(req.text, 'lxml')
  table = soup.find_all('table')[-4]

  data: Dict[str, List[SettlementNamesForPeriod]] = defaultdict(list)
  last_key: str = ''

  oldest_record_date = datetime.strptime('31.12.1899', '%d.%m.%Y')
  rows = table.find_all('tr')
  for row in rows[2:]:
    cells = row.find_all('td')
    num_cells = len(cells)

    if num_cells == 2:
      last_key = cells[0].text
    elif num_cells == 3:
      dates: str = cells[2].text.split('-')
      start = datetime.strptime(dates[0].strip(), '%d.%m.%Y')
      end = datetime.max

      if len(dates[1].strip()) > 0:
        end = datetime.strptime(dates[1].strip(), '%d.%m.%Y')

      name_tuple: Tuple[str, ...] = tuple(map(lambda s: s.strip(), cells[1].text.split(',')[::-1]))

      if end > oldest_record_date and len(name_tuple) > 2:
        data[last_key].append(
            SettlementNamesForPeriod(
                name_tuple,
                start,
                end,
            )
        )

  return SettlementDataTuple(settlement.key, data)


def mach_key_with_code(settlement: SettlementDataTuple) -> SettlementDataTuple:
  ker_region = settlement.key[0].lower()
  ker_municipality = settlement.key[1].lower()
  ker_settlement = settlement.key[2].lower()

  data_dict = settlement.data
  result = SettlementDataTuple(settlement.key)
  result_list = []

  for code, names_list in data_dict.items():
    for name_data in names_list:
      region_name = name_data.name[0].lower()
      municipality_name = name_data.name[1].lower()
      settlement_name = name_data.name[2].lower()

      straight_case = region_name.find(ker_region) != -1

      # HACK!!! used as a workaround for broken data
      hack_sf = all([
        ker_region == 'софийска',
        region_name.find('софия') != -1
      ])

      hack_sm = all([
        ker_region == 'смолян',
        region_name.find('пловдивска') != -1
      ])

      hack_pa = all([
        ker_region == 'пазарджик',
        region_name.find('пазарджишки') != -1
      ])

      region_match = any([
        straight_case,
        hack_sf,
        hack_sm,
        hack_pa
      ])

      municipality_and_settlement_match = all([
        municipality_name.find(ker_municipality) != -1,
        settlement_name.split('.')[-1].strip() == ker_settlement
      ])

      if all([
              region_match,
              municipality_and_settlement_match
         ]):
        result_list.append((name_data.end, SettlementDataTuple(settlement.key, code)))

  # if there are multiple matching names take the most recent one
  result_list = sorted(result_list)
  if len(result_list) > 0:
    result = result_list[-1][1]

  return result
