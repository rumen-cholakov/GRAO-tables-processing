#! /usr/bin/env python3

"""## Imports"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import argparse
import datetime
import requests
import shutil
import urllib
import random
import pickle
import regex
import time
import enum
import json
import os

from typing import TypeVar, Callable, Sequence, List, Optional, Tuple
from collections import namedtuple, defaultdict
from functools import reduce
from itertools import groupby, chain

from bs4 import BeautifulSoup
from wikidataintegrator import wdi_core, wdi_login

"""## Type Declarations"""

class HeaderEnum(enum.IntEnum):
  Old = 0
  New = 1

class TableTypeEnum(enum.IntEnum):
  Quarterly = 0
  Yearly = 1

DataTuple = namedtuple('DataTuple', 'data header_type table_type')
SettlementDataTuple = namedtuple('SettlementDataTuple', 'key data')
SettlementNamesData = namedtuple('SettlementNamesData', 'name first last')
MunicipalityIdentifier = namedtuple('MunicipalityIdentifier', 'region municipality')
SettlementInfo = namedtuple('SettlementInfo', 'name permanent_residents current_residents')
FullSettlementInfo = namedtuple('FullSettlementInfo', 'region municipality settlement permanent_residents current_residents')
PopulationInfo = namedtuple('PopulationInfo', 'permanent current')
ParsedLines = namedtuple('ParsedLines', 'municipality_ids settlements_info')

T = TypeVar('T')

class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class RegexPatternWrapper(metaclass=Singleton):
  def __init__(self):
    # Building regex strings
    cap_letter = '\p{Lu}'
    low_letter = '\p{Ll}'
    separator = '[\||\!]\s*'
    number = '\d+'

    self.year_group = '(\d{4})'
    self.date_group = '(\d{2}-\d{4})'

    name_part = f'[\s|-]*{cap_letter}*'
    name_part_old = f'[\.|\s|-]{cap_letter}*'
    type_abbr = f'{cap_letter}{{1,2}}\.'
    name = f'{cap_letter}+{name_part * 3}'
    name_old = f'{cap_letter}+{name_part_old * 3}'
    word = f'{low_letter}+'
    number_group = f'{separator}({number})\s*'

    self.old_reg = f'ОБЛАСТ:({name_old})'
    self.old_mun = f'ОБЩИНА:({name_old})'
    self.region_name_new = f'{word} ({name}) {word} ({name})'
    self.settlement_info_quarterly = f'({type_abbr}{name})\s*{number_group * 3}'
    self.settlement_info_yearly = f'({type_abbr}{name})\s*{number_group * 6}'

class PickleWrapper(metaclass=Singleton):

  def __init__(self, directory):
    self.directory = directory

  def pickle_data(self, data, name):
    if not os.path.exists(self.directory):
      os.makedirs(self.directory)

    with open(f'{self.directory}/{name}.pkl', 'wb') as f:
      pickle.dump(data, f)

  def load_data(self, name):
    path = f'{self.directory}/{name}.pkl'

    if not os.path.exists(path):
      return None

    with open(path, 'rb') as f:
      return pickle.load(f)

class Pipeline(object):
  def __init__(self, functions: Sequence[Callable[[T], T]]):
    self.pipeline = (lambda value: self._pipeline(value, function_pipeline=functions))

  def __call__(self, value: T) -> T:
    return self.pipeline(value)

  def _pipeline(
    self,
    value: T,
    function_pipeline: Sequence[Callable[[T], T]],
  ) -> T:
    '''A generic Unix-like pipeline

    :param value: the value you want to pass through a pipeline
    :param function_pipeline: an ordered list of functions that
        comprise your pipeline
    '''
    return reduce(lambda v, f: f(v), function_pipeline, value)

class Configuration(object):
  def __init__(self,
               data_configuration_path,
               processed_tables_path,
               matched_tables_path,
               combined_tables_path,
               pickled_data_path):
    self.data_configuration_path = data_configuration_path
    self.processed_tables_path = processed_tables_path
    self.matched_tables_path = matched_tables_path
    self.combined_tables_path = combined_tables_path
    self.pickled_data_path = pickled_data_path
    self.extra_params = {}

  def process_data_configuration(self) -> List[DataTuple]:
    with open(self.data_configuration_path) as file:
      data = json.load(file)

      output = []
      for entry in data:
        output.append(self._data_tuple_from_entry(entry))

      return output

  def _data_tuple_from_entry(self, entry: str) -> DataTuple:
    if regex.search(RegexPatternWrapper().date_group, entry) is not None:
      return DataTuple(entry, HeaderEnum.New, TableTypeEnum.Quarterly)

    date = regex.search(RegexPatternWrapper().year_group, entry)
    if date is not None:
      header_type = HeaderEnum(int(date.group(1)) > 2005)
      return DataTuple(entry, header_type, TableTypeEnum.Yearly)

"""## Helper Functions"""

def static_vars_function(**kwargs):
  def decorate(func):
      for k in kwargs:
          setattr(func, k, kwargs[k])
      return func
  return decorate

def fetch_raw_data(url: str):
  headers = requests.utils.default_headers()
  headers.update({
      'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:52.0) Gecko/20100101 Firefox/52.0'
  })

  req = requests.get(url, headers)
  req.encoding = 'windows-1251'

  return req

def fix_names(name: str) -> str:
    new_name = name
    prob_pos = name.find('Ь')

    # Some years in the names of settlements 'Ь' was used instead of 'Ъ'
    if prob_pos != -1:
      o_pos = name.find('О', prob_pos)
      if o_pos != prob_pos + 1:
        new_name = name.replace('Ь', 'Ъ')

    # Some years in the names of settlements were spelled wrong
    # correct names were taken from https://www.nsi.bg/nrnm/index.php?f=6&ezik=bul
    names = {
        'БОБОВДОЛ': 'БОБОВ ДОЛ',
        'ВЪЛЧИДОЛ': 'ВЪЛЧИ ДОЛ',
        'КАПИТАН ПЕТКО ВОЙВО': 'КАПИТАН ПЕТКО ВОЙВОДА',
        'ДОБРИЧКА': 'ДОБРИЧ-СЕЛСКА',
        'ДОБРИЧ СЕЛСКА': 'ДОБРИЧ-СЕЛСКА',
        'БЕРАИНЦИ': 'БЕРАЙНЦИ',
        'ФЕЛТФЕБЕЛ ДЕНКОВО': 'ФЕЛДФЕБЕЛ ДЕНКОВО',
        'УРУЧОВЦИ': 'УРУЧЕВЦИ',
        'ПОЛИКРАЙЩЕ': 'ПОЛИКРАИЩЕ',
        'КАМЕШИЦА': 'КАМЕЩИЦА',
        'БОГДАНОВДОЛ': 'БОГДАНОВ ДОЛ',
        'СИНЬО БЬРДО': 'СИНЬО БЪРДО',
        'ЗЕЛЕН ДОЛ': 'ЗЕЛЕНДОЛ',
        'МАРИКОСТЕНОВО': 'МАРИКОСТИНОВО',
        'САНСТЕФАНО': 'САН-СТЕФАНО',
        'САН СТЕФАНО': 'САН-СТЕФАНО',
        'ПЕТРОВДОЛ': 'ПЕТРОВ ДОЛ',
        'ЧАПАЕВО': 'ЦАРСКИ ИЗВОР',
        'ЕЛОВДОЛ': 'ЕЛОВ ДОЛ',
        'В. ТЪРНОВО': 'ВЕЛИКО ТЪРНОВО',
        'В.ТЪРНОВО': 'ВЕЛИКО ТЪРНОВО',
        'ГЕНЕРАЛ-ТОШОВО': 'ГЕНЕРАЛ ТОШЕВО',
        'ГЕНЕРАЛ ТОШОВО': 'ГЕНЕРАЛ ТОШЕВО',
        'ГЕНЕРАЛ-ТОШЕВО': 'ГЕНЕРАЛ ТОШЕВО',
        'БЕДЖДЕНЕ': 'БЕДЖЕНЕ',
        'ТАЙМИШЕ': 'ТАЙМИЩЕ',
        'СТОЯН ЗАИМОВО': 'СТОЯН-ЗАИМОВО',
        'ДАСКАЛ АТАНАСОВО': 'ДАСКАЛ-АТАНАСОВО',
        'СЛАВЕИНО': 'СЛАВЕЙНО',
        'КРАЛЕВДОЛ': 'КРАЛЕВ ДОЛ',
        'ФЕЛДФЕБЕЛ ДЯНКОВО': 'ФЕЛДФЕБЕЛ ДЕНКОВО',
        'ДЛЪХЧЕВО САБЛЯР': 'ДЛЪХЧЕВО-САБЛЯР',
        'СТОЯН ЗАИМОВО': 'СТОЯН-ЗАИМОВО',
        'ГОЛЕМ ВЪРБОВНИК': 'ГОЛЯМ ВЪРБОВНИК',
        'ПОЛКОВНИК ЖЕЛЕЗОВО': 'ПОЛКОВНИК ЖЕЛЯЗОВО',
        'ДОБРИЧ ГРАД': 'ДОБРИЧ',
        'ЦАР ПЕТРОВО': 'ЦАР-ПЕТРОВО',
        'ВЪЛЧАНДОЛ': 'ВЪЛЧАН ДОЛ',
        'ПАНАГЮРСКИ КОЛОНИ': 'ПАНАГЮРСКИ КОЛОНИИ',
        'ГОРСКИ ГОРЕН ТРЪМБЕ': 'ГОРСКИ ГОРЕН ТРЪМБЕШ',
        'ГОРСКИ ДОЛЕН ТРЪМБЕ': 'ГОРСКИ ДОЛЕН ТРЪМБЕШ',
        'ГЕНЕРАЛ-КАНТАРДЖИЕВ': 'ГЕНЕРАЛ КАНТАРДЖИЕВО',
        'ГЕНЕРАЛ КАНТАРДЖИЕВ': 'ГЕНЕРАЛ КАНТАРДЖИЕВО',
        'АЛЕКСАНДЪР СТАМБОЛИ': 'АЛЕКСАНДЬР СТАМБОЛИЙСКИ',
        'ПОЛКОВНИК-ЛАМБРИНОВ': 'ПОЛКОВНИК ЛАМБРИНОВО',
        'ПОЛКОВНИК ЛАМБРИНОВ': 'ПОЛКОВНИК ЛАМБРИНОВО',
        'ПОЛКОВНИК-СЕРАФИМОВ': 'ПОЛКОВНИК СЕРАФИМОВО',
        'ПОЛКОВНИК СЕРАФИМОВ': 'ПОЛКОВНИК СЕРАФИМОВО'
    }

    if new_name.find('-') != -1:
      new_name = new_name.replace('-', ' ')

    new_name = names.get(new_name,new_name)

    return new_name

"""## Settlement Disambiguation Pipeline"""

oldest_record_date = datetime.datetime.strptime('31.12.1899', '%d.%m.%Y')

def fetch_raw_settlement_data(settlement: SettlementDataTuple) -> SettlementDataTuple:
  name = settlement.data

  # HACK!!! used to circumvent stripping of non-letter chars from the name
  if name.find('-') != -1:
    name = name.split('-')[1]

  encoded_name = urllib.parse.quote(name.encode('windows-1251'))
  data = fetch_raw_data(f'https://www.nsi.bg/nrnm/index.php?ezik=bul&f=6&name={encoded_name}&code=&kind=-1')
  req = data

  if req.status_code != 200:
    raise ValueError

  return SettlementDataTuple(settlement.key, req)

def parse_raw_settlement_data(settlement: SettlementDataTuple) -> SettlementDataTuple:
  req = settlement.data
  soup = BeautifulSoup(req.text, 'lxml')
  table = soup.find_all('table')[-4]

  data = defaultdict(list)
  last_key = ''

  for row in table.find_all('tr')[2:]:
    cells = row.find_all('td')
    num_cells = len(cells)

    if num_cells == 2:
      last_key = cells[0].text
    elif num_cells == 3:
      dates = cells[2].text.split('-')
      start = datetime.datetime.strptime(dates[0].strip(), '%d.%m.%Y')
      end = datetime.datetime.max

      if len(dates[1].strip()) > 0 :
        end = datetime.datetime.strptime(dates[1].strip(), '%d.%m.%Y')

      name_tuple = tuple(map(lambda s: s.strip(), cells[1].text.split(',')[::-1]))

      if end > oldest_record_date and len(name_tuple) > 2:
        data[last_key].append(
            SettlementNamesData(
                name_tuple,
                start,
                end,
                )
            )

  return SettlementDataTuple(settlement.key, data)

def mach_key_with_code(settlement: SettlementDataTuple) -> SettlementDataTuple:
  key = settlement.key
  data_dict = settlement.data
  result = None
  result_list = []

  for code, names_list in data_dict.items():
    for name_data in names_list:
      full_name = name_data.name

      if all([
              full_name[0].lower().find(key[0].lower()) != -1,
              full_name[1].lower().find(key[1].lower()) != -1,
              full_name[2].split('.')[-1].strip().lower() == key[2].lower()
              ]):
        result_list.append((name_data.last, SettlementDataTuple(key, code)))
      elif key[0].lower() == 'СОФИЙСКА'.lower(): # HACK!!! ;(
        if all([
              full_name[0].lower().find('софия') != -1,
              full_name[1].lower().find(key[1].lower()) != -1,
              full_name[2].split('.')[-1].strip().lower() == key[2].lower()
              ]):
          result_list.append((name_data.last, SettlementDataTuple(key, code)))

      elif key[0].lower() == 'СМОЛЯН'.lower(): # HACK!!! ;(
        if all([
              full_name[0].lower().find('пловдивска') != -1,
              full_name[1].lower().find(key[1].lower()) != -1,
              full_name[2].split('.')[-1].strip().lower() == key[2].lower()
              ]):
          result_list.append((name_data.last, SettlementDataTuple(key, code)))

      elif key[0].lower() == 'ПАЗАРДЖИК'.lower(): # HACK!!! ;(
        if all([
              full_name[0].lower().find('пазарджишки') != -1,
              full_name[1].lower().find(key[1].lower()) != -1,
              full_name[2].split('.')[-1].strip().lower() == key[2].lower()
              ]):
          result_list.append((name_data.last, SettlementDataTuple(key, code)))

  # if there are multiple matching names take the most recent one
  sorted(result_list)
  if len(result_list) > 0:
    result = result_list[-1][1]

  return result

"""## Table Parsing Pipeline"""

def fetch_raw_table(data_tuple: DataTuple) -> DataTuple:
  url = data_tuple.data
  req = fetch_raw_data(url)

  return DataTuple(req, data_tuple.header_type, data_tuple.table_type)

def raw_table_to_lines(data_tuple: DataTuple) -> DataTuple:
  req = data_tuple.data
  soup = BeautifulSoup(req.text, 'lxml').prettify()
  split = soup.split('\r\n')

  return DataTuple(split, data_tuple.header_type, data_tuple.table_type)

def parse_lines(data_tuple: DataTuple) -> DataTuple:

  def parse_data_line(line: str, table_type: TableTypeEnum) -> Optional[SettlementInfo]:
    settlement_info_re = ''
    permanent_population_position = -1
    current_population_position = -1

    if table_type == TableTypeEnum.Quarterly:
      settlement_info_re = RegexPatternWrapper().settlement_info_quarterly
      permanent_population_position = 2
      current_population_position = 3
    elif table_type == TableTypeEnum.Yearly:
      settlement_info_re = RegexPatternWrapper().settlement_info_yearly
      permanent_population_position = 2
      current_population_position = 6

    settlement_info = regex.search(settlement_info_re, line)

    if settlement_info:
      name, permanent, current = settlement_info.group(1,
                                                       permanent_population_position,
                                                       current_population_position)

      name_parts = name.split('.')
      name = '. '.join([name_parts[0], fix_names(name_parts[1])])
      settlement_info = SettlementInfo(name.strip(), permanent, current)

    return settlement_info

  @static_vars_function(region=None)
  def parse_header_line(line: str, header_type: HeaderEnum) -> Optional[MunicipalityIdentifier]:
    region_name = None

    if header_type == HeaderEnum.New:
      region_name_re = RegexPatternWrapper().region_name_new
      region_gr = regex.search(region_name_re, line)

      if region_gr:
        region, municipality = region_gr.group(1, 2)
        region_name = MunicipalityIdentifier(region.strip(), municipality.strip())

    elif header_type == HeaderEnum.Old:
      if parse_header_line.region is None:
        parse_header_line.region = regex.search(RegexPatternWrapper().old_reg, line)
        region_name = None
      else:
        mun_gr = regex.search(RegexPatternWrapper().old_mun, line)
        if mun_gr:
          region, municipality = parse_header_line.region.group(1), mun_gr.group(1)
          region_name = MunicipalityIdentifier(fix_names(region.strip()), fix_names(municipality.strip()))

        parse_header_line.region = None

    return region_name

  municipality_ids = {}
  settlements_info = {}

  for line_num, line in enumerate(data_tuple.data):
    municipality_id = parse_header_line(line, data_tuple.header_type)
    if municipality_id:
      municipality_ids[line_num] = municipality_id
      continue

    settlement_info = parse_data_line(line, data_tuple.table_type)
    if settlement_info:
      settlements_info[line_num] = settlement_info

  return DataTuple(ParsedLines(municipality_ids, settlements_info), data_tuple.header_type, data_tuple.table_type)

def parssed_lines_to_full_info_list(data_tuple: DataTuple) -> DataTuple:

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
      fnsi = FullSettlementInfo(fix_names(reg.region),
                                fix_names(reg.municipality),
                                fix_names(set_info.name),
                                set_info.permanent_residents,
                                set_info.current_residents)
      full_name_settlement_infos.append(fnsi)

      sk_index += 1

  return DataTuple(full_name_settlement_infos, data_tuple.header_type, data_tuple.table_type)

def full_info_list_to_data_frame(data_tuple: DataTuple) -> DataTuple:
  df = pd.DataFrame(data_tuple.data)
  df.set_index(['region', 'municipality', 'settlement'], drop=True, inplace=True)

  return DataTuple(df, data_tuple.header_type, data_tuple.table_type)

"""## Data Processing Pipeline"""

def process_data(data_source: List[DataTuple], config: Configuration) -> List[DataTuple]:
  parsed_data = None
  data_frame_list = []
  parsing_pipeline = config.extra_params['table_parsing']

  for data_tuple in data_source:
    if data_tuple.table_type == TableTypeEnum.Quarterly:
      date_group = RegexPatternWrapper().date_group
    else:
      date_group = RegexPatternWrapper().year_group

    date_string = regex.search(date_group, data_tuple.data).group(1).replace('-', '_')
    data_frame = parsing_pipeline(data_tuple).data
    data_frame = data_frame.rename(columns={'permanent_residents':f'permanent_{date_string}',
                                            'current_residents':f'current_{date_string}'})
    # print(date_string)
    data_frame_list.append(DataTuple(data_frame, data_tuple.header_type, data_tuple.table_type))

  PickleWrapper().pickle_data(data_frame_list, 'data_frames_list')

  return data_frame_list

def disambiguate_data(data_frame_list: List[DataTuple], config: Configuration) -> List[DataTuple]:

  settlement_disambiguation_pipeline = config.extra_params['settlement_disambiguation']

  processed_sdts = PickleWrapper().load_data('triple_to_ekatte')
  if (processed_sdts is None) or (not isinstance(processed_sdts, dict)):
    processed_sdts = {}

  reverse_dict = PickleWrapper().load_data('ekatte_to_triple')
  if (reverse_dict is None) or (not isinstance(reverse_dict, dict)):
    reverse_dict = {}

  sdt_list = list(map(lambda tup: (SettlementDataTuple(tup[0], tup[0][2]), tup[1]),
                      map(lambda name: ((fix_names(name[0].strip()),
                                         fix_names(name[1].strip()),
                                         fix_names(name[2].split('.')[1].strip())),
                                        name),
                          set(chain.from_iterable(
                                  map(lambda dt: dt.data.index.values.tolist(),
                                      data_frame_list))))))
  failiures = set()

  for i, sdt in enumerate(sdt_list):
    if sdt[0].key in processed_sdts and processed_sdts[sdt[0].key] in reverse_dict:
      continue

    try:
      val = settlement_disambiguation_pipeline(sdt[0])
    except ValueError:
      time.sleep(10)
      try:
        val = settlement_disambiguation_pipeline(sdt[0])
      except ValueError:
        time.sleep(15)
        try:
          val = settlement_disambiguation_pipeline(sdt[0])
        except ValueError:
          time.sleep(20)
          val = settlement_disambiguation_pipeline(sdt[0])
    finally:
      if val is not None:
        # print(i, val.key, val.data)
        processed_sdts[val.key] = val.data
        PickleWrapper().pickle_data(processed_sdts, 'triple_to_ekatte')

        reverse_dict[val.data] = sdt[1]
        PickleWrapper().pickle_data(reverse_dict, 'ekatte_to_triple')
      else:
        failiures.add(sdt[0])

  PickleWrapper().pickle_data(failiures, 'failiures')

  disambiguated_data = []
  for dt in data_frame_list:
    df = dt.data
    df.reset_index(inplace=True)
    df['ekatte'] = df['settlement']
    cols = df.columns

    def updt(x):
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

    df_list_updated = [updt(x) for x in df.to_numpy()]

    df = pd.DataFrame(df_list_updated)

    df.columns = cols
    df.dropna(inplace=True)
    df.set_index(['ekatte'], drop=True, inplace=True)

    df = df.loc[~df.index.duplicated(keep='first')]
    # print(df.shape)
    disambiguated_data.append(DataTuple(df, dt.header_type, dt.table_type))

  PickleWrapper().pickle_data(disambiguated_data, 'data_frames_list_disambiguated')

  return disambiguated_data

def combine_data(processed_data: List[DataTuple], config: Configuration) -> List[DataTuple]:
  combined = None
  names = ['region',	'municipality', 'settlement']

  for dt in processed_data:
    if combined is None:
      combined = dt.data.drop(labels=names,axis=1)
    else:
      df = dt.data.drop(labels=names,axis=1)
      combined = combined.merge(df,
                                how='outer',
                                copy=False,
                                left_index=True,
                                right_index=True)

  combined.fillna(value=0, inplace=True)
  for column in combined.columns.to_list():
    combined[column] = combined[column].astype(int)

  PickleWrapper().pickle_data(combined, 'combined_tables')

  return [DataTuple(combined, 0, 0)]

def store_data_list(processed_data: List[DataTuple], config: Configuration) -> List[DataTuple]:
  for dt in processed_data:
    df = dt.data

    name = f'grao_data_{"_".join(df.columns[-1].split("_")[1:])}'
    directory = config.processed_tables_path
    if not os.path.exists(directory):
      os.makedirs(directory)
    # print(f'{directory}/{name}.csv')
    df.to_csv(f'{directory}/{name}.csv')

  return processed_data

def store_combined_data(processed_data: List[DataTuple], config: Configuration) -> List[DataTuple]:
  combined_data = processed_data[0].data

  directory = config.combined_tables_path
  if not os.path.exists(directory):
    os.makedirs(directory)

  # print(f'{directory}/grao_data_combined.csv')
  combined_data.to_csv(f'{directory}/grao_data_combined.csv')

  return processed_data

"""## Visualizations"""

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

def create_visualizations():
  ekatte_to_triple = PickleWrapper().load_data('ekatte_to_triple')
  triple_to_ekatte = PickleWrapper().load_data('triple_to_ekatte')
  combined         = PickleWrapper().load_data('combined_tables')

  combined_dict    = combined.to_dict(orient='index')

  plt.rcParams['figure.figsize'] = [45, 15]

  visualisaion_directory = 'visualization'

  labels = list(combined_dict[list(combined_dict.keys())[0]].keys())
  date_labels = list(map(lambda l: ' '.join(l.split('_')[1:]),labels[0::2]))
  date_labels.reverse()

  x = np.arange(len(date_labels))  # the label locations
  width = 0.4  # the width of the bars

  for item in combined_dict:
    triple = ekatte_to_triple[item]
    save_dir = f'{visualisaion_directory}/{triple[0]}/{triple[1]}'.replace(' ', '_')
    full_path = f'{os.getcwd()}/{save_dir}'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt_path_permanent = f'{full_path}/{triple[2]}_permanent'.replace('.', '').replace(' ', '_')
    plt_path_current = f'{full_path}/{triple[2]}_current'.replace('.', '').replace(' ', '_')
    plt_path_compare = f'{full_path}/{triple[2]}_compare'.replace('.', '').replace(' ', '_')
    name = f'обл. {triple[0]}, общ. {triple[1]}, {triple[2]}'

    fig, ax = plt.subplots()

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Number of residents')
    ax.set_xticks(x)
    ax.set_xticklabels(date_labels)
    ax.set_title(name)

    values = list(combined_dict[item].values())

    permanent_values = values[0::2]
    permanent_values.reverse()

    rects1 = ax.bar(x - width/20, permanent_values, width, label='Permanent', align='center')
    autolabel(rects1)

    ax.legend()

    plt.savefig(plt_path_permanent)
    print(plt_path_permanent)

    # Clear the current axes.
    plt.cla()

    # Clear the current figure.
    plt.clf()
    plt.ioff()

    fig, ax = plt.subplots()

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Number of residents')
    ax.set_xticks(x)
    ax.set_xticklabels(date_labels)
    ax.set_title(name)

    current_values = values[1::2]
    current_values.reverse()

    rects2 = ax.bar(x + width/20, current_values, width, label='Current', align='center')
    autolabel(rects2)

    ax.legend()

    plt.savefig(plt_path_current)
    print(plt_path_current)

    # Clear the current axes.
    plt.cla()

    # Clear the current figure.
    plt.clf()
    plt.ioff()

    fig, ax = plt.subplots()
    ax.set_xticks(x)
    ax.set_xticklabels(date_labels)
    plt.plot(permanent_values)
    plt.plot(current_values)

    plt.title('Comparison between permanent and current')
    plt.xlabel('Year')
    plt.ylabel('Number of residents')
    plt.legend(['Permanent', 'Current'], loc='upper right')

    plt.savefig(plt_path_compare)
    print(plt_path_compare)

    # Clear the current axes.
    plt.cla()
    # Clear the current figure.
    plt.clf()
    plt.ioff()

    # Closes all the figure windows.
    plt.close('all')
    # break

"""# Update bot"""

def login_with_credentials(credentials_path):
  credentials = pd.read_csv(credentials_path)
  username, password = tuple(credentials)

  return wdi_login.WDLogin(username, password)

def update_item(login, settlement_qid, population):
    ref = wdi_core.WDUrl(prop_nr="P854", value="https://www.grao.bg/tna/t41nm-15-06-2020_2.txt", is_reference=True)

    determination_method = wdi_core.WDItemID(value='Q90878157', prop_nr="P459", is_qualifier=True)
    point_in_time = wdi_core.WDTime(time='+2020-06-15T00:00:00Z', prop_nr='P585', is_qualifier=True)
    # publisher = wdi_core.WDItemID(value=login.consumer_key, prop_nr="P123", is_reference=True)

    qualifiers = []
    qualifiers.append(point_in_time)
    qualifiers.append(determination_method)
    data = []
    prop = wdi_core.WDQuantity(prop_nr='P1082', value=population, qualifiers=qualifiers, references=[[ref]])
    data.append(prop)

    item = wdi_core.WDItemEngine(wd_item_id=settlement_qid, data=data)
    item.write(login, False)
    time.sleep(15)

def update_all_settlements():
  data = pd.read_csv("matched_data/matched_data.csv")
  data.columns = ['ekatte', 'region', 'municipality', 'settlement', 'pop_permanent', 'pop_current']

  login = login_with_credentials('data/credentials.csv')

  error_logs = []
  for _, row in data.iterrows():
      try:
          settlement_qid = row['settlement']
          population = row['pop_permanent']
          update_item(login, settlement_qid, population)
      except:
          error_logs.append(settlement_qid)
          print("An error occured for item : " + settlement_qid)

  print("Summarizing failures for specific IDs")
  for error in error_logs:
      print("Error for : " + error)

def main():

  current_dir = os.path.dirname(os.path.abspath(__file__))

  example_text = """Examples:
    python3  grao_tables_processing.py

    python3  grao_tables_processing.py
      --data_configuration_path <path to folder>
      --processed_tables_path <path to folder>
      --matched_tables_path <path to folder>
      --combined_tables_path <path to folder>
      --pickled_data_path <path to folder>
      --produce_graphics
      --update_wiki_data
  """

  parser = argparse.ArgumentParser(description="Processes the tables provided by GRAO and"
                                               "extracts th einformation from them to csv files",
                                   epilog=example_text,
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--data_configuration_path",
                      type=str, default=f'{current_dir}/config/data_config.json',
                      help="Path to the JSON file containing the configuration for which tables should be processed.")
  parser.add_argument("--processed_tables_path",
                      type=str, default=f'{current_dir}/processed_tables',
                      help="Path to the folder where the processed tables will be stored.")
  parser.add_argument("--matched_tables_path",
                      type=str, default=f'{current_dir}/matched_data',
                      help="Path to the folder where the matched tables are be stored.")
  parser.add_argument("--combined_tables_path",
                      type=str, default=f'{current_dir}/combined_tables',
                      help="Path to the folder where the combined tables will be stored.")
  parser.add_argument("--pickled_data_path",
                      type=str, default=f'{current_dir}/pickled_data',
                      help="Path to the folder where pickled objects will be stored.")
  parser.add_argument("--produce_graphics",
                      default=False, action="store_true",
                      help="If set the script will produce grafics from the processed tables.")
  parser.add_argument("--update_wiki_data",
                      default=False, action="store_true",
                      help="If set the script will update WikiData with the processed tables.")


  args = parser.parse_args()
  PickleWrapper(args.pickled_data_path)

  configuration = Configuration(
    args.data_configuration_path,
    args.processed_tables_path,
    args.matched_tables_path,
    args.combined_tables_path,
    args.pickled_data_path
  )

  settlement_disambiguation = Pipeline(functions=(
    fetch_raw_settlement_data,
    parse_raw_settlement_data,
    mach_key_with_code
  ))
  configuration.extra_params['settlement_disambiguation'] = settlement_disambiguation

  table_parsing = Pipeline(functions=(
    fetch_raw_table,
    raw_table_to_lines,
    parse_lines,
    parssed_lines_to_full_info_list,
    full_info_list_to_data_frame,
  ))
  configuration.extra_params['table_parsing'] = table_parsing


  processing_pipeline = Pipeline(functions=(
    (lambda data: process_data(data, configuration)),
    (lambda data: disambiguate_data(data, configuration)),
    (lambda data: store_data_list(data, configuration)),
    (lambda data: combine_data(data, configuration)),
    (lambda data: store_combined_data(data, configuration)),
  ))

  data_source = configuration.process_data_configuration()

  processed_data = processing_pipeline(data_source)

  if args.produce_graphics:
    create_visualizations()

  if args.update_wiki_data:
    update_all_settlements()


if __name__ == "__main__":
  main()