from os.path import basename, join
from typing import List, Tuple
from datetime import datetime
from regex import search   # type: ignore
from os import listdir

from grao_tables_processing.common.helper_functions import execute_in_parallel
from grao_tables_processing.common.regex_pattern_wrapper import RegexPatternWrapper
from grao_tables_processing.common.pipeline import Pipeline
from grao_tables_processing.common.custom_types import UnexpectedNoneError


def find_ref_url(path_to_file: str, file_prefix: str, url_list: List[str]) -> str:
  processing_pipline = Pipeline((
    (lambda path: basename(path)),
    (lambda name: name.split('.')[0]),
    (lambda name: name.replace(file_prefix, '')),
    (lambda name: name.replace('_', '-')),
    (lambda date_str: next(filter((lambda url: url.find(date_str) > -1), url_list))),
  ))

  result = processing_pipline(path_to_file)
  return result


def date_from_url(url: str) -> datetime:
  date_str: str = ''

  if date_group := search(RegexPatternWrapper().full_date_group, url):
    date_str = date_group.group(1)
  elif date_group := search(RegexPatternWrapper().year_group, url):
    date_str = date_group.group(1)
    date_str = f'31-12-{date_str}'

  date = datetime.strptime(date_str, '%d-%m-%Y')

  return date


def file_prefix_for_directory(directory: str):
  return f'{basename(directory)}_'


def find_date_suffix(url: str) -> str:
    date = date_from_url(url)
    date_suffix = f'{date.year}'

    if date.day != 31 and date.month != 12:
      date_suffix = f'{date.month:02}_{date_suffix}'

    return date_suffix


def single_processed_file_info(input_data: Tuple[str, str, List[str]]) -> Tuple[datetime, str, str]:
  file, storage_directory, url_list = input_data
  file_prefix = file_prefix_for_directory(storage_directory)

  url = find_ref_url(file, file_prefix, url_list)
  date = date_from_url(url)

  return (date, url, join(storage_directory, file))


def find_latest_processed_file_info(storage_directory: str, url_list: List[str]) -> Tuple[datetime, str, str]:
  wrapped_data_generator = ((file, storage_directory, url_list) for file in listdir(storage_directory))

  processed_files = execute_in_parallel(single_processed_file_info, wrapped_data_generator)

  if processed_files is None:
    raise UnexpectedNoneError(f'Couldn\'t processed files in {storage_directory}')

  processed_files = sorted(processed_files)
  return processed_files[-1]
