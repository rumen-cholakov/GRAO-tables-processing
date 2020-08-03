from typing import Any, Callable, Optional, List, Generator
from joblib import Parallel, delayed  # type: ignore
from requests import get as get_request
from requests.utils import default_headers

from grao_tables_processing.common.custom_types import T, U


def execute_in_parallel(
  function: Callable[[T], U],
  data_source: Generator[T, None, None],
  num_jobs: int = -1
) -> Optional[List[U]]:
  result: Optional[List[U]] = []

  with Parallel(n_jobs=num_jobs) as parallel:
    result = parallel(map(delayed(function), data_source))

  return result


def fetch_raw_data(url: str, encoding: str = 'windows-1251') -> Any:
  headers = default_headers()
  headers.update({
      'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:52.0) Gecko/20100101 Firefox/52.0'
  })

  req = get_request(url, headers)
  req.encoding = encoding

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

    new_name = names.get(new_name, new_name)

    return new_name
