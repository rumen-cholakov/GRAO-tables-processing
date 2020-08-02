import pandas as pd
import time

from typing import List, Any
from datetime import datetime
from wikidataintegrator import wdi_core, wdi_login


from grao_tables_processing.common.configuration import Configuration
from grao_tables_processing.wikidata_interaction.common import find_latest_processed_file_info


def create_qualifiers(date: datetime) -> List[Any]:
  ref_time_str = f'+{date.isoformat()}Z'
  point_in_time = wdi_core.WDTime(time=ref_time_str, prop_nr='P585', is_qualifier=True)
  determination_method = wdi_core.WDItemID(value='Q90878157', prop_nr="P459", is_qualifier=True)

  return [point_in_time, determination_method]


def login_with_credentials(credentials_path: str) -> wdi_login.WDLogin:
  credentials = pd.DataFrame(pd.read_csv(credentials_path))
  username, password = tuple(credentials)

  return wdi_login.WDLogin(username, password)


def update_item(login: wdi_login.WDLogin, settlement_qid: str, data: List[wdi_core.WDQuantity]):
  item = wdi_core.WDItemEngine(wd_item_id=settlement_qid, data=data)
  item.write(login, False)
  time.sleep(15)


def update_all_settlements(config: Configuration):
  login = login_with_credentials(config.credentials_path)

  ref_time, ref_url, path = find_latest_processed_file_info(config.matched_tables_path, config.data)

  ref = wdi_core.WDUrl(prop_nr="P854", value=ref_url, is_reference=True)
  # publisher = wdi_core.WDItemID(value=login.consumer_key, prop_nr="P123", is_reference=True)

  qualifiers = create_qualifiers(ref_time)

  error_logs = []

  data = pd.DataFrame(pd.read_csv(path))
  for _, row in data.iterrows():
    settlement_qid: str = row['settlement']
    population: str = row['permanent_population']
    prop = wdi_core.WDQuantity(
      prop_nr='P1082',
      value=population,
      qualifiers=qualifiers,
      references=[[ref]]
    )

    try:
      update_item(login, settlement_qid, [prop])
    except BaseException:
      error_logs.append(settlement_qid)
      print("An error occurred for item : " + settlement_qid)

  if len(error_logs) > 0:
    print("Summarizing failures for specific IDs")
    for error in error_logs:
      print("Error for : " + error)
