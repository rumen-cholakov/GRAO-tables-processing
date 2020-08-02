import grao_tables_processing.common.configuration as cnf
import grao_tables_processing.common.pickle_wrapper as pw

import grao_tables_processing.settlement_disambiguation as sd
import grao_tables_processing.table_parsing as tpr
import grao_tables_processing.table_processing as tp
import grao_tables_processing.visualization as v
import grao_tables_processing.wikidata_interaction as wi


Configuration = cnf.Configuration
PickleWrapper = pw.PickleWrapper

settlement_disambiguation = sd.settlement_disambiguation
table_parser = tpr.table_parser
create_table_processor = tp.create_table_processor
create_visualizations = v.create_visualizations
update_matched_data = wi.update_matched_data
update_all_settlements = wi.update_all_settlements
