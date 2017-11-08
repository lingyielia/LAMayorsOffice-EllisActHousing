"""Read in a bunch of excel sheets and build a single giant dataset"""

import os
import string
import re
import pickle

import numpy as np

import pandas as pd
import recordlinkage
import networkx as nx

from tqdm import tqdm
import usaddress as ua

DATA_DIR = os.path.expanduser('~/GitHub/la_mayors_office/data')
SAVE_DIR = os.path.expanduser('~/GitHub/la_mayors_office/data/processed')

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)


ADDRESS_FIELD_PARSE_NAMES = {
    'AddressNumber': 'Address Number',
    'StreetName': 'Street Name',
    'StreetNamePostType': 'Street Suffix',
    'StreetNamePreDirectional': 'Street Direction',
    'PlaceName': 'City',
    'StateName': 'State'
}

STREET_SUFFIXES = set([
    'ave', 'st', 'pl', 'blvd', 'dr', 'way', 'ter', 'road', 'rd',
   'park', 'walk', 'cir', 'ct', 'hwy', 'cl', 'av', 'est',
   'ss', 'bl', 'place', 'stst', 'bend', 'blblvd', 'ndr',
   'al', 'summit', 'nn', 'green', 'creek', 'avenue', 'parkway', 'cyn',
   'street', 'npl', 'vista', 'glen', 'plaza', 'rcd', 'lane', 'ridge',
   'ast', 'heights', 'crossroad', 'tr', 'magdalena', 'pavia',
   'pass', 'pkwy', 'mall', 'pz', 'terrace', 'crt', 
   'court', 'cove', 'boulevard', 'sq',
   'lvl', 'ln', 'ctr', 'grd', 'promenade', 'ck', 'circle',
   'loop', 'mt', 'tsf', 'fls', 'flrs', 'mar',
   'annex', 'rdg', 'strt', 'terr', 'pt', 'haven', 'trl', 'drive'
])

PUNCT_TO_SPACE = string.maketrans(string.punctuation, " "*len(string.punctuation))
def punct_to_space(in_string):
    """Replace punctuation from string"""
    return str(in_string).translate(PUNCT_TO_SPACE)


NON_DECIMAL_RE = re.compile(r'[^\d.]+')
def string_to_float(in_string):
    """Get numeric house number"""
    try:
        just_numeric = NON_DECIMAL_RE.sub('', punct_to_space(in_string).split()[0])
        if just_numeric:
            return float(just_numeric)
    except IndexError:
        pass
    return None

def string_to_int(in_string):
    """Get int"""
    try:
        just_numeric = NON_DECIMAL_RE.sub('', punct_to_space(in_string).split()[0])
        if just_numeric:
            return int(just_numeric)
    except ValueError:
        pass
    return None

# The processing functions in this file are designed to handle these
# particular files


ALL_RSO_XLS_FILE = os.path.join(
    DATA_DIR, "RSO inventory May 24 2017.xlsx")

WITHDRAWAL_XLS_FILE = os.path.join(
    DATA_DIR, 'To Mayor - Ellis 7-16-2007 - 7-31-2017 Ran 8-18-2017.xlsx')

ENTITLEMENTS_XLS_FILE = os.path.join(
    DATA_DIR, 'DCP Applications Filed 10Yrs.xlsx')

ENTITLEMENTS_XLS_FILE = os.path.join(
    DATA_DIR, 'RawDataApplicationsFiled10Yrs_RunOn11072017.xlsx')

DEMOBUILD_PERMIT_XLS_FILE = os.path.join(
    DATA_DIR, "Building Permit Records 2007-2017.xls")


TEMP_OUTPUT_FILE = os.path.join(SAVE_DIR, "la_housing_dataset_no_geo.csv")
FEATURES_OUTPUT_FILE = os.path.join(SAVE_DIR, "features.pickle")
FINAL_OUTPUT_FILE = os.path.join(SAVE_DIR, "la_housing_dataset_no_geo_property_id.csv")


# This is the list of colums used in the final file
COLUMN_ORDER = [
    "Property ID",
    "APN",
    "General Category",
    "Status",
    "Status Date",
    "Completion Date",
    "Permit #",
    "Permit Type",
    "Permit Sub-Type",
    "Work Description",
    "Address Full",
    "Address Number",
    "Address Number (float)",
    "Street Direction",
    "Street Name",
    "Street Suffix",
    "City",
    "State",
    "Zip Code",
    "Unit Count",
    "Unit Number",
    "Unit Type",
    "Council District",
]


def strip_punct(in_string):
    """Delete punctuation from string"""
    return str(in_string).translate(None, string.punctuation)

def clean_spaces(in_string):
    """Remove multi-spaces"""
    return ' '.join(str(in_string).split())


def parse_addy(str_addy):
    """Parse a string address into a series"""

    str_addy = str(str_addy).lower()
    try:
        parse_tuples = ua.parse(str_addy)
    except:
        parse_tuples = {}

    parse_dict = {
        ADDRESS_FIELD_PARSE_NAMES[t]:v
        for v, t in parse_tuples
        if t in ADDRESS_FIELD_PARSE_NAMES
    }
    return pd.Series(parse_dict)


def make_apn(df_in):
    """ This function will form an APN by concatenting the partial APNs"""
    apn_cols = ['Assessor Book', 'Assessor Page', 'Assessor Parcel']
    series_apn = df_in[apn_cols].astype(str).apply(lambda x: ''.join(x), axis=1)
    series_apn[series_apn.str.contains('nan')] = ''
    series_apn[series_apn.str.contains('\*')] = ''
    return series_apn


def process_withdrawal_file(xls_filename):
    """Read in and apply formatting to the ellis withdrawal file"""

    df_ellis = pd.read_excel(xls_filename, sheetname=None)
    df_ellis = pd.concat(df_ellis, axis=0).reset_index(drop=True)

    df_ellis.APN = df_ellis.APN.astype(str)
    df_ellis.Zip = df_ellis.Zip.astype(str)

    df_ellis.Address = df_ellis.Address.apply(strip_punct)
    df_ellis.Address = df_ellis.Address.apply(lambda x: clean_spaces(x.split(' CA ')[0]))

    df_ellis = df_ellis.rename(
        columns={
            'APN': 'APN',
            'Date Filed': 'Status Date',
            'Address': 'Address Full',
            'Zip': 'Zip Code'
        }
    )

    # parsing address
    df_addy = df_ellis['Address Full'].apply(parse_addy)
    df_ellis = pd.concat([df_ellis, df_addy], axis=1)

    # Fix datetime dtype
    df_ellis['Status Date'] = pd.to_datetime(df_ellis['Status Date'])

    # Add in 'General Category'
    df_ellis["General Category"] = 'Ellis Withdrawal'

    df_ellis = df_ellis[[c for c in COLUMN_ORDER if c in df_ellis]]

    return df_ellis


def OLD_process_entitlements_file(xls_filename):
    """Read in and apply formatting to the entitlenments file"""

    df_ent = pd.read_excel(xls_filename, sheetname='Export Worksheet')

    df_ent.ADDRESS = df_ent.ADDRESS.apply(strip_punct)
    df_ent.ADDRESS = df_ent.ADDRESS.apply(lambda x: clean_spaces(x.split(' CA ')[0]))

    df_ent = df_ent.rename(
        columns={
            'FILING_DT': 'Status Date',
            'COMPLETION_DT': 'Completion Date',
            'ADDRESS': 'Address Full',
            'PROJ_DESC': 'Work Description',
            'PROCESSINGUNIT': 'Permit Type',
            'CASE_NBR': 'Permit #'
        }
    )

    # parsing address
    df_addy = df_ent['Address Full'].apply(parse_addy)
    df_ent = pd.concat([df_ent, df_addy], axis=1)

    # Fix datetime dtype
    df_ent['Status Date'] = pd.to_datetime(df_ent['Status Date'])
    df_ent['Completion Date'] = pd.to_datetime(df_ent['Completion Date'])

    # Add in 'General Category'
    df_ent["General Category"] = 'Entitlement Change'

    df_ent = df_ent[[c for c in COLUMN_ORDER if c in df_ent]]

    return df_ent


def stop_repeating_yourself(in_string):
    """When did Jimmy Two-times get a dta entry job?
    https://www.youtube.com/watch?v=CfW-MPUjC_0
    """
    sz = len(in_string)
    half1, half2 = in_string[:sz//2], in_string[sz//2:]
    if half1 == half2:
        return half1
    return in_string


def drop_multiple_street_suffixes(tok_list):
    suffix_idx_list = []
    for key, val in enumerate(tok_list):
        if val in STREET_SUFFIXES:
            suffix_idx_list.append(key)
    for to_drop in suffix_idx_list[1:][::-1]:
        tok_list.pop(to_drop)
    return tok_list


def parse_addy_DCP(str_addy):
    """Parse a string address from DCP file into a series"""

    tok_addy = str(str_addy).lower().split()
    tok_addy = drop_multiple_street_suffixes(tok_addy)
    
    if not tok_addy:
        return pd.Series({})

    try:
        zipcode = int(tok_addy[-1])
        str_addy = ' '.join(tok_addy[:-1])
    except ValueError:
        zipcode = np.nan
        str_addy = ' '.join(tok_addy)

    try:
        parse_tuples = ua.parse(str_addy)
    except:
        parse_tuples = {}

    parse_dict = {
        ADDRESS_FIELD_PARSE_NAMES[t]:v
        for v, t in parse_tuples
        if t in ADDRESS_FIELD_PARSE_NAMES
    }

    if "Street Name" in parse_dict:
        parse_dict["Street Name"] = stop_repeating_yourself(parse_dict["Street Name"])

    if zipcode:
        parse_dict["Zip Code"] = zipcode

    return pd.Series(parse_dict)


def process_entitlements_file(xls_filename):
    df_ent = pd.read_excel(xls_filename, sheetname='Export Worksheet')   
    
    # Parse Address
    df_ent.ADDRESS = df_ent.ADDRESS.apply(strip_punct)
    df_addy = df_ent.ADDRESS.apply(parse_addy_DCP)
    df_ent = pd.concat([df_ent, df_addy], axis=1)
    
    df_ent = df_ent.rename(
        columns={
            'APN': 'APN',
            'FILING_DT': 'Status Date',
            'COMPLETION_DT': 'Completion Date',
            'ADDRESS': 'Address Full',
            'PROJ_DESC': 'Work Description',
            'PROCESSINGUNIT': 'Permit Type',
            'CASE_NBR': 'Permit #',
            'Address Number': 'Address Number',
            'Street Direction': 'Street Direction',
            'Street Name': 'Street Name',
            'Street Suffix': 'Street Suffix',
            'City': 'City',
            'State': 'State',
            'Zip Code': 'Zip Code',
            'Unit Count': 'Unit Count',
            'Unit Number': 'Unit Number',
            'Unit Type': 'Unit Type',            
        }
    )

    # Fix datetime dtype
    df_ent['Status Date'] = pd.to_datetime(df_ent['Status Date'])
    df_ent['Completion Date'] = pd.to_datetime(df_ent['Completion Date'])

    # Add in 'General Category'
    df_ent["General Category"] = 'Entitlement Change'

    df_ent = df_ent[[c for c in COLUMN_ORDER if c in df_ent]]

    return df_ent


def process_building_and_demolition_file(xls_filename):
    """Read in and apply formatting to the demolitions file"""

    df_demo = pd.read_excel(xls_filename, sheetname=None)
    df_demo = pd.concat(df_demo, axis=0).reset_index(drop=True)

    df_demo = df_demo.rename(
        columns={
            'PERMIT TYPE': 'Permit Type',
            'PERMIT NUMBER': 'Permit #',
            'ADDRESS': 'Address Full',
            'PERMIT SUB-TYPE': 'Permit Sub-Type',
            'WORK DESCRIPTION': 'Work Description',
            'STATUS': 'Status',
            'ISSUE DATE': 'Status Date',
            'ZIP CODE': 'Zip Code'
        }
    )

    demolition_idx = df_demo['Permit Type'].isin(['Bldg-Demolition', 'NonBldg-Demolition'])
    
    # Add in 'General Category'
    df_demo["General Category"] = ''
    df_demo.loc[demolition_idx, "General Category"] = 'Demolition Permits'
    df_demo.loc[~demolition_idx, "General Category"] = 'Building Permits'

    # parsing address
    df_addy = df_demo['Address Full'].apply(parse_addy)
    df_demo = pd.concat([df_demo, df_addy], axis=1)

    # Fix datetime dtype
    df_demo['Status Date'] = pd.to_datetime(df_demo['Status Date'])

    df_demo = df_demo[[c for c in COLUMN_ORDER if c in df_demo]]

    return df_demo


def process_rso_inventory(xls_filename):
    """Read in and apply formatting to the rso inventory file"""

    df_rso = pd.read_excel(xls_filename, sheetname=None)
    df_rso = pd.concat(df_rso, axis=0).reset_index(drop=True)

    df_rso.APN = df_rso.APN.astype(str)
    df_rso.Property_Zip_Code = df_rso.Property_Zip_Code.astype(str)
    df_rso.Property_Street_Address = df_rso.Property_Street_Address.apply(strip_punct)

    df_rso = df_rso.rename(
        columns={
            'Property_Zip_Code': 'Zip Code',
            'Property_Street_Address': 'Address Full',
            'Unit_Count': 'Unit Count',
            'Council_District': 'Council District'
        }
    )

    df_addy = df_rso['Address Full'].apply(parse_addy)
    df_rso = pd.concat([df_rso, df_addy], axis=1)
    df_rso["General Category"] = "Is in RSO Inventory"
    df_rso = df_rso[[c for c in COLUMN_ORDER if c in df_rso]]

    return df_rso


def concat_datasets_and_save():
    """Concatenate and save all the dataset. ALL THE DATASETS!"""

    dataset_list = []

    print("Loading withdrawals file")
    dataset_list.append(process_withdrawal_file(WITHDRAWAL_XLS_FILE))

    print("Loading entitlement file")
    dataset_list.append(process_entitlements_file(ENTITLEMENTS_XLS_FILE))

    print("Loading demolitions + building file")
    dataset_list.append(process_building_and_demolition_file(DEMOBUILD_PERMIT_XLS_FILE))

    print('Loading inventory of RSO units')
    dataset_list.append(process_rso_inventory(ALL_RSO_XLS_FILE))

    df_full = pd.concat(dataset_list, axis=0).fillna(value='').reset_index(drop=True)

    df_full.to_csv(TEMP_OUTPUT_FILE, index=False, encoding='utf-8')

    return df_full


def compute_record_linkage(df_full):
    """Given the fully-concatenated table of records
    calculate which pairs are address-matches using
    fuzzy matching on street name, unit number and zipcode
    """

    print("Setting up blocking for pairwise comparisons")
    blocking_indices = [
        recordlinkage.BlockIndex(on="APN (int)"),
        recordlinkage.BlockIndex(on=["Address Number (float)", "Zip Code (int)"]),
    ]

    print("Finding blocked pairs")
    pairs = None
    for df_subset in tqdm(np.array_split(df_full, 10)):
        for bi in blocking_indices:
            _new_pairs = bi.index(df_full, df_subset)

            if pairs is not None:
                pairs = pairs.union(_new_pairs)
            else:
                pairs = _new_pairs

    print("Setting up similarity calculations")
    compare_cl = recordlinkage.Compare()

    compare_cl.exact('APN (int)', 'APN (int)', label='APN')
    compare_cl.exact('Zip Code (int)', 'Zip Code (int)', label='Zip')
    compare_cl.exact('Address Number (float)', 'Address Number (float)', label='number')

    #compare_cl.numeric('Address Number (float)', 'Address Number (float)',
    #                   offset=3, scale=2,
    #                   label='number')

    compare_cl.string('Street Name', 'Street Name',
                      method='levenshtein',
                      threshold=0.9,
                      label='street')

    print("Calculating similarities")
    features = compare_cl.compute(pairs, df_full)
    features.to_pickle(FEATURES_OUTPUT_FILE)
    
    return features


def form_clusters(df_full, features):
    """Given the full dataframe and a matrix of feature linkages,
    append a column to the dataset that contains the property_id
    that connects all properties toegether
    """

    features['APN'] = 3 * features['APN']
    matches = features[features.sum(axis=1) >= LINKAGE_THRESHOLD]

    graph = nx.Graph()
    for node_num in xrange(df_full.shape[0]):
        graph.add_node(node_num)
    print('Num nodes: {}'.format(graph.number_of_nodes()))

    for pair, _ in matches.iterrows():
        graph.add_edge(*pair)    
    print('Num edges: {}'.format(graph.number_of_edges()))

    graph = graph.to_undirected()

    # Finding connected clusters in the graph
    comp_num = 0
    row_to_group = {}
    for comp in nx.connected_components(graph):
        for row_num in list(comp):
            row_to_group[row_num] = comp_num
        comp_num += 1

    df_full['Property ID'] = pd.Series(df_full.index).apply(lambda x: row_to_group[x])
    df_full = df_full[[c for c in COLUMN_ORDER if c in df_full]]
    df_full.to_csv(FINAL_OUTPUT_FILE, index=False, encoding='utf-8')
    
    return df_full


if __name__ == '__main__':

    # Telling pylint to ignore the non-global naming scheme in here
    # pylint: disable=C0103

    if os.path.exists(TEMP_OUTPUT_FILE):
        print('Reading temp file from {}'.format(TEMP_OUTPUT_FILE))
        df_data = pd.read_csv(TEMP_OUTPUT_FILE)
    else:
        df_data = concat_datasets_and_save()

    df_data['Address Full'] = df_data['Address Full'].astype(unicode)
    df_data['Street Name'] = df_data['Street Name'].astype(unicode)
    df_data['Address Number (float)'] = df_data['Address Number'].apply(string_to_float)
    df_data['Zip Code (int)'] = df_data['Zip Code'].apply(string_to_int)
    df_data['APN (int)'] = df_data['APN'].apply(string_to_int)

    if os.path.exists(FEATURES_OUTPUT_FILE):
        print('Loading features from saved file {}'.format(FEATURES_OUTPUT_FILE))
        with open(FEATURES_OUTPUT_FILE, 'rb') as file:
            features = pickle.load(file)
    else:
        features = compute_record_linkage(df_data)

    # Look at the distribution of feature-scores
    #features.sum(axis=1).value_counts(bins=50).sort_index(ascending=False)

    LINKAGE_THRESHOLD = 3.0
    df_data = form_clusters(df_data, features)

    #aws s3 cp ~/GitHub/la_mayors_office/data/processed/la_housing_dataset_no_geo_property_id.csv s3://datadive-democraticfreedom-nyc/LA\ Mayor\'s\ Office\ -\ Housing/Cleaned\ Data/ 