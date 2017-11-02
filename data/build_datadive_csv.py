"""Read in a bunch of excel sheets and build a single giant dataset"""

import os
import string
import re

import usaddress as ua
import pandas as pd

import networkx as nx
import recordlinkage

DATA_DIR = os.path.expanduser('~/GitHub/la_mayors_office/data')
SAVE_DIR = os.path.expanduser('~/GitHub/la_mayors_office/data/processed')

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)


ADDRESS_FIELD_PARSE_NAMES = {
    'AddressNumber': 'Address Number',
    'StreetName': 'Street Name',
    'StreetNamePostType': 'Street Suffix',
    'StreetNamePreDirectional': 'Street Direction',
    'OccupancyType': 'Unit Type',
    'OccupancyIdentifier': 'Unit Number',
    'PlaceName': 'City',
    'StateName': 'State'
}


PUNCT_TO_SPACE = string.maketrans(string.punctuation, " "*len(string.punctuation))
def punct_to_space(in_string):
    """Replace punctuation from string"""
    return str(in_string).translate(PUNCT_TO_SPACE)


NON_DECIMAL_RE = re.compile(r'[^\d.]+')
def string_to_numeric(in_string):
    """Get numeric house number"""
    try:
        just_numeric = NON_DECIMAL_RE.sub('', punct_to_space(in_string).split()[0])
        if just_numeric:
            return float(just_numeric)
    except IndexError:
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

#DEMO_PERMIT_CSV_FILE = os.path.join(
#    DATA_DIR, "Building_and_Safety_Permit_Information.csv")

#BUILDING_PERMIT_CSV_FILE = os.path.join(
#    DATA_DIR, "Building_and_Safety_Permit_Information.csv")

DEMOBUILD_PERMIT_XLS_FILE = os.path.join(
    DATA_DIR, "Building Permit Records 2007-2017.xls")


#OCCUPANCY_INSPECTION_CSV_FILE = os.path.join(
#    DATA_DIR, "Building_and_Safety_Certificate_of_Occupancy.csv")


TEMP_OUTPUT_FILE = os.path.join(SAVE_DIR, "la_housing_dataset_no_geo.csv")
FINAL_OUTPUT_FILE = os.path.join(SAVE_DIR, "la_housing_dataset_no_geo_property_id.csv")


# This is the list of colums used in the final file
COLUMN_ORDER = [
    "General Category",
    "APN",
    "Latitude/Longitude",

    "Address Full",
    "Address Number",
    "Street Direction",
    "Street Name",
    "Street Suffix",
    "Unit Count",
    "Unit Number",
    "Unit Type",
    "City",
    "State",
    "Zip Code",
    "Council District",

    "Status",
    "Status Date",
    "Original Issue Date",
    "Completion Date",
    "Permit #",
    "Permit Type",
    "Permit Sub-Type",
    "Work Description",
]


def strip_punct(in_string):
    """Delete punctuation from string"""
    return str(in_string).translate(None, string.punctuation)

PUNCT_TO_SPACE = string.maketrans(string.punctuation, " "*len(string.punctuation))
def punct_to_space(in_string):
    """Replace punctuation from string"""
    return str(in_string).translate(PUNCT_TO_SPACE)

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

    # Fix datetime dtype
    df_ellis = df_ellis.rename(
        columns={
            'Date Filed': 'Status Date',
            'Address': 'Address Full',
            'Zip': 'Zip Code'
        }
    )

    # parsing address
    df_addy = df_ellis['Address Full'].apply(parse_addy)
    df_ellis = pd.concat([df_ellis, df_addy], axis=1)

    df_ellis['Status Date'] = pd.to_datetime(df_ellis['Status Date'])

    # Add in 'General Category'
    df_ellis["General Category"] = 'Ellis Withdrawal'

    df_ellis = df_ellis[[c for c in COLUMN_ORDER if c in df_ellis]]

    return df_ellis


def process_entitlements_file(xls_filename):
    """Read in and apply formatting to the entitlenments file"""

    df_ent = pd.read_excel(xls_filename, sheetname='Export Worksheet')

    df_ent.ADDRESS = df_ent.ADDRESS.apply(strip_punct)
    df_ent.ADDRESS = df_ent.ADDRESS.apply(lambda x: clean_spaces(x.split(' CA ')[0]))

    # Fix datetime dtype
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

    # Fix datetime dtype
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

    df_rso = pd.read_excel(ALL_RSO_XLS_FILE, sheetname=None)
    df_rso = pd.concat(df_rso, axis=0).reset_index(drop=True)

    df_rso.APN = df_rso.APN.astype(str)
    df_rso.Property_Zip_Code = df_rso.Property_Zip_Code.astype(str)

    df_rso.Property_Street_Address = df_rso.Property_Street_Address.apply(strip_punct)

    # Fix datetime dtype
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
    _blocking_indices = [
        #recordlinkage.BlockIndex(on="APN"),
        recordlinkage.BlockIndex(on="Address Full"),
        recordlinkage.BlockIndex(on=["Street Name", "Zip Code"]),
    ]

    print("Finding blocked pairs")
    pairs = None
    for bi in _blocking_indices:
        if pairs is not None:
            pairs = pairs.union(bi.index(df_full))
        else:
            pairs = bi.index(df_full)

    print("Setting up similarity calculations")
    compare_cl = recordlinkage.Compare()

    compare_cl.numeric('Address Number (float)', 'Address Number (float)',
                       offset=3, scale=2,
                       label='address_number'
    )
    compare_cl.string('Street Name', 'Street Name',
                      method='levenshtein',
                      threshold=0.9,
                      label='street name')

    compare_cl.exact('Address Full', 'Address Full', label='addy_full')
    compare_cl.exact('Zip Code', 'Zip Code', label='zip')

    print("Calculating similarities")
    features = compare_cl.compute(pairs, df_full)
    features.to_pickle(os.path.join(SAVE_DIR, "features.pickle"))
    
    return features


def form_clusters(df_full, features):
    """Given the full dataframe and a matrix of feature linkages,
    append a column to the dataset that contains the property_id
    that connects all properties toegether
    """

    matches = features[features.sum(axis=1) >= LINKAGE_THRESHOLD]

    graph = nx.Graph()
    for i in xrange(df_full.shape[0]):
        graph.add_node(i)
    print('Num nodes: {}'.format(graph.number_of_nodes()))

    for pair, match_feats in matches.iterrows():
        graph.add_edge(*pair)    
    print('Num edges: {}'.format(graph.number_of_edges()))

    graph = graph.to_undirected()

    # Finding connected clusters in the graph
    comp_num = 0
    row_to_group = {}
    for comp in nx.connected_components(graph):
        for r in list(comp):
            row_to_group[r] = comp_num
        comp_num += 1

    df_full['Property ID'] = pd.Series(df_full.index).apply(lambda x: row_to_group[x])
    df_full.to_csv(FINAL_OUTPUT_FILE, index=False, encoding='utf-8')
    
    return df_full

if __name__ == '__main__':

    #df_full = concat_datasets_and_save()
    df_full = pd.read_csv(TEMP_OUTPUT_FILE)

    df_full['Address Full'] = df_full['Address Full'].astype(unicode)
    df_full['Street Name'] = df_full['Street Name'].astype(unicode)
    df_full['Address Number (float)'] = df_full['Address Number'].apply(string_to_numeric)

    features = compute_record_linkage(df_full)

    # Look at the distribution of feature-scores
    #features.sum(axis=1).value_counts(bins=50).sort_index(ascending=False)

    LINKAGE_THRESHOLD = 3.5
    df_full = form_clusters(df_full, features)
