"""Read in a bunch of excel sheets and build a single giant dataset"""

import os
import string

import usaddress as ua
import pandas as pd
import census_geocode as cg

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


def clean_spaces(in_string):
    """Remove multi-spaces"""
    return ' '.join(str(in_string).split())


def parse_addy(str_addy):
    """Parse a string address into a series"""

    str_addy = str_addy.lower()
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
            'Address': 'Address Full'
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
            'ZIP': 'Zip Code'
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



def DEPRECATED_process_demolition_file(csv_filename):
    """Read in and apply formatting to the demolitions file"""

    df_demo = pd.read_csv(csv_filename, low_memory=False, dtype=str)

    demolition_idx = df_demo['Permit Type'].isin(['Bldg-Demolition', 'NonBldg-Demolition'])

    # Make APN
    df_demo["APN"] = make_apn(df_demo)

    # Add in 'General Category'
    df_demo["General Category"] = 'Demolition Permits'

    df_demo = df_demo.loc[demolition_idx, COLUMN_ORDER]

    # Fix datetime dtype
    df_demo['Status Date'] = pd.to_datetime(df_demo['Status Date'])
    df_demo = df_demo[[c for c in COLUMN_ORDER if c in df_demo]]

    return df_demo


def DEPRECATED_process_building_file(csv_filename):
    """Read in and apply formatting to the building permits file"""

    df_building = pd.read_csv(csv_filename, low_memory=False, dtype=str)

    building_idx = ~df_building['Permit Type'].isin(['Bldg-Demolition', 'NonBldg-Demolition'])

    # Make APN
    df_building["APN"] = make_apn(df_building)

    # Add in 'General Category'
    df_building["General Category"] = 'Building Permits'

    df_building = df_building.loc[building_idx, COLUMN_ORDER]

    # Fix datetime dtype
    df_building['Status Date'] = pd.to_datetime(df_building['Status Date'])
    df_building = df_building[[c for c in COLUMN_ORDER if c in df_building]]

    return df_building


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
            'Property_Zip_Code': 'Zip',
            'Property_Street_Address': 'Address Full',
            'Unit_Count': 'Unit Count',
            'Council_District': 'Council District'
        }
    )

    df_addy = df_rso['Address Full'].apply(parse_addy)
    df_rso = pd.concat([df_rso, df_addy], axis=1)
    df_rso = df_rso[[c for c in COLUMN_ORDER if c in df_rso]]

    return df_rso


def process_inspections_file(csv_filename):
    """Read in and apply formatting to the building inspections file"""

    df_inspect = pd.read_csv(csv_filename, low_memory=False, dtype=str)
    df_inspect = df_inspect.rename(
        columns={
            'Status Date': 'Status Date',
            'Permit Issue Date': 'Original Issue Date',
            'ADDRESS': 'Address Full',
            'PROJ_DESC': 'Work Description',
            'PROCESSINGUNIT': 'Permit Type',
            'CASE_NBR': 'Permit #',
            "Address Start": "Address Number",
            "Unit Range Start": "Unit Number",
        }
    )

    # Fix datetime dtype
    df_inspect['Original Issue Date'] = pd.to_datetime(df_inspect['Original Issue Date'])
    df_inspect['Status Date'] = pd.to_datetime(df_inspect['Status Date'])

    # Make APN
    df_inspect["APN"] = make_apn(df_inspect)

    df_inspect = df_inspect[[c for c in COLUMN_ORDER if c in df_inspect]]
    return df_inspect


def concat_datasets_and_save():
    """Concatenate and save all the dataset. ALL THE DATASETS!"""

    dataset_list = []

    print("Loading withdrawals file")
    dataset_list.append(process_withdrawal_file(WITHDRAWAL_XLS_FILE))

    print("Loading entitlement file")
    dataset_list.append(process_entitlements_file(ENTITLEMENTS_XLS_FILE))

    print("Loading demolitions + building file")
    dataset_list.append(process_building_and_demolition_file(DEMOBUILD_PERMIT_XLS_FILE))

    df_full = pd.concat(dataset_list, axis=0).fillna(value='').reset_index(drop=True)
    df_full.to_csv(TEMP_OUTPUT_FILE, index=False, encoding='utf-8')

    return df_full


if __name__ == '__main__':

    df_full = concat_datasets_and_save()

    print('Loading inventory of RSO units')
    df_rso = process_rso_inventory()

    # TODO: run geocoding on parsed addresses after we get APNs/ZIPs for everybody

    # Assign new geocoded coordinates to every row?
    #df_full_geocode = cg.geocode_full_dataframe(df_full, 'v1_full')
    #df_rso_geocode = cg.geocode_full_dataframe(df_rso, 'v1_rso')

    # TODO: merge rso inventory into full data file

    #df_full['Address for geocoding'] = df_new_geo['Address for geocoding']
    #df_full['Latitude/Longitude (2)'] = df_new_geo['Latitude/Longitude (2)']

    # Assign new property id to each row based on geocoding and street name/zip match

