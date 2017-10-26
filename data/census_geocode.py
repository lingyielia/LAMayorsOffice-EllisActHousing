"""Tools for geocoding via the US Census website"""

import os
from glob import iglob

import pandas as pd


DATA_DIR = os.path.expanduser('~/GitHub/la_mayors_office/data')
SAVE_DIR = os.path.expanduser('~/GitHub/la_mayors_office/data/processed')
TEMPSAVE_DIR = os.path.expanduser('~/GitHub/la_mayors_office/data/temp')

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

if not os.path.exists(TEMPSAVE_DIR):
    os.makedirs(TEMPSAVE_DIR)

BATCH_SIZE = 1000


def make_address_for_API(row_num, row):
    """Make an address for geocoding out of the normalized parts"""
    row_dict = {'Unique ID': row_num}

    row_dict['Street address'] = ' '.join([
        row["Address Number",],
        row["Street Direction"],
        row["Street Name"],
        row["Street Suffix"]
    ])

    row_dict['Street address'] = drop_fractions(row_dict['Street address'])

    if row['Zip Code']:
        row_dict['ZIP'] = row['Zip Code']
        row_dict['City'] = ''
    else:
        row_dict['ZIP'] = ''
        row_dict['City'] = 'Los Angeles'

    row_dict['State'] = 'CA'
    
    return row_dict


def flip_latlon(val):
    """Flip the latlong in a string of coords"""
    if val:
        return "({}, {})".format(*val.split(',')[::-1])
    return ''


def drop_fractions(in_string):
    """Drop fractions for an address"""
    return ' '.join([part for part in in_string.split() if not '/' in part])


def break_df_into_batches(df_in, file_prefix):
    """Break a dataframe into smaller files of size 'BATCH_SIZE',
    save them to disk, and return a list with the relevant filepaths
    """

    # delete old files with this batch prefix
    search_string = os.path.join(TEMPSAVE_DIR, '{}_input-batch*.csv'.format(file_prefix))
    for old_file in iglob(search_string):
        os.remove(old_file)

    # save the new batches
    file_list = []
    for batch in range(0, df_in.shape[0], BATCH_SIZE):
        new_file = os.path.join(TEMPSAVE_DIR, '{}_input-batch{}.csv'.format(file_prefix, batch))
        df_in.iloc[batch: (batch + BATCH_SIZE)].to_csv(new_file, index=False, encoding='utf-8')
        file_list.append(new_file)

    return file_list


CURL_CMD = """
    curl --form addressFile=@{file} \
         --form benchmark={benchmark} https://geocoding.geo.census.gov/geocoder/locations/addressbatch \
         --output {return_file}
"""

def run_api_batches(file_list):
    """Send a list of files to be geocoded"""
    print("Running batches of Geocoding")
    return_file_list = []
    for file in file_list:
        out_file = '_output-batch'.join(file.split('_input-batch'))
        print("   {}".format(out_file))
        
        _cmd = CURL_CMD.format(
            file=file,
            benchmark='Public_AR_Current',
            return_file=out_file
        )

        return_file_list.append(out_file)
        os.system(_cmd)

    return return_file_list


def geocode_full_dataframe(df_in, data_prefix):
    """Re-geocode a big dataframe"""

    # Set up dataframe to geocode
    df_to_geo = []
    for row_num, row in df_in.iterrows():
        df_to_geo.append(make_address_for_API(row_num, row))

    df_to_geo = pd.DataFrame(df_to_geo)[['Unique ID', 'Street address', 'City', 'State', 'ZIP']]
    df_to_geo = df_to_geo[df_to_geo['Street address'] != '']

    # Save batch files
    to_api_files = break_df_into_batches(df_to_geo, data_prefix)

    # Run batches via API
    from_api_files = run_api_batches(to_api_files)

    # Load results
    columns = [
        'Row', 'in_address', 'has_match', 'is_exact_match',
        'Address for geocoding', 'Latitude/Longitude',
        'tmp1', 'tmp2']

    df_from_geo = pd.concat([pd.read_csv(f, names=columns) for f in from_api_files], axis=0)
    df_from_geo['Latitude/Longitude'] = (
        df_from_geo['Latitude/Longitude'].
        fillna(value='').apply(flip_latlon)
        )

    return df_from_geo