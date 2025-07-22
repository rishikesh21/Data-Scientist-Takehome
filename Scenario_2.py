#!/usr/bin/env python
# coding: utf-8

# In[5]:


get_ipython().run_line_magic('pip', 'install pyrosm geopandas folium')


# In[6]:


import warnings
warnings.filterwarnings(action="ignore")
import requests
import time
import geopandas as gpd
import tempfile
import os
from io import StringIO


# In[ ]:


BIKE_DATASET_ID = "d_9326f791b521187f503149712fc400ef"
ROAD_DATASET_ID = "d_95a29fbb10cf94a3c263d33861d7b6c6"
ACRA_DATASET_ID='d_3f960c10fed6145404ca7b821f263b87'


# In[7]:


def load_geojson_as_gdf(DATASET_ID, max_tries=3):
    INITIATE_URL = f"https://api-open.data.gov.sg/v1/public/api/datasets/{DATASET_ID}/initiate-download"
    POLL_URL = f"https://api-open.data.gov.sg/v1/public/api/datasets/{DATASET_ID}/poll-download"

    init_resp = requests.get(INITIATE_URL)
    init_resp.raise_for_status()

    for _ in range(max_tries):
        time.sleep(2)
        poll_resp = requests.get(POLL_URL)
        poll_resp.raise_for_status()
        download_url = poll_resp.json().get("data", {}).get("url")
        if download_url:
            break
    else:
        raise TimeoutError("Timed out waiting for dataset download URL.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".geojson") as tmpfile:
        resp = requests.get(download_url)
        resp.raise_for_status()
        tmpfile.write(resp.content)
        tmpfile.flush()
        tmpfile.close()
        gdf = gpd.read_file(tmpfile.name)
    os.remove(tmpfile.name)
    return gdf


# In[ ]:


bike_gdf = load_geojson_as_gdf(BIKE_DATASET_ID)
road_gdf=load_geojson_as_gdf(ROAD_DATASET_ID)


# In[8]:


road_gdf.head()


# In[11]:


import requests
import pandas as pd
import zipfile
import io

headers = {
    "AccountKey": "rkRkeQReQ/CPdWz43lZtRA==",
    "accept": "application/json"
}

params = {"Date": "202506"}
url = "https://datamall2.mytransport.sg/ltaodataservice/PV/ODTrain"
resp = requests.get(url, headers=headers, params=params)
resp.raise_for_status()
download_link = resp.json()["value"][0]["Link"]

zip_resp = requests.get(download_link)
zip_file = zipfile.ZipFile(io.BytesIO(zip_resp.content))

csv_name = [name for name in zip_file.namelist() if name.endswith(".csv")][0]
with zip_file.open(csv_name) as csvfile:
    od_train_df = pd.read_csv(csvfile)


# In[12]:


od_train_df.head()


# In[13]:


import requests
import pandas as pd
import zipfile
import io

headers = {
    "AccountKey": "rkRkeQReQ/CPdWz43lZtRA==",
    "accept": "application/json"
}

params = {"Date": "202506"}
url = "https://datamall2.mytransport.sg/ltaodataservice/PV/ODBus"
resp = requests.get(url, headers=headers, params=params)
resp.raise_for_status()
download_link = resp.json()["value"][0]["Link"]

zip_resp = requests.get(download_link)
zip_file = zipfile.ZipFile(io.BytesIO(zip_resp.content))

csv_name = [name for name in zip_file.namelist() if name.endswith(".csv")][0]
with zip_file.open(csv_name) as csvfile:
    od_bus_df = pd.read_csv(csvfile)

print(od_bus_df.head())


# In[14]:


od_bus_df


# In[15]:


def load_dataset_as_dataframe(DATASET_ID,max_tries=3):
    INITIATE_URL = f"https://api-open.data.gov.sg/v1/public/api/datasets/{DATASET_ID}/initiate-download"
    POLL_URL = f"https://api-open.data.gov.sg/v1/public/api/datasets/{DATASET_ID}/poll-download"
    init_resp = requests.get(INITIATE_URL)
    init_resp.raise_for_status()

    for _ in range(max_tries):
        time.sleep(2)
        poll_resp = requests.get(POLL_URL)
        poll_resp.raise_for_status()
        download_url = poll_resp.json().get("data", {}).get("url")
        if download_url:
            break
    else:
        raise TimeoutError("Timed out waiting for dataset download URL.")

    csv_resp = requests.get(download_url)
    csv_resp.raise_for_status()
    df = pd.read_csv(StringIO(csv_resp.text))
    return df


# In[36]:


acra_df = load_dataset_as_dataframe(ACRA_DATASET_ID)


# In[37]:


len(acra_df['reg_postal_code'].unique())


# In[20]:


import gzip
import json
import pandas as pd
import requests

url = "https://github.com/isen-ng/singapore-postal-codes-1/raw/master/singpostcode.json.gz"
response = requests.get(url)
with open("singpostcode.json.gz", "wb") as f:
    f.write(response.content)

with gzip.open("singpostcode.json.gz", "rt", encoding="utf-8") as f:
    data = json.load(f)

postal_df = pd.DataFrame(data)
print(postal_df.head())


# In[39]:


acra_df=acra_df[['entity_name','reg_postal_code','entity_name']]


# In[38]:


acra_df


# In[40]:


acra_df['reg_postal_code'] = acra_df['reg_postal_code'].astype(str)
postal_df['POSTAL'] = postal_df['POSTAL'].astype(str)
merged = pd.merge(
    acra_df,
    postal_df,
    left_on='reg_postal_code',
    right_on='POSTAL',
    how='left'
)


# In[41]:


acra_df=merged[['entity_name','ADDRESS','LATITUDE','LONGITUDE']]


# In[25]:


mrt=pd.read_csv('https://raw.githubusercontent.com/elliotwutingfeng/singapore_train_station_coordinates/refs/heads/main/all_stations.csv')


# In[26]:


mrt


# In[27]:


mrt_df = mrt[['station_code','station_name','lat','lon']]
print(mrt_df.head())


# In[28]:


bus_stops = []
skip = 0

while True:
    url = f"https://datamall2.mytransport.sg/ltaodataservice/BusStops?$skip={skip}"
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    batch = resp.json()['value']
    if not batch:
        break
    bus_stops.extend(batch)
    skip += len(batch)

bus_stops_df = pd.DataFrame(bus_stops)
print(bus_stops_df.head())


# In[51]:


acra_df_lat_lon=acra_df[['LATITUDE','LONGITUDE','entity_name']]


# In[51]:





# In[52]:


acra_df_lat_lon['LATITUDE'] = pd.to_numeric(acra_df_lat_lon['LATITUDE'], errors='coerce')
acra_df_lat_lon['LONGITUDE'] = pd.to_numeric(acra_df_lat_lon['LONGITUDE'], errors='coerce')


# In[53]:


import numpy as np
from sklearn.neighbors import BallTree
acra_latlon = np.deg2rad(acra_df_lat_lon[['LATITUDE', 'LONGITUDE']].values)
bus_latlon = np.deg2rad(bus_stops_df[['Latitude', 'Longitude']].values)
mrt_latlon = np.deg2rad(mrt_df[['lat', 'lon']].values)


# In[62]:


acra_latlon = np.deg2rad(acra_df_lat_lon[['LATITUDE', 'LONGITUDE']].values)
bus_latlon = np.deg2rad(bus_stops_df[['Latitude', 'Longitude']].values)
mrt_latlon = np.deg2rad(mrt_df[['lat', 'lon']].values)

bus_tree = BallTree(bus_latlon, metric='haversine')
mrt_tree = BallTree(mrt_latlon, metric='haversine')

dist_bus, idx_bus = bus_tree.query(acra_latlon, k=5)
dist_mrt, idx_mrt = mrt_tree.query(acra_latlon, k=5)

dist_bus_m = dist_bus[:, 0] * 6371000
dist_mrt_m = dist_mrt[:, 0] * 6371000

acra_df_lat_lon['nearest_bus_stop'] = bus_stops_df.iloc[idx_bus[:, 0]].reset_index(drop=True)['BusStopCode']
acra_df_lat_lon['bus_distance_m'] = dist_bus_m
acra_df_lat_lon['nearest_mrt'] = mrt_df.iloc[idx_mrt[:, 0]].reset_index(drop=True)['station_code']
acra_df_lat_lon['mrt_distance_m'] = dist_mrt_m

print(acra_df_lat_lon.head())
all_bus_indices = np.unique(idx_bus.flatten())
all_mrt_indices = np.unique(idx_mrt.flatten())
unique_bus_stops = bus_stops_df.iloc[all_bus_indices].copy()
unique_mrt_stations = mrt_df.iloc[all_mrt_indices].copy()
unique_bus_stops = unique_bus_stops.rename(
    columns={
        'BusStopCode': 'stop_id',
        'Description': 'stop_name',
        'Latitude': 'lat',
        'Longitude': 'lon'
    }
)
unique_bus_stops['type'] = 'bus_stop'

unique_mrt_stations = unique_mrt_stations.rename(
    columns={
        'station_name': 'stop_name',
        'lat': 'lat',
        'lon': 'lon'
    }
)
unique_mrt_stations['stop_id'] = unique_mrt_stations.get('station_code', unique_mrt_stations.index)
unique_mrt_stations['type'] = 'mrt_station'

cols = ['stop_id', 'stop_name', 'lat', 'lon', 'type']

all_stops_df = pd.concat([
    unique_bus_stops[cols],
    unique_mrt_stations[cols]
], ignore_index=True)



# In[68]:


all_stops_df


# In[64]:


od_bus_df['ORIGIN_PT_CODE'] = od_bus_df['ORIGIN_PT_CODE'].astype(str)
od_bus_df['DESTINATION_PT_CODE'] = od_bus_df['DESTINATION_PT_CODE'].astype(str)
all_stops_df['stop_id'] = all_stops_df['stop_id'].astype(str)


# In[69]:


valid_mrt_stations = set(all_stops_df.query('type == "mrt_station"')['stop_id'])

filtered_od_train = od_train_df[
    od_train_df['ORIGIN_PT_CODE'].isin(valid_mrt_stations) &
    od_train_df['DESTINATION_PT_CODE'].isin(valid_mrt_stations)
].copy()


# In[70]:


valid_bus_stops = set(all_stops_df.query('type == "bus_stop"')['stop_id'])

filtered_od_bus = od_bus_df[
    od_bus_df['ORIGIN_PT_CODE'].isin(valid_bus_stops) &
    od_bus_df['DESTINATION_PT_CODE'].isin(valid_bus_stops)
].copy()


# In[102]:


# Filter for weekday AM peak (commute-to-work)
am_peak_hours = [7,8,9]
pm_peak_hours = [18,19,20]

filtered_od_train = od_train_df[
    (od_train_df['DAY_TYPE'] == 'WEEKDAY') &
    (od_train_df['TIME_PER_HOUR'].isin(am_peak_hours))
]

filtered_od_bus = od_bus_df[
    (od_bus_df['DAY_TYPE'] == 'WEEKDAY') &
    (od_bus_df['TIME_PER_HOUR'].isin(am_peak_hours))
]



# In[103]:


top_od_bus = (
    filtered_od_bus.groupby(['ORIGIN_PT_CODE', 'DESTINATION_PT_CODE'])['TOTAL_TRIPS']
    .sum()
    .reset_index()
    .sort_values('TOTAL_TRIPS', ascending=False)
)

top_od_train = (
    filtered_od_train.groupby(['ORIGIN_PT_CODE', 'DESTINATION_PT_CODE'])['TOTAL_TRIPS']
    .sum()
    .reset_index()
    .sort_values('TOTAL_TRIPS', ascending=False)
)


# In[105]:





# In[106]:


top_od=pd.concat([top_od_bus,top_od_train])


# In[107]:


top_od.head()


# In[76]:


threshold = top_od['TOTAL_TRIPS'].quantile(0.75)
top_od = top_od[top_od['TOTAL_TRIPS'] > threshold]


# In[108]:





# In[109]:


all_stops_df['stop_id'] = all_stops_df['stop_id'].astype(str)
top_od['ORIGIN_PT_CODE'] = top_od['ORIGIN_PT_CODE'].astype(str)
top_od['DESTINATION_PT_CODE'] = top_od['DESTINATION_PT_CODE'].astype(str)

top_od = top_od.merge(
    all_stops_df[['stop_id', 'lat', 'lon']],
    left_on='ORIGIN_PT_CODE',
    right_on='stop_id',
    how='left'
).rename(columns={'lat': 'origin_lat', 'lon': 'origin_lon'})

top_od = top_od.merge(
    all_stops_df[['stop_id', 'lat', 'lon']],
    left_on='DESTINATION_PT_CODE',
    right_on='stop_id',
    how='left'
).rename(columns={'lat': 'dest_lat', 'lon': 'dest_lon'})

top_od = top_od.drop(columns=['stop_id_x', 'stop_id_y'])


# In[112]:


top_od=top_od.dropna()


# In[113]:


top_od.head(10)


# In[114]:


print(top_od.columns)


# In[115]:


import geopandas as gpd
from shapely.geometry import LineString

top_od['od_line'] = top_od.apply(
    lambda row: LineString([(row['origin_lon'], row['origin_lat']), (row['dest_lon'], row['dest_lat'])]),
    axis=1
)

od_gdf = gpd.GeoDataFrame(top_od, geometry='od_line', crs='EPSG:4326')


# In[116]:


od_gdf = od_gdf.to_crs(3414)
bike_gdf = bike_gdf.to_crs(3414)
road_gdf = road_gdf.to_crs(3414)


# In[117]:


def check_od_coverage_with_index(od_gdf, bike_gdf):
    bike_gdf = bike_gdf.explode(index_parts=False, ignore_index=True)
    bike_gdf = bike_gdf[~bike_gdf.geometry.is_empty]
    bike_gdf = bike_gdf[bike_gdf.geometry.type == 'LineString']
    bike_sindex = bike_gdf.sindex

    def is_covered(line):
        possible = list(bike_sindex.intersection(line.bounds))
        if not possible:
            return False
        return bike_gdf.iloc[possible].geometry.intersects(line).any()

    return od_gdf.geometry.apply(is_covered)


od_gdf['bike_covered'] = check_od_coverage_with_index(od_gdf, bike_gdf)


# In[117]:





# In[118]:


summary = (
    od_gdf['bike_covered']
    .value_counts(normalize=True)
    .rename_axis('bike_covered')
    .reset_index(name='percentage')
)
summary['percentage'] = summary['percentage'] * 100


# In[119]:


summary


# In[120]:


uncovered_od = od_gdf[~od_gdf['bike_covered']].copy()


# In[121]:


uncovered_od['buffer'] = uncovered_od.geometry.buffer(40)


# In[122]:


road_gdf = road_gdf.explode(index_parts=False, ignore_index=True)
road_gdf = road_gdf[~road_gdf.geometry.is_empty]
road_gdf = road_gdf[road_gdf.geometry.type == 'LineString']


# In[123]:


buffer_gdf = gpd.GeoDataFrame(uncovered_od[['buffer']], geometry='buffer', crs=road_gdf.crs)
road_to_buffer = gpd.sjoin(
    road_gdf, buffer_gdf, how='inner', predicate='intersects'
).drop_duplicates(subset=road_gdf.columns.tolist())


# In[124]:


bike_gdf = bike_gdf.explode(index_parts=False, ignore_index=True)
bike_gdf = bike_gdf[~bike_gdf.geometry.is_empty]
bike_gdf = bike_gdf[bike_gdf.geometry.type == 'LineString']

road_to_buffer['has_bike'] = road_to_buffer.geometry.apply(lambda geom: bike_gdf.geometry.intersects(geom).any())

proposed_roads = road_to_buffer[~road_to_buffer['has_bike']].copy()


# In[125]:


proposed_roads['length_m'] = proposed_roads.geometry.length
proposed_roads = proposed_roads.sort_values('length_m', ascending=False)
proposed_roads['cum_length_km'] = proposed_roads['length_m'].cumsum() / 1000

final_recommend = proposed_roads[proposed_roads['cum_length_km'] <= 100]


# In[126]:


import folium

final_recommend_wgs84 = final_recommend.to_crs(4326)

center_lat = final_recommend_wgs84.geometry.centroid.y.mean()
center_lon = final_recommend_wgs84.geometry.centroid.x.mean()

m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles="cartodbpositron")

for line in final_recommend_wgs84.geometry:
    if line.geom_type == 'LineString':
        coords = [(lat, lon) for lon, lat, *_ in line.coords]
        folium.PolyLine(coords, color='orange', weight=5, opacity=0.8, tooltip="Proposed Cycling Path").add_to(m)
    elif line.geom_type == 'MultiLineString':
        for seg in line:
            coords = [(lat, lon) for lon, lat, *_ in seg.coords]
            folium.PolyLine(coords, color='orange', weight=5, opacity=0.8, tooltip="Proposed Cycling Path").add_to(m)


m


# In[127]:


m.save('map.png')


# In[ ]:




