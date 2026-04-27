import json

import pandas as pd
from libpysal.weights import Queen
import geopandas as gpd
from shapely.geometry import shape
from spreg import ML_Lag

lsoa_shapes =  pd.read_csv('geography_lookup.csv')

# Convert GeoJSON string to Shapely geometry
lsoa_shapes['geometry'] = lsoa_shapes['geo_shape'].apply(lambda x: shape(json.loads(x)))
lsoa_shapes = lsoa_shapes[['lsoa_code', 'geometry']]
gdf = gpd.GeoDataFrame(lsoa_shapes, geometry='geometry')
gdf.crs = "EPSG:4326"  # sets the cooordinat ereference system to lat/long


# Build adjacency
w = Queen.from_dataframe(gdf)
w.transform = "r"  # row-standardized
adj_matrix, ids = w.full() # check what ids is



model = ML_Lag(y, X, w=w, name_w='w')