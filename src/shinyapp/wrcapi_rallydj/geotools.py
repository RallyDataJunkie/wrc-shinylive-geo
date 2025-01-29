import geopandas as gpd

from shapely.geometry import Point, LineString
from ipyleaflet import Map, Marker, GeoData, GeoJSON, Popup, DivIcon
from ipywidgets import HTML

class RallyGeoTools:
    def __init__(self):
        pass

    # https://gis.stackexchange.com/a/90554
    def explode(self, coords):
        """Explode a GeoJSON geometry's coordinates object and yield coordinate tuples.
        As long as the input is conforming, the type of the geometry doesn't matter."""
        for e in coords:
            if isinstance(e, (float, int)):
                yield coords
                break
            else:
                for f in self.explode(e):
                    yield f

    def bbox(self, f):
        """Find bounding box around a geojson feature route."""
        x, y, z = zip(*list(self.explode(f["geometry"]["coordinates"])))
        return [[min(y), min(x)], [max(y), max(x)]]

    def geojson_to_gpd(self, gj, crs="EPSG:4326"):
        _gdf = gpd.GeoDataFrame.from_features(gj["features"], crs=crs)
        _gdf = self.add_start_end_coords(_gdf)
        retcols = ["name", "stages", "start", "finish", "geometry"]
        retcols = [c for c in retcols if c in _gdf.columns]
        return _gdf[retcols]

    @staticmethod
    def get_gdf_from_lat_lon_df(df, lat="latitude", lon="longitude", crs="EPSG:4326"):
        # Create a GeoDataFrame
        _df = gpd.GeoDataFrame(
            df, geometry=gpd.points_from_xy(df[lon], df[lat])
        )

        # Set the coordinate reference system (CRS) to WGS84 (EPSG:4326)
        _df.set_crs(crs, inplace=True)
        return df

    # ## Stage Metadata From JSON
    #
    # What can we derive from JSON data?
    def add_start_end_coords(self, gdf):
        """Add start and end cords to geopandas df geomety Linestring."""

        gdf["start"] = gdf["geometry"].apply(
            lambda x: x.coords[0] if isinstance(x, LineString) else None
        )
        gdf["finish"] = gdf["geometry"].apply(
            lambda x: x.coords[-1] if isinstance(x, LineString) else None
        )
        return gdf

    @staticmethod
    def simple_stage_map(stages_gdf, stages=None, poi_gdf=None, zoom=9):
        # TO DO - we need to handle/ignore dupliacte stage routes
        if stages is not None:
            stages = {stages} if isinstance(stages,str) else set(stages)
            stages_gdf = stages_gdf[
                stages_gdf["stages"].apply(lambda x: bool(set(x) & stages))
            ]
        # The centroid calc prefers a different projection
        #crs = stages_gdf.crs
        # centroid = stages_gdf.to_crs("+proj=cea").centroid.to_crs(crs).iloc[0]
        # Get the total bounding box of all geometries
        minx, miny, maxx, maxy = (
            stages_gdf.total_bounds
        )  # total_bounds returns (minx, miny, maxx, maxy)
        buffer_percentage = 0.05  # Adjust as needed
        x_buffer = (maxx - minx) * buffer_percentage
        y_buffer = (maxy - miny) * buffer_percentage

        # Expand the bounding box
        minx -= x_buffer
        maxx += x_buffer
        miny -= y_buffer
        maxy += y_buffer
        # Create map centered at the bounding box midpoint
        m = Map(center=[(miny + maxy) / 2, (minx + maxx) / 2], zoom=zoom)
        # Auto-fit to the bounding box
        m.fit_bounds([[miny, minx], [maxy, maxx]])

        geo_data = GeoData(
            geo_dataframe=stages_gdf,
            hover_style={"fillColor": "red", "fillOpacity": 0.2},
            name="name",
        )

        m.add(geo_data)
        if poi_gdf is not None:
            poi_gdf = poi_gdf[poi_gdf['desription'].isin(stages)]
            for idx, row in poi_gdf.dropna(subset="label").iterrows():
                icon_text = row["name"].split(" ")[-1]
                icon = DivIcon(
                    html=icon_text, bg_pos=[0, 0], icon_size=[8 * len(icon_text), 15]
                )
                marker = Marker(location=(row["latitude"], row["longitude"]), icon=icon)
                message = HTML()
                message.value = str(row["label"])
                marker.popup = Popup(
                    location=(row["latitude"], row["longitude"]), child=message
                )

                m.add_layer(marker)
        return m

    @staticmethod
    def linestringFomLatLonCols(df, latcol="lat", loncol="lon", groupby=None):
        """Generate a Linestring object from a dataframe containing
        lat and lon columns."""
        geometry = [Point(xy) for xy in zip(df[latcol], df[loncol])]
        dfg = gpd.GeoDataFrame(df, geometry=geometry)
        if groupby is not None:
            return dfg.groupby(groupby)["geometry"].apply(
                lambda x: LineString(x.tolist())
            )
        return LineString(dfg["geometry"].tolist())

    @staticmethod
    def cut_line_by_distance_meters(line, start_meters, end_meters):
        """
        Cut a LineString between two distances measured in meters.

        Parameters:
        -----------
        line : shapely.geometry.LineString
            Input LineString in WGS84 (EPSG:4326)
        start_meters : float
            Starting distance in meters from the beginning of the line
        end_meters : float
            Ending distance in meters from the beginning of the line

        Returns:
        --------
        shapely.geometry.LineString
            The portion of the line between start_meters and end_meters
        """
        # TO DO  - assuming this is the projection
        WGS84 = "EPSG:4326"
        # Convert the line to a local UTM projection for accurate measurements
        gdf = gpd.GeoDataFrame(geometry=[line], crs=WGS84)

        # Get the centroid to determine appropriate UTM zone
        centroid = line.centroid
        utm_crs = gpd.GeoDataFrame(
            geometry=[Point(centroid.x, centroid.y)], crs=WGS84
        ).estimate_utm_crs()

        # Project to UTM for accurate measurements
        line_utm = gdf.to_crs(utm_crs).geometry.iloc[0]

        # Validate distances
        if (
            start_meters < 0
            or end_meters > line_utm.length
            or start_meters >= end_meters
        ):
            raise ValueError("Invalid distances provided")

        # Get all coordinates as tuples
        coords = [(t[0], t[1]) for t in line_utm.coords]
        result_coords = []

        # Find start point
        if start_meters > 0:
            start_point = line_utm.interpolate(start_meters)
            start_found = False

            for i, p in enumerate(coords):
                pd = line_utm.project(Point(p))
                if pd >= start_meters:
                    result_coords = [(start_point.x, start_point.y)] + coords[i:]
                    start_found = True
                    break

            if not start_found:
                result_coords = [(start_point.x, start_point.y)]
        else:
            result_coords = coords[:]

        # Find end point
        if end_meters < line_utm.length:
            end_point = line_utm.interpolate(end_meters)
            temp_coords = []

            for i, p in enumerate(result_coords):
                pd = line_utm.project(Point(p))
                if pd <= end_meters:
                    temp_coords.append(p)
                else:
                    temp_coords.append((end_point.x, end_point.y))
                    break

            result_coords = temp_coords

        # Create new LineString with the selected coordinates
        if len(result_coords) < 2:
            raise ValueError("Not enough points to create a valid LineString")

        line_utm = LineString(result_coords)

        # Convert back to WGS84 and extract the geometry object
        result = (
            gpd.GeoDataFrame(geometry=[line_utm], crs=utm_crs)
            .to_crs(WGS84)
            .geometry.iloc[0]
        )

        return result
