import geopandas as gpd

from shapely.geometry import Point, LineString
from ipyleaflet import Map, Marker, GeoData, GeoJSON, Popup, DivIcon, Polyline
from ipywidgets import HTML

import json
import requests # TO DO support corsproxy?

import shapely


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

        def expand_hyphenated_stages(stages_list):
            """
            Expands stages with hyphens (e.g., 'SS5-8' to ['SS5', 'SS8']) while preserving
            non-hyphenated stages.

            Args:
                stages_list (list): List of stage identifiers

            Returns:
                list: Expanded list with hyphenated stages split into separate elements
            """
            result = []

            for stage in stages_list:
                if "-" in stage:
                    # Extract the prefix (SS) and the numbers
                    prefix = stage[: stage.find("0")] if "0" in stage else "SS"
                    start, end = map(int, stage[len(prefix) :].split("-"))
                    # Add start and end stages with proper prefix
                    result.extend([f"{prefix}{start}", f"{prefix}{end}"])
                else:
                    result.append(stage)

            return result

        _gdf = gpd.GeoDataFrame.from_features(gj["features"], crs=crs)
        _gdf = self.add_start_end_coords(_gdf)
        retcols = ["name", "stages", "start", "finish", "geometry"]
        retcols = [c for c in retcols if c in _gdf.columns]
        _gdf["stages"] = _gdf["stages"].apply(expand_hyphenated_stages)
        return _gdf[retcols]

    @staticmethod
    def get_gdf_from_lat_lon_df(df, lat="latitude", lon="longitude", crs="EPSG:4326"):
        # Create a GeoDataFrame
        _df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[lon], df[lat]))

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
    def simple_stage_map(stages_gdf, stages=None, poi_gdf=None, zoom=9, buffer_percentage=0.05):
        # TO DO - we need to handle/ignore duplicate stage routes
        if stages is not None:
            stages = {stages} if isinstance(stages, str) else set(stages)
            stages_gdf = stages_gdf[
                stages_gdf["stages"].apply(lambda x: bool(set(x) & stages))
            ]
        # The centroid calc prefers a different projection
        # crs = stages_gdf.crs
        # centroid = stages_gdf.to_crs("+proj=cea").centroid.to_crs(crs).iloc[0]
        # Get the total bounding box of all geometries
        minx, miny, maxx, maxy = (
            stages_gdf.total_bounds
        )  # total_bounds returns (minx, miny, maxx, maxy)
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
            # desription (sic)
            poi_gdf = poi_gdf[poi_gdf["desription"].isin(stages)]
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
    def cut_line_by_distance_meters(line, start_meters, end_meters=None):
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

        if end_meters is None:
            end_meters = line_utm.length

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

    # Via Claude
    def route_elevations(self, route_input, mode="elevations"):
        """
        Retrieve and process elevation data with flexible output modes
        
        Usage:

            routes_gdf.loc[index, 'geometry'] = route_elevations(routes_gdf.iloc[index], mode='augmented')

            routes_gdf['geometry'] = routes_gdf.apply(
                lambda row: route_elevations(row, mode='augmented'), 
                axis=1
            )

        Parameters:
        route_input: Can be GeoPandas row, Shapely Geometry, GeoJSON dict/string
        mode: Output mode - 'elevations', 'coords', 'augmented'
        
        Returns:
        Depends on mode:
        - 'elevations': List of elevation values
        - 'coords': List of [lon, lat, elevation] tuples
        - 'augmented': Augmented input (geometry/GeoJSON with elevation)
        """
        def extract_coordinates(input_route):
            # Robust coordinate extraction
            def get_base_coords(geometry):
                # Handle different geometry types and coordinate dimensions
                if geometry.geom_type == 'LineString':
                    coords = list(geometry.coords)
                elif geometry.geom_type == 'MultiLineString':
                    # Take first linestring if multiple exist
                    coords = list(geometry[0].coords)
                else:
                    raise ValueError(f"Unsupported geometry type: {geometry.geom_type}")

                # Truncate to 2D coordinates (lon, lat)
                return [coord[:2] for coord in coords]

            # GeoPandas row
            if hasattr(input_route, 'geometry'):
                geometry = input_route.geometry
                coords = get_base_coords(geometry)
            # Shapely geometry
            elif hasattr(input_route, 'geom_type'):
                geometry = input_route
                coords = get_base_coords(geometry)
            # GeoJSON dictionary
            elif isinstance(input_route, dict):
                # Check for FeatureCollection or Feature
                if input_route.get('type') == 'FeatureCollection':
                    geometry = shapely.from_geojson(json.dumps(input_route['features'][0]))
                    coords = get_base_coords(geometry)
                elif input_route.get('type') == 'Feature':
                    geometry = shapely.from_geojson(json.dumps(input_route))
                    coords = get_base_coords(geometry)
                elif input_route.get('type') == 'LineString':
                    # Extract coordinates, truncating to 2D
                    coords = [coord[:2] for coord in input_route['coordinates']]
                else:
                    raise ValueError(f"Unsupported GeoJSON type: {input_route.get('type')}")
            # GeoJSON string
            elif isinstance(input_route, str):
                try:
                    geojson = json.loads(input_route)
                    return extract_coordinates(geojson)
                except json.JSONDecodeError:
                    raise ValueError("Invalid GeoJSON string")
            else:
                raise ValueError(f"Unsupported input type: {type(input_route)}")

            return coords, geometry

        # Validate mode
        valid_modes = ['elevations', 'coords', 'augmented']
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode. Choose from {valid_modes}")

        # Extract coordinates and original geometry
        try:
            coordinates, original_geom = extract_coordinates(route_input)
        except Exception as e:
            print(f"Coordinate extraction error: {e}")
            return None

        # Prepare locations for elevation lookup
        locations = [{"latitude": coord[1], "longitude": coord[0]} for coord in coordinates]

        try:
            # Use Open-Elevation API (free, no key required)
            response = requests.post(
                "https://api.open-elevation.com/api/v1/lookup", 
                json={"locations": locations}
            )
            response.raise_for_status()
            elevation_data = response.json()

            # Extract elevations
            elevations = [loc['elevation'] for loc in elevation_data['results']]

            # Create augmented coordinates with elevation
            augmented_coords = [
                list(coord) + [elev] 
                for coord, elev in zip(coordinates, elevations)
            ]

            # Return based on mode
            if mode == 'elevations':
                return elevations

            if mode == 'coords':
                return augmented_coords

            # Augmented mode
            if hasattr(route_input, 'geometry') or hasattr(route_input, 'geom_type'):
                # For GeoPandas row or Shapely geometry, return augmented geometry
                return LineString(augmented_coords)

            # For GeoJSON, reconstruct with elevation information
            if isinstance(route_input, dict):
                # Preserve original GeoJSON structure
                if route_input.get('type') == 'FeatureCollection':
                    route_input['features'][0]['geometry']['coordinates'] = augmented_coords
                elif route_input.get('type') == 'Feature':
                    route_input['geometry']['coordinates'] = augmented_coords
                elif route_input.get('type') == 'LineString':
                    route_input['coordinates'] = augmented_coords

                return route_input

            # For GeoJSON string or other inputs
            return {
                "type": "LineString",
                "coordinates": augmented_coords
            }

        except Exception as e:
            print(f"Elevation retrieval error: {e}")
            return None

    # Diagnostic function to inspect coordinate dimensions
    def inspect_coordinates(input_route):
        """
        Inspect the coordinates of an input route
        
        Parameters:
        input_route: Input geometry or GeoJSON
        
        Returns:
        Dictionary with coordinate information
        """
        def get_coord_info(coords):
            return {
                'total_coords': len(coords),
                'first_coord_dimensions': len(coords[0]) if coords else None,
                'coord_dimensions': [len(coord) for coord in coords[:5]]
            }

        if hasattr(input_route, 'geometry'):
            coords = list(input_route.geometry.coords)
        elif hasattr(input_route, 'geom_type'):
            coords = list(input_route.coords)
        elif isinstance(input_route, dict):
            if input_route.get('type') == 'FeatureCollection':
                coords = input_route['features'][0]['geometry']['coordinates']
            elif input_route.get('type') == 'Feature':
                coords = input_route['geometry']['coordinates']
            elif input_route.get('type') == 'LineString':
                coords = input_route['coordinates']
        elif isinstance(input_route, str):
            coords = json.loads(input_route)['coordinates']
        else:
            return "Unable to extract coordinates"

        return get_coord_info(coords)

    def swap_linestring_coords(self, linestring):
        """Swap latitude and longitude in a Shapely LineString, preserving altitude if present."""
        swapped_coords = []

        for coord in linestring.coords:
            if len(coord) == 3:  # (lon, lat, alt) format
                lon, lat, alt = coord
                swapped_coords.append((lat, lon, alt))
            else:  # (lon, lat) format
                lon, lat = coord
                swapped_coords.append((lat, lon))

        return LineString(swapped_coords)

    def route_segment_meters(self, line, start, end=None):
        first_segment = self.cut_line_by_distance_meters(line, 0, start)
        highlighted_segment = self.cut_line_by_distance_meters(line, start, end)
        if end is None:
            segments = [first_segment, highlighted_segment]
        else:
            end_segment = self.cut_line_by_distance_meters(line, end)
            segments = [first_segment, highlighted_segment, end_segment]

        gdf_segments = gpd.GeoDataFrame(list(range(len(segments))), geometry=segments)
        gdf_segments.columns = ["index", "geometry"]

        return gdf_segments

    def leaflet_highlight_section(self, gdf_segments, colour=None, swap_coords=False):
        # TO DO - generalise to n sections with interleave
        base_colour = "blue"
        highlight_colour = "red"
        if colour is None:
            colour = ["blue", "red"]
        if isinstance(colour, str):
            if colour == base_colour:
                base_colour = highlight_colour
                highlight_colour = colour
        else:
            base_colour = colour[0]
            highlight_colour = colour[1]

        # Need to better generalise map center based on bounds TO DO
        route_coords = (
            list(self.swap_linestring_coords(gdf_segments.geometry.iloc[1]).coords)
            if swap_coords
            else list(gdf_segments.geometry.iloc[1].coords)
        )
        m = Map(center=route_coords[0], zoom=13)

        highlighted_route = Polyline(
            locations=route_coords,
            color=highlight_colour,
            fill=False,
        )
        m.add_layer(highlighted_route)

        start_coords = (
            list(self.swap_linestring_coords(gdf_segments.geometry.iloc[0]).coords)
            if swap_coords
            else list(gdf_segments.geometry.iloc[0].coords)
        )
        if len(gdf_segments) == 3:
            end_coords = (
                list(
                    self.swap_linestring_coords(
                        gdf_segments.geometry.iloc[2]
                    ).coords
                )
                if swap_coords
                else list(gdf_segments.geometry.iloc[2].coords)
            )
            segment = Polyline(
                locations=[start_coords, end_coords],
                color=base_colour,
                fill=False,
                weight=5,
            )
        else:
            segment = Polyline(
                locations=start_coords, color=base_colour, fill=False, weight=5
            )
        m.add_layer(segment)
        return m

    def leaflet_highlight_route(self, line, start, end=None, swap_coords=False):
        gdf_segments = self.route_segment_meters(line, start, end)
        m = self.leaflet_highlight_section(gdf_segments, colour=None, swap_coords=swap_coords)
        return m

    # Via claude.ai
    def calculate_route_distance(self, gdf, row_index, lat, lon):
        """
        Calculate distance along a route using automatically estimated UTM CRS

        Parameters:
        -----------
        gdf : GeoDataFrame
            Input GeoDataFrame with latitude/longitude route
        row_index : int
            Index of the specific route
        lat : float
            Latitude of the point
        lon : float
            Longitude of the point

        Returns:
        --------
        dict
            Contains distance in meters and other relevant information
        """
        # Ensure CRS is set (if not already)
        if gdf.crs is None or gdf.crs.name == "undefined":
            gdf = gdf.set_crs("EPSG:4326")

        # Create point in the same CRS as the dataframe
        point = gpd.GeoDataFrame(geometry=[Point(lon, lat)], crs="EPSG:4326")

        # Automatically estimate the best UTM CRS for the point
        utm_crs = point.estimate_utm_crs()
        #print(f"Estimated UTM CRS: {utm_crs}")

        # Project both route and point to the estimated UTM CRS
        projected_gdf = gdf.to_crs(utm_crs)
        projected_point = point.to_crs(utm_crs).geometry.iloc[0]

        # Get the route geometry
        route = projected_gdf.loc[row_index, "geometry"]

        # Find the nearest point on the route
        nearest_point = route.interpolate(route.project(projected_point))

        # Calculate distances
        try:
            # Distance along the route to the nearest point
            distance_along_route = route.project(projected_point)
            total_route_length = route.length

            return {
                "estimated_utm_crs": utm_crs,
                "distance_along_route_meters": distance_along_route,
                "total_route_length_meters": total_route_length,
                "percent_along_route": (
                    (distance_along_route / total_route_length) * 100
                    if total_route_length > 0
                    else 0
                ),
                "nearest_point_on_route": nearest_point,
            }
        except Exception as e:
            return {"error": str(e), "latitude": lat, "longitude": lon}
