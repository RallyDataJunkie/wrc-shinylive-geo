import geopandas as gpd
from pandas import DataFrame
from shapely.geometry import Point, LineString, MultiLineString
from ipyleaflet import Map, Marker, GeoData, GeoJSON, Popup, DivIcon, Polyline
from ipywidgets import HTML

import json
import osmnx as ox
import numpy as np

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

        def fix_encoding(text):
            if isinstance(text, str):
                try:
                    return text.encode('latin1').decode('utf-8')
                except UnicodeDecodeError:
                    return text  # Return as-is if decoding fails
            return text

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
                    # TO DO - need handlers for badly behaved names
                    prefix = stage[: stage.find("0")] if "0" in stage and "SS" not in stage else "SS"
                    start, end = map(int, stage[len(prefix) :].split("-"))
                    # Add start and end stages with proper prefix
                    result.extend([f"{prefix}{start}", f"{prefix}{end}"])
                else:
                    result.append(stage)

            return result

        _gdf = gpd.GeoDataFrame.from_features(gj["features"], crs=crs)
        _gdf = self.add_start_end_coords(_gdf)
        if "name" in _gdf.columns:
            _gdf["name"] = _gdf["name"].apply(fix_encoding)
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
                #marker.popup = Popup(
                #    location=(row["latitude"], row["longitude"]), child=message
                #)

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

    def route_N_segments_meters(self, line, points, toend=False):
        """
        Create N-1 segments from N points measured in meters along a line.

        Parameters:
        -----------
        line : shapely.geometry.LineString
            Input LineString in WGS84 (EPSG:4326)
        points : list
            List of distances in meters from the beginning of the line
        toend : bool, default=False
            If True, automatically add the line's length as the final point

        Returns:
        --------
        gpd.GeoDataFrame
            GeoDataFrame containing all segments with their index and geometry
        """
        # Ensure points are sorted
        points = sorted(points)

        # Get line length in meters (UTM projection)
        WGS84 = "EPSG:4326"
        gdf = gpd.GeoDataFrame(geometry=[line], crs=WGS84)
        utm_crs = gdf.estimate_utm_crs()
        line_utm = gdf.to_crs(utm_crs).geometry.iloc[0]
        line_length = line_utm.length

        # Add line's length as the final point if toend=True
        if toend and (not points or points[-1] < line_length):
            points.append(line_length)

        # Ensure 0 is the first point if not already included
        if not points or points[0] > 0:
            points.insert(0, 0)

        # Create segments between consecutive points
        segments = []
        for i in range(len(points) - 1):
            segment = self.cut_line_by_distance_meters(line, points[i], points[i + 1])
            segments.append(segment)

        # Create GeoDataFrame with segments
        gdf_segments = gpd.GeoDataFrame(list(range(len(segments))), geometry=segments)
        gdf_segments.columns = ["index", "geometry"]

        return gdf_segments

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
        mode: Output mode - 'elevations', 'coords', 'augmented', 'elevationdistance', 'elevationdistance_df'

        Returns:
        Depends on mode:
        - 'elevations': List of elevation values
        - 'coords': List of [lon, lat, elevation] tuples
        - 'augmented': Augmented input (geometry/GeoJSON with elevation)
        - 'elevationdistance': List of (distance_along_route_meters, elevation) tuples
        - 'elevationdistance_df': Dataframe with distance along route and elevation columns (in meters)
        """
        import geopandas as gpd
        from shapely.geometry import LineString, Point
        import json
        import requests
        import shapely

        def extract_coordinates(input_route):
            # Robust coordinate extraction
            def get_base_coords(geometry):
                # Handle different geometry types and coordinate dimensions
                if geometry.geom_type == "LineString":
                    coords = list(geometry.coords)
                elif geometry.geom_type == "MultiLineString":
                    # Take first linestring if multiple exist
                    coords = list(geometry[0].coords)
                else:
                    raise ValueError(f"Unsupported geometry type: {geometry.geom_type}")

                # Truncate to 2D coordinates (lon, lat)
                return [coord[:2] for coord in coords]

            # GeoPandas row
            if hasattr(input_route, "geometry"):
                geometry = input_route.geometry
                coords = get_base_coords(geometry)
            # Shapely geometry
            elif hasattr(input_route, "geom_type"):
                geometry = input_route
                coords = get_base_coords(geometry)
            # GeoJSON dictionary
            elif isinstance(input_route, dict):
                # Check for FeatureCollection or Feature
                if input_route.get("type") == "FeatureCollection":
                    geometry = shapely.from_geojson(json.dumps(input_route["features"][0]))
                    coords = get_base_coords(geometry)
                elif input_route.get("type") == "Feature":
                    geometry = shapely.from_geojson(json.dumps(input_route))
                    coords = get_base_coords(geometry)
                elif input_route.get("type") == "LineString":
                    # Extract coordinates, truncating to 2D
                    coords = [coord[:2] for coord in input_route["coordinates"]]
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
        valid_modes = ["elevations", "coords", "augmented", "elevationdistance"]
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
                json={"locations": locations},
            )
            response.raise_for_status()
            elevation_data = response.json()

            # Extract elevations
            elevations = [loc["elevation"] for loc in elevation_data["results"]]

            # Create augmented coordinates with elevation
            augmented_coords = [
                list(coord) + [elev] for coord, elev in zip(coordinates, elevations)
            ]

            # Return based on mode
            if mode == "elevations":
                return elevations

            if mode == "coords":
                return augmented_coords

            if mode.startswith("elevationdistance"):
                # For elevationdistance mode, we need to calculate distances in meters
                # Create a GeoDataFrame with the route in WGS84 (EPSG:4326)
                route_line = LineString(coordinates)
                route_gdf = gpd.GeoDataFrame(geometry=[route_line], crs="EPSG:4326")

                # Get the center point of the route to determine appropriate UTM zone
                center_point = route_line.centroid
                center_gdf = gpd.GeoDataFrame(geometry=[center_point], crs="EPSG:4326")

                # Estimate appropriate UTM CRS based on the center of the route
                utm_crs = center_gdf.estimate_utm_crs()

                # Project route to UTM for accurate measurements in meters
                projected_route_gdf = route_gdf.to_crs(utm_crs)
                projected_route = projected_route_gdf.geometry[0]

                # Project each point and calculate distance along the route in meters
                distances = []
                prev_point = None
                current_distance = 0.0

                for coord in coordinates:
                    # Create point in WGS84
                    point = Point(coord)
                    point_gdf = gpd.GeoDataFrame(geometry=[point], crs="EPSG:4326")

                    # Project point to same UTM CRS
                    projected_point = point_gdf.to_crs(utm_crs).geometry[0]

                    if prev_point is None:
                        # First point has distance 0
                        distances.append(0.0)
                    else:
                        # Add segment distance to running total
                        segment_distance = projected_point.distance(prev_point)
                        current_distance += segment_distance
                        distances.append(current_distance)

                    prev_point = projected_point

                # Create distance-elevation pairs dataframe
                distance_elevation_pairs = list(zip(distances, elevations))
                if mode == "elevationdistance_df":
                    distance_elevation_pairs = DataFrame(
                    distance_elevation_pairs,
                    columns=["distance", "elevation"],
                )
                return distance_elevation_pairs

            # Augmented mode
            if hasattr(route_input, "geometry") or hasattr(route_input, "geom_type"):
                # For GeoPandas row or Shapely geometry, return augmented geometry
                return LineString(augmented_coords)

            # For GeoJSON, reconstruct with elevation information
            if isinstance(route_input, dict):
                # Preserve original GeoJSON structure
                if route_input.get("type") == "FeatureCollection":
                    route_input["features"][0]["geometry"]["coordinates"] = augmented_coords
                elif route_input.get("type") == "Feature":
                    route_input["geometry"]["coordinates"] = augmented_coords
                elif route_input.get("type") == "LineString":
                    route_input["coordinates"] = augmented_coords

                return route_input

            # For GeoJSON string or other inputs
            return {"type": "LineString", "coordinates": augmented_coords}

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
        # print(f"Estimated UTM CRS: {utm_crs}")

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

    def enhance_route_resolution_osm(self,
        geodf, point_spacing_meters=10, route_simplify_tolerance=None, use_osm=True
    ):
        """
        Enhance route resolution by adding points at regular meter intervals while PRESERVING original route points.
        
        Args:
            geodf: GeoDataFrame containing LineString geometries representing routes
            point_spacing_meters: Distance between points in meters (default 10m)
            route_simplify_tolerance: Optional tolerance for simplifying the route after enhancement
            use_osm: Whether to use OSM data for route enhancement (if False, just does regular spacing)
            
        Returns:
            Enhanced GeoDataFrame with more detailed route geometries
        """

        def get_osm_roads_in_bbox(bounds, buffer_deg):
            """Get OSM road network within the bounds of a route"""
            minx, miny, maxx, maxy = bounds
            # Add buffer to ensure we capture nearby roads
            bbox = (
                miny - buffer_deg,  # south
                minx - buffer_deg,  # west
                maxy + buffer_deg,  # north
                maxx + buffer_deg,  # east
            )

            try:
                # Get all highway features
                roads = ox.features_from_bbox(bbox, {"highway": True})
                return roads
            except Exception as e:
                print(f"Error fetching OSM data: {e}")
                return None

        def enhance_line_with_regular_points(line, spacing_meters, roads=None, buffer_deg=0.0005):
            """
            Enhance a LineString with points at regular meter intervals while preserving original points.
            
            If roads is provided, also adds points from OSM road network.
            """
            # Ensure we're working with a proper LineString
            if not isinstance(line, LineString):
                print(f"Expected LineString, got {type(line)}")
                return line

            # Check if line is valid
            if not line.is_valid:
                print("Invalid LineString, attempting to fix...")
                try:
                    line = line.buffer(0).boundary
                    if not isinstance(line, LineString):
                        print("Could not fix LineString, returning original")
                        return line
                except Exception:
                    return line

            # Handle empty or near-empty linestrings
            if len(list(line.coords)) < 2:
                print("LineString has fewer than 2 points, cannot enhance")
                return line

            # Original route points - these are GROUND TRUTH and must be preserved exactly
            try:
                # Make sure we use only x,y coordinates
                original_points = [(float(p[0]), float(p[1])) for p in line.coords]
            except Exception as e:
                print(f"Error extracting coordinates: {e}")
                return line

            try:
                # Create a GeoDataFrame with the line for projection to UTM
                line_gdf = gpd.GeoDataFrame(geometry=[line], crs="EPSG:4326")

                # Get the centroid of the line to determine UTM zone
                centroid = line.centroid
                center_gdf = gpd.GeoDataFrame(geometry=[centroid], crs="EPSG:4326")

                # Determine the appropriate UTM CRS for accurate measurements
                utm_crs = center_gdf.estimate_utm_crs()

                # Project line to UTM for accurate measurements
                line_utm = line_gdf.to_crs(utm_crs).geometry.iloc[0]

                # Get total line length in meters
                total_length = line_utm.length
            except Exception as e:
                print(f"Error projecting to UTM: {e}")
                return line

            # Add OSM road points if available
            osm_points = []
            if use_osm and roads is not None and not roads.empty:
                try:
                    # Extract all valid geometries from the roads
                    road_lines = []
                    for geom in roads.geometry:
                        if geom.geom_type == "LineString":
                            road_lines.append(geom)
                        elif geom.geom_type == "MultiLineString":
                            road_lines.extend(list(geom.geoms))

                    # Create a buffer around our route line
                    route_buffer = line.buffer(buffer_deg)

                    # For each segment in the original route
                    for i in range(len(original_points) - 1):
                        # Create a LineString for this segment
                        segment = LineString([original_points[i], original_points[i + 1]])
                        segment_buffer = segment.buffer(buffer_deg)

                        # Find OSM roads that might provide additional detail for this segment
                        for road in road_lines:
                            if road.intersects(segment_buffer):
                                # Get points from this road segment
                                road_coords = list(road.coords)
                                for pt in road_coords:
                                    point = Point(pt)
                                    if segment_buffer.contains(point):
                                        # Calculate position along the segment (0-1)
                                        position = segment.project(point, normalized=True)
                                        if 0 < position < 1:  # Only add points between our original points
                                            # Convert to UTM to get actual distance
                                            point_gdf = gpd.GeoDataFrame(geometry=[point], crs="EPSG:4326")
                                            point_utm = point_gdf.to_crs(utm_crs).geometry.iloc[0]

                                            # Calculate actual distance in meters from start of segment
                                            segment_gdf = gpd.GeoDataFrame(geometry=[segment], crs="EPSG:4326")
                                            segment_utm = segment_gdf.to_crs(utm_crs).geometry.iloc[0]

                                            # Store the point with its distance along the route
                                            segment_start_dist = line_utm.project(Point(segment_utm.coords[0]))
                                            point_dist = segment_start_dist + segment_utm.project(point_utm)
                                            osm_points.append((point_dist, (float(pt[0]), float(pt[1]))))
                except Exception as e:
                    print(f"Error processing OSM roads: {e}")
                    # Continue without OSM points

            # Sort OSM points by distance
            osm_points.sort(key=lambda x: x[0])

            # Generate regular interval points
            regular_points = []
            try:
                # Calculate number of segments needed
                num_segments = max(1, int(total_length / spacing_meters))

                # Generate points at regular intervals
                for i in range(1, num_segments):
                    # Calculate distance along the line
                    distance = i * spacing_meters

                    if distance >= total_length:
                        break

                    # Interpolate point at this distance
                    point_utm = line_utm.interpolate(distance)

                    # Convert back to WGS84
                    point_gdf = gpd.GeoDataFrame(geometry=[point_utm], crs=utm_crs)
                    point_wgs84 = point_gdf.to_crs("EPSG:4326").geometry.iloc[0]

                    # Store point with distance
                    regular_points.append((distance, (float(point_wgs84.x), float(point_wgs84.y))))
            except Exception as e:
                print(f"Error generating regular points: {e}")
                # Continue with whatever points we have

            # Merge original, OSM, and regular points
            all_points_with_dist = []

            # Add original points with distances
            try:
                for pt in original_points:
                    # Project original point to UTM to get distance
                    pt_gdf = gpd.GeoDataFrame(geometry=[Point(pt)], crs="EPSG:4326")
                    pt_utm = pt_gdf.to_crs(utm_crs).geometry.iloc[0]
                    dist = line_utm.project(pt_utm)
                    all_points_with_dist.append((dist, pt))
            except Exception as e:
                print(f"Error processing original points: {e}")
                # Make sure original points are still included
                all_points_with_dist.extend([(0, pt) for pt in original_points])

            # Add OSM and regular points
            all_points_with_dist.extend(osm_points)
            all_points_with_dist.extend(regular_points)

            # Sort by distance
            all_points_with_dist.sort(key=lambda x: x[0])

            # Extract just the coordinates, remove duplicates while preserving order
            all_points = []
            prev_point = None

            for _, pt in all_points_with_dist:
                if prev_point is None or pt != prev_point:
                    all_points.append((float(pt[0]), float(pt[1])))
                    prev_point = pt

            # Ensure we have at least 2 points for a valid LineString
            if len(all_points) < 2:
                print("Not enough valid points to create LineString, returning original line")
                return line

            # Create enhanced LineString
            try:
                enhanced_line = LineString(all_points)
            except ValueError as e:
                print(f"Error creating LineString: {e}")
                # Return original line as fallback
                return line

            # Optionally simplify, but ENSURE all original points remain
            if route_simplify_tolerance is not None:
                try:
                    # Simplify first
                    simplified = enhanced_line.simplify(route_simplify_tolerance, preserve_topology=True)

                    # Get simplified coords
                    simplified_coords = list(simplified.coords)

                    # Ensure all original points are included
                    final_coords = list(simplified_coords)  # Start with simplified points

                    for orig_pt in original_points:
                        # Check if original point is already in simplified coords
                        if any(np.allclose([orig_pt[0], orig_pt[1]], [sc[0], sc[1]]) for sc in simplified_coords):
                            continue

                        # Find where to insert this original point
                        inserted = False
                        for i in range(len(simplified_coords) - 1):
                            segment = LineString([simplified_coords[i], simplified_coords[i + 1]])
                            if segment.distance(Point(orig_pt)) < buffer_deg:
                                # Insert at appropriate position
                                final_coords.insert(i + 1, orig_pt)
                                inserted = True
                                break

                        # If not inserted, add it at the end (shouldn't happen but just in case)
                        if not inserted:
                            final_coords.append(orig_pt)

                    # Create final LineString with all required points
                    enhanced_line = LineString(final_coords)
                except Exception as e:
                    print(f"Error during simplification: {e}")
                    # Keep unsimplified version

            return enhanced_line

        # Process each geometry in the GeoDataFrame
        enhanced_geometries = []

        # Ensure we're working with a GeoDataFrame in WGS84 (EPSG:4326)
        if geodf.crs is None:
            geodf = geodf.set_crs("EPSG:4326")
        elif geodf.crs != "EPSG:4326":
            geodf = geodf.to_crs("EPSG:4326")

        # Reasonable buffer in degrees (roughly 50-100m depending on latitude)
        buffer_deg = 0.0005

        # Process each geometry
        for i, geom in enumerate(geodf.geometry):
            try:
                if geom.geom_type == "LineString":
                    if use_osm:
                        # Get OSM roads around this line
                        roads = get_osm_roads_in_bbox(geom.bounds, buffer_deg)
                    else:
                        roads = None

                    enhanced_line = enhance_line_with_regular_points(geom, point_spacing_meters, roads, buffer_deg)
                    enhanced_geometries.append(enhanced_line)
                elif geom.geom_type == "MultiLineString":
                    # Process each part of the MultiLineString
                    enhanced_parts = []
                    for part in geom.geoms:
                        if use_osm:
                            roads = get_osm_roads_in_bbox(part.bounds, buffer_deg)
                        else:
                            roads = None

                        enhanced_part = enhance_line_with_regular_points(part, point_spacing_meters, roads, buffer_deg)
                        enhanced_parts.append(enhanced_part)
                    # Keep it as a MultiLineString
                    enhanced_geometries.append(MultiLineString(enhanced_parts))
                else:
                    # For non-line geometries, keep as is
                    enhanced_geometries.append(geom)
            except Exception as e:
                print(f"Error processing geometry {i}: {e}")
                # Keep original geometry as fallback
                enhanced_geometries.append(geom)

        # Create a new GeoDataFrame with enhanced geometries
        result = gpd.GeoDataFrame(
            geodf.drop(columns="geometry", errors="ignore"),
            geometry=enhanced_geometries,
            crs="EPSG:4326",
        )

        return result


    def smooth_geojson_route(self, route_input, max_distance_meters=10):
        """
        Smooths a route by adding interpolated points so that no two consecutive points
        are further apart than max_distance_meters.

        Parameters:
        -----------
        route_input : str, dict, GeoDataFrame, LineString, or list
            The input route in various formats
        max_distance_meters : float
            Maximum allowed distance between consecutive points in meters

        Returns:
        --------
        GeoDataFrame containing the smoothed route with the same CRS as input
        """


        def extract_coordinates(route_input):
            """Extract coordinates from various input formats"""
            if isinstance(route_input, str):
                route_input = json.loads(route_input)
            if isinstance(route_input, dict):  # GeoJSON
                if "features" in route_input:
                    return [
                        feature["geometry"]["coordinates"]
                        for feature in route_input["features"]
                    ]
                elif "coordinates" in route_input:
                    return [route_input["coordinates"]]
            elif isinstance(route_input, gpd.GeoDataFrame):
                return [list(geom.coords) for geom in route_input.geometry]
            elif isinstance(route_input, LineString):
                return [list(route_input.coords)]
            elif isinstance(route_input, list) and all(
                isinstance(pt, (list, tuple)) for pt in route_input
            ):
                return [route_input]  # Single LineString as list of points
            raise ValueError("Unsupported route input format")

        def calculate_distance(p1, p2, projected_crs="EPSG:3857"):
            """Calculate distance between two points in meters using projection"""
            # Create points
            point1 = Point(p1)
            point2 = Point(p2)

            # Create GeoSeries with the points
            points_gs = gpd.GeoSeries([point1, point2], crs="EPSG:4326")

            # Project to a metric CRS
            projected = points_gs.to_crs(projected_crs)

            # Calculate distance
            return projected.iloc[0].distance(projected.iloc[1])

        def interpolate_points(p1, p2, num_segments):
            """Interpolate points between p1 and p2"""
            # Make sure we're working with arrays of the same dimension
            if len(p1) != len(p2):
                raise ValueError(f"Points have different dimensions: {p1} and {p2}")

            # Generate interpolated points
            points = []
            for t in np.linspace(0, 1, num_segments + 1)[1:-1]:
                # Linear interpolation for each dimension
                point = [p1[i] + (p2[i] - p1[i]) * t for i in range(len(p1))]
                points.append(point)
            return points

        def interpolate_line(coords, max_distance):
            """Interpolate a line based on maximum distance"""
            if len(coords) < 2:
                return coords

            result = [coords[0]]  # Start with the first point

            for i in range(1, len(coords)):
                p1 = coords[i - 1]
                p2 = coords[i]

                # Calculate distance in meters
                distance = calculate_distance(p1, p2)

                if distance > max_distance:
                    # Calculate number of segments needed
                    num_segments = int(np.ceil(distance / max_distance))

                    # Get interpolated points
                    interp_points = interpolate_points(p1, p2, num_segments)
                    result.extend(interp_points)

                # Always add the original end point
                result.append(p2)

            return result

        # Determine input CRS
        input_crs = "EPSG:4326"
        if isinstance(route_input, gpd.GeoDataFrame):
            input_crs = route_input.crs

        # Extract coordinates from input
        all_coords = extract_coordinates(route_input)

        # Interpolate each line
        smoothed_coords = [
            interpolate_line(coords, max_distance_meters) for coords in all_coords
        ]

        # Create geometries
        smoothed_geometries = [LineString(coords) for coords in smoothed_coords]

        # Return as GeoDataFrame
        return gpd.GeoDataFrame(geometry=smoothed_geometries, crs=input_crs)
