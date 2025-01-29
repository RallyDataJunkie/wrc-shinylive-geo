import kml2geojson
from ipyleaflet import Map

from shinywidgets import render_widget 
import requests

from io import StringIO


def kml_url_to_json(kml_url):
    text = StringIO(requests.get(kml_url).text)
    return kml2geojson.main.convert(text)

# ## Stage Metadata From JSON
#
# What can we derive from JSON data?


def get_map_stage_details(_df):
    """Return some details about the stage."""

    example_map_coords = _df["features"][0]["geometry"]["coordinates"]
    _name = _df["features"][0]["properties"]["name"]
    # stage_length
    # start location
    _start = example_map_coords[0]
    _end = example_map_coords[-1]
    # end location

    # From elsewhere we should be able to get
    # the split locations.

    return _name, _start, _end
