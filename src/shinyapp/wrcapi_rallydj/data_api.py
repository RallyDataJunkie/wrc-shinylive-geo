# Tools for working with the WRC data api

import datetime
from requests_cache import CachedSession
from datetime import timedelta
import pandas as pd
from geopandas import GeoDataFrame
from io import StringIO
import kml2geojson
from typing import Dict, Any
import re

# from time import sleep
# import random
import json
import gzip
import base64

# TO DO - maybe see what we can learn from
# https://webapps2.wrc.com/2020/obc/js/wrc/api.js


class WRCDataAPIClient:
    """Client for accessing WRC Telemetry Rally API data."""

    WRC_DATA_API_BASE = "https://webappsdata.wrc.com/srv/wrc/json/api/wrcsrv/{query}"
    WRC_ASSETS_PATH = "https://webapps2.wrc.com/2020/web"
    # We can use the WRC_ASSETS_PATH to retrieve the get_rallies_data "logo"
    WRC_KML_PATH = WRC_ASSETS_PATH + "/live/kml/{kmlfile}.xml"
    CATEGORY_MAP = {"ALL": "all", "WRC": "wrc", "WRC2": "wrc2", "WRC3": "wrc3"}

    def __init__(self, year: int = datetime.date.today().year, usegeo: bool = False):
        """
        Initialize the WRC Data API client.
        """
        if usegeo:
            from .geotools import RallyGeoTools

            self.GeoTools = RallyGeoTools()
        else:
            self.GeoTools = None

        self.year = year
        self.championshipType = None
        self.eventId = None
        self.rallyId = None
        # Simple requests session
        # TO DO - we could use a proxy here
        self.r = CachedSession("demo_cache", expire_after=timedelta(hours=1))
        self.alldata = {}

    def initialise(self, year=None):
        """Initialise with the calendar."""
        if year:
            self.year = year

    def kmlfile_to_json(self, kmlfile):
        if not isinstance(kmlfile, str) or not kmlfile:
            return {}
        kmlurl = self.WRC_KML_PATH.format(kmlfile=kmlfile)
        text = StringIO(
            re.sub(
                r"[\u200b\u200e\u200f]",
                "",
                self.r.get(kmlurl).content.decode("utf-8"),
            )
        )
        return kml2geojson.main.convert(text)

    def read_kmlfile(self, kmlfile):
        def _simpleStageList(stages):
            if stages.startswith("SS"):
                # Match SS + one or more number parts separated by / or -
                match = re.match(r"SS\s*(\d+(?:\s*[-/]\s*\d+)*)", stages)
                if not match:
                    return []

                segment = match.group(1)

                # Split on - or /, keep only endpoints
                numbers = re.split(r"[-/]", segment)
                stages = [f"SS{int(n.strip())}" for n in numbers if n.strip().isdigit()]

                return stages
            else:
                return [stages]

        if not isinstance(kmlfile, str) or not kmlfile:
            return GeoDataFrame() if self.GeoTools else {}

        gj = self.kmlfile_to_json(kmlfile)
        if not gj:
            return GeoDataFrame() if self.GeoTools else {}
        gj = gj[0]
        for feature in gj["features"]:
            stage = feature["properties"].get("name", "")
            feature["properties"]["stages"] = _simpleStageList(stage)

        if self.GeoTools:
            _gdf = self.GeoTools.geojson_to_gpd(gj)
            return _gdf
        return gj

    def get_map_stages(self, gj):
        """"""
        gff = []
        for gf in gj["features"]:
            # Handle SS 1/2 as SS1-2
            gf["properties"]["name"] = (
                gf["properties"]["name"].replace("/", "-").replace(" ", "").strip()
            )
            # print(gf["properties"]["name"])
            gff.append({"type": "FeatureCollection", "features": [gf]})
        return gff

    # Utility functions

    @staticmethod
    def listify(item):
        return item if isinstance(item, list) else [item]

    @staticmethod
    def getkv(p):
        """Return key / value pair from dict"""
        return p["n"], p["v"]

    def getkvp(self, d):
        """Generate key / value dict"""
        r = {}
        for p in d:
            k, v = self.getkv(p)
            r[k] = v
        return r

    def record_parse(self, r, datekey=False):
        """Parse basic downloaded record and return base record dict.
        Returns record _id, name, type and metadata.
        """
        d = {
            "type": r["type"],
            "name": r["name"],
            "_id": r["_id"],
            "_meta": self.getkvp(r["_meta"]),
        }
        if datekey:
            year = d["_meta"]["date-start"].split("-")[0]
            return year, d
        else:
            return d

    def get_rally_attribute(
        self, df_rallies, rallyid=None, rallyname=None, year=None, attr="rosterid"
    ):
        if rallyid is not None or (rallyid is None and rallyname is None):
            rallyid = rallyid if rallyid else self.rallyId
            _rally_df = df_rallies.loc[
                df_rallies["sas-rallyid"] == rallyid,
                attr,
            ]
        elif rallyname:
            year = year if year else self.year
            _rally_df = df_rallies.loc[
                (df_rallies["year"] == str(year))
                & (df_rallies["name"].str.contains(rallyname)),
                attr,
            ]
        return _rally_df.values[0]

    def _df_from_record(self, rd):
        """Get dataframe from base record dict."""
        rid = rd["_id"]
        rname = rd["name"]
        rtype = rd["type"]

        df = pd.DataFrame.from_dict(rd).reset_index()[["index", "_meta"]]
        df.columns = [rtype, rid]
        df = df.set_index(rtype).T
        df["_record_name"] = rname
        return df

    def get_base_data(self, typ="WRC", retval=False):
        """Get base rally data, keyed by year."""
        # The "availability now" list adds entries when a rally is running.
        # This DOES NOT include shakedown or the days prior to the rally start.
        # q = "queryMeta?t=%22Event%22&p=%7B%22n%22%3A%22availability%22%2C%22v%22%3A%22now%22%7D&maxdepth=1"
        # TO DO the following filters on category
        q = f"queryMeta?t=%22Event%22&p=%7B%22n%22%3A%22category%22%2C%22v%22%3A%22{typ.upper()}%22%7D&maxdepth=1"
        # TO DO - to get the kmlurl we need to set the depth to 2 and parse down

        # TO DO - we can get event by category:
        # https://webappsdata.wrc.com/srv/wrc/json/api/wrcsrv/queryMeta?t=%22Event%22&p=%7B%22n%22%3A%22category%22%2C%22v%22%3A%22WRC%22%7D&maxdepth=1
        # TO DO - query season by category
        # https://webappsdata.wrc.com/srv/wrc/json/api/wrcsrv/queryMeta?t=%22Season%22&p=%7B%22n%22%3A%22category%22%2C%22v%22%3A%22WRC%22%7D&maxdepth=1
        # TO DO query event by category eg ERC
        # https://webappsdata.wrc.com/srv/wrc/json/api/wrcsrv/queryMeta?t=%22Event%22&p=%7B%22n%22%3A%22category%22%2C%22v%22%3A%22ERC%22%7D&maxdepth=1
        # TO DO by surface
        # https://webappsdata.wrc.com/srv/wrc/json/api/wrcsrv/queryMeta?t=%22Event%22&p=%7B%22n%22%3A%22info-surface%22%2C%22v%22%3A%22Gravel%22%7D&maxdepth=1
        # TO DO - is there a way to query on multiple params?
        url = self.WRC_DATA_API_BASE.format(query=q)

        basedata = self.r.get(url, verify=False).json()
        self.championshipType = typ

        # Useful - type, name, _id, _meta
        alldata = {}
        for r in basedata:
            year, d = self.record_parse(r, datekey=True)
            if year not in alldata:
                alldata[year] = {}
            alldata[year][d["name"]] = d
        self.alldata = alldata
        if retval:
            return alldata

    def _wrc_events_to_df(self, data: Dict[str, Any]) -> pd.DataFrame:
        """
        Convert WRC event JSON data to a pandas DataFrame using existing processing methods.

        Args:
            data: Dictionary containing WRC season/event data

        Returns:
            DataFrame with all events and their metadata
        """
        all_events = []

        for year, year_data in data.items():
            # Parse year data
            year_record = self.record_parse(year_data)
            year_df = self._df_from_record(year_record)

            # Process each event in the year
            for event in year_data["_dchildren"]:
                # Parse event data
                event_record = self.record_parse(event)
                event_df = self._df_from_record(event_record)

                # Add year information
                event_df["year"] = year
                if "_meta" in year_df:
                    event_df["category"] = year_df["_meta"].get("category", "")

                all_events.append(event_df)

        # Combine all events into single DataFrame
        result_df = pd.concat(all_events, axis=0)

        # Convert date columns to datetime if they exist
        date_cols = ["date-start", "date-finish"]
        # for col in result_df.columns:
        #    if any(date_col in col for date_col in date_cols):
        #        result_df[col] = pd.to_datetime(result_df[col])

        return result_df

    def get_seasons(self, typ="WRC"):
        """Get season data."""
        q = "byType?t=%22Season%22&maxdepth=2"
        url = self.WRC_DATA_API_BASE.format(query=q)
        seasondata = self.r.get(url, verify=False).json()
        seasons = {}
        for s in seasondata:
            event = {"name": s["name"]}
            for _m in s["_meta"]:
                (k, v) = self.getkv(_m)
                event[k] = v
            # season[event['category']] = {event['name']: s}
            if not event["category"] in seasons:
                seasons[event["category"]] = {event["name"]: s}
            else:
                seasons[event["category"]][event["name"]] = s
        # The seasons data
        # return { s['name']:s for s in seasondata }
        # return seasons[typ]
        if not typ or typ.lower() == "all" or typ not in seasons:
            all_champs = pd.concat(
                [
                    self._wrc_events_to_df(seasons["WRC"]),
                    self._wrc_events_to_df(seasons["ERC"]),
                ],
                axis=0,
                ignore_index=True,
            )
            return all_champs.reset_index(drop=True)

        return self._wrc_events_to_df(seasons[typ])

    def get_poilist_data(self, poilist, as_gdf=True):
        """Get poi list data."""
        q = f"byId?id=%22{poilist}%22"
        url = self.WRC_DATA_API_BASE.format(query=q)
        poidata = self.r.get(url, verify=False).json()["_dchildren"]
        pois = []
        for poi in poidata:
            pois.append(self.record_parse(poi))
        _df = pd.DataFrame(pois)
        _df = _df.drop(columns=["_meta"]).join(
            pd.json_normalize(_df["_meta"]), rsuffix="_meta"
        )

        _df[["latitude", "longitude"]] = (
            _df["position"].str.split(",", expand=True).astype(float)
        )

        if as_gdf and self.GeoTools:
            _df = self.GeoTools.get_gdf_from_lat_lon_df(_df)
            _df.dropna(subset="latitude", inplace=True)
        _df.drop_duplicates(inplace=True)
        return _df

    def get_rallies_data(self, year=None, typ="WRC", alldata=None, as_dict=False):
        """Get rally data as dataframe."""
        # If we provide a year for the rallies, assume this is the year we are using
        if year is not None:
            self.year = year
        else:
            year = self.year
        df_rallies = pd.DataFrame()
        if typ != self.championshipType or year != self.year:
            alldata = self.get_base_data(typ=typ, retval=True)
        elif alldata is None:
            alldata = (
                self.alldata
                if self.alldata
                else self.get_base_data(typ=typ, retval=True)
            )
        years = year if year else alldata.keys()
        years = [years] if not isinstance(years, list) else years
        years = [str(year) for year in years]
        for year in years:
            if year in alldata:
                rallies_year = alldata[year]
                for rid in rallies_year.keys():
                    _df_rallies = self._df_from_record(rallies_year[rid])
                    _df_rallies["year"] = year
                    df_rallies = pd.concat([df_rallies, _df_rallies], sort=False)

        # TO DO - note this call returns two possible datatypes
        # A dict for a single year, or a df for multiple years
        if as_dict:
            return df_rallies[df_rallies["year"] == year].to_dict(orient="records")
        return df_rallies

    def get_rally_by_name(self, df_rallies, rally, year=None):
        year = year if year else self.year
        return df_rallies[
            (df_rallies["year"] == str(year)) & (df_rallies["name"] == rally)
        ].index.values[0]

    def get_rally_by_rallyid(self, df_rallies, rallyid=None):
        rallyid = rallyid if rallyid else self.rallyId
        return df_rallies[df_rallies["sas-rallyid"] == rallyid].index.values[0]

    # + tags=["active-ipynb"]
    # get_rally_by_name(df_rallies, YEAR, rally=_sample_rally_name)
    # -

    def _process_roster_data(self, rosterdata, _rosterID, category):  # , _rallyID):

        def _concat_df(_df, _carID, record):
            _df_tmp = self._df_from_record(record)
            _df_tmp["_carID"] = _carID
            return pd.concat([_df, _df_tmp], sort=False)

        df_rosterEntry = pd.DataFrame()
        df_rosterEntry["_rallies_rosterid"] = _rosterID

        df_car = pd.DataFrame()
        df_biog = pd.DataFrame()
        df_team = pd.DataFrame()

        if not rosterdata or not "_dchildren" in rosterdata:
            return df_rosterEntry, df_car, df_biog, df_team

        for r in rosterdata["_dchildren"]:
            # rosterdata['_dchildren'] -> df_rosterEntry
            _rosterentryID = r["_id"]
            _df_rosterEntry = self._df_from_record(self.record_parse(r))
            df_rosterEntry = pd.concat([df_rosterEntry, _df_rosterEntry], sort=False)
            for c in r["_dchildren"]:
                _df_car = self._df_from_record(self.record_parse(c))
                _df_car["_rosterentryID"] = _rosterentryID

                _carID = c["_id"]
                _driver = self.record_parse(c["_dchildren"][0])
                _codriver = self.record_parse(c["_dchildren"][1])
                _team = self.record_parse(c["_dchildren"][2])

                # I must be grabbing the wrong data here somewhere?
                # The site shows driver and co-driver names
                # but the data is sometimes missing in the feed I grab?

                # Patch missing data
                # print(_df_car.columns)
                if "driver" not in _df_car or not _df_car["driver"].iloc[0]:
                    _df_car["driver"] = _driver["name"]
                    # print('Patched driver {}'.format(_driver['name']))
                if "codriver" not in _df_car or not _df_car["codriver"].iloc[0]:
                    _df_car["codriver"] = _codriver["name"]
                    # print('Patched codriver {}'.format(_codriver['name']))
                if category and category != "all":
                    _df_car = _df_car[_df_car["category"] == category.upper()]
                df_car = pd.concat([df_car, _df_car], sort=False)

                if _driver["type"] == "Driver":
                    df_biog = _concat_df(df_biog, _carID, _driver)
                # _df_biog=_df_from_record(_driver)
                # _df_biog['_carID']=_carID
                # df_biog = pd.concat([df_biog, _df_biog ])
                else:
                    print("oops... wrong type: {}".format(_driver["type"]))
                if _codriver["type"] == "Driver":
                    df_biog = _concat_df(df_biog, _carID, _codriver)
                    # _df_biog= _df_from_record(_codriver)
                    # _df_biog['_carID']=_carID
                    # df_biog = pd.concat([df_biog,_df_biog ])
                else:
                    print("oops... wrong type: {}".format(_driver["type"]))
                if _team["type"] == "Team":
                    _team["_meta"]["_rosterentryID"] = _rosterentryID
                    df_team = _concat_df(df_team, _carID, _team)
                    if category and category != "all":
                        df_team = df_team[df_team["category"] == category.upper()]
                    # _df_team = _df_from_record(_team)
                    # _df_team['_carID']=_carID
                    # df_team = pd.concat([df_team, _df_team ])
                else:
                    print("oops... wrong type: {}".format(_driver["type"]))
                """for b in c['_dchildren']:
                    parsed_b = record_parse(b)
                    if parsed_b['type']=='Driver':
                        _df_biog = _df_from_record(parsed_b)
                        _df_biog['_carID']=_carID
                        df_biog = pd.concat([df_biog, _df_biog ])
                    elif parsed_b['type']=='Team':
                        _df_team = _df_from_record(parsed_b)
                        _df_team['_carID']=_carID
                        df_team = pd.concat([df_team, _df_team ])
                    else:
                        print('**NEW TYPE TO ME...***: {}'.format(parsed_b['type']))
    """
        # df_rosterEntry['_rallies_rosterid'] = _rosterID
        # df_rosterEntry['_rallies_rallyid'] = _rallyID
        df_rosterEntry.reset_index(inplace=True)
        df_rosterEntry.rename(
            columns={"filename": "driverCode", "index": "_rosterentryID"}, inplace=True
        )

        df_car = pd.merge(
            df_car,
            df_rosterEntry[["_rosterentryID", "driverCode"]],
            on="_rosterentryID",
        )

        df_car = pd.merge(
            df_car,
            df_team[["_rosterentryID", "seasonname"]],
            on="_rosterentryID",
        )

        return (
            df_rosterEntry.dropna(how="all", axis=1)
            .dropna(how="all", axis=0)
            .drop_duplicates(),
            df_car.dropna(how="all", axis=1)
            .dropna(how="all", axis=0)
            .drop_duplicates(),
            df_biog.dropna(how="all", axis=1)
            .dropna(how="all", axis=0)
            .drop_duplicates(),
            df_team.dropna(how="all", axis=1)
            .dropna(how="all", axis=0)
            .drop_duplicates(),
        )

    def _get_roster_data(self, _rosterID, stub_roster=None):
        """Get roster data given a rosterID."""
        if stub_roster is None:
            stub_roster = (
                "https://webappsdata.wrc.com/srv/wrc/json/api/wrcsrv/byId?id=%22{}%22"
            )

        rosterdata = self.r.get(stub_roster.format(_rosterID), verify=False).json()
        return rosterdata

    def get_roster_data(self, df_rallies, rallyid=None, rallyname=None, year=None):
        """Get roster data for specific rally."""
        if rallyid is not None or (rallyid is None and rallyname is None):
            rallyid = rallyid if rallyid else self.rallyId
            _rosterID = df_rallies[df_rallies["sas-rallyid"] == rallyid][
                "rosterid"
            ].iloc[0]
        elif rallyname:
            year = year if year else self.year
            _rosterID = df_rallies[
                (df_rallies["year"] == str(year)) & (df_rallies["name"] == rallyname)
            ]["rosterid"].iloc[0]
        else:
            return {}

        return self._get_roster_data(_rosterID)

    def process_roster_data(
        self, df_rallies, rallyid=None, rallyname=None, year=None, category="WRC"
    ):
        year = year if year else self.year
        year = str(year)
        rosterdata = self.get_roster_data(df_rallies, rallyid, rallyname, year)
        _rosterID = self.get_rally_attribute(
            df_rallies, rallyid, rallyname, year, attr="rosterid"
        )
        return self._process_roster_data(rosterdata, _rosterID, category)

    def _get_rally_data(
        self, df_rallies, rallyid=None, rallyname=None, year=None, stub_rally=None
    ):
        year = year if year else self.year
        if rallyid or not rallyname:
            rallyid = rallyid if rallyid else self.rallyId
            _rallyID = self.get_rally_by_rallyid(df_rallies, rallyid)
        elif rallyname:
            _rallyID = self.get_rally_by_name(df_rallies, rallyname, year)
        if stub_rally is None:
            stub_rally = "https://webappsdata.wrc.com/srv/wrc/json/api/wrcsrv/byId?id=%22{}%22&maxdepth=2"
        rallydata = self.r.get(stub_rally.format(_rallyID), verify=False).json()
        # print(stub_rally.format(_rallyID))
        return _rallyID, rallydata

    def _process_rally_data(self, rallydata, _rallyID):  # , _rallyID):
        df_rallydata = pd.DataFrame()
        for r in rallydata["_dchildren"]:
            _df_rallydata = self._df_from_record(self.record_parse(r))
            df_rallydata = pd.concat([df_rallydata, _df_rallydata], sort=False)

        df_rallydata["_rallies_rallyid"] = _rallyID
        return (
            df_rallydata.dropna(how="all", axis=1)
            .dropna(how="all", axis=0)
            .drop_duplicates()
        )

    def process_rally_data(self, df_rallies, rallyid=None, rallyname=None, year=None):
        if rallyid or (not rallyid and not rallyname):
            rallyid = rallyid if rallyid else self.rallyId
            _rallyID, rallydata = self._get_rally_data(df_rallies, rallyid)
        elif rallyname:
            year = year if year else self.year
            _rallyID, rallydata = self._get_rally_data(
                df_rallies, rallyname=rallyname, year=year
            )
        _df_rallydata = self._process_rally_data(rallydata, _rallyID)
        return _df_rallydata

    def get_stage_id(self, df_rallydata, name="SS1", stageId=None):
        if stageId and not df_rallydata.empty and "nr" in df_rallydata.columns:
            _result = df_rallydata[(df_rallydata["nr"] == str(stageId))]
        elif not df_rallydata.empty and "name" in df_rallydata.columns:
            _result = df_rallydata[(df_rallydata["name"] == name)]
        if not _result.empty:
            return _result.index.values[0]

    # + tags=["active-ipynb"]
    # get_stage_id(df_rallydata, name='SS1')
    # -

    def _get_stage_data(self, df_rallydata, name=None, stageId=None, stub_stage=None):
        _stageID = self.get_stage_id(df_rallydata, name=name, stageId=stageId)
        if not _stageID:
            return None, pd.DataFrame()
        if stub_stage is None:
            stub_stage = "https://webappsdata.wrc.com/srv/wrc/json/api/wrcsrv/byId?id=%22{}%22&maxdepth=50"
        stagedata = self.r.get(stub_stage.format(_stageID), verify=False).json()
        return _stageID, stagedata

    def _process_stage_data(self, stagedata, _stageID):  # , _rallyID):
        if not stagedata:
            return
        df_stagedata = pd.DataFrame()
        if "_dchildren" in stagedata:
            for r in stagedata["_dchildren"]:
                _df_stagedata = self._df_from_record(self.record_parse(r))
                df_stagedata = pd.concat([df_stagedata, _df_stagedata], sort=False)

        df_stagedata["_rally_stageid"] = _stageID  # same as first part of index
        df_stagedata["_carentryid"] = [x.split("_")[1] for x in df_stagedata.index]
        if "telemetry_merged" not in df_stagedata:
            df_stagedata["telemetry_merged"] = None
        # _carentryid references index in df_car
        # _record_name references driver in df_car
        return (
            df_stagedata.dropna(how="all", axis=1)
            .dropna(how="all", axis=0)
            .drop_duplicates()
        )

    def process_stage_data(self, df_rallydata, name="SS1", stageId=None):
        _stageID, stagedata = self._get_stage_data(df_rallydata, name, stageId)

        return self._process_stage_data(stagedata, _stageID)

    def get_telemetry_id(self, df_stagedata, name="Neuville"):
        if "telemetry" in df_stagedata.columns:
            _tel_id = df_stagedata[(df_stagedata["_record_name"] == name)][
                "telemetry"
            ].values[0]
        else:
            _tel_id = None
        if "telemetry_merged" in df_stagedata.columns:
            _merged_tel_id = df_stagedata[(df_stagedata["_record_name"] == name)][
                "telemetry_merged"
            ].values[0]
        else:
            _merged_tel_id = None
        _car_entry_id = df_stagedata[(df_stagedata["_record_name"] == name)][
            "_carentryid"
        ].values[0]
        _rally_stage_id = df_stagedata[(df_stagedata["_record_name"] == name)][
            "_rally_stageid"
        ].values[0]
        return _rally_stage_id, _car_entry_id, _tel_id, _merged_tel_id

    # + tags=["active-ipynb"]
    # sample_telem_id = get_telemetry_id(df_stagedata)
    # sample_telem_id

    # +
    # the name can come from the _record_name in df_stagedata

    # Note that the telemetry does not appear to include hybrid status
    def _get_telemetry_data(self, df_stagedata, name="Neuville", stub_telemetry=None):
        print("Checking telemetry...")
        _rally_stage_id, _car_entry_id, _telemetryID, _telemetrymergedID = (
            self.get_telemetry_id(df_stagedata, name)
        )
        # print("_rally_stage_id, _car_entry_id, _telemetryID, _telemetrymergedID")
        if stub_telemetry is None:
            stub_telemetry = "https://webappsdata.wrc.com/srv/fs/pull{}"
            # print(stub_telemetry.format(_telemetryID))
        telemetrydata = self.r.get(stub_telemetry.format(_telemetryID), verify=False)

        if telemetrydata.text:
            try:
                telemetrydata = telemetrydata.json()
            except:
                # print(telemetrydata.text)
                telemetrydata = None
        else:
            telemetrydata = None

        # https://stackoverflow.com/a/28642346/454773
        telemetrymergeddata = None
        if _telemetrymergedID:
            r = self.r.get(stub_telemetry.format(_telemetrymergedID), verify=False)
            if r.text:
                try:
                    telemetrymergeddata = json.loads(
                        gzip.decompress(base64.b64decode(r.text))
                    )
                except:
                    telemetrymergeddata = {"_entries": []}
        return (
            _rally_stage_id,
            _car_entry_id,
            _telemetryID,
            _telemetrymergedID,
            telemetrydata,
            telemetrymergeddata,
        )

    # + tags=["active-ipynb"]
    # _get_telemetry_data(df_stagedata)
    # -

    def _process_driver_telemetry_data(self, telemetrydata):  # , _rallyID):
        df_telemetrydata = pd.DataFrame()
        if telemetrydata is None:
            return df_telemetrydata
        _df_telemetrydata = pd.DataFrame(telemetrydata["_entries"])
        df_telemetrydata = pd.concat([df_telemetrydata, _df_telemetrydata], sort=False)

        return df_telemetrydata

    def process_driver_telemetry_data(self, df_stagedata, name="Neuville"):
        (
            _rally_stage_id,
            _car_entry_id,
            _telemetryID,
            _telemetrymergedID,
            telemetrydata,
            telemetrymergeddata,
        ) = self._get_telemetry_data(df_stagedata, name)
        if telemetrydata is not None:
            df_telemetrydata = self._process_driver_telemetry_data(telemetrydata)
            df_telemetrydata["_rally_stageid"] = _rally_stage_id
            df_telemetrydata["_carentryid"] = _car_entry_id
            df_telemetrydata["_telemetryID"] = _telemetryID
            df_telemetrydata["_name"] = name
        else:
            df_telemetrydata = pd.DataFrame()

        if telemetrymergeddata is not None:
            df_telemetrymergeddata = self._process_driver_telemetry_data(
                telemetrymergeddata
            )
            df_telemetrymergeddata["_rally_stageid"] = _rally_stage_id
            df_telemetrymergeddata["_carentryid"] = _car_entry_id
            df_telemetrymergeddata["_telemetrymergedID"] = _telemetrymergedID
            df_telemetrymergeddata["_name"] = name
        else:
            df_telemetrymergeddata = pd.DataFrame()

        return (
            df_telemetrydata.dropna(how="all", axis=1)
            .dropna(how="all", axis=0)
            .drop_duplicates(),
            df_telemetrymergeddata.dropna(how="all", axis=1)
            .dropna(how="all", axis=0)
            .drop_duplicates(),
        )
