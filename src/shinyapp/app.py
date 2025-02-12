from wrcapi_rallydj.data_api import WRCDataAPIClient
from shiny.express import ui, input
from shiny import render, reactive
from shinywidgets import render_widget

wrcapi = WRCDataAPIClient(usegeo=True)


ui.panel_title("RallyDataJunkie WRC API Data Browser", "WRC-RallyDJ")


with ui.sidebar(open="desktop"):
    # Create season selector
    # Currently offers a hard coded set of season year options
    ui.input_select(
        "season", "Season:", [str(i) for i in range(2020, 2026)], selected="2025"
    )

    # Create rally selector
    # Dynamically populated using a list of events
    # based on the season selection
    ui.input_select("event", "Event:", {}, selected=wrcapi.rallyId)

    ## Limit things to WRC for now
    # # Create championship selector
    # ui.input_select(
    #     "championship",
    #     "Championship:",
    #     {v: k for k, v in wrcapi.CATEGORY_MAP.items()},
    #     selected="wrc",
    # )

    # Create stages selector
    ui.input_select(
        "stage",
        "Stage:",
        {},
    )


@reactive.calc
@reactive.event(input.event)
def rally_id_var():
    rally_id = input.event()
    wrcapi.rallyId = rally_id
    return wrcapi.rallyId


@reactive.calc
@reactive.event(input.season, input.event)
def rally_geodata():
    kmlstub = season_rally_data()["kmlfile"].iloc[0]
    geostages = wrcapi.read_kmlfile(kmlstub)
    return geostages
# TO DO - the stage view is based on stages from the rally data
# which may not exist before the event starts
# Instead, we could pull the tracks from the kml file

@reactive.calc
@reactive.event(input.season, input.event)
def rally_data():
    season_df = season_data()
    rallydata = wrcapi.process_rally_data(season_df, rally_id_var())
    return rallydata


@reactive.calc
@reactive.event(input.season)
def season_data():
    wrcapi.initialise(year=int(input.season()))
    # WRC API data fetch
    season = wrcapi.get_rallies_data()
    return season

# Get season_rally_data which is filtered season data
@reactive.calc
@reactive.event(input.event)
def season_rally_data():
    season = season_data()
    season_filtered = season[season["sas-rallyid"] == input.event()]
    return season_filtered

@reactive.effect
@reactive.event(input.season)
def update_events_select():
    season = season_data()
    print(season)
    # events = season["EventName"].to_list()
    events = season[["sas-rallyid", "name"]].set_index("sas-rallyid")["name"].to_dict()
    ui.update_select("event", choices=events)


@reactive.effect
@reactive.event(input.event)
def update_stages_select():
    rallydata = rally_data()
    if rallydata.empty:
        ui.update_select("stage", choices={})
    else:
        stages = rallydata["name"].to_list()
        ui.update_select("stage", choices=stages)


with ui.accordion(open=False):
    with ui.accordion_panel("Season info"):

        @render.data_frame
        def rallies_frame():
            season = season_data()
            retcols = [
                "name",
                "info-date",
                "date-start",
                "date-finish",
                "info-based",
                "info-surface",
            ]
            return render.DataGrid(season[retcols])


# Card pattern, no tooltip
# with ui.card(class_="mt-3"):
#    ui.card_header("Card title")

with ui.accordion(open=False):
    with ui.accordion_panel("Event info"):
        with ui.accordion(open=False):
            with ui.accordion_panel("Stages overview"):

                @render.data_frame
                @reactive.event(input.event)
                def rallies_display():
                    rallydata = rally_data()
                    displayCols = ["date", "name", "location", "distance", "firstCar"]
                    return render.DataGrid(rallydata[displayCols])

            ## Roster info seems to be a bit flaky?
            # with ui.card(class_="mt-3"):
            #     ui.card_header("Roster")

            #     @render.data_frame
            #     @reactive.event(input.event, input.championship)
            #     def roster_display():
            #         season_df = season_data()
            #         df_rosterEntry, df_car, df_biog, df_team = wrcapi.process_roster_data(
            #             season_df, input.event(), category=input.championship()
            #         )
            #         displayCols = [
            #             c
            #             for c in ["driver", "codriver", "driverCode", "seasonname"]
            #             if c in df_car.columns
            #         ]
            #         return render.DataGrid(df_car[displayCols])
        with ui.accordion(open=False):
            with ui.accordion_panel("Stages map"):

                @render_widget
                @reactive.event(input.season)
                def allstages_map():
                    geostages = rally_geodata()
                    m = wrcapi.GeoTools.simple_stage_map(geostages)
                    return m


with ui.accordion(open=False):
    with ui.accordion_panel("Stage info"):
        with ui.accordion(open=False):
            with ui.accordion_panel("DRIVER INFO TO MOVE"):

                @render.data_frame
                @reactive.event(input.event, input.stage)
                def stage_display():
                    stage = input.stage()
                    season_df = season_data()
                    rallydata = wrcapi.process_rally_data(season_df, rally_id_var())
                    _out = wrcapi.process_stage_data(rallydata, name=stage)
                    if _out.empty:
                        return
                    displaycols = [
                        c
                        for c in ["_record_name", "time", "racetime", "position"]
                        if c in _out.columns
                    ]
                    # TO DO - the cateogry mapping to WRC is incorrect in provided data?
                    # e.g. we also get WRC2 cars
                    return render.DataGrid(_out[displaycols])

        # TO DO:
        # - map with all stages view;
        # - map with single stage view
        # - stage route analyses (there is no code ported over for this yet)
        #   consider doing a live version of https://github.com/RallyDataJunkie/visualising-rally-stages
        # - move the driver / roster stuff etc out to a telemetry app,
        #   and just keep this app purely focussed on matters relating to the stage

        with ui.accordion(open=False):
            with ui.accordion_panel("Stage map"):

                @render_widget
                @reactive.event(input.season, input.stage)
                def single_stage_map():
                    geostages = rally_geodata()
                    m = wrcapi.GeoTools.simple_stage_map(geostages, input.stage())
                    return m
