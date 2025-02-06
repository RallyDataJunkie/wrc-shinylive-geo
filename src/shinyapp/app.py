from wrcapi_rallydj.data_api import WRCDataAPIClient
from shiny.express import ui, input
from shiny import render, reactive

wrcapi = WRCDataAPIClient()


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

@reactive.calc
@reactive.event(input.season)
def season_data():
    wrcapi.initialise(year=int(input.season()))
    # WRC API data fetch
    season = wrcapi.get_rallies_data()
    return season

@reactive.effect
@reactive.event(input.season)
def update_events_select():
    season = season_data()
    # events = season["EventName"].to_list()
    events = (
        season[["sas-rallyid", "name"]].set_index("sas-rallyid")["name"].to_dict()
    )
    ui.update_select("event", choices=events)

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
                "info-surface"
            ]
            return render.DataGrid(season[retcols])

# Card pattern, no tooltip
#with ui.card(class_="mt-3"):
#    ui.card_header("Card title")

with ui.accordion(open=False):
    with ui.accordion_panel("Event info"):
        with ui.accordion(open=False):
            with ui.accordion_panel("Stages overview"):
                @render.data_frame
                @reactive.event(input.event)
                def rallies_display():
                    season_df = season_data()
                    rallydata = wrcapi.process_rally_data(season_df, input.event())
                    displayCols = ["date", "name", "location", "distance",  "firstCar"]
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
