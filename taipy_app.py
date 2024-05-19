from taipy.gui import Gui
import pandas as pd

penguin_file_url = "https://raw.githubusercontent.com/allisonhorst/palmerpenguins/master/inst/extdata/penguins.csv"
penguin_df = pd.read_csv(penguin_file_url)

target_names = list(penguin_df.species.unique())
species = target_names[0]
df = penguin_df[penguin_df.species == species]

chart_properties = {
    "height": "35vh",
    "width": "40vw",
    "mode": "markers",
    "marker": {
        "size": 10,
        "color": "orange",
        "line": {"width": 3, "color": "black"},
    },
    "layout": {"margin": {"t": 0}},
    "options": {"unselected": {"marker": {"opacity": 1}}},
}

# dialog pop-up
def toggle_table_dialog(state):
    state.show_table_dialog = not state.show_table_dialog


show_table_dialog = False


md = """
    <|toggle|theme|>
    # Taipy Demo - [palmerpenguins](https://github.com/allisonhorst/palmerpenguins) 🐧
    ------------------------------
    <|layout.start|columns=1 3 3|gap=1.5rem|>
    <|part.start|>
    ## Selections
    ### Penguin species: 
    <|{species}|selector|lov={target_names}|dropdown|width=100%|>
    ------------------------------
    Selected penguins out of all penguins:  
    <br />
    <|{len(df)}|indicator|value={len(df)}|max={len(penguin_df)}|>
    ------------------------------
    <|Show Raw Data|button|on_action=toggle_table_dialog|>
    <|{show_table_dialog}|dialog|on_action=toggle_table_dialog|width=90vw|labels={["Cancel"]}|
    <center><|{penguin_df}|table|width=fit-content|height=65vh|></center>
    |>
    <|part.end|>
    <|part.start|>
    **Chart 1:** bill_depth_mm against bill_length_mm 
    <|{df}|chart|x=bill_length_mm|y=bill_depth_mm|properties={chart_properties}|>
    **Chart 3:** bill_depth_mm against body_mass_g 
    <|{df}|chart|x=body_mass_g|y=bill_depth_mm|properties={chart_properties}|>
    <|part.end|>
    <|part.start|>
    **Chart 2:** flipper_length_mm against bill_length_mm 
    <|{df}|chart|x=bill_length_mm|y=flipper_length_mm|properties={chart_properties}|>
    **Chart 4:** flipper_length_mm against body_mass_g
    <|{df}|chart|x=body_mass_g|y=flipper_length_mm|properties={chart_properties}|>
    <|part.end|>
    <|layout.end|>
"""


def on_change(state, var_name, var_value):
    print(var_name, type(var_value), var_value)
    if var_name == "species":
        state.df = penguin_df[penguin_df.species == var_value]


if __name__ == "__main__":
    gui = Gui(page=md)
    gui.on_change = on_change
    gui.run(title="Palmer Penguins 🐧", dark_mode=False, port=8080)