import pandas as pd
import seaborn as sns
df = sns.load_dataset("tips")

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(name="Simple Analytics Server",external_stylesheets=external_stylesheets)

colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}

def generate_table(dataframe, max_rows=10):
    return html.Table(
        [html.Tr([ html.Th(col) for col in dataframe.columns])] +

        [html.Tr([
            html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
        ]) for i in range(min(len(dataframe), max_rows))]
    )

app.layout = html.Div(children=[
    html.H1("Hello world to Dash"),
    # dcc.Input(id="TextBox1Id", value='initial value', type="text"),

    html.Div(children=
        [
            dcc.Dropdown(
            id="gender_drop_down_id", 
            options=[{'label': i, 'value': i} for i in df.sex.unique()],
            value="Male",
            style={"width": 200}),

            dcc.Dropdown(
            id="time_drop_down_id", 
            options=[{'label': i, 'value': i} for i in df.time.unique()],
            value="Dinner",
            style={"width": 200})
        ],
    ),
    dcc.Graph(id="chartid")
    ])

@app.callback(
    Output(component_id='chartid', component_property='figure'),
    [Input(component_id='gender_drop_down_id', component_property='value'),
    Input(component_id='time_drop_down_id', component_property='value')]
)
def update_graph(gender_value, time_value):
    print(gender_value, time_value)
    res = df.query(f"sex=='{gender_value}' & time=='{time_value}'").groupby("day")["tip"]
    #x,y = res.index, res.values
    figure={
            "data": [
                {'x': res.mean().index, 'y': res.mean().values, 'type': "bar", "name": "Avg tip by day"},
                {'x': res.min().index, 'y': res.min().values, 'type': "bar", "name": "Min tip by day"},
                {'x': res.max().index, 'y': res.max().values, 'type': "bar", "name": "Max tip by day"},
                ],
            "layout": {
                "title": "Simple Chart",
                "xaxis": { "title" : "Days"},
                "yaxis": { "title" : "Tips"}
            }    
    }
    return figure

if __name__ == "__main__":
    print("Starting serve 123")
    app.run_server(port=8080,debug=True)
