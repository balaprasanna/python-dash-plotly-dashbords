import  pandas as pd
import networkx as nx
from matplotlib import colors as matcolors
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout
import community
import random

fname = '/mnt/hdd/Projects/bala/papers/VS/cleandata4.csv'
clean_data = pd.read_csv(fname, delimiter=';')
clean_data.head(4)
print(clean_data.shape)

def get_custom_pos():
    cord_df =  clean_data[['EntryGate','EntryCoordinates']].drop_duplicates().reset_index(drop=True)
    cord_df.set_index('EntryGate',inplace=True)
    cord_df.EntryCoordinates = cord_df.EntryCoordinates.apply(lambda x: eval(x))
    cord_df['cord'] = cord_df.EntryCoordinates.apply(lambda x: (x[0]/960, x[1]/960))
    custompos = cord_df.cord.to_dict()
    return custompos

custompos = get_custom_pos()

def make_gate_pair_dict(subset):
    ## Prepare the data
    gate_pair_dict = {}
    for index, series in subset.iterrows():
        from_gate, to_gate = series['ExitGate'],series['EntryGate']
        gate_pair_dict[(from_gate,to_gate)] = gate_pair_dict.get((from_gate,to_gate), 0) + 1
    return gate_pair_dict

def add_edge(G, metadata,weight):
    ## Add the metadata to each edge
    G.add_edge(metadata['from_gate'], metadata['to_gate'], weight=weight, **metadata)
        
def make_graph_G(subset):
    gate_pair_dict = make_gate_pair_dict(subset)
    
    #G = nx.DiGraph()
    G = nx.Graph()
    # G.add_edge(1,2)
    for index, series in subset.iterrows():
        car_type = series['car-type']
        from_gate, to_gate = series['ExitGate'],series['EntryGate']
        DistanceMiles = series['DistanceMiles']
        EntryTime = series['EntryTime']
        metadata = {
            "cat_type": car_type,
            "from_gate": from_gate,
            "to_gate": to_gate,
            "DistanceMiles": DistanceMiles,
            "EntryTime": EntryTime
            
        }
        weight = gate_pair_dict.get((from_gate,to_gate), 1)
        add_edge(G,metadata, weight)

    return G,gate_pair_dict


def Draw_Graph_Util_With_Degree_Centrality_and_Cluster(G, gate_feq_dict, custompos,figsize=None):
    # plot it
    random.seed(43)
    values = list(colors.cnames.keys())
    random.shuffle(values)
    # BASED ON wight
    edge_width = [0.5215 * G[u][v]['weight'] for u, v in G.edges()] 
    ## with deg_centrality
    deg_centrality = nx.degree_centrality(G)
    node_size = [ 10000.0*deg_centrality.get(v,0) for v in G] 
    
    #COMPUTE THE COMMUNITY - CLSUTER
    parts = community.best_partition(G)
    c_values = [parts.get(node) for node in G.nodes()]
    node_color = [ values[n] for n in c_values]
    df = DF(parts)
    df['node_color'] = node_color
    node_color_dict = df.node_color.to_dict()
    node_color = [node_color_dict.get(v, "red") for v in G] 
    
    if figsize:
        fig = plt.figure(figsize =figsize)
    else:
        fig = plt.figure(figsize =(10, 10)) 
#     nx.draw_networkx(G, pos=custompos,
#                      node_size = node_size,
#                      node_color = node_color, alpha = 0.8, 
#                      with_labels = True, width = edge_width, 
#                      #pos=nx.circular_layout(G),
#                      edge_color ='.9', cmap = plt.cm.Blues) 
    nx.draw_networkx(G, pos=custompos,
                     node_size = node_size,
                     node_color = node_color, alpha = .8, 
                     with_labels = True, width = edge_width, 
                     edge_color ='.5', cmap = plt.cm.Blues) 
    plt.axis('off') 
    plt.tight_layout();
    return fig
    
def filter_date(df,Date,figsize=None):
    ## plot the graph h,ere...
    subset = df.loc[df.Date == Date]
    G,gate_feq_dict = make_graph_G(subset)
    #return Draw_Graph_Util_With_Degree_Centrality_and_Cluster(G, gate_feq_dict, custompos,figsize)
    return subset,G,gate_feq_dict

df = clean_data.copy()

import seaborn as sns
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objects as go

#df = sns.load_dataset("tips")


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(name=__name__,external_stylesheets=external_stylesheets)

colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}


app.layout = html.Div(children=[
    html.H1("Hello world to Dash"),
    # dcc.Input(id="TextBox1Id", value='initial value', type="text"),

    html.Div(children=
        [
            dcc.Dropdown(
            id="date_drop_down_id", 
            options=[{'label': i, 'value': i} for i in df.Date.unique()],
            value=df.Date.unique()[0],
            style={"width": 400}),
        ],
    ),
    #dcc.Graph(id="barchart1", style={"display":None}),
    html.Br(),
    html.Hr(),
    dcc.Graph(id="networkchart1",
        style={"width": 800, "height": 800, 'display': 'inline-block','vertical-align': 'middle'}
        )
    ])

# @app.callback(
#     Output(component_id='barchart1', component_property='figure'),
#     [Input(component_id='date_drop_down_id', component_property='value')],
#     #Input(component_id='time_drop_down_id', component_property='value')]
# )
# def update_barchart(date_value):
#     print(date_value)
    
#     subset,G,gate_feq_dict = filter_date(df,date_value)

#     res = pd.Series(subset.EntryGate.value_counts().to_dict())
    
#     figure={
#             "data": [
#                 {'x': res.index, 'y': res.values, 'type': "bar", "name": "Entry Gate Valuecount"},
#                 ],
#             "layout": {
#                 "title": "Simple Chart",
#                 "xaxis": { "title" : "Entry Gates"},
#                 "yaxis": { "title" : "Value counts"}
#             }    
#     }
#     return figure


@app.callback(
    Output(component_id='networkchart1', component_property='figure'),
    [Input(component_id='date_drop_down_id', component_property='value')],
    #Input(component_id='time_drop_down_id', component_property='value')]
)
def update_network_graph(date_value):
    print(date_value)
    
    subset,G,gate_feq_dict = filter_date(df,date_value)

    ## set custom pos
    for node in G.nodes:
        G.nodes[node]['pos'] = list(custompos[node])

    ## set edges and its pos
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = G.nodes[edge[0]]['pos']
        x1, y1 = G.nodes[edge[1]]['pos']
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)
    
    edge_trace = go.Scatter( x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='text',
        mode='lines')

    # setup node_trace
    node_x,node_y = [], []
    for node in G.nodes():
        x, y = G.nodes[node]['pos']
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            # colorscale options
            #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
            #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
            #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
            colorscale='YlGnBu',
            reversescale=True,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))
    

    random.seed(43)
    color_values = list(matcolors.cnames.keys())
    random.shuffle(color_values)

    # Setting coloring for nodes
    node_adjacencies = []
    node_text = []
    node_size = []
    node_color_based_on_cluster = []
    deg_central = nx.degree_centrality(G)
    parts = community.best_partition(G)


    for node_index, adjacencies in enumerate(G.adjacency()):
        node , adj_values = adjacencies
        node_adjacencies.append(len(adj_values))
        node_text.append(f' #{node} -> Deg centrality: '+ str(deg_central.get(node,0)))  #+str(len(adj_values)))
        scale_factor = 100.0
        node_size.append( scale_factor*deg_central.get(node,0) )
        
        #cluster it and get a color
        node_color_based_on_cluster.append( color_values[parts.get(node)] )

    node_trace.marker.color = node_color_based_on_cluster #node_adjacencies
    node_trace.marker.size = node_size
    node_trace.text = node_text    

    fig = go.Figure(
            data = [edge_trace, node_trace],
            layout = go.Layout(
                title='Network graph made with Python',
                titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                xaxis=dict(showgrid=False, 
                    zeroline=False, 
                    showticklabels=False),
                yaxis=dict(showgrid=False, 
                    zeroline=False, 
                    showticklabels=False)
                ))
    return fig

if __name__ == "__main__":
    print("Starting serve 123")
    app.run_server(port=8080,debug=False)
