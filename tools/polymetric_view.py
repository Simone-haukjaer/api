import os
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import networkx as nx
import numpy as np

# 1. Extract project file modules
def gather_dependencies(project_root):
    dependencies = {}
    project_modules = set()

    # First pass: collect all module names
    for root, _, files in os.walk(project_root):
        for file in files:
            if file.endswith('.py'):
                rel_path = os.path.relpath(os.path.join(root, file), project_root)
                module_name = rel_path.replace(os.sep, '.').replace('.py', '')
                project_modules.add(module_name)

    # Second pass: extract in-project imports
    for root, _, files in os.walk(project_root):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, project_root)
                current_module = rel_path.replace(os.sep, '.').replace('.py', '')

                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

                imports = []
                for line in lines:
                    line = line.strip()
                    if line.startswith('import '):
                        parts = line.split()
                        if len(parts) >= 2:
                            imports.append(parts[1].split('.')[0])
                    elif line.startswith('from '):
                        parts = line.split()
                        if len(parts) >= 2:
                            imports.append(parts[1].split('.')[0])

                # Only keep in-project module dependencies
                filtered_imports = [imp for imp in imports if imp in project_modules]
                dependencies[rel_path] = {
                    "imports": filtered_imports,
                    "path": file_path
                }

    return dependencies

# 2. Create graph and compute metrics
def create_dependency_graph(dependencies):
    G = nx.DiGraph()
    metrics = {}

    for file, data in dependencies.items():
        G.add_node(file)
        loc = count_loc(data['path'])
        num_imports = len(data['imports'])
        metrics[file] = {
            "LOC": loc,
            "NumImports": num_imports,
            "DependsOnMe": 0  # will be filled later
        }
        for dep in data['imports']:
            if dep in dependencies:
                G.add_edge(file, dep)

    # Update reverse dependencies count
    for target in G.nodes():
        metrics[target]["DependsOnMe"] = len(list(G.predecessors(target)))

    return G, metrics

def count_loc(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return len(f.readlines())
    except Exception:
        return 0

# 3. Plot with plotly
def generate_figure(G, metrics):
    pos = nx.spring_layout(G, seed=42)
    nodes = list(G.nodes())
    edges = list(G.edges())

    node_x = [pos[node][0] for node in nodes]
    node_y = [pos[node][1] for node in nodes]
    node_color = [metrics[node]['LOC'] for node in nodes]
    node_size = [10 + metrics[node]['NumImports'] * 4 for node in nodes]
    node_text = [f"{node}<br>LOC: {metrics[node]['LOC']}<br>Imports: {metrics[node]['NumImports']}<br>Used by: {metrics[node]['DependsOnMe']}" for node in nodes]

    edge_x = []
    edge_y = []
    for src, dst in edges:
        x0, y0 = pos[src]
        x1, y1 = pos[dst]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    return {
        'data': [
            go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(width=0.5, color='#888'), hoverinfo='none'),
            go.Scatter(
                x=node_x, y=node_y, mode='markers',
                marker=dict(size=node_size, color=node_color, colorscale='Viridis', showscale=True, colorbar=dict(title="Lines of Code")),
                text=node_text, hoverinfo='text'
            )
        ],
        'layout': go.Layout(
            title='Code Dependency Network',
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False)
        )
    }

# 4. Dash app
app = dash.Dash(__name__)
app.title = "Polymetric View"
app.layout = html.Div([
    html.H2("Project Dependency Visualizer"),
    html.Div([
        dcc.Input(id='folder-input', type='text', placeholder='Enter path (e.g., /workspaces/api)', style={'width': '60%'}),
        html.Button('Load', id='load-button', n_clicks=0),
    ], style={'margin': '10px'}),
    dcc.Graph(id='graph'),
    html.H4("Module Metrics"),
    dash_table.DataTable(
        id='metrics-table',
        columns=[
            {"name": "File", "id": "file"},
            {"name": "Lines of Code", "id": "loc"},
            {"name": "Num Imports", "id": "imports"},
            {"name": "Used By (Dependencies)", "id": "dependents"},
        ],
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'left'},
    )
])

@app.callback(
    [Output('graph', 'figure'),
     Output('metrics-table', 'data')],
    [Input('load-button', 'n_clicks')],
    [State('folder-input', 'value')]
)
def update_view(n_clicks, folder):
    if not folder or not os.path.isdir(folder):
        return go.Figure(), []

    dependencies = gather_dependencies(folder)
    G, metrics = create_dependency_graph(dependencies)
    fig = generate_figure(G, metrics)

    table_data = [{
        "file": k,
        "loc": v["LOC"],
        "imports": v["NumImports"],
        "dependents": v["DependsOnMe"]
    } for k, v in metrics.items()]

    return fig, table_data

if __name__ == '__main__':
    app.run(debug=True)
