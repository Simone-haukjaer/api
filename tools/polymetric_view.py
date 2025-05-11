import os
import re
import ast
import networkx as nx
import dash
import dash.html as html
import dash.dcc as dcc
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from dash.dependencies import Input, Output, State
import warnings
warnings.simplefilter("error", SyntaxWarning)

def gather_dependencies(project_root):
    dependencies = {}
    module_map = {}

    for root, dirs, files in os.walk(project_root):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, project_root)
                parts = relative_path.split(os.sep)
                module_name = parts[0] if len(parts) > 1 else "root"
                module_map[file_path] = module_name

    for file_path, module_name in module_map.items():
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        imports = set()
        try:
            import_statements = re.findall(r"^\s*(?:import|from)\s+([a-zA-Z0-9_\.]+)", content, re.MULTILINE)
        except re.error as e:
            print(f"Regex error in file {file_path}: {e}")
            import_statements = []

        for imp in import_statements:
            for mod_name, mod_path in module_map.items():
                if imp == mod_name or imp.startswith(mod_name + "."):
                    imports.add(mod_path)
                    break

        dependencies[file_path] = list(imports)

    return dependencies, module_map

def create_dependency_graph(dependencies):
    G = nx.DiGraph()
    for file, deps in dependencies.items():
        G.add_node(file, loc=count_loc(file))
        for dep in deps:
            G.add_node(dep)
            G.add_edge(file, dep)
    return G

def count_loc(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return sum(1 for line in f if line.strip() and not line.strip().startswith('#'))
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return 0

def generate_dependency_figure(G, module_map):
    module_files = {}
    for file_path, module in module_map.items():
        module_files.setdefault(module, []).append(file_path)

    module_positions = {}
    spacing = 5
    for i, module in enumerate(module_files):
        module_positions[module] = (spacing * i, 0)

    file_positions = {}
    offset_range = 0.4
    for module, files in module_files.items():
        mod_x, mod_y = module_positions[module]
        for file in files:
            dx, dy = np.random.uniform(-offset_range, offset_range, 2)
            file_positions[file] = (mod_x + dx, mod_y + dy)

    for module, pos in module_positions.items():
        mod_node = f"__module__{module}"
        G.add_node(mod_node, is_module=True, loc=0)

        for file in module_files[module]:
            G.add_edge(mod_node, file)

    pos = {**file_positions, **{f"__module__{m}": p for m, p in module_positions.items()}}

    edge_trace = go.Scatter(x=[], y=[], line=dict(width=0.5, color='gray'),
                            hoverinfo='none', mode='lines')
    for src, dst in G.edges():
        if src in pos and dst in pos:
            x0, y0 = pos[src]
            x1, y1 = pos[dst]
            edge_trace['x'] += (x0, x1, None)
            edge_trace['y'] += (y0, y1, None)

    node_trace = go.Scatter(
        x=[], y=[], text=[], mode='markers', hoverinfo='text',
        marker=dict(showscale=True, colorscale='Viridis', size=15,
                    color=[], colorbar=dict(thickness=15, title=dict(text='Lines of Code'), xanchor='left'))
    )

    for node in G.nodes():
        if G.nodes[node].get("is_module"):
            continue
        if node not in pos:
            continue
        x, y = pos[node]
        loc = G.nodes[node].get('loc', 0)
        node_trace['x'] += (x,)
        node_trace['y'] += (y,)
        node_trace['marker']['color'] += (loc,)
        node_trace['text'] += (f"{os.path.basename(node)}\nLOC: {loc}",)  # Show LOC on hover

    module_trace = go.Scatter(
        x=[], y=[], text=[], mode='markers+text', textposition='top center', hoverinfo='text',
        marker=dict(size=60, color='lightgrey', line=dict(width=2, color='black'))
    )

    for module, (mod_x, mod_y) in module_positions.items():
        module_trace['x'] += (mod_x,)
        module_trace['y'] += (mod_y,)
        module_trace['text'] += (module,)

    fig = go.Figure(data=[edge_trace, node_trace, module_trace],
                    layout=go.Layout(
                        title=dict(text='Python File Dependency Graph', font=dict(size=20)),
                        showlegend=False, hovermode='closest', margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False), yaxis=dict(showgrid=False, zeroline=False),
                        paper_bgcolor='#edf1f5'
                    ))

    return fig

def create_metrics_table(G, project_root, module_map):
    data = []
    for node in G.nodes():
        if G.nodes[node].get("is_module"):
            continue
        name = os.path.basename(node)
        loc = G.nodes[node].get('loc', 0)
        num_imports = G.out_degree(node)  # Outgoing Coupling
        used_by = G.in_degree(node)  # Incoming Coupling
        coupling = num_imports + used_by  # Combined Coupling (incoming + outgoing)
        cyclomatic_complexity = calculate_cyclomatic_complexity(node)
        module_name = module_map.get(node, "Unknown")
        data.append({
            "File": name,
            "Module": module_name,
            "LOC": loc,  
            "Coupling": coupling,  
            "Cyclomatic Complexity": cyclomatic_complexity
        })

    df = pd.DataFrame(data)
    df.sort_values(by=['Module', 'LOC'], ascending=[True, False], inplace=True)

    return html.Table([  
        html.Thead(html.Tr([html.Th(col, style={'padding': '10px'}) for col in df.columns])),
        html.Tbody([html.Tr([html.Td(df.iloc[i][col], style={'padding': '10px'}) for col in df.columns]) for i in range(len(df))])
    ])

def calculate_cyclomatic_complexity(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            source = file.read()
            tree = ast.parse(source, filename=file_path)
    except Exception:
        return 0

    complexity = 1
    for node in ast.walk(tree):
        if isinstance(node, (ast.If, ast.For, ast.While, ast.Try, ast.With, ast.And, ast.Or, ast.ExceptHandler)):
            complexity += 1
    return complexity

app = dash.Dash(__name__)
app.title = "Python Dependency Visualizer"

app.layout = html.Div([
    html.H1("Python Dependency Visualizer"),
    dcc.Input(id='folder-input', type='text', placeholder='Enter project folder path', debounce=True),
    html.Button("Generate", id='generate-btn', n_clicks=0),
    html.Div(id='feedback'),
    dcc.Graph(id='dependency-graph'),
    html.H2("Metrics Table"),
    html.Div(id='metrics-table')
])

@app.callback(
    [Output('feedback', 'children'),
     Output('dependency-graph', 'figure'),
     Output('metrics-table', 'children')],
    [Input('generate-btn', 'n_clicks')],
    [State('folder-input', 'value')]
)

def update_graph(n_clicks, folder_path):
    if not folder_path:
        return "Please provide a valid project folder path.", {}, None

    try:
        dependencies, module_map = gather_dependencies(folder_path)
        G = create_dependency_graph(dependencies)
        fig = generate_dependency_figure(G, module_map)
        metrics_table = create_metrics_table(G, folder_path, module_map)
        return "", fig, metrics_table
    except Exception as e:
        return f"An error occurred: {e}", {}, None

if __name__ == '__main__':
    app.run(debug=True)
