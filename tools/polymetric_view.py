import os
import ast
import networkx as nx
import dash
import dash.html as html
import dash.dcc as dcc
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import pandas as pd
import numpy as np

def gather_dependencies(project_root):
    module_map = {}
    dependencies = {}

    # Map all Python files to modules
    for root, _, files in os.walk(project_root):
        for file in files:
            if file.endswith('.py'):
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, project_root)
                module_name = rel_path.split(os.sep)[0] if os.sep in rel_path else 'root'
                module_map[full_path] = module_name

    file_paths = set(module_map.keys())

    for file_path in file_paths:
        local_deps = set()

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                tree = ast.parse(f.read(), filename=file_path)
        except Exception:
            dependencies[file_path] = list(local_deps)
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.name.split('.')[0]
                    match = [f for f in file_paths if os.path.basename(f).startswith(name + '.')]
                    if match:
                        local_deps.update(match)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    name = node.module.split('.')[0]
                    match = [f for f in file_paths if os.path.basename(f).startswith(name + '.')]
                    if match:
                        local_deps.update(match)

        dependencies[file_path] = list(local_deps)

    return dependencies, module_map

def count_loc(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return sum(1 for line in f if line.strip() and not line.strip().startswith('#'))
    except Exception:
        return 0

def calculate_cyclomatic_complexity(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            tree = ast.parse(file.read(), filename=file_path)
    except Exception:
        return 0

    complexity = 1
    for node in ast.walk(tree):
        if isinstance(node, (ast.If, ast.For, ast.While, ast.Try, ast.With, ast.And, ast.Or, ast.ExceptHandler)):
            complexity += 1
    return complexity

def create_dependency_graph(dependencies):
    G = nx.DiGraph()
    for file, deps in dependencies.items():
        G.add_node(file, loc=count_loc(file))
        for dep in deps:
            G.add_node(dep)
            G.add_edge(file, dep)
    return G

def generate_dependency_figure(G, module_map, dependencies):
    module_files = {}
    for file_path, module in module_map.items():
        module_files.setdefault(module, []).append(file_path)

    module_positions = {module: (i * 5, 0) for i, module in enumerate(module_files)}
    file_positions = {}
    for module, files in module_files.items():
        base_x, base_y = module_positions[module]
        for file in files:
            dx, dy = np.random.uniform(-0.5, 0.5, 2)
            file_positions[file] = (base_x + dx, base_y + dy)

    for module in module_files:
        G.add_node(f"__module__{module}", is_module=True)

    for file, module in module_map.items():
        G.add_edge(f"__module__{module}", file)

    pos = {**file_positions, **{f"__module__{m}": p for m, p in module_positions.items()}}

    real_edges = go.Scatter(x=[], y=[], line=dict(width=0.7, color='gray'), mode='lines', hoverinfo='none')
    container_edges = go.Scatter(x=[], y=[], line=dict(width=1, color='lightgray', dash='dot'), mode='lines', hoverinfo='none')

    for src, dst in G.edges():
        if src.startswith("__module__"):
            trace = container_edges
        else:
            trace = real_edges

        if src in pos and dst in pos:
            x0, y0 = pos[src]
            x1, y1 = pos[dst]
            trace['x'] += (x0, x1, None)
            trace['y'] += (y0, y1, None)

    node_trace = go.Scatter(
        x=[], y=[], text=[], mode='markers', hoverinfo='text',
        marker=dict(size=15, color=[], colorscale='Viridis', showscale=True,
                    colorbar=dict(title="LOC"))
    )

    for node in G.nodes():
        if G.nodes[node].get("is_module") or node not in pos:
            continue
        x, y = pos[node]
        loc = G.nodes[node].get('loc', 0)
        node_trace['x'] += (x,)
        node_trace['y'] += (y,)
        node_trace['marker']['color'] += (loc,)
        label = os.path.basename(node)
        node_trace['text'] += (f"{label}<br>LOC: {loc}",)

    module_trace = go.Scatter(
        x=[], y=[], text=[], mode='markers+text', hoverinfo='text',
        marker=dict(size=60, color='lightgrey', line=dict(width=2, color='black')),
        textposition='middle center'
    )

    for module, (x, y) in module_positions.items():
        module_trace['x'] += (x,)
        module_trace['y'] += (y,)
        module_trace['text'] += (module,)

    fig = go.Figure(data=[real_edges, container_edges, node_trace, module_trace],
                    layout=go.Layout(
                        title="Python Dependency Graph",
                        showlegend=False,
                        hovermode='closest',
                        xaxis=dict(showgrid=False, zeroline=False),
                        yaxis=dict(showgrid=False, zeroline=False),
                        paper_bgcolor='#f7f7f7',
                        margin=dict(b=20, l=5, r=5, t=40)
                    ))

    return fig

def create_metrics_table(G, project_root, module_map):
    rows = []
    for node in G.nodes():
        if G.nodes[node].get("is_module"):
            continue
        loc = G.nodes[node].get('loc', 0)
        complexity = calculate_cyclomatic_complexity(node)
        coupling = G.out_degree(node) + G.in_degree(node)
        module = module_map.get(node, "Unknown")
        rows.append({
            "File": os.path.basename(node),
            "Module": module,
            "LOC": loc,
            "Coupling": coupling,
            "Cyclomatic Complexity": complexity
        })
    df = pd.DataFrame(rows)
    df.sort_values(by='Coupling', ascending=False, inplace=True)

    return html.Table([
        html.Thead(html.Tr([html.Th(col, style={'padding': '8px'}) for col in df.columns])),
        html.Tbody([
            html.Tr([html.Td(df.iloc[i][col], style={'padding': '8px'}) for col in df.columns])
            for i in range(len(df))
        ])
    ])

app = dash.Dash(__name__)
app.title = "Python Dependency Visualizer"

app.layout = html.Div([
    html.H1("Python Dependency Visualizer"),
    dcc.Input(id='folder-input', type='text', placeholder='Enter folder path', debounce=True),
    html.Button("Generate", id='generate-btn', n_clicks=0),
    html.Div(id='feedback'),
    dcc.Graph(id='dependency-graph'),
    html.H2("Code Metrics"),
    html.Div(id='metrics-table')
])

@app.callback(
    [Output('feedback', 'children'),
     Output('dependency-graph', 'figure'),
     Output('metrics-table', 'children')],
    [Input('generate-btn', 'n_clicks')],
    [State('folder-input', 'value')]
)

def update(n_clicks, folder):
    if not folder:
        return "Please provide a folder path.", {}, None
    try:
        deps, mod_map = gather_dependencies(folder)
        G = create_dependency_graph(deps)
        fig = generate_dependency_figure(G, mod_map, deps)
        metrics = create_metrics_table(G, folder, mod_map)
        return "", fig, metrics
    except Exception as e:
        return f"Error: {e}", {}, None

if __name__ == '__main__':
    app.run(debug=True)
