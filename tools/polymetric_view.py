import os
import re
import ast
import networkx as nx
import dash
import dash.html as html
import dash.dcc as dcc
import plotly.graph_objs as go
import pandas as pd
from dash.dependencies import Input, Output, State
import warnings
warnings.simplefilter("error", SyntaxWarning)

# Step 1: Gather file dependencies
def gather_dependencies(project_root):
    dependencies = {}
    module_map = {}

    # Walk through the directory and find Python files
    for root, dirs, files in os.walk(project_root):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)

                # Get the relative path of the file
                relative_path = os.path.relpath(file_path, project_root)

                # Split the relative path to get the directory and file
                parts = relative_path.split(os.sep)

                # If the length of parts is greater than 1, this means it's inside a subfolder
                if len(parts) > 1:
                    # The immediate parent folder is the module name
                    module_name = parts[0]  # This will give you the folder name as the module
                else:
                    # If it's a root file, classify it as 'root'
                    module_name = "root"

                # Store the file path with its corresponding module name
                module_map[file_path] = module_name

    # Read through each Python file and extract import statements
    for file_path, module_name in module_map.items():
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        imports = set()

        try:
            import_statements = re.findall(
                r"^\s*(?:import|from)\s+([a-zA-Z0-9_\.]+)",
                content,
                re.MULTILINE
            )
        except re.error as e:
            print(f"Regex error in file {file_path}: {e}")
            import_statements = []

        # Map the found imports to the files in the module_map
        for imp in import_statements:
            for mod_name, mod_path in module_map.items():
                # If the import statement matches any known module path, add the file to dependencies
                if imp == mod_name or imp.startswith(mod_name + "."):
                    imports.add(mod_path)
                    break

        dependencies[file_path] = list(imports)

    return dependencies, module_map

# Step 2: Create graph
def create_dependency_graph(dependencies):
    G = nx.DiGraph()

    # Add nodes and dependencies as edges in the graph
    for file, deps in dependencies.items():
        G.add_node(file, loc=count_loc(file))
        for dep in deps:
            G.add_edge(file, dep)

    return G

# Count lines of code (LOC)
def count_loc(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return sum(1 for line in f if line.strip() and not line.strip().startswith('#'))
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return 0

def generate_dependency_figure(G, module_map):
    pos = nx.spring_layout(G, seed=42)

    edge_shapes = []
    edge_trace = go.Scatter(
        x=[],
        y=[],
        line=dict(width=0.5, color='gray'),
        hoverinfo='none',
        mode='lines'
    )

    # Draw edges and arrow shapes
    for src, dst in G.edges():
        x0, y0 = pos[src]
        x1, y1 = pos[dst]

        edge_trace['x'] += [x0, x1, None]
        edge_trace['y'] += [y0, y1, None]

        # Add arrow as shape
        edge_shapes.append(
            dict(
                type="line",
                x0=x0,
                y0=y0,
                x1=x1,
                y1=y1,
                line=dict(color="black", width=1),
                opacity=1,
                layer="above",
                arrowhead=3,
                arrowsize=1,
            )
        )

    node_trace = go.Scatter(
        x=[],
        y=[],
        text=[],
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='Viridis',
            size=15,
            color=[],
            colorbar=dict(
                thickness=15,
                title=dict(text='Lines of Code'),
                xanchor='left'
            ),
        )
    )

    x_vals = []
    y_vals = []
    colors = []
    texts = []

    for node in G.nodes():
        x, y = pos[node]
        x_vals.append(x)
        y_vals.append(y)
        loc = G.nodes[node].get('loc', 0)  # Retrieve LOC (Lines of Code)
        colors.append(loc)

        # Get the module name for the node from the module map
        module_name = module_map.get(node, "Unknown")
        texts.append(f"{os.path.basename(node)}\nModule: {module_name}")

    node_trace['x'] = x_vals
    node_trace['y'] = y_vals
    node_trace['marker']['color'] = colors
    node_trace['text'] = texts

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=dict(text='Python File Dependency Graph', font=dict(size=20)),
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False),
            shapes=edge_shapes
        )
    )

    return fig

def calculate_cyclomatic_complexity(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            source = file.read()
            try:
                tree = ast.parse(source, filename=file_path)
            except (SyntaxError, SyntaxWarning) as e:
                print(f"Syntax issue in {file_path}: {e}")
                return 0
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return 0

    complexity = 1
    for node in ast.walk(tree):
        if isinstance(node, (ast.If, ast.For, ast.While, ast.Try, ast.With, ast.And, ast.Or, ast.ExceptHandler)):
            complexity += 1
    return complexity

def create_metrics_table(G, project_root, module_map):
    data = []

    for node in G.nodes():
        name = os.path.basename(node)
        loc = G.nodes[node].get('loc', 0)  # Cohesion = Lines of Code
        num_imports = G.out_degree(node)  # Outgoing Coupling (dependencies this file imports)
        used_by = G.in_degree(node)  # Incoming Coupling (how many files depend on this file)

        # Cyclomatic Complexity calculation (Placeholder logic)
        cyclomatic_complexity = calculate_cyclomatic_complexity(node)

        # Get the module name for the node
        module_name = module_map.get(node, "Unknown")

        data.append({
            "File": name,
            "Module": module_name,
            "Cohesion (LOC)": loc,
            "Outgoing Coupling": num_imports,
            "Incoming Coupling": used_by,
            "Cyclomatic Complexity": cyclomatic_complexity
        })

    df = pd.DataFrame(data)
    df.sort_values(by=['Module', 'Cohesion (LOC)'], ascending=[True, False], inplace=True)
    return html.Table([ 
        html.Thead(html.Tr([html.Th(col) for col in df.columns])),
        html.Tbody([ 
            html.Tr([html.Td(df.iloc[i][col]) for col in df.columns]) 
            for i in range(len(df)) 
        ])
    ])

# Step 5: Dash app layout
app = dash.Dash(__name__)
app.title = "Python Dependency Visualizer"

app.layout = html.Div([html.H1("Python Dependency Visualizer"),
                       dcc.Input(id='folder-input', type='text', placeholder='Enter project folder path', debounce=True),
                       html.Button("Generate", id='generate-btn', n_clicks=0),
                       html.Div(id='feedback'),
                       dcc.Graph(id='dependency-graph'),
                       html.H2("Metrics Table"),
                       html.Div(id='metrics-table')])

# Step 6: Callback logic
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
        # Step 1: Gather dependencies
        dependencies, module_map = gather_dependencies(folder_path)

        # Step 2: Create graph
        G = create_dependency_graph(dependencies)

        # Step 3: Generate figure
        fig = generate_dependency_figure(G, module_map)

        # Step 4: Create metrics table
        metrics_table = create_metrics_table(G, folder_path, module_map)

        return "", fig, metrics_table

    except Exception as e:
        return f"An error occurred: {e}", {}, None

if __name__ == '__main__':
    app.run(debug=True)
