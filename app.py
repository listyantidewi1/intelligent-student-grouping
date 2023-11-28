import dash
from dash import dcc
from dash import html
from dash import dash_table
from dash.dependencies import Input, Output, State
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import base64
from dash.exceptions import PreventUpdate
from dash.dash_table.Format import Group

'''
pip install -U scikit-learn
pip install -U pandas
pip install -U dash
'''

# Initialize Dash app
app = dash.Dash(__name__)

# Initialize with an empty DataFrame
data = pd.DataFrame()

# Define layout of the app
app.layout = html.Div([
    html.H1("Intelligent Student Grouping"),

    # Upload CSV file component
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        multiple=False
    ),

    # Output for uploaded data
    html.Div(id='uploaded-data'),

    # 3D Scatter plot for Final Grouping
    dcc.Graph(
        id='scatter-final-grouping-3d',
        figure={},
    ),

    # 3D Scatter plot with Multiple Intelligences
    dcc.Graph(
        id='scatter-multiple-intelligences',
        figure={},
    ),

    # Bar chart for Cluster Performance
    dcc.Graph(
        id='bar-cluster-performance',
        figure={},
    ),

    # Table for displaying student data with pagination
    dash_table.DataTable(
        id='table',
        columns=[
            {'name': col, 'id': col} for col in data.columns
        ],
        # page_size=10,  # Set the number of rows per page
        style_table={'height': '1px', 'overflowY': 'auto'},
        data=data.to_dict('records'),  # Load data directly from the DataFrame
    ),

    # Download link for CSV
    html.A('Download CSV', id='download-link', href='', download='final_grouping_results.csv', target='_blank'),

   # HTML component for displaying cluster descriptions
    html.Div(id='cluster-descriptions'),  # Placeholder for cluster descriptions

    # Visualizations for Cluster Homogeneity/Heterogeneity
    dcc.Graph(
        id='intra-cluster-similarity',
        figure={},
    ),

    dcc.Graph(
        id='inter-cluster-similarity',
        figure={},
    ),

    # # Accuracy Analysis
    # html.Div([
    #     html.H3("Accuracy Analysis"),
    #     html.P("Accuracy analysis results will be displayed here."),
    # ], id='accuracy-analysis'),
])

# Callback to process uploaded data
@app.callback(
    [Output('uploaded-data', 'children'),
     Output('scatter-final-grouping-3d', 'figure'),
     Output('scatter-multiple-intelligences', 'figure'),
     Output('bar-cluster-performance', 'figure'),
     Output('table', 'data'),
     Output('download-link', 'href'),
     Output('intra-cluster-similarity', 'figure'),
     Output('inter-cluster-similarity', 'figure')
     ],
    [Input('upload-data', 'contents')],
    [State('upload-data', 'filename')]
)

def update_uploaded_data(contents, filename):
    if contents is None:
        raise PreventUpdate

    print("Callback triggered!")

    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)

    # Read the uploaded CSV file into a DataFrame
    uploaded_data = pd.read_csv(BytesIO(decoded))

    # Process the uploaded data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(uploaded_data[['Study_Hours', 'Assignments_Completed', 'Exam_Scores',
                                                      'Learning_Style', 'Linguistic', 'Logical_Mathematical',
                                                      'Spatial', 'Musical', 'Bodily_Kinesthetic',
                                                      'Interpersonal', 'Intrapersonal', 'Naturalistic', 'Existential']])

    # Experiment with different numbers of clusters
    min_clusters = 2
    max_clusters = 6

    cluster_performance = []

    for num_clusters in range(min_clusters, max_clusters + 1):
        kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=42)
        uploaded_data['Cluster'] = kmeans.fit_predict(scaled_data)

        # Analyze the average performance of each cluster
        cluster_avg_performance = uploaded_data.groupby('Cluster')['Exam_Scores'].mean().mean()
        cluster_performance.append(cluster_avg_performance)

    # Choose the optimal number of clusters based on the analysis
    optimal_num_clusters = cluster_performance.index(max(cluster_performance)) + min_clusters

    # Use the optimal number of clusters to create the final grouping
    kmeans_final = KMeans(n_clusters=optimal_num_clusters, random_state=42)
    uploaded_data['Final_Cluster'] = kmeans_final.fit_predict(scaled_data)

    # Save the final results to a CSV file
    result_csv_path = 'final_grouping_results.csv'
    uploaded_data.to_csv(result_csv_path, index=False)

    # Read results from the CSV file
    data = pd.read_csv(result_csv_path)

    # Analyze the characteristics of each final cluster
    cluster_characteristics = uploaded_data.groupby('Final_Cluster').mean()

    # Generate descriptive text for each cluster
    cluster_descriptions = []
    for cluster_id, characteristics in cluster_characteristics.iterrows():
        description = f"Cluster {cluster_id} Characteristics:\n"
        description += f"  - Average Study Hours: {characteristics['Study_Hours']:.2f}\n"
        description += f"  - Average Assignments Completed: {characteristics['Assignments_Completed']:.2f}\n"
        description += f"  - Average Exam Scores: {characteristics['Exam_Scores']:.2f}\n"
        description += f"  - Dominant Learning Style: {characteristics['Learning_Style']:.0f}\n"

        # Determine the dominant intelligence
        dominant_intelligence = characteristics[['Linguistic', 'Logical_Mathematical', 'Spatial', 'Musical',
                                                 'Bodily_Kinesthetic', 'Interpersonal', 'Intrapersonal', 'Naturalistic',
                                                 'Existential']].idxmax()
        description += f"  - Dominant Intelligence: {dominant_intelligence}\n"

        cluster_descriptions.append(description)

    # Update the scatter plot
    scatter_final_grouping_3d = px.scatter_3d(uploaded_data, x='Study_Hours', y='Exam_Scores', z='Learning_Style',
                                              color='Final_Cluster', size_max=20,
                                              title=f'3D Scatter Plot with Learning Style and Final Grouping',
                                              labels={'Study_Hours': 'Study Hours', 'Exam_Scores': 'Exam Scores',
                                                      'Learning_Style': 'Learning Style'})

    # Update the scatter plot for multiple intelligences
    scatter_multiple_intelligences = px.scatter_3d(uploaded_data, x='Linguistic', y='Logical_Mathematical',
                                                   z='Spatial', color='Final_Cluster', size_max=20,
                                                   title='3D Scatter Plot with Multiple Intelligences',
                                                   labels={'Linguistic': 'Linguistic', 'Logical_Mathematical': 'Logical Mathematical',
                                                           'Spatial': 'Spatial', 'Musical': 'Musical',
                                                           'Bodily_Kinesthetic': 'Bodily Kinesthetic', 'Interpersonal': 'Interpersonal',
                                                           'Intrapersonal': 'Intrapersonal', 'Naturalistic': 'Naturalistic',
                                                           'Existential': 'Existential'})

    # Update the bar chart for cluster performance
    bar_cluster_performance = {
        'data': [
            go.Bar(x=list(range(min_clusters, max_clusters + 1)),
                   y=cluster_performance,
                   marker=dict(color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']),  # Specify colors
                   type='bar',
                   name='Average Exam Scores')
        ],
        'layout': dict(title='Average Exam Scores for Different Numbers of Clusters',
                       xaxis=dict(title='Number of Clusters'), yaxis=dict(title='Average Exam Scores'))
    }

    # Update the table data
    table_data = uploaded_data.to_dict('records')

    # Update the download link for CSV
    csv_buffer = BytesIO()
    uploaded_data.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    csv_data = base64.b64encode(csv_buffer.read()).decode('utf-8')
    download_link = f'data:text/csv;base64,{csv_data}'

    # Update the intra-cluster similarity bar chart
    intra_cluster_similarity = {
        'data': [
            go.Bar(x=cluster_characteristics.index,
                   y=uploaded_data.groupby('Final_Cluster').std()['Exam_Scores'],
                   marker=dict(color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']),  # Specify colors
                   name='Standard Deviation of Exam Scores')
        ],
        'layout': dict(title='Intra-Cluster Homogeneity/Heterogeneity',
                       xaxis=dict(title='Final Cluster'), yaxis=dict(title='Standard Deviation of Exam Scores'))
    }

    # Update the inter-cluster similarity bar chart
    inter_cluster_similarity = {
        'data': [
            go.Bar(x=uploaded_data.groupby('Final_Cluster').mean().index,
                   y=cluster_performance,
                   marker=dict(color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']),  # Specify colors
                   name='Average Exam Scores')
        ],
        'layout': dict(title='Inter-Cluster Homogeneity/Heterogeneity',
                       xaxis=dict(title='Final Cluster'), yaxis=dict(title='Average Exam Scores'))
    }

    # # Update the accuracy analysis
    # accuracy_analysis = html.Div([
    #     html.H4("Accuracy Analysis"),
    #     html.P("Accuracy analysis results will be displayed here."),
    #     # Add your accuracy analysis components here
    # ])

    return html.Div([
        html.H4(f'Uploaded Data: {filename}'),
        dash_table.DataTable(
            columns=[
                {'name': col, 'id': col} for col in uploaded_data.columns
            ],
            style_table={'height': '400px', 'overflowY': 'auto', 'overflowX':'auto'},
            data=uploaded_data.to_dict('records'),
            page_size=10
        )
    ]), scatter_final_grouping_3d, scatter_multiple_intelligences, bar_cluster_performance, table_data, download_link, intra_cluster_similarity, inter_cluster_similarity

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
