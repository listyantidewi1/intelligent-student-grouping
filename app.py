import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import base64

# Function to generate synthetic data
def generate_synthetic_data(num_students):
    np.random.seed(42)

    student_ids = np.arange(1, num_students + 1)
    study_hours = np.random.uniform(1, 10, num_students)
    assignments_completed = np.random.randint(0, 10, num_students)
    exam_scores = 30 + 3 * study_hours + 2 * assignments_completed + np.random.normal(0, 5, num_students)

    # Simulating learning style (0: Visual, 1: Auditory, 2: Kinesthetic)
    learning_styles = np.random.choice([0, 1, 2], num_students)

    # Simulating Gardner's Multiple Intelligences
    linguistic = np.random.uniform(1, 10, num_students)
    logical_mathematical = np.random.uniform(1, 10, num_students)
    spatial = np.random.uniform(1, 10, num_students)
    musical = np.random.uniform(1, 10, num_students)
    bodily_kinesthetic = np.random.uniform(1, 10, num_students)
    interpersonal = np.random.uniform(1, 10, num_students)
    intrapersonal = np.random.uniform(1, 10, num_students)
    naturalistic = np.random.uniform(1, 10, num_students)
    existential = np.random.uniform(1, 10, num_students)

    data = pd.DataFrame({
        'Student_ID': student_ids,
        'Study_Hours': study_hours,
        'Assignments_Completed': assignments_completed,
        'Exam_Scores': exam_scores,
        'Learning_Style': learning_styles,
        'Linguistic': linguistic,
        'Logical_Mathematical': logical_mathematical,
        'Spatial': spatial,
        'Musical': musical,
        'Bodily_Kinesthetic': bodily_kinesthetic,
        'Interpersonal': interpersonal,
        'Intrapersonal': intrapersonal,
        'Naturalistic': naturalistic,
        'Existential': existential,
    })

    return data

# Generate synthetic educational data
num_students = 500
data = generate_synthetic_data(num_students)

# Standardize the features
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data[['Study_Hours', 'Assignments_Completed', 'Exam_Scores',
                                         'Learning_Style', 'Linguistic', 'Logical_Mathematical',
                                         'Spatial', 'Musical', 'Bodily_Kinesthetic',
                                         'Interpersonal', 'Intrapersonal', 'Naturalistic', 'Existential']])

# Experiment with different numbers of clusters
min_clusters = 2
max_clusters = 6

cluster_performance = []

for num_clusters in range(min_clusters, max_clusters + 1):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    data['Cluster'] = kmeans.fit_predict(scaled_data)
    
    # Analyze the average performance of each cluster
    cluster_avg_performance = data.groupby('Cluster')['Exam_Scores'].mean().mean()
    cluster_performance.append(cluster_avg_performance)

# Choose the optimal number of clusters based on the analysis
optimal_num_clusters = cluster_performance.index(max(cluster_performance)) + min_clusters

# Use the optimal number of clusters to create the final grouping
kmeans_final = KMeans(n_clusters=optimal_num_clusters, random_state=42)
data['Final_Cluster'] = kmeans_final.fit_predict(scaled_data)

# Save the final results to a CSV file
result_csv_path = 'final_grouping_results.csv'
data.to_csv(result_csv_path, index=False)

# Analyze the characteristics of each final cluster
cluster_characteristics = data.groupby('Final_Cluster').mean()

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

# Initialize Dash app
app = dash.Dash(__name__)

# Define layout of the app
app.layout = html.Div([
    html.H1("Smart Student Grouping"),
    
    # 3D Scatter plot for Final Grouping
    dcc.Graph(
        id='scatter-final-grouping-3d',
        figure=px.scatter_3d(data, x='Study_Hours', y='Exam_Scores', z='Learning_Style',
                             color='Final_Cluster', size_max=10,
                             title=f'3D Scatter Plot with Learning Style and Final Grouping',
                             labels={'Study_Hours': 'Study Hours', 'Exam_Scores': 'Exam Scores',
                                     'Learning_Style': 'Learning Style'}),
    ),

    # 3D Scatter plot with Multiple Intelligences
    dcc.Graph(
        id='scatter-multiple-intelligences',
        figure=px.scatter_3d(data, x='Linguistic', y='Logical_Mathematical', z='Spatial',
                             color='Final_Cluster', size_max=10,
                             title='3D Scatter Plot with Multiple Intelligences',
                             labels={'Linguistic': 'Linguistic', 'Logical_Mathematical': 'Logical Mathematical',
                                     'Spatial': 'Spatial', 'Musical': 'Musical',
                                     'Bodily_Kinesthetic': 'Bodily Kinesthetic', 'Interpersonal': 'Interpersonal',
                                     'Intrapersonal': 'Intrapersonal', 'Naturalistic': 'Naturalistic',
                                     'Existential': 'Existential'}),
    ),
    
    # Bar chart for Cluster Performance
    dcc.Graph(
        id='bar-cluster-performance',
        figure={
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
    ),
    
    # Table for displaying student data with pagination
    dash_table.DataTable(
        id='table',
        columns=[
            {'name': col, 'id': col} for col in data.columns
        ],
        page_size=10,  # Set the number of rows per page
        data=data.to_dict('records'),  # Load data directly from the DataFrame
    ),

    # Download link for CSV
    html.A('Download CSV', id='download-link', href='', download='final_grouping_results.csv', target='_blank'),
    
    # HTML component for displaying cluster descriptions
    html.Div([
        html.H3("Cluster Descriptions"),
        *[html.P(description) for description in cluster_descriptions]
    ]),
    
    # Visualizations for Cluster Homogeneity/Heterogeneity
    dcc.Graph(
        id='intra-cluster-similarity',
        figure={
            'data': [
                go.Bar(x=cluster_characteristics.index, 
                       y=data.groupby('Final_Cluster').std()['Exam_Scores'], 
                       marker=dict(color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']),  # Specify colors
                       name='Standard Deviation of Exam Scores')
            ],
            'layout': dict(title='Intra-Cluster Homogeneity/Heterogeneity',
                           xaxis=dict(title='Final Cluster'), yaxis=dict(title='Standard Deviation of Exam Scores'))
        }
    ),
    
    
    dcc.Graph(
        id='inter-cluster-similarity',
        figure={
            'data': [
                go.Bar(x=data.groupby('Final_Cluster').mean().index, 
                       y=cluster_performance, 
                       marker=dict(color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']),  # Specify colors
                       name='Average Exam Scores')
            ],
            'layout': dict(title='Inter-Cluster Homogeneity/Heterogeneity',
                           xaxis=dict(title='Final Cluster'), yaxis=dict(title='Average Exam Scores'))
        }
    ),
])
# Define callback for CSV download
@app.callback(
    Output('download-link', 'href'),
    [Input('scatter-final-grouping-3d', 'relayoutData')]  # Use an arbitrary input, so the callback is triggered on page load
)
def update_download_link(relayout_data):
    # Create a download link for the CSV file
    csv_buffer = BytesIO()
    data.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    csv_data = base64.b64encode(csv_buffer.read()).decode('utf-8')
    href_value = f'data:text/csv;base64,{csv_data}'
    return href_value

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
