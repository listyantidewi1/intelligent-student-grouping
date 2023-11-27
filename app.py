import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash.dependencies import Input, Output
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px
from io import BytesIO
import base64
import numpy as np

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

# Initialize Dash app
app = dash.Dash(__name__)

# Define layout of the app
app.layout = html.Div([
    # Scatter plot for Final Grouping
    dcc.Graph(
        id='scatter-final-grouping',
        figure=px.scatter(data, x='Study_Hours', y='Exam_Scores', color='Final_Cluster',
                          title=f'Final Grouping of Students (Optimal Clusters: {optimal_num_clusters})',
                          labels={'Study_Hours': 'Study Hours', 'Exam_Scores': 'Exam Scores'}),
    ),
    
    # Bar chart for Cluster Performance
    dcc.Graph(
        id='bar-cluster-performance',
        figure={
            'data': [
                {'x': list(range(min_clusters, max_clusters + 1)), 'y': cluster_performance, 'type': 'bar', 'name': 'Average Exam Scores'}
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
    html.A('Download CSV', href='/download-csv'),
])

# Define callback for CSV download
@app.server.route('/download-csv')
def download_csv():
    csv_buffer = BytesIO()
    data.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    return base64.b64encode(csv_buffer.read()).decode('utf-8')

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
