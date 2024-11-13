# Import necessary libraries
import matplotlib
matplotlib.use('Agg')  # Set the backend of matplotlib to 'Agg' for compatibility with Dash
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import io
import base64

# Load the dataset
file_path = r'C:\Users\USER\Desktop\banford\barry\credit_ risk_dataset\credit_risk_dataset.csv'
credit_risk_df = pd.read_csv(file_path)

# Define age groups
# Age groups are defined using bins and labels
bins = [0, 25, 35, 45, 55, 65, np.inf]  # Define the age bins
labels = ['<25', '25-34', '35-44', '45-54', '55-64', '65+']  # Define the labels for each age group
credit_risk_df['age_group'] = pd.cut(credit_risk_df['person_age'], bins=bins, labels=labels)  # Assign each individual to an age group

# Instantiate the Dash app
app = dash.Dash(__name__)

# Define custom colors for the app
colors = {
    'background': '#f4f4f4',
    'text': '#333333',
    'accent': '#1f77b4'
}

# Define the layout of the app
app.layout = html.Div(style={'backgroundColor': colors['background'], 'height': '100vh'}, children=[
    html.H1("CREDIT RISK ANALYSIS", style={'color': colors['text']}),  # Title of the dashboard
    
    # Dropdowns for selecting filters
    html.Div([
        html.Div([
            html.Label("Select Home Ownership:", style={'color': colors['text']}),
            dcc.Dropdown(
                id='home-ownership-dropdown',
                options=[{'label': ownership, 'value': ownership} for ownership in credit_risk_df['person_home_ownership'].unique()],
                value=credit_risk_df['person_home_ownership'].unique()[0],
                style={'width': '50%', 'marginBottom': '20px'}  # Reduced width to 50%
            ),
            html.Label("Select Age Group:", style={'color': colors['text']}),
            dcc.Dropdown(
                id='age-group-dropdown',
                options=[{'label': group, 'value': group} for group in credit_risk_df['age_group'].unique()],
                value=credit_risk_df['age_group'].unique()[0],
                style={'width': '50%', 'marginBottom': '20px'}  # Reduced width to 50%
            ),
        ], style={'width': '45%', 'display': 'inline-block'}),
        
        html.Div([
            html.Label("Select Loan Intent:", style={'color': colors['text']}),
            dcc.Dropdown(
                id='loan-intents-dropdown',
                options=[{'label': intent, 'value': intent} for intent in credit_risk_df['loan_intent'].unique()],
                value=credit_risk_df['loan_intent'].unique()[0],
                style={'width': '50%', 'marginBottom': '20px'}  # Reduced width to 50%
            ),
        ], style={'width': '45%', 'display': 'inline-block'})
    ]),
    
    # Containers for displaying charts
    html.Div([
        html.Div(id='loan-intents-chart-container', style={'width': '45%', 'height': '50vh', 'display': 'inline-block', 'backgroundColor': colors['background']}),
        html.Div(id='loan-grade-chart-container', style={'width': '45%', 'height': '50vh', 'display': 'inline-block', 'backgroundColor': colors['background']})
    ], style={'width': '100%', 'display': 'flex', 'justifyContent': 'space-between'})
])

# Set seaborn plot background color and adjust plot margins
sns.set(style="whitegrid", rc={"axes.facecolor": colors['background'], 'figure.figsize':(8,6)})

# Callback for updating the loan intents chart based on selected home ownership and age group
@app.callback(
    Output('loan-intents-chart-container', 'children'),
    [Input('home-ownership-dropdown', 'value'),
     Input('age-group-dropdown', 'value')]
)
def update_loan_intents_chart(selected_ownership, selected_age_group):
    # Filter the dataset based on selected filters
    if selected_ownership and selected_age_group:
        filtered_data = credit_risk_df[(credit_risk_df['person_home_ownership'] == selected_ownership) & 
                                       (credit_risk_df['age_group'] == selected_age_group)]
    elif selected_ownership:
        filtered_data = credit_risk_df[credit_risk_df['person_home_ownership'] == selected_ownership]
    elif selected_age_group:
        filtered_data = credit_risk_df[credit_risk_df['age_group'] == selected_age_group]
    else:
        filtered_data = credit_risk_df
    
    # Check if filtered dataset is empty
    if filtered_data.empty:
        return html.Div("No data available for the selected criteria.")
    
    # Create a countplot of loan intents
    plt.figure(facecolor=colors['background'])
    ax = sns.countplot(x='loan_intent', data=filtered_data, palette='viridis', order=filtered_data['loan_intent'].value_counts().index.sort_values())
    ax.set_title('Loan Intents', color=colors['text'], fontsize=16)
    ax.set_xlabel('Loan Intent', color=colors['text'], fontsize=10)
    ax.set_ylabel('Count', color=colors['text'], fontsize=10)
    ax.tick_params(axis='x', colors=colors['text'], labelrotation=18)  # Rotate x-axis labels by 18 degrees
    ax.tick_params(axis='y', colors=colors['text'])
    plt.tight_layout()
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    
    # Convert the plot to base64 format to display in Dash
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    # Return the image as a Dash component
    return html.Img(src='data:image/png;base64,{}'.format(img_base64), style={'width': '100%'})

# Callback for updating the loan grade chart based on selected loan intent
@app.callback(
    Output('loan-grade-chart-container', 'children'),
    [Input('loan-intents-dropdown', 'value')]
)
def update_loan_grade_chart(selected_loan_intent):
    # Check if any loan intent is selected
    if selected_loan_intent is None:
        filtered_data = credit_risk_df  # If no loan intent is selected, show all data
        title = 'Loan Grades'  # Set title for the chart
    else:
        filtered_data = credit_risk_df[credit_risk_df['loan_intent'] == selected_loan_intent]  # Filter data based on selected loan intent
        title = f'Loan Grades for Loan Intent "{selected_loan_intent}"'  # Set title for the chart
    
    # Check if filtered dataset is empty
    if filtered_data.empty:
        return html.Div("No data available for the selected loan intent.")
    
    # Create a countplot of loan grades
    plt.figure(facecolor=colors['background'])
    ax = sns.countplot(x='loan_grade', data=filtered_data, palette='viridis', order=filtered_data['loan_grade'].value_counts().index.sort_values())
    ax.set_title(title, color=colors['text'], fontsize=16)
    ax.set_xlabel('Loan Grade', color=colors['text'], fontsize=10)
    ax.set_ylabel('Count', color=colors['text'], fontsize=10)
    ax.tick_params(axis='x', colors=colors['text'], labelrotation=45)  # Rotate x-axis labels by 45 degrees
    ax.tick_params(axis='y', colors=colors['text'])
    plt.tight_layout()
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    
    # Convert the plot to base64 format to display in Dash
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    # Return the image as a Dash component
    return html.Img(src='data:image/png;base64,{}'.format(img_base64), style={'width': '100%'})

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
