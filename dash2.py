# Import necessary libraries
import matplotlib
matplotlib.use('Agg')  # Optional: Set the backend of matplotlib to 'Agg' for compatibility with Dash
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
import pickle

# Read the file path for visualization data
with open('file_path.txt', 'r') as file:
    data_path = file.readline().strip()

# Load the visualization dataset
credit_df = pd.read_csv(data_path)

# Read the file path for preprocessed dummy data
with open('dummy_df_path.txt', 'r') as file:
    dummy_path = file.readline().strip()

# Load the ML-ready dataset
dummy_df = pd.read_csv(dummy_path)

# Define age groups
bins = [0, 32, 55, np.inf]
labels = ['20-32', '33-55', '56+']
credit_df['age_group'] = pd.cut(credit_df['person_age'], bins=bins, labels=labels)

# Instantiate the Dash app
app = dash.Dash(__name__)

# Define custom colors for the app
colors = {
    'background': '#f4f4f4',
    'text': '#333333',
    'accent': '#1f77b4'
}

# Define the layout of the app
app.layout = html.Div(style={'backgroundColor': colors['background'], 'height': '120vh'}, children=[
    html.H1("CREDIT RISK ANALYSIS", style={'color': colors['text']}),

    # Dropdown filters
    html.Div([
        html.Div([
            html.Label("Select Home Ownership:", style={'color': colors['text']}), 
            dcc.Dropdown(
                id='home-ownership-dropdown',
                options=[{'label': ownership, 'value': ownership} for ownership in credit_df['person_home_ownership'].unique()],
                value=credit_df['person_home_ownership'].unique()[0],
                style={'width': '40%', 'marginBottom': '20px'}
            ),
            html.Label("Select Age Group:", style={'color': colors['text']}), 
            dcc.Dropdown(
                id='age-group-dropdown',
                options=[{'label': group, 'value': group} for group in credit_df['age_group'].unique()],
                value=credit_df['age_group'].unique()[0],
                style={'width': '40%', 'marginBottom': '20px'}
            ),
        ], style={'width': '45%', 'display': 'inline-block'}),
        html.Div([
            html.Label("Select Loan Grade:", style={'color': colors['text']}), 
            dcc.Dropdown(
                id='loan-grade-dropdown',
                options=[{'label': grade, 'value': grade} for grade in credit_df['loan_grade'].unique()],
                value=credit_df['loan_grade'].unique()[0],
                style={'width': '40%', 'marginBottom': '20px'}
            ),
        ], style={'width': '45%', 'display': 'inline-block'})
    ]),

    # Containers for displaying charts
    html.Div([
        html.Div(id='loan-grades-chart-container', style={'width': '45%', 'height': '50vh', 'display': 'inline-block', 'backgroundColor': colors['background']}), 
        html.Div(id='loan-intents-chart-container', style={'width': '45%', 'height': '50vh', 'display': 'inline-block', 'backgroundColor': colors['background']})
    ], style={'width': '100%', 'display': 'flex', 'justifyContent': 'space-between'}),

    # Prediction section
    html.Div([
        html.H1("Loan Default Prediction", style={'textAlign': 'center', 'color': 'darkblue'}),
        html.Div([
            html.Label("Loan Interest Rate (e.g., 0.05 for 5%)", style={'color': 'black'}),
            dcc.Input(id='input-interest-rate', type='number', step=0.01, placeholder='0.05', style={'width': '100%'}),
            html.Label("Loan Percent of Income (e.g., 20 for 20%)", style={'color': 'black'}),
            dcc.Input(id='input-percent-income', type='number', placeholder='20', style={'width': '50%'}),
            html.Label("Loan Amount", style={'color': 'black'}),
            dcc.Input(id='input-loan-amount', type='number', placeholder='10000', style={'width': '50%'}),
            html.Button('Predict', id='predict-button', n_clicks=0),
            html.Div(id='prediction-output', style={'color': 'black', 'marginTop': '20px'})
        ])
    ])
])

# Load the trained logistic regression model and scaler from files
with open('logistic_model.pkl', 'rb') as model_file:
    logistic_model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Callback for updating loan grades chart
@app.callback(
    Output('loan-grades-chart-container', 'children'),
    [Input('home-ownership-dropdown', 'value'), Input('age-group-dropdown', 'value')]
)
def update_loan_grades_chart(selected_ownership, selected_age_group):
    filtered_df = credit_df[(credit_df['person_home_ownership'] == selected_ownership) & 
                            (credit_df['age_group'] == selected_age_group)]
    plt.figure(figsize=(6, 4))
    sns.countplot(data=filtered_df, x='loan_grade', palette='viridis')
    plt.title("Loan Grades by Home Ownership & Age Group")
    plt.xlabel("Loan Grade")
    plt.ylabel("Count")
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    encoded_image = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    return html.Img(src=f"data:image/png;base64,{encoded_image}")

# Callback for predictions
@app.callback(
    Output('prediction-output', 'children'),
    [Input('predict-button', 'n_clicks')],
    [Input('input-interest-rate', 'value'), Input('input-percent-income', 'value'), Input('input-loan-amount', 'value')]
)
def make_prediction(n_clicks, interest_rate, percent_income, loan_amount):
    if n_clicks > 0 and all(v is not None for v in [interest_rate, percent_income, loan_amount]):
        # Prepare data for prediction
        input_data = pd.DataFrame([[interest_rate, percent_income, loan_amount]], columns=['loan_int_rate', 'person_emp_length', 'loan_amnt'])
        scaled_data = scaler.transform(input_data)
        prediction = logistic_model.predict(scaled_data)
        return f"Prediction: {'Default' if prediction[0] == 1 else 'No Default'}"
    return "Enter all inputs and click Predict."

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
