# Import necessary libraries
import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend for compatibility with Dash; prevents GUI dependencies for rendering
import dash
from dash import dcc, html  # Import Dash components for layout and interactivity
from dash.dependencies import Input, Output  # Import callback dependencies
import pandas as pd  # Library for data manipulation
import seaborn as sns  # Library for advanced visualization
import matplotlib.pyplot as plt  # Core plotting library
import numpy as np  # Library for numerical computations
import io  # Library for handling in-memory streams
import base64  # For encoding images into base64 strings
import pickle  # For loading pre-trained models and scalers

# Step 1: Load the dataset path for visualization
with open('file_path.txt', 'r') as file:
    data_path = file.readline().strip()

# Step 2: Read the main dataset used for visualizations
credit_df = pd.read_csv(data_path)

# Step 3: Load the preprocessed dataset path for ML predictions
with open('dummy_df_path.txt', 'r') as file:
    dummy_path = file.readline().strip()

# Step 4: Read the ML-ready dataset
dummy_df = pd.read_csv(dummy_path)

# Step 5: Add age group categorization to the visualization dataset
# Age bins and labels are defined for better data grouping
bins = [0, 32, 55, np.inf]
labels = ['20-32', '33-55', '56+']
credit_df['age_group'] = pd.cut(credit_df['person_age'], bins=bins, labels=labels)

# Step 6: Create a Dash application instance
app = dash.Dash(__name__)

# Step 7: Define custom color palette for the app's theme
colors = {
    'background': '#f4f4f4',  # Light grey background
    'text': '#333333',  # Dark text color
    'accent': '#1f77b4'  # Accent color for emphasis
}

# Step 8: Define the layout of the app
app.layout = html.Div(style={'backgroundColor': colors['background'], 'height': '120vh'}, children=[
    html.H1("CREDIT RISK ANALYSIS", style={'color': colors['text']}),  # Title of the dashboard

    # Dropdown filters for interactivity
    html.Div([
        # Filter for home ownership
        html.Div([
            html.Label("Select Home Ownership:", style={'color': colors['text']}), 
            dcc.Dropdown(
                id='home-ownership-dropdown',
                options=[{'label': ownership, 'value': ownership} for ownership in credit_df['person_home_ownership'].unique()],
                value=credit_df['person_home_ownership'].unique()[0],  # Default value
                style={'width': '40%', 'marginBottom': '20px'}
            ),
            # Filter for age group
            html.Label("Select Age Group:", style={'color': colors['text']}), 
            dcc.Dropdown(
                id='age-group-dropdown',
                options=[{'label': group, 'value': group} for group in credit_df['age_group'].unique()],
                value=credit_df['age_group'].unique()[0],  # Default value
                style={'width': '40%', 'marginBottom': '20px'}
            ),
        ], style={'width': '45%', 'display': 'inline-block'}),
        # Filter for loan grade
        html.Div([
            html.Label("Select Loan Grade:", style={'color': colors['text']}), 
            dcc.Dropdown(
                id='loan-grade-dropdown',
                options=[{'label': grade, 'value': grade} for grade in credit_df['loan_grade'].unique()],
                value=credit_df['loan_grade'].unique()[0],  # Default value
                style={'width': '40%', 'marginBottom': '20px'}
            ),
        ], style={'width': '45%', 'display': 'inline-block'})
    ]),

    # Containers for displaying visualizations
    html.Div([
        html.Div(id='loan-grades-chart-container', style={'width': '45%', 'height': '50vh', 'display': 'inline-block', 'backgroundColor': colors['background']}), 
        html.Div(id='loan-intents-chart-container', style={'width': '45%', 'height': '50vh', 'display': 'inline-block', 'backgroundColor': colors['background']})
    ], style={'width': '100%', 'display': 'flex', 'justifyContent': 'space-between'}),

    # Section for loan default prediction
    html.Div([
        html.H1("Loan Default Prediction", style={'textAlign': 'center', 'color': 'darkblue'}),  # Section title
        html.Div([
            # Input fields for prediction parameters
            html.Label("Loan Interest Rate (e.g., 0.05 for 5%)", style={'color': 'black'}),
            dcc.Input(id='input-interest-rate', type='number', step=0.01, placeholder='0.05', style={'width': '100%'}),
            html.Label("Loan Percent of Income (e.g., 20 for 20%)", style={'color': 'black'}),
            dcc.Input(id='input-percent-income', type='number', placeholder='20', style={'width': '50%'}),
            html.Label("Loan Amount", style={'color': 'black'}),
            dcc.Input(id='input-loan-amount', type='number', placeholder='10000', style={'width': '50%'}),
            # Prediction button
            html.Button('Predict', id='predict-button', n_clicks=0),
            # Output field for displaying prediction result
            html.Div(id='prediction-output', style={'color': 'black', 'marginTop': '20px'})
        ])
    ])
])

# Step 9: Load pre-trained logistic regression model and scaler
with open('logistic_model.pkl', 'rb') as model_file:
    logistic_model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Step 10: Define callback for updating loan grades chart
@app.callback(
    Output('loan-grades-chart-container', 'children'),
    [Input('home-ownership-dropdown', 'value'), Input('age-group-dropdown', 'value')]
)
def update_loan_grades_chart(selected_ownership, selected_age_group):
    # Filter dataset based on user selections
    filtered_df = credit_df[(credit_df['person_home_ownership'] == selected_ownership) & 
                            (credit_df['age_group'] == selected_age_group)]
    # Create a count plot of loan grades
    plt.figure(figsize=(6, 4))
    sns.countplot(data=filtered_df, x='loan_grade', palette='viridis')
    plt.title("Loan Grades by Home Ownership & Age Group")
    plt.xlabel("Loan Grade")
    plt.ylabel("Count")
    # Convert plot to base64 image
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    encoded_image = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    return html.Img(src=f"data:image/png;base64,{encoded_image}")

# Step 11: Define callback for loan default prediction
@app.callback(
    Output('prediction-output', 'children'),
    [Input('predict-button', 'n_clicks')],
    [Input('input-interest-rate', 'value'), Input('input-percent-income', 'value'), Input('input-loan-amount', 'value')]
)
def make_prediction(n_clicks, interest_rate, percent_income, loan_amount):
    if n_clicks > 0 and all(v is not None for v in [interest_rate, percent_income, loan_amount]):
        # Prepare input data for model prediction
        input_data = pd.DataFrame([[interest_rate, percent_income, loan_amount]], columns=['loan_int_rate', 'person_emp_length', 'loan_amnt'])
        scaled_data = scaler.transform(input_data)
        prediction = logistic_model.predict(scaled_data)
        # Return prediction result
        return f"Prediction: {'Default' if prediction[0] == 1 else 'No Default'}"
    return "Enter all inputs and click Predict."

# Step 12: Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
