# Standard library imports
import numpy as np

# Third-party imports
import dash
from dash import html, dcc, no_update
from dash.dependencies import Input, Output, State, ALL
import plotly.graph_objects as go
import torch

# Local imports
from models import get_model
from data import get_base_data, get_fine_tune_data, combine_datasets
from training import Trainer

# Set the default values
DEFAULT_HIDDEN_SIZE = 20
DEFAULT_BATCH_SIZE = 512
DEFAULT_FINE_TUNE_METHOD = 'full'
DEFAULT_TRAINING_DATA_SIZE = 10000
DEFAULT_FINE_TUNE_DATA_SIZE = 1000
DEFAULT_TRAINING_STEPS = 20
DEFAULT_LEARNING_RATE = 0.01
DEFAULT_N_LAYERS = 10

def create_data_plot(base_data, fine_tune_data, model):
    fig = go.Figure()
    
    if base_data is not None:
        # Convert tensors to numpy arrays if they're not already
        x_base_train = base_data['train']['x'].numpy() if isinstance(base_data['train']['x'], torch.Tensor) else base_data['train']['x']
        y_base_train = base_data['train']['y'].numpy() if isinstance(base_data['train']['y'], torch.Tensor) else base_data['train']['y']
        x_base_val = base_data['val']['x'].numpy() if isinstance(base_data['val']['x'], torch.Tensor) else base_data['val']['x']
        y_base_val = base_data['val']['y'].numpy() if isinstance(base_data['val']['y'], torch.Tensor) else base_data['val']['y']
        
        # Plot base training data
        fig.add_trace(go.Scatter(
            x=x_base_train.flatten(), 
            y=y_base_train.flatten(),
            mode='markers', 
            name='Base Training',
            marker=dict(color='blue', size=8, symbol='circle')
        ))
        
        # Plot base validation data
        fig.add_trace(go.Scatter(
            x=x_base_val.flatten(), 
            y=y_base_val.flatten(),
            mode='markers', 
            name='Base Validation',
            marker=dict(color='lightblue', size=12, symbol='star')
        ))
    
    if fine_tune_data is not None:
        # Convert tensors to numpy arrays if they're not already
        x_fine_train = fine_tune_data['train']['x'].numpy() if isinstance(fine_tune_data['train']['x'], torch.Tensor) else fine_tune_data['train']['x']
        y_fine_train = fine_tune_data['train']['y'].numpy() if isinstance(fine_tune_data['train']['y'], torch.Tensor) else fine_tune_data['train']['y']
        x_fine_val = fine_tune_data['val']['x'].numpy() if isinstance(fine_tune_data['val']['x'], torch.Tensor) else fine_tune_data['val']['x']
        y_fine_val = fine_tune_data['val']['y'].numpy() if isinstance(fine_tune_data['val']['y'], torch.Tensor) else fine_tune_data['val']['y']
        
        # Plot fine-tune training data
        fig.add_trace(go.Scatter(
            x=x_fine_train.flatten(), 
            y=y_fine_train.flatten(),
            mode='markers', 
            name='Fine-tune Training',
            marker=dict(color='red', size=8, symbol='circle')
        ))
        
        # Plot fine-tune validation data
        fig.add_trace(go.Scatter(
            x=x_fine_val.flatten(), 
            y=y_fine_val.flatten(),
            mode='markers', 
            name='Fine-tune Validation',
            marker=dict(color='lightcoral', size=12, symbol='star')
        ))
    
    if model is not None:
        # Add model predictions
        x_range = torch.linspace(0, 4, 200).reshape(-1, 1)
        with torch.no_grad():
            y_pred = model(x_range)
        
        fig.add_trace(go.Scatter(
            x=x_range.numpy().flatten(),
            y=y_pred.numpy().flatten(),
            mode='lines',
            name='Model Predictions',
            line=dict(color='black', width=2)
        ))
    
    fig.update_layout(
        title={
            'text': 'Training and Validation Data',
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title='x',
        yaxis_title='y',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig

def create_loss_plot(history):
    fig = go.Figure()
    
    if history['base']:
        fig.add_trace(go.Scatter(
            y=history['base']['train_loss'],
            name='Base Training Loss',
            line=dict(color='blue')
        ))
        fig.add_trace(go.Scatter(
            y=history['base']['val_loss'],
            name='Base Validation Loss',
            line=dict(color='blue', dash='dash')
        ))
    
    if history['fine_tune']:
        # Add vertical line to separate base training and fine-tuning
        if history['base']:
            fig.add_vline(x=len(history['base']['train_loss']) - 0.5, 
                         line_dash="dash", line_color="gray")
        
        fig.add_trace(go.Scatter(
            y=history['fine_tune']['train_loss'],
            name='Fine-tune Training Loss',
            line=dict(color='red')
        ))
        fig.add_trace(go.Scatter(
            y=history['fine_tune']['val_loss'],
            name='Fine-tune Validation Loss',
            line=dict(color='red', dash='dash')
        ))
    
    fig.update_layout(
        title={
            'text': 'Training and Validation Loss',
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title='Epoch',
        yaxis_title='Loss',
        yaxis_type='log',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig

app = dash.Dash(__name__)

# Initialize data at startup
initial_base_data = get_base_data(n_samples=DEFAULT_TRAINING_DATA_SIZE)
initial_fine_tune_data = get_fine_tune_data(n_samples=DEFAULT_FINE_TUNE_DATA_SIZE)

app.layout = html.Div([
    html.Div([
        dcc.Graph(id='data-plot', 
                 figure=create_data_plot(initial_base_data, initial_fine_tune_data, None)),
        dcc.Graph(id='loss-plot',
                 figure=create_loss_plot({'base': [], 'fine_tune': []})),
    ], style={'width': '60%', 'display': 'inline-block'}),
    
    html.Div([
        html.H4('Model Configuration'),
        
        html.Label('Hidden Layer Size:'),
        dcc.Dropdown(
            id='hidden-size',
            options=[
                {'label': '10 neurons', 'value': 10},
                {'label': '20 neurons', 'value': 20},
                {'label': '40 neurons', 'value': 40}
            ],
            value=DEFAULT_HIDDEN_SIZE
        ),
        
        html.Label('Number of Hidden Layers:'),
        dcc.Dropdown(
            id='n-layers',
            options=[
                {'label': '2 layers', 'value': 2},
                {'label': '10 layers', 'value': 10},
                {'label': '20 layers', 'value': 20},
                {'label': '30 layers', 'value': 30}
            ],
            value=DEFAULT_N_LAYERS
        ),
        
        html.Label('Batch Size:'),
        dcc.Dropdown(
            id='batch-size',
            options=[
                {'label': '64', 'value': 64},
                {'label': '128', 'value': 128},
                {'label': '256', 'value': 256},
                {'label': '512', 'value': 512},
                {'label': '1024', 'value': 1024},
                {'label': '2048', 'value': 2048},
                {'label': '4096', 'value': 4096}
            ],
            value=DEFAULT_BATCH_SIZE
        ),
        
        html.Label('Fine-tuning Method:'),
        dcc.Dropdown(
            id='fine-tune-method',
            options=[
                {'label': 'Full Fine-tuning', 'value': 'full'},
                {'label': 'Adapter', 'value': 'adapter'},
                {'label': 'LoRA', 'value': 'lora'}
            ],
            value=DEFAULT_FINE_TUNE_METHOD
        ),
        
        html.Label('Training Steps:'),
        dcc.Dropdown(
            id='training-steps',
            options=[
                {'label': '1 step', 'value': 1},
                {'label': '10 steps', 'value': 10},
                {'label': '20 steps', 'value': 20},
                {'label': '50 steps', 'value': 50},
                {'label': '100 steps', 'value': 100},
                {'label': '500 steps', 'value': 500}
            ],
            value=DEFAULT_TRAINING_STEPS
        ),
        
        html.Label('Learning Rate:'),
        dcc.Dropdown(
            id='learning-rate',
            options=[
                {'label': '0.1', 'value': 0.1},
                {'label': '0.01', 'value': 0.01},
                {'label': '0.001', 'value': 0.001},
                {'label': '0.0001', 'value': 0.0001}
            ],
            value=DEFAULT_LEARNING_RATE
        ),
        
        html.Div([
            html.Button('Train Base Model', id='train-base-button', n_clicks=0),
            html.Button('Fine-tune', id='fine-tune-button', n_clicks=0, disabled=True),
            html.Button('Reset Training', id='reset-button', n_clicks=0)
        ], style={'marginTop': '20px'}),
        
        html.Div([
            html.H4('Training Status'),
            html.Pre(id='status-display', 
                    style={
                        'border': '1px solid #ddd',
                        'padding': '10px',
                        'backgroundColor': '#f8f9fa',
                        'fontFamily': 'monospace',
                        'whiteSpace': 'pre-wrap',
                        'fontSize': '12px'
                    })
        ])
    ], style={'width': '35%', 'float': 'right', 'padding': '20px'})
])

# Initialize global state
current_model = None
base_data = None
fine_tune_data = None
training_history = {'base': [], 'fine_tune': []}

@app.callback(
    Output('train-base-button', 'n_clicks'),
    Output('fine-tune-button', 'n_clicks'),
    Output('fine-tune-button', 'disabled'),
    Output('data-plot', 'figure'),
    Output('loss-plot', 'figure'),
    Output('status-display', 'children'),
    Input('hidden-size', 'value'),
    Input('n-layers', 'value'),
    Input('train-base-button', 'n_clicks'),
    Input('fine-tune-button', 'n_clicks'),
    Input('reset-button', 'n_clicks'),
    State('batch-size', 'value'),
    State('fine-tune-method', 'value'),
    State('training-steps', 'value'),
    State('learning-rate', 'value'),
    prevent_initial_call=True
)
def update_model(hidden_size, n_layers, base_clicks, fine_tune_clicks, 
                reset_clicks, batch_size, fine_tune_method, training_steps,
                learning_rate):
    global current_model, base_data, fine_tune_data, training_history
    
    ctx = dash.callback_context
    if not ctx.triggered:
        return 0, 0, True, create_data_plot(base_data, fine_tune_data, None), \
               create_loss_plot({'base': [], 'fine_tune': []}), \
               "Click 'Train Base Model' to start"
    
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Initialize data if not already done
    if base_data is None:
        base_data = get_base_data(n_samples=DEFAULT_TRAINING_DATA_SIZE)
        fine_tune_data = get_fine_tune_data(n_samples=DEFAULT_FINE_TUNE_DATA_SIZE)
    
    if trigger_id == 'reset-button':
        # Reset everything except the data
        current_model = None
        training_history = {'base': [], 'fine_tune': []}
        return 0, 0, True, create_data_plot(base_data, fine_tune_data, None), \
               create_loss_plot({'base': [], 'fine_tune': []}), \
               "Training reset. Click 'Train Base Model' to start"
    
    if trigger_id in ['hidden-size', 'n-layers']:  # Reset on either parameter change
        # Reset everything when architecture changes
        current_model = None
        training_history = {'base': [], 'fine_tune': []}
        return 0, 0, True, create_data_plot(base_data, fine_tune_data, None), \
               create_loss_plot({'base': [], 'fine_tune': []}), \
               "Model reset. Click 'Train Base Model' to start"
    
    elif trigger_id == 'train-base-button':
        if current_model is None:
            # Initialize new model if none exists
            current_model = get_model(hidden_size, fine_tune_method, n_layers)
            training_history['base'] = []
        
        # Continue training the existing model
        trainer = Trainer(current_model, fine_tune_method, learning_rate=learning_rate)
        new_history = trainer.train(base_data, batch_size=batch_size, epochs=training_steps)
        
        # Append new training history to existing history
        if not training_history['base']:
            training_history['base'] = new_history
        else:
            training_history['base']['train_loss'].extend(new_history['train_loss'])
            training_history['base']['val_loss'].extend(new_history['val_loss'])
            
        status = "Base model training continued"
        enable_fine_tune = False
        
    elif trigger_id == 'fine-tune-button':
        if current_model is None:
            return no_update
        
        # Fine-tune the model using only fine-tune data
        trainer = Trainer(current_model, fine_tune_method, learning_rate=learning_rate)
        new_history = trainer.train(fine_tune_data, batch_size=batch_size, epochs=training_steps)
        
        # Append new training history to existing history
        if not training_history['fine_tune']:
            training_history['fine_tune'] = new_history
        else:
            training_history['fine_tune']['train_loss'].extend(new_history['train_loss'])
            training_history['fine_tune']['val_loss'].extend(new_history['val_loss'])
            
        status = f"Model fine-tuned on new data using {fine_tune_method}"
        enable_fine_tune = False
    
    # Create figures
    data_fig = create_data_plot(base_data, fine_tune_data, current_model)
    loss_fig = create_loss_plot(training_history)
    
    # Always enable fine-tune button if we have a trained model
    should_disable_fine_tune = current_model is None
    
    return base_clicks, fine_tune_clicks, should_disable_fine_tune, data_fig, loss_fig, status

if __name__ == '__main__':
    app.run_server(debug=True) 