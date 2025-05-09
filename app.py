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
from data import get_base_data, get_fine_tune_data, combine_datasets, combine_datasets_with_replacement
from training import Trainer

# Set the default values
DEFAULT_HIDDEN_SIZE = 20
DEFAULT_BATCH_SIZE = 512
DEFAULT_FINE_TUNE_METHOD = 'none'
DEFAULT_TRAINING_DATA_SIZE = 10000
DEFAULT_FINE_TUNE_DATA_SIZE = 1000
DEFAULT_TRAINING_STEPS = 50
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_N_LAYERS = 11
DEFAULT_FINE_TUNE_DATA_MODE = 'combined'

def create_data_plot(base_data, fine_tune_data, model, fine_tune_method='none'):
    fig = go.Figure()
    
    if base_data is not None:
        x_base_train = base_data['train']['x'].numpy() if isinstance(base_data['train']['x'], torch.Tensor) else base_data['train']['x']
        y_base_train = base_data['train']['y'].numpy() if isinstance(base_data['train']['y'], torch.Tensor) else base_data['train']['y']
        x_base_val = base_data['val']['x'].numpy() if isinstance(base_data['val']['x'], torch.Tensor) else base_data['val']['x']
        y_base_val = base_data['val']['y'].numpy() if isinstance(base_data['val']['y'], torch.Tensor) else base_data['val']['y']
        
        if fine_tune_method == 'none':
            # For 'none', show all blue points
            fig.add_trace(go.Scatter(
                x=x_base_train.flatten(), 
                y=y_base_train.flatten(),
                mode='markers', 
                name='Base Training',
                marker=dict(color='blue', size=8, symbol='circle')
            ))
            fig.add_trace(go.Scatter(
                x=x_base_val.flatten(), 
                y=y_base_val.flatten(),
                mode='markers', 
                name='Base Validation',
                marker=dict(color='lightblue', size=12, symbol='star')
            ))
        else:
            # For other methods, filter out base points in [2,3] range
            train_mask = ~((x_base_train >= 2) & (x_base_train <= 3))
            val_mask = ~((x_base_val >= 2) & (x_base_val <= 3))
            
            fig.add_trace(go.Scatter(
                x=x_base_train[train_mask].flatten(), 
                y=y_base_train[train_mask].flatten(),
                mode='markers', 
                name='Base Training',
                marker=dict(color='blue', size=8, symbol='circle')
            ))
            fig.add_trace(go.Scatter(
                x=x_base_val[val_mask].flatten(), 
                y=y_base_val[val_mask].flatten(),
                mode='markers', 
                name='Base Validation',
                marker=dict(color='lightblue', size=12, symbol='star')
            ))
    
    # Always show fine-tune data if method is not 'none'
    if fine_tune_data is not None and fine_tune_method != 'none':
        x_fine_train = fine_tune_data['train']['x'].numpy() if isinstance(fine_tune_data['train']['x'], torch.Tensor) else fine_tune_data['train']['x']
        y_fine_train = fine_tune_data['train']['y'].numpy() if isinstance(fine_tune_data['train']['y'], torch.Tensor) else fine_tune_data['train']['y']
        x_fine_val = fine_tune_data['val']['x'].numpy() if isinstance(fine_tune_data['val']['x'], torch.Tensor) else fine_tune_data['val']['x']
        y_fine_val = fine_tune_data['val']['y'].numpy() if isinstance(fine_tune_data['val']['y'], torch.Tensor) else fine_tune_data['val']['y']
        
        fig.add_trace(go.Scatter(
            x=x_fine_train.flatten(), 
            y=y_fine_train.flatten(),
            mode='markers', 
            name='Fine-tune Training',
            marker=dict(color='red', size=8, symbol='circle')
        ))
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

def count_trainable_parameters(model):
    """Count the total number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_total_parameters(model):
    """Count the total number of parameters in the model."""
    return sum(p.numel() for p in model.parameters())

app = dash.Dash(__name__)

# Initialize data at startup
initial_base_data = get_base_data(n_samples=DEFAULT_TRAINING_DATA_SIZE)
initial_fine_tune_data = get_fine_tune_data(n_samples=DEFAULT_FINE_TUNE_DATA_SIZE)
initial_model = get_model(DEFAULT_HIDDEN_SIZE, DEFAULT_FINE_TUNE_METHOD, DEFAULT_N_LAYERS)  # Create initial model

app.layout = html.Div([
    html.Div([
        dcc.Graph(id='data-plot', 
                 figure=create_data_plot(initial_base_data, initial_fine_tune_data, initial_model)),  # Pass initial_model
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
                {'label': '11 layers', 'value': 11}
            ],
            value=11
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
                {'label': 'None', 'value': 'none'},
                {'label': 'Full Fine-tuning', 'value': 'full'},
                {'label': 'Freeze First 6', 'value': 'freeze6'},
                {'label': 'Freeze First 8', 'value': 'freeze8'},
                {'label': 'Freeze First 10', 'value': 'freeze10'}
            ],
            value=DEFAULT_FINE_TUNE_METHOD
        ),
        
        html.Label('Fine-tuning Data:'),
        dcc.Dropdown(
            id='fine-tune-data-mode',
            options=[
                {'label': 'Only New Data', 'value': 'only_new'},
                {'label': 'Combined Data', 'value': 'combined'}
            ],
            value=DEFAULT_FINE_TUNE_DATA_MODE
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
        ]),
        
        # Add new div for parameter count
        html.Div([
            html.H4('Model Parameters'),
            html.Pre(id='parameter-display', 
                    style={
                        'border': '1px solid #ddd',
                        'padding': '10px',
                        'backgroundColor': '#f8f9fa',
                        'fontFamily': 'monospace',
                        'whiteSpace': 'pre-wrap',
                        'fontSize': '12px'
                    })
        ]),
        
        # Add the neural network architecture image
        html.Div([
            html.Img(id='arch-image',  # Add this ID
                    src='assets/nnarch.png',
                    style={
                        'width': '100%',
                        'marginTop': '20px',
                        'marginBottom': '20px'
                    })
        ])
    ], style={'width': '35%', 'float': 'right', 'padding': '20px'})
])

# Initialize global state
current_model = None
base_data = None
fine_tune_data = None
training_history = {'base': [], 'fine_tune': []}

def get_arch_image(method):
    if method == 'freeze6':
        return 'assets/6.png'
    elif method == 'freeze8':
        return 'assets/8.png'
    elif method == 'freeze10':
        return 'assets/10.png'
    else:
        return 'assets/nnarch.png'

@app.callback(
    Output('train-base-button', 'n_clicks'),
    Output('fine-tune-button', 'n_clicks'),
    Output('fine-tune-button', 'disabled'),
    Output('data-plot', 'figure'),
    Output('loss-plot', 'figure'),
    Output('status-display', 'children'),
    Output('parameter-display', 'children'),
    Output('arch-image', 'src'),  # Add this line
    Input('hidden-size', 'value'),
    Input('n-layers', 'value'),
    Input('train-base-button', 'n_clicks'),
    Input('fine-tune-button', 'n_clicks'),
    Input('reset-button', 'n_clicks'),
    Input('fine-tune-method', 'value'),
    State('batch-size', 'value'),
    State('training-steps', 'value'),
    State('learning-rate', 'value'),
    State('fine-tune-data-mode', 'value'),
    prevent_initial_call=True
)
def update_model(hidden_size, n_layers, base_clicks, fine_tune_clicks, 
                reset_clicks, fine_tune_method, batch_size, training_steps,
                learning_rate, fine_tune_data_mode):
    global current_model, base_data, fine_tune_data, training_history
    
    ctx = dash.callback_context
    if not ctx.triggered:
        return 0, 0, True, create_data_plot(base_data, fine_tune_data, None), \
               create_loss_plot({'base': [], 'fine_tune': []}), \
               "Click 'Train Base Model' to start", \
               "No model loaded", \
               'assets/nnarch.png'  # Default image
    
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Initialize data if not already done
    if base_data is None:
        base_data = get_base_data(n_samples=DEFAULT_TRAINING_DATA_SIZE)
        fine_tune_data = get_fine_tune_data(n_samples=DEFAULT_FINE_TUNE_DATA_SIZE)
    
    try:
        if trigger_id == 'reset-button':
            current_model = get_model(hidden_size, fine_tune_method, n_layers)
            total_params = count_total_parameters(current_model)
            training_history = {'base': [], 'fine_tune': []}
            param_info = f"Total parameters: {total_params:,}\nTrainable parameters: {total_params:,}"
            return 0, 0, True, create_data_plot(base_data, fine_tune_data, current_model), \
                   create_loss_plot({'base': [], 'fine_tune': []}), \
                   "Training reset. Click 'Train Base Model' to start", \
                   param_info, \
                   get_arch_image(fine_tune_method)  # Add image source
        
        if trigger_id in ['hidden-size', 'n-layers']:  # Reset on either parameter change
            # Reset everything when architecture changes
            current_model = get_model(hidden_size, fine_tune_method, n_layers)  # Create new model instead of None
            training_history = {'base': [], 'fine_tune': []}
            return 0, 0, True, create_data_plot(base_data, fine_tune_data, current_model), \
                   create_loss_plot({'base': [], 'fine_tune': []}), \
                   "Model reset. Click 'Train Base Model' to start", \
                   "No model loaded", \
                   get_arch_image(fine_tune_method)  # Add image source
        
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
                
            total_params = count_total_parameters(current_model)
            param_info = f"Total parameters: {total_params:,}\nTrainable parameters: {total_params:,}"
            status = "Base model training continued"
            
        elif trigger_id == 'fine-tune-button':
            if current_model is None:
                return no_update
            
            # Mark that we're fine-tuning
            current_model.is_fine_tuning = True
            
            # Prepare fine-tuning data based on selected mode
            if fine_tune_data_mode == 'only_new':
                train_data = fine_tune_data
                mode_desc = "new data only"
            else:  # combined
                train_data = combine_datasets_with_replacement(base_data, fine_tune_data)
                mode_desc = "combined data"
            
            # Fine-tune the model
            trainer = Trainer(current_model, fine_tune_method, learning_rate=learning_rate)
            new_history = trainer.train(train_data, batch_size=batch_size, epochs=training_steps)
            
            # Append new training history to existing history
            if not training_history['fine_tune']:
                training_history['fine_tune'] = new_history
            else:
                training_history['fine_tune']['train_loss'].extend(new_history['train_loss'])
                training_history['fine_tune']['val_loss'].extend(new_history['val_loss'])
                
            if fine_tune_method in ['none', 'full']:
                param_count = count_total_parameters(current_model)
                trainable_count = param_count
            else:
                param_count = count_total_parameters(current_model)
                trainable_count = count_trainable_parameters(current_model)
            
            param_info = f"Total parameters: {param_count:,}\nTrainable parameters: {trainable_count:,}"
            status = f"Model fine-tuned on {mode_desc} using {fine_tune_method}"
        
        # Create figures
        data_fig = create_data_plot(base_data, fine_tune_data, current_model, fine_tune_method)
        loss_fig = create_loss_plot(training_history)
        
        # Always enable fine-tune button if we have a trained model
        should_disable_fine_tune = current_model is None
        
        if trigger_id == 'fine-tune-method':
            if current_model is not None:  # Only update parameters if we have a model
                # Store current weights
                state_dict = current_model.state_dict()
                
                # Create new model with the new fine-tuning method
                current_model = get_model(hidden_size, fine_tune_method, n_layers)
                current_model.load_state_dict(state_dict)  # Restore weights
                current_model.is_fine_tuning = True  # Mark as fine-tuning to apply freezing
                
                # Re-apply the fine-tuning method to ensure proper freezing
                if fine_tune_method == 'none':
                    # Freeze all parameters
                    for param in current_model.parameters():
                        param.requires_grad = False
                elif fine_tune_method == 'freeze6':
                    # Freeze first 6 layers
                    for i in range(min(6, len(current_model.layers))):
                        for param in current_model.layers[i].parameters():
                            param.requires_grad = False
                        for param in current_model.bn_layers[i].parameters():
                            param.requires_grad = False
                elif fine_tune_method == 'freeze8':
                    # Freeze first 8 layers
                    for i in range(min(8, len(current_model.layers))):
                        for param in current_model.layers[i].parameters():
                            param.requires_grad = False
                        for param in current_model.bn_layers[i].parameters():
                            param.requires_grad = False
                elif fine_tune_method == 'freeze10':
                    # Freeze first 10 layers
                    for i in range(min(10, len(current_model.layers))):
                        for param in current_model.layers[i].parameters():
                            param.requires_grad = False
                        for param in current_model.bn_layers[i].parameters():
                            param.requires_grad = False
                
                # Count parameters
                param_count = count_total_parameters(current_model)
                trainable_count = count_trainable_parameters(current_model)
                param_info = f"Total parameters: {param_count:,}\nTrainable parameters: {trainable_count:,}"
            else:
                param_info = "No model loaded"
            
            return base_clicks, fine_tune_clicks, should_disable_fine_tune, \
                   create_data_plot(base_data, fine_tune_data, current_model, fine_tune_method), \
                   loss_fig, \
                   "Fine-tuning method changed", \
                   param_info, \
                   get_arch_image(fine_tune_method)  # Add image source
        
        return base_clicks, fine_tune_clicks, should_disable_fine_tune, \
               data_fig, loss_fig, status, param_info, \
               get_arch_image(fine_tune_method)  # Add image source
    
    except Exception as e:
        print(f"Error in callback: {str(e)}")
        return no_update

if __name__ == '__main__':
    app.run_server(debug=True, port=8051) 