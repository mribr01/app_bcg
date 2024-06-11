from scipy.fftpack import fft
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from scipy.signal import find_peaks
from scipy.signal import correlate
from scipy.fft import fft, fftfreq
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import dash_bootstrap_components as dbc


def get_fft_values(y_values, T, N, f_s):
    f_values = np.linspace(0.0, 1.0/(2.0*T), N//2)
    fft_values_ = fft(y_values)
    fft_values = 2.0/N * np.abs(fft_values_[0:N//2])
    return f_values, fft_values

def generate_composite_signal(randomTime, respiration_amplitude, respiration_frequency, noise_amplitude) :
    # Signal parameters
    t_n = 50; N = 10000; T = t_n / N

    # Time values
    t = np.linspace(0, t_n, N)

    # Generate composite signal without respiration
    amplitudes = [4, 6, 8, 20, 14]; frequencies = [10, 5, 3, 1.5, 1]
    y_values = [amplitudes[ii] * np.sin(2 * np.pi * frequencies[ii] * t) for ii in range(len(amplitudes))]
    composite_signal = np.sum(y_values, axis=0)
    #----------------------------------------------------------

    # Trouver les pics dans le signal composite
    peaks, _ = find_peaks(composite_signal, distance=200)  # distance minimale entre
    peaks = peaks[1:]

    # Créer une liste pour stocker les parties du signal correspondant à chaque pic
    peaks_signals = []
    t_peaks_signals = []

    # Pour chaque pic détecté, extraire la partie correspondante du signal
    peak_signal = composite_signal[(t >= t[peaks[0]] - 0.19) & (t <= t[peaks[0]] + 1.3)]
    noisePeak_signal = composite_signal[(t >= t[peaks[0]] + 1.3) & (t <= t[peaks[0]] + 1.8)]

    newComposite_signal = []
    newt = []

    for i in range(60):
        # Décalage aléatoire du temps
        random_time = np.random.uniform(randomTime[0], randomTime[1])  # Modifier l'intervalle selon votre besoin
        # Créer un signal bruité pour remplir les intervalles entre les peak_signal
        noise_amplitude = 5
        noise = noise_amplitude * np.random.randn(len(noisePeak_signal))

        TempSignal = np.hstack((noisePeak_signal + noise, peak_signal))

        if len(newt) > 0:
            shifted = newt[-1][-1]  # Obtenir la dernière valeur de temps ajoutée
            TempTimeNoise = np.linspace(shifted, shifted + random_time, len(noisePeak_signal))
            TempTimeSignal = np.linspace(shifted + random_time, shifted + random_time + 0.5, len(peak_signal))
            TempTime = np.hstack((TempTimeNoise, TempTimeSignal))
        else:
            TempTimeNoise = np.linspace(0, random_time, len(noisePeak_signal))
            TempTimeSignal = np.linspace(random_time, random_time + 0.5, len(peak_signal))
            TempTime = np.hstack((TempTimeNoise, TempTimeSignal))

        newComposite_signal.append(TempSignal)
        newt.append(TempTime)

    # Concaténer les listes en un seul tableau
    newComposite_signal = np.hstack(newComposite_signal)
    newt = np.hstack(newt)

    # Add respiratory disturbance
    respiration_signal = respiration_amplitude * np.sin(2 * np.pi * respiration_frequency * newt)
    noisy_composite_signal = newComposite_signal + respiration_signal

    # Add noise to the composite signal
    noise = noise_amplitude * np.random.randn(len(newt))

    signalBCG = noisy_composite_signal + noise

    # Calculer la corrélation croisée entre newComposite_signal et peak_signal
    cross_correlation = correlate(newComposite_signal, peak_signal, mode='valid')

    # Trouver les indices où la corrélation est maximale
    max_indices = np.where(cross_correlation == np.max(cross_correlation))[0]

    #-----------------------------------------------------
    bpm = len(newt[max_indices])*60/(max(newt[max_indices]) - min(newt[max_indices]))
    intersections = np.argwhere(np.diff(np.sign(respiration_signal)) != 0).squeeze()
    bpmRespi = len(newt[intersections[1::2]])*60/(max(newt[intersections[1::2]]) - min(newt[intersections[1::2]]))

    return newt, signalBCG, respiration_signal, max_indices, bpm, intersections, bpmRespi

randomTime = [-0.2, 1.0] # Ce paramètre joue sur la variabilité de la fréquence instantanée; combien de fois apparaît le signal BCG
respiration_amplitude = 20
respiration_frequency = 0.2
noise_amplitude = 2 # valeur qui va de 0 à 5

newt, signalBCG, respiration_signal, max_indices, bpm, intersections, bpmRespi = generate_composite_signal(randomTime, respiration_amplitude, respiration_frequency, noise_amplitude)

# Create Dash app

app = dash.Dash(__name__)
server = app.server  # Explicitly name the Dash app instance as 'server'


# Create figure object
fig = go.Figure()

# Set up initial layout
fig.update_layout(
    xaxis=dict(range=[0, 4000]),
    yaxis=dict(range=[-65, 65]),
    xaxis_title='Time',
    yaxis_title='Amplitude',
    title='Signal BCG Animation',
    # add bpm annotations in the middle of the plot
    annotations= [dict(x=0.5, y=1.1, showarrow=False, text='BPM :' +str(int(bpm*100)/100) + '  Respiratory rate :' + str(int(bpmRespi)), xref='paper', yref='paper')],
)

# Plot the initial empty line
fig.add_trace(go.Scatter(x=[], y=[], mode='lines', line=dict(color='blue'), name='Signal BCG'))

# Plot the respiration signal
fig.add_trace(go.Scatter(x=[], y=np.random.rand(4000), mode='lines', line=dict(color='black'), name='Respiration'))

# Define app layout
app.layout = html.Div([
    dbc.Alert(id='bpm-alert', color="danger"),
    dcc.Graph(id='bcg-graph', figure=fig),
    html.Div([
        html.Label('Variabilité fréquence 1'),
        dcc.Slider(
            id='VarFrq-slider-1',
            min=-0.5,
            max=0.5,
            step=0.05,
            value=-0.2,
        ),
        html.Label('Variabilité fréquence 2'),
        dcc.Slider(
            id='VarFrq-slider-2',
            min=-0.5,
            max=0.5,
            step=0.05,
            value=1.0,
        ),
        html.Label('Amplitude Respiratoire'),
        dcc.Slider(
            id='respiration_amplitude-slider-2',
            min=5,
            max=25,
            step=5,
            value=20,
        ),
        html.Label('Frequence Respiratoire'),
        dcc.Slider(
            id='respiration_frequency-slider-2',
            min=0.,
            max=1.0,
            step=0.1,
            value=0.2,
        ),
        html.Label('Amplitude Bruit'),
        dcc.Slider(
            id='noise_amplitude-slider-3',
            min=0,
            max=5,
            step=0.1,
            value=2,
        ),
    ]),
    html.Div([
        html.Button('Play', id='play-button', n_clicks=0),
        html.Button('Pause', id='pause-button', n_clicks=0),
        html.Button('Stop', id='stop-button', n_clicks=0),
    ]),

    # html.Div([
    #     html.Button('newborn', id='newborn', n_clicks=0),
    #     html.Button('infant', id='infant', n_clicks=0),
    #     html.Button('child', id='child', n_clicks=0),
    #     html.Button('adult', id='adult', n_clicks=0),
    #     html.Button('senior', id='senior', n_clicks=0),
    # ]),

    dcc.Interval(
        id='interval-component',
        interval=300,  # Augmentez l'intervalle pour ralentir le défilement
        n_intervals=20
    )
])

# Initialize play status
play_status = False

@app.callback(
    [Output('bcg-graph', 'figure'),
     Output('bpm-alert', 'style')],
    [Input('interval-component', 'n_intervals'),
     Input('play-button', 'n_clicks'),
     Input('pause-button', 'n_clicks'),
     Input('stop-button', 'n_clicks'),
     Input('VarFrq-slider-1', 'value'),
     Input('VarFrq-slider-2', 'value'),
     Input('respiration_amplitude-slider-2', 'value'),
     Input('respiration_frequency-slider-2', 'value'),
     Input('noise_amplitude-slider-3', 'value')]
)


def update_graph(n_intervals, play_clicks, pause_clicks, stop_clicks, VarFrq1, VarFrq2, respAmpl, respFrq, noiseAmpl):
    global play_status
    ctx = dash.callback_context
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    randomTime = [VarFrq1, VarFrq2]

    newt, signalBCG, respiration_signal, max_indices, bpm, intersections, bpmRespi = generate_composite_signal(randomTime, respAmpl, respFrq, noiseAmpl)

    # Initialize alert_style here
    alert_style = {'display': 'none'}

    if button_id == 'interval-component' and play_status:
        i = n_intervals % 5000
        y = signalBCG[i:i+4000]
        respiration_y = respiration_signal[i:i+4000]
        x = np.linspace(0, 4000, len(y))
        fig.data[0].x = x
        fig.data[0].y = y
        fig.data[1].x = x
        fig.data[1].y = respiration_y
        fig.update_layout(annotations=[dict(x=0.5, y=1.1, showarrow=False, text='BPM : ' + str(int(bpm*100)/100) + '  Respiratory rate: ' + str(int(bpmRespi)), xref='paper', yref='paper', font=dict(color='black'))])

        # when an age button is clicked, check if bpm is odd
        if bpm < 50 or bpm > 110:
            fig.update_layout(annotations=[dict(x=0.5, y=1.1, showarrow=False, text='GO TO THE GP, your bpm is odd : ' + str(int(bpm)), xref='paper', yref='paper', font=dict(color='red'))])
            alert_style = {'display': 'block'}

    elif button_id == 'play-button':
        play_status = True
    elif button_id == 'pause-button':
        play_status = False
    elif button_id == 'stop-button':
        play_status = False
        fig.data[0].x = []
        fig.data[0].y = []
        fig.data[1].x = []
        fig.data[1].y = []

    return fig, alert_style


if __name__ == '__main__':
    app.run_server(debug=True)