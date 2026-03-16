clear; 
close all; 
clc;


%% 1. Define Simulation Parameters
fc = 3.5e9;                 % Center frequency (Hz)
num_tx = 1;                 % Number of transmit antennas (for SISO)
num_rx = 1;                 % Number of receive antennas (for SISO)
ue_speed_kmh = 3;           % User equipment (UE) speed (km/h)
num_snapshots = 20000;      % Number of channel snapshots (samples) to generate


%% 2. Create Quadriga Scenario and Layout
% Create simulation parameters object
s = qd_simulation_parameters;
s.center_frequency = fc;

% Create layout object
l = qd_layout(s);

% Configure the transmitter (Tx)
l.tx_array = qd_arrayant('omni');       % Omnidirectional antenna
l.tx_position = [0; 0; 25];             % Tx position at (0,0,25) meters

% Configure the receiver (Rx)
l.rx_array = qd_arrayant('omni');
l.no_rx = num_rx;

% Create the user's movement trajectory (track)
% Linear track, length is determined by snapshot count and speed
ue_speed_mps = ue_speed_kmh * 1000 / 3600;

% create the track:
track = qd_track('linear');
track.no_snapshots = num_snapshots;
track.set_speed(ue_speed_mps);
track.initial_position = [100; 0; 1.5];

l.rx_track = track;

% Using 3GPP 38.901 Urban Micro (UMi) NLOS scenario
l.set_scenario('3GPP_38.901_UMi_NLOS');

fprintf('Scenario: 3GPP_38.901_UMi_NLOS\n');
fprintf('Generating %d channel snapshots for a user moving at %.1f km/h...\n', num_snapshots, ue_speed_kmh);


%% 3. Generate Channel Coefficients
% Initialize the channel builder
b = l.init_builder;
% Generate large-scale and small-scale parameters
gen_parameters(b);

% Calculate the channel coefficients
% h is a qd_channel object containing coefficients and other info
h = get_channels(b);

% Extract the channel coefficient matrix
% The dimensions of h.coeff are [Rx_Ant, Tx_Ant, Num_Clusters, Num_Snapshots]
% For our SISO case, the dimensions are [1, 1, Num_Clusters, Num_Snapshots]
h_coeff = h.coeff;

% For a Rayleigh flat-fading model, we sum the contributions of all paths (clusters)
% The dimensions of h_flat become [Rx_Ant, Tx_Ant, Num_Snapshots]
h_flat = sum(h_coeff, 3);

% Since this is a SISO system, squeeze the dimensions to get a vector
% The final h_siso will have dimensions [1, Num_Snapshots] or [Num_Snapshots, 1]
h_siso = squeeze(h_flat);
fprintf('Processed SISO channel vector size: [%s]\n', num2str(size(h_siso)));



%% 4. Save the Dataset
dataset_filename = 'rayleigh_channel_dataset.mat';
save(dataset_filename, 'h_siso');

fprintf('\nDataset saved to ''%s''\n', dataset_filename);