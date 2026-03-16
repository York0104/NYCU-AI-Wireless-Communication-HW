clear;
close all;
clc;

addpath(genpath(fullfile(pwd,'QuaDRiGa-main')));
rehash;

disp('QuaDRiGa path added successfully.');

which qd_layout
which qd_simulation_parameters
which qd_arrayant

%% 1) Simulation parameters
fc = 3.5e9;                  % carrier frequency (Hz)
num_tx = 4;                  % number of Tx antennas
num_rx = 2;                  % number of Rx antennas
ue_speed_kmh = 3;            % UE speed (km/h)
num_snapshots = 20000;       % number of channel snapshots

%% 2) Create QuaDRiGa simulation parameters and layout
s = qd_simulation_parameters;
s.center_frequency = fc;

l = qd_layout(s);

%% 3) Configure Tx array
l.tx_array = qd_arrayant('omni');
l.tx_array.no_elements = num_tx;
l.tx_position = [0; 0; 25];

%% 4) Configure Rx array
l.rx_array = qd_arrayant('omni');
l.rx_array.no_elements = num_rx;
l.no_rx = 1;

%% 5) User trajectory
ue_speed_mps = ue_speed_kmh * 1000 / 3600;

track = qd_track('linear');
track.no_snapshots = num_snapshots;
track.set_speed(ue_speed_mps);
track.initial_position = [100; 0; 1.5];

l.rx_track = track;

%% 6) Scenario selection
l.set_scenario('3GPP_38.901_UMi_NLOS');

fprintf('Scenario: 3GPP_38.901_UMi_NLOS\n');
fprintf('Generating %d snapshots for %dx%d MIMO...\n', ...
    num_snapshots, num_rx, num_tx);

%% 7) Generate channel coefficients
b = l.init_builder;
gen_parameters(b);
h = get_channels(b);

%% 8) Extract channel coefficients
h_coeff = h.coeff;                 % raw multipath coefficients
h_mimo = squeeze(sum(h_coeff, 3)); % flat-fading MIMO channel

fprintf('Raw channel size   : [%s]\n', num2str(size(h_coeff)));
fprintf('Flat MIMO size     : [%s]\n', num2str(size(h_mimo)));

%% 9) Save dataset
dataset_filename = 'mimo_channel_dataset.mat';
save(dataset_filename, 'h_coeff', 'h_mimo');

fprintf('Dataset saved to %s\n', dataset_filename);