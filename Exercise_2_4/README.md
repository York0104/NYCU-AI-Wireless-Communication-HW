# Exercise 2.4(a)

This exercise uses QuaDRiGa to generate a realistic MIMO wireless channel for one specific configuration. QuaDRiGa supports geometry-based stochastic channel generation, MIMO modeling, and standardized scenarios such as 3GPP TR 38.901. In this implementation, we generate a 2 x 4 MIMO channel under the 3GPP_38.901_UMi_NLOS scenario.

## Problem Setup

The goal is to generate a realistic MIMO channel dataset for a specific wireless communication scenario using QuaDRiGa.

### Configuration

- Scenario: 3GPP_38.901_UMi_NLOS
- Carrier frequency: 3.5 GHz
- Number of transmit antennas: 4
- Number of receive antennas: 2
- UE speed: 3 km/h
- Number of channel snapshots: 20000
- BS position: [0; 0; 25]
- UE initial position: [100; 0; 1.5]
- UE trajectory: linear track

## Method

The implementation follows these steps:

1. Add the QuaDRiGa-main folder and all subfolders to the MATLAB path.
2. Verify that key QuaDRiGa classes are accessible:
   - qd_layout
   - qd_simulation_parameters
   - qd_arrayant
3. Create simulation parameters and set the center frequency.
4. Construct the layout and configure the transmitter and receiver arrays.
5. Define the UE linear trajectory and movement speed.
6. Select the 3GPP_38.901_UMi_NLOS scenario.
7. Generate channel coefficients using:
   - init_builder
   - gen_parameters
   - get_channels
8. Extract the raw multipath channel tensor h_coeff.
9. Sum over the path dimension to obtain a flat-fading MIMO channel h_mimo.
10. Save both the raw channel and processed MIMO channel into mimo_channel_dataset.mat.

## MATLAB File

The implementation is provided in [Exercise_2_4.m](./Exercise_2_4.m).

## Results

From the current script output:

* QuaDRiGa was successfully added to the MATLAB path
* The required QuaDRiGa constructors were correctly detected
* The selected scenario was 3GPP_38.901_UMi_NLOS
* The raw channel tensor size was 2 x 4 x 58 x 20000
* The flat-fading MIMO channel size was 2 x 4 x 20000
* The dataset was saved as mimo_channel_dataset.mat

This means:

* 2 receive antennas
* 4 transmit antennas
* 58 multipath components (before path summation)
* 20000 time snapshots

After summing over the multipath dimension, the final flat-fading channel dataset contains one 2 x 4 MIMO channel matrix for each of the 20000 snapshots.

### Console Output

```matlab
QuaDRiGa path added successfully.
D:\NYCU\class\Artificial Intelligence Wireless\NYCU-AI-Wireless-Communication-HW\Exercise_2_4\QuaDRiGa-main\quadriga_src\@qd_layout\qd_layout.m  % qd_layout constructor
D:\NYCU\class\Artificial Intelligence Wireless\NYCU-AI-Wireless-Communication-HW\Exercise_2_4\QuaDRiGa-main\quadriga_src\@qd_simulation_parameters\qd_simulation_parameters.m  % qd_simulation_parameters constructor
D:\NYCU\class\Artificial Intelligence Wireless\NYCU-AI-Wireless-Communication-HW\Exercise_2_4\QuaDRiGa-main\quadriga_src\@qd_arrayant\qd_arrayant.m  % qd_arrayant constructor
Scenario: 3GPP_38.901_UMi_NLOS
Generating 20000 snapshots for 2x4 MIMO...
Parameters   [oooooooooooooooooooooooooooooooooooooooooooooooooo]     0 seconds
Channels     [oooooooooooooooooooooooooooooooooooooooooooooooooo]    36 seconds
Raw channel size   : [2      4      58  20000]
Flat MIMO size     : [2      4  20000]
Dataset saved to mimo_channel_dataset.mat
```

###  Output File
The generated dataset file is:
* mimo_channel_dataset.mat

It contains:
* h_coeff: raw multipath MIMO channel coefficients

* h_mimo: processed flat-fading MIMO channel