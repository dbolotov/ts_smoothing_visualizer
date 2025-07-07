<p align="center">
  <img src="app_preview.gif" width="600" alt="App Preview">
</p>


# Time Series Smoothing: an Interactive Visualizer

[![Try the App](https://img.shields.io/badge/TRY%20THE%20APP-FF4B4B)](https://timeseriessmoothing.streamlit.app/)


This repository contains the code for a Streamlit app that visualizes time series smoothing techniques. It includes both real and synthetic datasets and lets you compare how different methods behave with adjustable parameters.

*Note: The app was developed while writing this Medium article: [Six Approaches to Time Series Smoothing](https://medium.com/@dmitriy.bolotov/six-approaches-to-time-series-smoothing-cc3ea9d6b64f)*

**Features**
- Adjustable smoothing parameters
- Visual comparison across methods
- 5 datasets

**Supported methods**: Moving Average, Exponential Moving Average, Savitzky-Golay, LOESS, Gaussian Filter, Kalman Filter


## Datasets

This project uses a mix of real-world and synthetic datasets. Below are the sources and licensing information:

- **Sunspots**  
  Daily total sunspot numbers from [SILSO](https://www.sidc.be/SILSO/datafiles). Licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).

- **Humidity (RH)** and **Wind Speed (WV)**  
  Weather time series from [Weather Long-term Time Series Forecasting](https://www.kaggle.com/datasets/alistairking/weather-long-term-time-series-forecasting) on Kaggle. Licensed under the [MIT License](https://www.mit.edu/~amini/LICENSE.md).

- **Noisy Sine**  
  Synthetic noisy sine wave, created for this project.

- **Process Anomalies**  
  Synthetic dataset simulating different industrial operating modes and injected anomalies, created for this project.



