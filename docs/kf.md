[Kalman Filter](https://en.wikipedia.org/wiki/Kalman_filter): Uses a probabilistic model to estimate system state from noisy observations. Good for dynamic smoothing, handling noisy or missing data, and adapting over time.

**Limitations:** More complex; sensitive to parameter tuning; assumes linear dynamics.

**Parameters:**
- `Tr std`: Transition standard deviation; expected noise in the process’s internal dynamics.  
- `Obs std`: Observation standard deviation; expected noise in the observed data.
