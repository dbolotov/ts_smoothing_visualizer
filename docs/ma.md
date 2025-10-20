[Moving Average (Rolling Mean) Smoothing](https://en.wikipedia.org/wiki/Moving_average): Replaces each point with the average of nearby values in a fixed window. Simple and fast, it removes short-term fluctuations and highlights trends. 

**Limitations**: Introduces lag and blunts sharp features.

**Parameters**:
- `Window Size`: Number of data points averaged at each step. Larger = smoother but more laggy.