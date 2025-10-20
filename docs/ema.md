[Exponential Moving Average (EMA)](https://en.wikipedia.org/wiki/Exponential_smoothing): Averages past values with more weight on recent data. Responsive and causal (only uses past and present), good for real-time use.  

**Limitations:** Can still lag and smooth too little if alpha is too small.

**Parameters:**
- `Alpha`: Smoothing factor between 0 and 1. Higher = quicker response, less smoothing.
