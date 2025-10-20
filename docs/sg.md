[Savitzky–Golay Filter](https://en.wikipedia.org/wiki/Savitzky–Golay_filter): Fits a polynomial to a local window of data. Good for preserving local features like peaks and valleys better than a simple moving average.

**Limitations:** Can amplify noise if parameters are poorly chosen; requires enough points to support the polynomial order.

**Parameters:**
- `Window Size`: Number of points used in each fit (must be odd).  
- `Polynomial Degree`: Controls how flexible the fit is; higher = more responsive to structure.
