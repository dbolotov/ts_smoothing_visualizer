- [LOESS (LOWESS)](https://en.wikipedia.org/wiki/Local_regression): Fits a local regression for each point using neighboring data. Good for flexible trend fitting, especially for nonlinear or slowly-varying trends.  

**Limitations:** Slower for large datasets; may overfit if the fraction is too low.

**Parameters:**
- `Frac`: Proportion of the dataset used to compute each local regression.