**RPR (Roughness Preservation Ratio)**: quantifies how much of the signal’s short-term variation (its “roughness”) is retained after smoothing. It’s based on the total variation (defined as the sum of absolute differences between consecutive values):

RPR = ∑ |ŷᵢ₊₁ − ŷᵢ| / ∑ |yᵢ₊₁ − yᵢ|

Values near 1 indicate minimal smoothing; values near 0 reflect strong smoothing. Values above 1 can occur if the method increases local variation.

This metric doesn’t measure whether the shape of the signal is preserved, only how much the jaggedness is reduced. It’s useful for comparing how aggressively different methods smooth the same data.