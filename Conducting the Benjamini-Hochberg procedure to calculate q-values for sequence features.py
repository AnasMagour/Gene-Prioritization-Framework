# Conducting the Benjamini-Hochberg procedure to calculate q-values for sequence features

# ============================================================
# Importing the required libraries
# ============================================================

import numpy as np

# ============================================================
# Importing the necessary modules
# ============================================================

from statsmodels.stats.multitest import multipletests

# =================================================================
# p-values from our independent t-tests for sequence features
# =================================================================

raw_p_values = [0.00028, 0.00045, 0.00328, 0.00888, 0.01695, 0.01708, 0.01708, 0.05723, 0.06689, 0.07771, 0.09792, 0.10036, 0.11485, 0.13561, 0.13998, 0.16315, 0.17378, 0.17694, 0.18345, 0.18709, 0.19392, 0.22352, 0.22593, 0.31287, 0.32952, 0.34960, 0.46977, 0.49257, 0.53013, 0.55764, 0.67429, 0.72504, 0.86421, 0.88803, 0.95434, 0.95615]

# ===================================================================
# Apply Benjamini-Hochberg correction (fdr_bh) to calculate q-value
# ===================================================================

reject, q_values, _, _ = multipletests(raw_p_values, alpha=0.05, method='fdr_bh')

# ===============
# Output
# ===============

print("Reject Null?", reject)
print("Adjusted P-values (q-values):", q_values)
