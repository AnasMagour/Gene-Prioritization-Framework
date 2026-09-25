# Conducting the Benjamini-Hochberg procedure to calculate q-values for topological features

# ============================================================
# Importing the required libraries
# ============================================================

import numpy as np

# ============================================================
# Importing the necessary modules
# ============================================================

from statsmodels.stats.multitest import multipletests

# =================================================================
# p-values from our independent t-tests for topological features
# =================================================================

raw_p_values = [8.90161E-69, 8.3813E-50, 5.32663E-40, 5.32663E-40, 5.81726E-25, 2.25298E-21, 7.59855E-13, 1.77865E-05, 3.01537E-05, 0.00147]

# ===================================================================
# Apply Benjamini-Hochberg correction (fdr_bh) to calculate q-value
# ===================================================================

reject, q_values, _, _ = multipletests(raw_p_values, alpha=0.05, method='fdr_bh')

# ===============
# Output
# ===============

print("Reject Null?", reject)
print("Adjusted P-values (q-values):", q_values)
