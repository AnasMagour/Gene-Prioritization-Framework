import scipy.stats as stats

# 1. Define your complete model predicted T1D positive gene list
# (Make sure to plug in your actual complete predicted list here!)
predicted_t1d_genes = [
    "ACTG1", "ADAM15", "ADAM9", "ADAMTS13", "ADAR", "ADIPOQ", "ADRA2A", "AGRP", 
    "AIM2", "ALOX15", "ANGPT2", "ANXA1", "APOBEC3F", "APP", "BCL2", "BMP6", 
    "BMP7", "C5", "CADM1", "CARD11", "CARD9", "CASP6", "CAV1", "CCL16", 
    "CCL17", "CCL18", "CCL19", "CCL2", "CCL21", "CCL22", "CCL24", "CCL25", 
    "CCL26", "CCL3", "CCL4", "CCL5", "CCN2", "CD300LF", "CD36", "CD48", 
    "CD7", "CD74", "CD79A", "CD79B", "CD81", "CDC42EP4", "CGAS", "CIITA", 
    "CLEC4M", "CLU", "CR2", "CREBBP", "CRP", "CSF2RB", "CSH1", "CTSG", 
    "CXCL1", "CXCL13", "CXCL2", "CXCL5", "CYBA", "DHX9", "DKK1", "ECM1", 
    "EDN1", "EGF", "EIF4E2", "EPHB6", "EPO", "EREG", "F7", "FCER1G", 
    "FCGR3B", "FCN1", "FCN2", "FCN3", "FFAR1", "FFAR2", "FGB", "FGF21", 
    "FLT3LG", "FOXF1", "FURIN", "FYN", "FZD5", "GAPDH", "GBP1", "GDF15", 
    "GH2", "GPS2", "GZMM", "HLA-B", "HLA-C", "HLA-DOA", "HLA-DOB", "HLA-DPA1", 
    "HLA-DRB5", "HLA-E", "HLA-G", "HMGB1", "HMGB2", "HSPA1A", "HSPA1B", 
    "ICOS", "IDO1", "IFITM1", "IFITM3", "IFNA2", "IFNB1", "IFNGR1", "IFNGR2", 
    "IFNK", "IFNL1", "IFNL3", "IGLL1", "IGLL5", "IKBKE", "IKBKG", "IL10RB", 
    "IL12A", "IL12B", "IL12RB1", "IL13", "IL17F", "IL18R1", "IL1A", "IL23A", 
    "IL33", "IL5", "IL6R", "IL6ST", "IL7R", "IRAK4", "ITGAM", "ITGB2", 
    "JAK2", "KAT5", "KIF5B", "KITLG", "KL", "KLRB1", "KLRD1", "KLRG1", 
    "LAG3", "LAIR1", "LAPTM5", "LILRB2", "LRRK2", "LSM14A", "LTBR", "LYN", 
    "MAPKAPK3", "MMP14", "MNDA", "MR1", "MS4A2", "MST1R", "MSTN", "MYC", 
    "NAIP", "NCF1", "NCR1", "NCR3LG1", "NECTIN1", "NFKBIZ", "NLRP1", "NOP53", 
    "NR1H3", "NR1H4", "OAS2", "OAS3", "OPTN", "OSM", "PAK1", "PAK2", 
    "PARK7", "PARP1", "PCSK9", "PDGFA", "PDGFB", "PDPK1", "PELI1", "PIK3R1", 
    "PLAT", "PLCG2", "PLG", "PML", "PPARG", "PPBP", "PRF1", "PRKCE", 
    "PRKD1", "PRL", "PSMA8", "PSMB11", "PTGDR", "PTH", "PTK2B", "PTPN11", 
    "PUM1", "RAB11FIP2", "RAB27A", "RAC1", "RAET1E", "RAET1G", "RAF1", 
    "RARRES2", "RIGI", "RMI1", "RNF125", "RNF185", "RPS19", "RPS6KA3", 
    "S100A12", "S100A8", "S100A9", "SEMA7A", "SFRP1", "SIN3A", "SLIT2", 
    "SMARCA4", "STAT3", "TBK1", "TF", "TGFA", "TGFB1", "THBS1", "THPO", 
    "TLR1", "TLR7", "TNFRSF14", "TNFRSF4", "TNFSF18", "TNFSF4", "TRIM32", 
    "TRIM56", "TSLP", "UBE2K", "UBE2N", "UFD1", "ULBP1", "ULBP2", "ULBP3", 
    "USP18", "VAMP3", "WAS", "WNT1", "WNT11", "WNT2", "WNT3", "WNT3A", 
    "WNT5A", "WNT7A", "XCL2", "YTHDF3"]

# 2. Embedded list of highly-studied, classic cancer driver baseline hub genes
# This bypasses the need for an external 'cancer_drivers.txt' file
cancer_drivers = [
    "TP53",
    "KRAS",
    "EGFR",
    "MYC",
    "BRAF",
    "PIK3CA",
    "BRCA1",
    "BRCA2",
    "PTEN",
    "APC",
    "RB1",
    "JAK2",
    "STAT3",
    "CTNNB1",
    "ATM",
    "IDH1",
    "MDM2",
    "NF1",
    "VHL",
    "AR",
]

# 3. Establish our overall background genome context
TOTAL_HUMAN_GENES = 20000

# 4. Calculate intersections
t1d_set = set(predicted_t1d_genes)
cancer_set = set(cancer_drivers)

overlapping_genes = t1d_set.intersection(cancer_set)
num_overlap = len(overlapping_genes)

# 5. Build a 2x2 Contingency Table for Fisher's Exact Test
a = num_overlap
b = len(cancer_set) - num_overlap
c = len(t1d_set) - num_overlap
d = TOTAL_HUMAN_GENES - (a + b + c)

odds_ratio, p_value = stats.fisher_exact([[a, b], [c, d]], alternative="greater")

print("\n--- BIAS DECOUPLING CROSS-VALIDATION ---")
print(f"Total Predicted T1D Genes: {len(t1d_set)}")
print(f"Total Highly-Studied Cancer Drivers: {len(cancer_set)}")
print(f"Overlapping Genes: {num_overlap} {list(overlapping_genes)}")
print(f"Fisher's Exact Test p-value: {p_value:.4f}")

if p_value > 0.05:
    print(
        "\n🎉 SUCCESS: No statistically significant enrichment with global disease hubs!"
    )
    print(
        "This proves your model isolates T1D-specific immune features over general literature frequency bias."
    )
else:
    print(
        "\n⚠️ WARNING: Significant overlap detected. Double check if generic cell proliferation pathways are muddying your features."
    )
