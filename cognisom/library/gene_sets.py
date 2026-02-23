"""
Curated Gene Sets and Entity Lists
====================================

Pre-built lists for one-click population of the entity database.
Each set targets a specific biological domain with genes, drugs,
and pathways that are well-characterized in public databases.

Usage:
    from cognisom.library.gene_sets import PROSTATE_CANCER_COMPREHENSIVE
    from cognisom.library.bulk_import import BulkImporter

    importer = BulkImporter(store)
    report = importer.import_gene_list(PROSTATE_CANCER_COMPREHENSIVE)
"""

# ── Prostate Cancer ──────────────────────────────────────────────────

PROSTATE_CANCER_COMPREHENSIVE = [
    # AR signaling axis
    "AR", "FOXA1", "HOXB13", "NKX3-1", "KLK3",  # PSA
    "TMPRSS2", "ERG", "ETV1", "ETV4", "ETV5",
    # Tumor suppressors
    "TP53", "RB1", "PTEN", "BRCA1", "BRCA2",
    "ATM", "CDH1", "CDKN1B", "APC", "SMAD4",
    # Oncogenes
    "MYC", "PIK3CA", "PIK3CB", "AKT1", "BRAF",
    "KRAS", "NRAS", "HRAS", "RAF1", "MTOR",
    # DNA repair
    "MSH2", "MSH6", "MLH1", "PMS2", "PALB2",
    "RAD51", "CHEK2", "NBN", "CDK12",
    # Cell cycle
    "CDK4", "CDK6", "CCND1", "CCNE1", "E2F1",
    "MDM2", "MDM4", "CDKN2A", "CDKN2B",
    # Epigenetic
    "EZH2", "KDM1A", "DNMT3A", "DNMT3B", "KMT2A",
    # Immune
    "CD274",  # PD-L1
    "PDCD1",  # PD-1
    "CTLA4", "LAG3", "TIGIT",
    # Metabolism
    "FASN", "ACLY", "SCD", "HK2", "LDHA",
    # Growth factors
    "EGFR", "ERBB2", "FGFR1", "FGFR2", "MET",
    "IGF1R", "VEGFA", "KDR",
    # WNT/Notch
    "CTNNB1", "WNT5A", "NOTCH1", "DLL3",
    # Neuroendocrine
    "SYP", "CHGA", "NCAM1", "SOX2", "ASCL1",
    "INSM1", "POU3F2",
    # Steroidogenesis
    "CYP17A1", "SRD5A1", "SRD5A2", "HSD3B1",
    "CYP11A1", "STAR",
]

PROSTATE_CANCER_DRUGS = [
    "Enzalutamide", "Abiraterone", "Docetaxel", "Cabazitaxel",
    "Apalutamide", "Darolutamide", "Olaparib", "Rucaparib",
    "Sipuleucel-T", "Radium-223", "Lutetium-177 PSMA",
    "Pembrolizumab", "Bicalutamide", "Flutamide",
    "Leuprolide", "Goserelin", "Degarelix",
    "Cabozantinib", "Talazoparib", "Niraparib",
]

PROSTATE_KEGG_PATHWAYS = [
    "hsa05215",  # Prostate cancer
    "hsa04115",  # p53 signaling pathway
    "hsa04151",  # PI3K-Akt signaling pathway
    "hsa04010",  # MAPK signaling pathway
    "hsa04110",  # Cell cycle
    "hsa04310",  # Wnt signaling pathway
    "hsa04330",  # Notch signaling pathway
    "hsa04350",  # TGF-beta signaling pathway
]

# ── Pan-Cancer Drivers ──────────────────────────────────────────────

PAN_CANCER_DRIVERS = [
    # Tumor suppressors (COSMIC Cancer Gene Census Tier 1)
    "TP53", "RB1", "PTEN", "APC", "BRCA1", "BRCA2",
    "VHL", "WT1", "NF1", "NF2", "SMAD4", "SMARCB1",
    "STK11", "TSC1", "TSC2", "CDH1", "CDKN2A",
    "BAP1", "ARID1A", "ARID2", "SETD2",
    # Oncogenes
    "KRAS", "NRAS", "HRAS", "BRAF", "PIK3CA",
    "MYC", "MYCN", "MYCL", "EGFR", "ERBB2",
    "ALK", "ROS1", "RET", "MET", "FGFR1", "FGFR2",
    "FGFR3", "KIT", "PDGFRA", "ABL1", "JAK2",
    "FLT3", "NOTCH1", "IDH1", "IDH2",
    "AKT1", "MTOR", "CCND1", "CDK4", "CDK6",
    "MDM2", "MDM4", "BCL2", "MCL1",
    # Epigenetic regulators
    "EZH2", "DNMT3A", "TET2", "ASXL1", "KMT2A",
    "KMT2D", "CREBBP", "EP300",
    # DNA damage repair
    "ATM", "ATR", "CHEK1", "CHEK2", "PALB2",
    "RAD51", "FANCA", "FANCC",
    # Immune evasion
    "CD274", "PDCD1LG2", "B2M", "JAK1",
    # Splicing
    "SF3B1", "U2AF1", "SRSF2",
]

# ── Breast Cancer ──────────────────────────────────────────────────

BREAST_CANCER_CORE = [
    "BRCA1", "BRCA2", "TP53", "ERBB2", "ESR1",
    "PGR", "PIK3CA", "AKT1", "PTEN", "CDH1",
    "GATA3", "MAP3K1", "MAP2K4", "CBFB", "RUNX1",
    "TBX3", "CDK4", "CDK6", "CCND1", "RB1",
    "MYC", "FGFR1", "NOTCH1", "SF3B1", "PALB2",
    "ATM", "CHEK2", "FOXA1", "KMT2C", "NF1",
    "ARID1A", "CDKN2A", "MDM2", "EGFR",
    "MET", "VEGFA", "MTOR", "RAD51",
]

BREAST_CANCER_DRUGS = [
    "Tamoxifen", "Letrozole", "Anastrozole", "Exemestane",
    "Trastuzumab", "Pertuzumab", "Lapatinib", "Tucatinib",
    "Palbociclib", "Ribociclib", "Abemaciclib",
    "Olaparib", "Talazoparib",
    "Capecitabine", "Doxorubicin", "Paclitaxel",
    "Atezolizumab", "Pembrolizumab",
    "Alpelisib", "Everolimus",
]

# ── DNA Repair ──────────────────────────────────────────────────────

DNA_REPAIR_GENES = [
    # Homologous recombination
    "BRCA1", "BRCA2", "RAD51", "RAD51B", "RAD51C",
    "RAD51D", "PALB2", "BARD1", "XRCC2", "XRCC3",
    # Mismatch repair
    "MSH2", "MSH3", "MSH6", "MLH1", "MLH3",
    "PMS1", "PMS2", "EPCAM",
    # Nucleotide excision repair
    "ERCC1", "ERCC2", "ERCC3", "ERCC4", "XPA", "XPC",
    # Base excision repair
    "OGG1", "MUTYH", "APEX1", "XRCC1", "POLB",
    # DDR signaling
    "ATM", "ATR", "CHEK1", "CHEK2", "NBN", "MRE11",
    "RAD50", "TP53BP1", "RNF168", "CDK12",
]

# ── Immune Checkpoint ────────────────────────────────────────────────

IMMUNE_CHECKPOINT_GENES = [
    "CD274",    # PD-L1
    "PDCD1",    # PD-1
    "PDCD1LG2", # PD-L2
    "CTLA4",
    "LAG3",
    "TIGIT",
    "TIM3",     # HAVCR2
    "HAVCR2",
    "VISTA",    # VSIR
    "VSIR",
    "BTLA",
    "CD47",
    "SIRPA",
    "CD80",
    "CD86",
    "ICOS",
    "TNFRSF18", # GITR
    "TNFRSF4",  # OX40
    "TNFRSF9",  # 4-1BB
    "IDO1",
]

# ── Kinase Targets (Druggable) ──────────────────────────────────────

KINASE_TARGETS = [
    # Receptor tyrosine kinases
    "EGFR", "ERBB2", "ERBB3", "ERBB4",
    "FGFR1", "FGFR2", "FGFR3", "FGFR4",
    "MET", "ALK", "ROS1", "RET", "NTRK1",
    "KIT", "PDGFRA", "PDGFRB", "FLT3",
    "KDR", "FLT1", "IGF1R", "INSR",
    "AXL", "MERTK",
    # Cytoplasmic kinases
    "ABL1", "ABL2", "JAK1", "JAK2", "JAK3",
    "SRC", "BTK", "SYK", "LCK",
    # Serine/threonine kinases
    "BRAF", "RAF1", "MAP2K1", "MAP2K2",
    "MAPK1", "MAPK3",
    "AKT1", "AKT2", "MTOR",
    "CDK4", "CDK6", "CDK2", "CDK1",
    "PLK1", "AURKA", "AURKB",
    "CHEK1", "CHEK2", "WEE1",
    "PIK3CA", "PIK3CB", "PIK3CD",
]

# ── Metabolic Enzymes ───────────────────────────────────────────────

METABOLIC_ENZYMES = [
    # Glycolysis
    "HK1", "HK2", "GPI", "PFKM", "ALDOA",
    "TPI1", "GAPDH", "PGK1", "ENO1", "PKM",
    # TCA cycle
    "CS", "ACO2", "IDH1", "IDH2", "OGDH",
    "SUCLA2", "SDHA", "FH", "MDH2",
    # Fatty acid synthesis
    "FASN", "ACACA", "ACLY", "SCD",
    # Amino acid metabolism
    "GLS", "GLUD1", "GOT1", "GOT2",
    # Nucleotide synthesis
    "DHFR", "TYMS", "RRM1", "RRM2",
    # Oxidative phosphorylation
    "NDUFS1", "UQCRC1", "COX4I1", "ATP5F1A",
    # Pentose phosphate
    "G6PD", "PGD", "TKT",
]

# ── FDA-Approved Cancer Drugs ───────────────────────────────────────

FDA_APPROVED_CANCER_DRUGS = [
    # Immunotherapy
    "Pembrolizumab", "Nivolumab", "Atezolizumab",
    "Durvalumab", "Avelumab", "Ipilimumab",
    "Cemiplimab",
    # Targeted therapy
    "Imatinib", "Erlotinib", "Gefitinib", "Osimertinib",
    "Crizotinib", "Alectinib", "Lorlatinib",
    "Vemurafenib", "Dabrafenib", "Trametinib",
    "Sotorasib", "Adagrasib",
    "Palbociclib", "Ribociclib", "Abemaciclib",
    "Olaparib", "Rucaparib", "Niraparib", "Talazoparib",
    "Venetoclax", "Ibrutinib", "Acalabrutinib",
    "Trastuzumab", "Bevacizumab", "Cetuximab",
    "Lenvatinib", "Sorafenib", "Sunitinib",
    "Cabozantinib", "Regorafenib", "Axitinib",
    "Alpelisib", "Everolimus", "Temsirolimus",
    "Larotrectinib", "Entrectinib",
    # Chemotherapy
    "Cisplatin", "Carboplatin", "Oxaliplatin",
    "Docetaxel", "Paclitaxel", "Cabazitaxel",
    "Doxorubicin", "Epirubicin",
    "5-Fluorouracil", "Capecitabine", "Gemcitabine",
    "Irinotecan", "Topotecan", "Etoposide",
    "Cyclophosphamide", "Temozolomide",
    "Methotrexate", "Pemetrexed",
    # Hormonal
    "Tamoxifen", "Letrozole", "Anastrozole",
    "Enzalutamide", "Abiraterone", "Apalutamide",
]

# ── KEGG Cancer Pathways ───────────────────────────────────────────

KEGG_CANCER_PATHWAYS = [
    "hsa05200",  # Pathways in cancer
    "hsa05215",  # Prostate cancer
    "hsa05224",  # Breast cancer
    "hsa05210",  # Colorectal cancer
    "hsa05225",  # Hepatocellular carcinoma
    "hsa05226",  # Gastric cancer
    "hsa05222",  # Small cell lung cancer
    "hsa05223",  # Non-small cell lung cancer
    "hsa05214",  # Glioma
    "hsa05221",  # Acute myeloid leukemia
    "hsa05220",  # Chronic myeloid leukemia
    "hsa05218",  # Melanoma
    "hsa05211",  # Renal cell carcinoma
    "hsa05219",  # Bladder cancer
    "hsa05213",  # Endometrial cancer
    "hsa05216",  # Thyroid cancer
    "hsa05217",  # Basal cell carcinoma
    "hsa05212",  # Pancreatic cancer
    # Signaling pathways
    "hsa04010",  # MAPK signaling
    "hsa04151",  # PI3K-Akt signaling
    "hsa04110",  # Cell cycle
    "hsa04115",  # p53 signaling
    "hsa04310",  # Wnt signaling
    "hsa04330",  # Notch signaling
    "hsa04064",  # NF-kappa B signaling
    "hsa04210",  # Apoptosis
    "hsa04370",  # VEGF signaling
    "hsa04620",  # Toll-like receptor signaling
    "hsa04630",  # JAK-STAT signaling
]

# ── Immunology: Innate Immunity ────────────────────────────────────

INNATE_IMMUNITY_GENES = [
    # Toll-like receptors
    "TLR1", "TLR2", "TLR3", "TLR4", "TLR5",
    "TLR6", "TLR7", "TLR8", "TLR9", "TLR10",
    # NOD-like receptors / inflammasome
    "NOD1", "NOD2", "NLRP1", "NLRP3", "NLRC4",
    "PYCARD", "CASP1", "IL1B", "IL18",
    # RIG-I-like receptors
    "DDX58", "IFIH1", "DHX58",  # RIG-I, MDA5, LGP2
    # cGAS-STING
    "MB21D1", "TMEM173",  # cGAS, STING
    # TLR signaling
    "MYD88", "TIRAP", "TICAM1", "TICAM2",  # MyD88, MAL, TRIF, TRAM
    "IRAK1", "IRAK4", "TRAF6", "TRAF3",
    "IRF3", "IRF7", "IRF5",
    "NFKB1", "NFKB2", "RELA", "NFKBIA",
    # Antimicrobial peptides
    "DEFA1", "DEFA4", "DEFB1", "CAMP",  # Defensins, LL-37
    # Acute phase
    "CRP", "SAA1", "LBP", "MBL2",
    # NK cell receptors
    "KLRK1", "NCR1", "NCR3", "KLRD1",  # NKG2D, NKp46, NKp30, CD94
    "KIR2DL1", "KIR3DL1",
]

# ── Immunology: Adaptive Immunity ──────────────────────────────────

ADAPTIVE_IMMUNITY_GENES = [
    # TCR complex
    "CD3D", "CD3E", "CD3G", "CD247",  # CD3 chains + CD3ζ
    "TRAC", "TRBC1",  # TCR alpha/beta constant
    # T cell coreceptors
    "CD4", "CD8A", "CD8B",
    # T cell signaling
    "LCK", "ZAP70", "LAT", "SLP76",  # PLCG1
    "CD28", "ICOS",
    "PTPRC",  # CD45
    # B cell / BCR
    "CD79A", "CD79B",  # Igα/Igβ
    "CD19", "MS4A1",  # CD20
    "CR2",  # CD21
    "SYK", "BTK", "BLNK",
    # Immunoglobulin genes
    "IGHM", "IGHG1", "IGHG2", "IGHG3", "IGHG4",
    "IGHA1", "IGHE", "IGHD",
    "IGKC", "IGLC1",
    # Class switching / somatic hypermutation
    "AICDA",  # AID
    "UNG",
    # Antigen processing
    "TAP1", "TAP2", "TAPBP",
    "PSMB8", "PSMB9", "PSMB10",  # Immunoproteasome
    "ERAP1", "ERAP2",
    # MHC / HLA
    "HLA-A", "HLA-B", "HLA-C",       # MHC class I
    "HLA-DRA", "HLA-DRB1", "HLA-DPA1", "HLA-DPB1",  # MHC class II
    "B2M",                             # Beta-2 microglobulin
    "CIITA", "CD74",                   # MHC class II transactivator
]

# ── Immunology: Cytokines & Chemokines ────────────────────────────

CYTOKINE_CHEMOKINE_GENES = [
    # Interleukins
    "IL1A", "IL1B", "IL1RN",
    "IL2", "IL2RA", "IL2RB",
    "IL4", "IL4R",
    "IL5",
    "IL6", "IL6R", "IL6ST",
    "IL7", "IL7R",
    "IL10", "IL10RA",
    "IL12A", "IL12B", "IL12RB1",
    "IL13",
    "IL15", "IL15RA",
    "IL17A", "IL17F", "IL17RA",
    "IL18", "IL18R1",
    "IL21", "IL21R",
    "IL22",
    "IL23A",
    "IL33", "IL1RL1",
    # TNF family
    "TNF", "TNFRSF1A", "TNFRSF1B",
    "LTA", "LTB",
    "FASLG", "FAS",
    "TRAIL",  # TNFSF10
    "TNFSF10",
    # Interferons
    "IFNA1", "IFNA2", "IFNB1",
    "IFNG", "IFNGR1",
    "IFNL1",  # IL29
    # Colony-stimulating factors
    "CSF1", "CSF2", "CSF3",
    "CSF1R", "CSF2RA", "CSF3R",
    # TGF-beta family
    "TGFB1", "TGFB2", "TGFB3",
    "TGFBR1", "TGFBR2",
    # Chemokines (CC)
    "CCL2", "CCL3", "CCL4", "CCL5",
    "CCL17", "CCL19", "CCL20", "CCL21", "CCL22",
    "CCR2", "CCR5", "CCR7",
    # Chemokines (CXC)
    "CXCL1", "CXCL2", "CXCL8",  # IL-8
    "CXCL9", "CXCL10", "CXCL11", "CXCL12", "CXCL13",
    "CXCR3", "CXCR4", "CXCR5",
]

# ── Immunology: T Cell Subset Markers ─────────────────────────────

T_CELL_SUBSET_MARKERS = [
    # Th1
    "TBX21", "IFNG", "IL12RB2", "CXCR3", "STAT4",
    # Th2
    "GATA3", "IL4", "IL5", "IL13", "IL4R", "STAT6",
    # Th17
    "RORC", "IL17A", "IL17F", "IL22", "IL23R", "CCR6", "STAT3",
    # Treg
    "FOXP3", "IL2RA", "CTLA4", "TGFB1", "IL10", "IKZF2",  # Helios
    # Tfh
    "BCL6", "CXCR5", "ICOS", "IL21", "PDCD1",
    # Cytotoxic CD8+
    "PRF1", "GZMA", "GZMB", "GZMK",
    "GNLY", "NKG7", "FASLG",
    # Exhaustion markers
    "PDCD1", "LAG3", "HAVCR2", "TIGIT", "TOX", "TOX2",
    "ENTPD1",  # CD39
    # Memory / naive
    "CCR7", "SELL", "IL7R",  # CD62L
    "CD44", "KLRG1",
]

# ── Immunology: B Cell Biology ────────────────────────────────────

B_CELL_MARKERS = [
    # Pan-B cell
    "CD19", "MS4A1", "CD22", "PAX5", "CD79A", "CD79B",
    # Mature / memory
    "CD27", "TNFRSF13B",  # TACI
    "TNFRSF13C",  # BAFF-R
    # Germinal center
    "BCL6", "AICDA",
    # Plasma cell differentiation
    "PRDM1", "XBP1", "IRF4", "SDC1",  # CD138
    # Immunoglobulin isotypes
    "IGHM", "IGHG1", "IGHG2", "IGHG3", "IGHG4",
    "IGHA1", "IGHA2", "IGHE", "IGHD",
    # Class switch recombination
    "UNG", "MSH2", "MSH6",
]

# ── Immunology: Complement System ─────────────────────────────────

COMPLEMENT_GENES = [
    # Classical pathway
    "C1QA", "C1QB", "C1QC", "C1R", "C1S", "C2", "C4A", "C4B",
    # Lectin pathway
    "MBL2", "MASP1", "MASP2", "FCN1", "FCN2", "FCN3",
    # Alternative pathway
    "CFB", "CFD", "CFP",  # Factor B, D, properdin
    # Terminal / MAC
    "C3", "C5", "C6", "C7", "C8A", "C8B", "C8G", "C9",
    # Regulators
    "CFH", "CFI", "CD46", "CD55", "CD59", "SERPING1",  # C1-INH
]

# ── Immunology: Antigen Presentation ──────────────────────────────

ANTIGEN_PRESENTATION_GENES = [
    # MHC class I
    "HLA-A", "HLA-B", "HLA-C",
    "B2M",
    "TAP1", "TAP2", "TAPBP",
    "PSMB8", "PSMB9", "PSMB10",  # Immunoproteasome
    "ERAP1", "ERAP2",
    # MHC class II
    "HLA-DRA", "HLA-DRB1",
    "HLA-DPA1", "HLA-DPB1",
    "HLA-DQA1", "HLA-DQB1",
    "CIITA", "CD74",
    "CTSS", "CTSL",  # Cathepsins for peptide processing
    # Cross-presentation
    "SEC61A1", "CYBB",
    # Non-classical MHC
    "HLA-E", "HLA-G", "HFE",
    "CD1A", "CD1B", "CD1C", "CD1D",  # Lipid antigen presentation
]

# ── Immunology Drug Targets ──────────────────────────────────────

IMMUNE_THERAPEUTIC_DRUGS = [
    # Checkpoint inhibitors
    "Pembrolizumab", "Nivolumab", "Atezolizumab",
    "Durvalumab", "Ipilimumab", "Cemiplimab",
    "Relatlimab",
    # Cytokine therapies
    "Aldesleukin",  # IL-2
    "Interferon alfa-2b",
    # TLR agonists
    "Imiquimod",  # TLR7
    # Anti-CD20
    "Rituximab", "Obinutuzumab",
    # Anti-TNF
    "Infliximab", "Adalimumab",
    # JAK inhibitors
    "Tofacitinib", "Baricitinib", "Ruxolitinib",
]


# ── Diapedesis / Leukocyte Migration ────────────────────────────────
# Adhesion molecules, junctional proteins, chemokine receptors, and
# signaling molecules involved in leukocyte extravasation (Fig 3.3,
# Abbas Cellular and Molecular Immunology 10th Ed).

DIAPEDESIS_ADHESION_GENES = [
    # Selectins (rolling)
    "SELE",    # E-selectin (CD62E) — endothelial, cytokine-induced
    "SELP",    # P-selectin (CD62P) — endothelial/platelet, histamine/thrombin-induced
    "SELL",    # L-selectin (CD62L) — leukocyte, constitutive
    # Selectin ligands
    "SELPLG",  # PSGL-1 — primary selectin ligand on leukocytes
    "CD44",    # hyaluronan receptor, E-selectin ligand
    "FUT7",    # fucosyltransferase VII — required for sialyl Lewis-x synthesis
    "GLG1",    # ESL-1 (E-selectin ligand 1)
    # Integrins (leukocyte — arrest)
    "ITGAL",   # CD11a — alpha-L integrin (LFA-1 α chain)
    "ITGB2",   # CD18 — beta-2 integrin (shared by LFA-1, MAC-1, p150,95)
    "ITGAM",   # CD11b — alpha-M integrin (MAC-1 α chain)
    "ITGA4",   # alpha-4 integrin (VLA-4 and α4β7)
    "ITGB1",   # CD29 — beta-1 integrin (VLA-4 β chain)
    "ITGB7",   # beta-7 integrin (α4β7 for gut homing)
    "ITGAX",   # CD11c — alpha-X integrin (p150,95)
    # Integrin ligands (endothelial — arrest)
    "ICAM1",   # CD54 — ICAM-1, ligand for LFA-1 and MAC-1
    "ICAM2",   # CD102 — ICAM-2, constitutive ligand for LFA-1
    "VCAM1",   # CD106 — VCAM-1, ligand for VLA-4
    "MADCAM1", # MAdCAM-1 — mucosal addressin, ligand for α4β7
    # Junctional molecules (transmigration)
    "PECAM1",  # CD31 — PECAM-1, homophilic binding at junctions
    "CDH5",    # VE-cadherin (CD144) — adherens junction, must be disrupted
    "F11R",    # JAM-A — junctional adhesion molecule A
    "JAM2",    # JAM-B
    "JAM3",    # JAM-C
    "CD99",    # CD99 — homophilic, required for final diapedesis step
    "ESAM",    # endothelial cell-selective adhesion molecule
    # Chemokine receptors (integrin inside-out activation)
    "CXCR1",   # IL-8 receptor α — neutrophil
    "CXCR2",   # IL-8 receptor β — neutrophil
    "CCR2",    # MCP-1 receptor — monocyte
    "CXCR3",   # IP-10 receptor — Th1/CD8 T cell
    "CCR7",    # lymphocyte homing receptor
    "CXCR4",   # SDF-1 receptor — lymphocyte, bone marrow homing
    # Integrin inside-out signaling
    "RAPGEF1", # RapGEF1 — Rap1 activator
    "RAP1A",   # Rap1A — GTPase, integrin activation
    "RAP1B",   # Rap1B
    "FERMT3",  # kindlin-3 — integrin activator (LAD-3 gene)
    "TLN1",    # talin-1 — links integrins to actin cytoskeleton
    # Cytokines that induce endothelial adhesion molecule expression
    "TNF",     # TNF-α — induces E-selectin, ICAM-1, VCAM-1
    "IL1B",    # IL-1β — induces E-selectin, ICAM-1
    "IL1A",    # IL-1α
    "IFNG",    # IFN-γ — induces ICAM-1, MHC-II on endothelium
]


# ── Predefined Import Sets ──────────────────────────────────────────

IMPORT_SETS = {
    "Prostate Cancer (80 genes + 20 drugs)": {
        "genes": PROSTATE_CANCER_COMPREHENSIVE,
        "drugs": PROSTATE_CANCER_DRUGS,
        "pathways": PROSTATE_KEGG_PATHWAYS,
        "description": "Comprehensive prostate cancer gene panel including AR signaling, tumor suppressors, oncogenes, DNA repair, and neuroendocrine markers.",
    },
    "Pan-Cancer Drivers (80 genes)": {
        "genes": PAN_CANCER_DRIVERS,
        "drugs": [],
        "pathways": [],
        "description": "COSMIC Cancer Gene Census Tier 1 genes — the most commonly mutated driver genes across all cancer types.",
    },
    "Breast Cancer (38 genes + 20 drugs)": {
        "genes": BREAST_CANCER_CORE,
        "drugs": BREAST_CANCER_DRUGS,
        "pathways": [],
        "description": "Core breast cancer gene panel with hormonal, HER2, and triple-negative pathway genes.",
    },
    "DNA Repair (35 genes)": {
        "genes": DNA_REPAIR_GENES,
        "drugs": [],
        "pathways": [],
        "description": "Homologous recombination, mismatch repair, NER, BER, and DDR signaling genes.",
    },
    "Immune Checkpoints (20 genes)": {
        "genes": IMMUNE_CHECKPOINT_GENES,
        "drugs": [],
        "pathways": [],
        "description": "Immune checkpoint and co-stimulatory receptors/ligands for immunotherapy.",
    },
    "Kinase Targets (50 genes)": {
        "genes": KINASE_TARGETS,
        "drugs": [],
        "pathways": [],
        "description": "Druggable kinases (RTKs, cytoplasmic, Ser/Thr) that are therapeutic targets.",
    },
    "FDA Cancer Drugs (70 drugs)": {
        "genes": [],
        "drugs": FDA_APPROVED_CANCER_DRUGS,
        "pathways": [],
        "description": "FDA-approved cancer drugs including immunotherapy, targeted, chemo, and hormonal agents.",
    },
    "Full Catalog (all genes + drugs)": {
        "genes": list(set(
            PROSTATE_CANCER_COMPREHENSIVE + PAN_CANCER_DRIVERS +
            BREAST_CANCER_CORE + DNA_REPAIR_GENES +
            IMMUNE_CHECKPOINT_GENES + KINASE_TARGETS +
            METABOLIC_ENZYMES
        )),
        "drugs": list(set(
            PROSTATE_CANCER_DRUGS + BREAST_CANCER_DRUGS +
            FDA_APPROVED_CANCER_DRUGS
        )),
        "pathways": KEGG_CANCER_PATHWAYS,
        "description": "All curated gene sets combined — ~350 unique genes, ~80 drugs, 30 pathways.",
    },
    # ── Immunology Import Sets ──────────────────────────────────────
    "Immunology Foundations (~130 genes)": {
        "genes": list(set(
            IMMUNE_CHECKPOINT_GENES + T_CELL_SUBSET_MARKERS +
            B_CELL_MARKERS + ANTIGEN_PRESENTATION_GENES
        )),
        "drugs": [],
        "pathways": [],
        "description": "Core adaptive immunity: checkpoints, T cell subsets, B cell markers, and antigen presentation.",
    },
    "Innate Immunity (~80 genes)": {
        "genes": list(set(
            INNATE_IMMUNITY_GENES + COMPLEMENT_GENES
        )),
        "drugs": [],
        "pathways": [],
        "description": "Pattern recognition receptors (TLRs, NLRs, RLRs), complement cascade, antimicrobial peptides, and NK cell receptors.",
    },
    "Adaptive Immunity (~90 genes)": {
        "genes": list(set(
            ADAPTIVE_IMMUNITY_GENES + T_CELL_SUBSET_MARKERS +
            B_CELL_MARKERS
        )),
        "drugs": [],
        "pathways": [],
        "description": "TCR/BCR signaling, MHC antigen processing, T/B cell differentiation markers, and immunoglobulin genes.",
    },
    "Cytokine Network (~70 genes)": {
        "genes": CYTOKINE_CHEMOKINE_GENES,
        "drugs": [],
        "pathways": [],
        "description": "Full interleukin, interferon, chemokine, TNF, and colony-stimulating factor panel with receptors.",
    },
    "Immune Therapeutic Targets (~50 genes + 17 drugs)": {
        "genes": list(set(
            IMMUNE_CHECKPOINT_GENES +
            ["TLR7", "TLR9", "CD19", "MS4A1", "TNF", "JAK1", "JAK2", "JAK3",
             "IL2", "IL2RA", "IFNA1", "IFNA2", "CD3E", "CD19", "CD22"]
        )),
        "drugs": IMMUNE_THERAPEUTIC_DRUGS,
        "pathways": [],
        "description": "Immune drug targets: checkpoint inhibitors, cytokine therapies, TLR agonists, anti-CD20, anti-TNF, JAK inhibitors.",
    },
    "Complete Immunology (~300 genes + 17 drugs)": {
        "genes": list(set(
            INNATE_IMMUNITY_GENES + ADAPTIVE_IMMUNITY_GENES +
            CYTOKINE_CHEMOKINE_GENES + T_CELL_SUBSET_MARKERS +
            B_CELL_MARKERS + COMPLEMENT_GENES +
            ANTIGEN_PRESENTATION_GENES + IMMUNE_CHECKPOINT_GENES
        )),
        "drugs": IMMUNE_THERAPEUTIC_DRUGS,
        "pathways": [],
        "description": "All immunology gene sets combined — innate, adaptive, cytokines, complement, and therapeutic drugs.",
    },
    "Diapedesis & Leukocyte Migration (~40 genes)": {
        "genes": DIAPEDESIS_ADHESION_GENES,
        "drugs": [],
        "pathways": [],
        "description": "Selectins, integrins, Ig-superfamily adhesion molecules, junctional proteins, chemokine receptors, and signaling molecules for leukocyte rolling, arrest, and transmigration.",
    },
}
