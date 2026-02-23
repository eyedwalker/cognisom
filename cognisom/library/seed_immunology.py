"""
Seed Data — Comprehensive Immunology Entity Catalog
====================================================

Pre-built catalog of immunology entities for Harvard immunology coursework
and immune system simulation. Includes immune cell subtypes, cytokines,
pathogens, antibodies, complement components, PRRs, MHC molecules,
immunotherapy drugs, immune pathways, and their relationships.

Usage:
    from cognisom.library.seed_immunology import seed_immunology_catalog
    from cognisom.library.store import EntityStore

    store = EntityStore()
    seed_immunology_catalog(store)
"""

from __future__ import annotations

import logging
from typing import Dict, List

from .models import (
    AdhesionMolecule,
    Antibody,
    Antigen,
    Bacterium,
    ComplementComponent,
    Cytokine,
    Drug,
    ImmuneCellEntity,
    MHCMolecule,
    Pathway,
    PatternRecognitionReceptor,
    Relationship,
    RelationshipType,
    Virus,
)
from .store import EntityStore

log = logging.getLogger(__name__)


def seed_immunology_catalog(store: EntityStore) -> Dict[str, int]:
    """Seed the entity library with a comprehensive immunology catalog.

    Args:
        store: EntityStore to populate.

    Returns:
        dict with counts of entities created per category.
    """
    counts = {
        "immune_cells": 0,
        "cytokines": 0,
        "viruses": 0,
        "bacteria": 0,
        "antibodies": 0,
        "complement": 0,
        "prrs": 0,
        "mhc": 0,
        "adhesion_molecules": 0,
        "endothelial_cells": 0,
        "drugs": 0,
        "pathways": 0,
        "relationships": 0,
    }

    log.info("Seeding immunology catalog...")

    counts["immune_cells"] += _seed_immune_cell_types(store)
    counts["cytokines"] += _seed_cytokines(store)
    counts["viruses"] += _seed_pathogens_viruses(store)
    counts["bacteria"] += _seed_pathogens_bacteria(store)
    counts["antibodies"] += _seed_antibody_types(store)
    counts["complement"] += _seed_complement_components(store)
    counts["prrs"] += _seed_prrs(store)
    counts["mhc"] += _seed_mhc_molecules(store)
    counts["adhesion_molecules"] += _seed_adhesion_molecules(store)
    counts["endothelial_cells"] += _seed_endothelial_cells(store)
    counts["drugs"] += _seed_immune_drugs(store)
    counts["pathways"] += _seed_immune_pathways(store)
    counts["relationships"] += _seed_immune_relationships(store)
    counts["relationships"] += _seed_diapedesis_relationships(store)

    total = sum(counts.values())
    log.info("Immunology seed complete: %d total items created", total)
    return counts


# ── Immune Cell Types ──────────────────────────────────────────────

def _seed_immune_cell_types(store: EntityStore) -> int:
    cells = [
        # (name, immune_type, subtype, markers, cytokines_secreting, description)
        ("CD8+ Naive T cell", "T_cell", "naive",
         ["CD8", "CD45RA", "CCR7", "CD62L"],
         [],
         "Naive cytotoxic T cell. Not yet encountered antigen. Circulates between blood and lymph nodes."),
        ("CD8+ Effector T cell", "T_cell", "effector",
         ["CD8", "CD45RO", "KLRG1", "granzyme_B", "perforin"],
         ["IFNg", "TNFa", "granzyme_B"],
         "Activated cytotoxic T lymphocyte. Kills virus-infected and tumor cells via perforin/granzyme."),
        ("CD8+ Memory T cell", "T_cell", "memory",
         ["CD8", "CD45RO", "CD127", "CCR7"],
         ["IFNg"],
         "Long-lived memory CTL. Rapid recall response upon re-encounter with antigen."),
        ("CD4+ Th1 cell", "T_cell", "Th1",
         ["CD4", "CXCR3", "CCR5", "T-bet"],
         ["IFNg", "TNFa", "IL2"],
         "Type 1 helper T cell. Drives cell-mediated immunity against intracellular pathogens. Master TF: T-bet."),
        ("CD4+ Th2 cell", "T_cell", "Th2",
         ["CD4", "CCR4", "CCR8", "GATA3"],
         ["IL4", "IL5", "IL13"],
         "Type 2 helper T cell. Drives humoral immunity and anti-helminth responses. Master TF: GATA3."),
        ("CD4+ Th17 cell", "T_cell", "Th17",
         ["CD4", "CCR6", "IL23R", "RORgt"],
         ["IL17A", "IL17F", "IL22"],
         "IL-17-producing helper T cell. Mucosal defense against extracellular bacteria/fungi. Master TF: RORgt."),
        ("Regulatory T cell (Treg)", "T_cell", "Treg",
         ["CD4", "CD25", "FOXP3", "CTLA4", "CD127lo"],
         ["IL10", "TGFb", "IL35"],
         "Immunosuppressive T cell. Maintains self-tolerance and prevents autoimmunity. Master TF: FOXP3."),
        ("T follicular helper (Tfh)", "T_cell", "Tfh",
         ["CD4", "CXCR5", "PD-1", "ICOS", "BCL6"],
         ["IL21", "IL4"],
         "Germinal center helper T cell. Provides help to B cells for affinity maturation and class switching."),
        ("Gamma-delta T cell", "T_cell", "gamma_delta",
         ["TCRgd", "CD3", "NKG2D"],
         ["IFNg", "IL17A"],
         "Unconventional T cell with gamma-delta TCR. Bridges innate and adaptive immunity. Tissue-resident sentinel."),
        ("NK cell", "NK_cell", "NK",
         ["CD56", "CD16", "NKG2D", "NKp46", "KIR"],
         ["IFNg", "TNFa", "perforin", "granzyme_B"],
         "Natural killer cell. Innate cytotoxic lymphocyte. Kills via missing-self (low MHC-I) and ADCC."),
        ("NKT cell", "NK_cell", "NKT",
         ["CD3", "Va24Ja18TCR", "CD1d-reactive", "CD56"],
         ["IFNg", "IL4"],
         "Natural killer T cell. Recognizes lipid antigens on CD1d. Bridges innate and adaptive immunity."),
        ("Macrophage M1", "macrophage", "M1",
         ["CD68", "CD80", "CD86", "iNOS", "HLA-DR"],
         ["TNFa", "IL1b", "IL6", "IL12", "NO"],
         "Classically activated macrophage. Pro-inflammatory, microbicidal. Activated by IFNg + LPS."),
        ("Macrophage M2", "macrophage", "M2",
         ["CD68", "CD163", "CD206", "arginase-1"],
         ["IL10", "TGFb"],
         "Alternatively activated macrophage. Anti-inflammatory, tissue repair. Activated by IL-4/IL-13."),
        ("Conventional dendritic cell (cDC)", "dendritic", "cDC",
         ["CD11c", "HLA-DR", "CD1c", "CD141"],
         ["IL12", "IL6", "TNFa"],
         "Professional antigen-presenting cell. Cross-presents antigen on MHC-I/II. Activates naive T cells."),
        ("Plasmacytoid dendritic cell (pDC)", "dendritic", "pDC",
         ["CD123", "CD303", "CD304"],
         ["IFNa", "IFNb"],
         "Type I interferon-producing dendritic cell. Key antiviral sentinel via TLR7/9 sensing."),
        ("Naive B cell", "B_cell", "naive",
         ["CD19", "CD20", "IgM", "IgD", "CD23"],
         [],
         "Mature naive B cell expressing surface IgM/IgD. Not yet encountered antigen."),
        ("Memory B cell", "B_cell", "memory",
         ["CD19", "CD20", "CD27", "class-switched_Ig"],
         [],
         "Long-lived memory B cell. Rapid recall response with high-affinity class-switched antibodies."),
        ("Plasma cell", "B_cell", "plasma_cell",
         ["CD138", "CD38", "BLIMP-1", "XBP1"],
         [],
         "Antibody-secreting cell. Terminally differentiated B cell producing ~10,000 antibodies/second."),
        ("Neutrophil", "neutrophil", "neutrophil",
         ["CD15", "CD16", "CD66b", "MPO"],
         ["IL8", "TNFa"],
         "Most abundant granulocyte. First responder to bacterial infection. Phagocytosis, NETs, degranulation."),
        ("Mast cell", "mast_cell", "mast_cell",
         ["FceRI", "CD117", "tryptase", "chymase"],
         ["histamine", "TNFa", "IL4", "IL13"],
         "Tissue-resident granulocyte. IgE-mediated degranulation. Allergic responses and anti-parasite defense."),
        ("Basophil", "basophil", "basophil",
         ["FceRI", "CD123", "CD203c"],
         ["IL4", "IL13", "histamine"],
         "Rare blood granulocyte. Type 2 immunity, anti-parasite. Th2 differentiation via early IL-4."),
        ("Eosinophil", "eosinophil", "eosinophil",
         ["CD193", "Siglec-8", "MBP", "EPO"],
         ["IL5", "IL13", "MBP"],
         "Granulocyte for anti-helminth defense. Eosinophilic granule proteins damage large parasites."),
        ("ILC1", "ILC", "ILC1",
         ["T-bet", "NK1.1", "NKp46", "CD127"],
         ["IFNg", "TNFa"],
         "Group 1 innate lymphoid cell. IFNg-producing, mirrors Th1 function. Anti-intracellular pathogens."),
        ("ILC2", "ILC", "ILC2",
         ["GATA3", "CD127", "CRTH2", "ST2"],
         ["IL5", "IL13", "IL4"],
         "Group 2 innate lymphoid cell. Type 2 cytokines, mirrors Th2. Anti-helminth and allergic inflammation."),
        ("ILC3", "ILC", "ILC3",
         ["RORgt", "CD127", "NKp44"],
         ["IL22", "IL17A"],
         "Group 3 innate lymphoid cell. IL-22/IL-17 producing, mirrors Th17. Mucosal barrier defense."),
    ]
    count = 0
    for name, itype, subtype, markers, cytokines, desc in cells:
        entity = ImmuneCellEntity(
            name=name,
            display_name=name,
            description=desc,
            immune_type=itype,
            immune_subtype=subtype,
            surface_markers=markers,
            cytokines_secreting=cytokines,
            source="curated",
            tags=["immunology", itype, subtype],
        )
        store.add_entity(entity)
        count += 1
    return count


# ── Cytokines ──────────────────────────────────────────────────────

def _seed_cytokines(store: EntityStore) -> int:
    cytokines = [
        # (name, family, receptor, producing, target, function, pathway, pro_inflammatory, gene, mw_kda)
        ("IL-1beta", "interleukin", "IL1R1/IL1RAP", ["macrophage_M1", "dendritic"], ["T_cell", "endothelial"],
         "Potent pro-inflammatory cytokine. Fever, acute phase response, inflammasome effector.",
         "NF-kB", True, "IL1B", 17.4),
        ("IL-2", "interleukin", "IL2RA/IL2RB/IL2RG", ["Th1", "CD8_effector"], ["T_cell", "NK_cell", "Treg"],
         "T cell growth factor. Drives clonal expansion of activated T cells. Essential for Treg survival.",
         "JAK-STAT", True, "IL2", 15.5),
        ("IL-4", "interleukin", "IL4R/IL13RA1", ["Th2", "basophil", "mast_cell"], ["B_cell", "macrophage"],
         "Th2 master cytokine. Drives IgE class switch, M2 macrophage polarization.",
         "JAK-STAT", False, "IL4", 15.0),
        ("IL-6", "interleukin", "IL6R/IL6ST", ["macrophage", "T_cell", "fibroblast"], ["hepatocyte", "B_cell", "T_cell"],
         "Pleiotropic cytokine. Acute phase response, B cell differentiation, Th17 differentiation.",
         "JAK-STAT", True, "IL6", 21.0),
        ("IL-10", "interleukin", "IL10RA/IL10RB", ["Treg", "macrophage_M2", "Th2"], ["macrophage", "dendritic", "T_cell"],
         "Master anti-inflammatory cytokine. Suppresses pro-inflammatory responses. Immune regulation.",
         "JAK-STAT", False, "IL10", 18.6),
        ("IL-12", "interleukin", "IL12RB1/IL12RB2", ["dendritic", "macrophage_M1"], ["T_cell", "NK_cell"],
         "Drives Th1 differentiation and IFNg production. Key bridge between innate and adaptive immunity.",
         "JAK-STAT", True, "IL12B", 70.0),
        ("IL-17A", "interleukin", "IL17RA/IL17RC", ["Th17", "gamma_delta_T", "ILC3"], ["epithelial", "neutrophil"],
         "Mucosal defense cytokine. Recruits neutrophils, induces antimicrobial peptides.",
         "NF-kB", True, "IL17A", 15.5),
        ("IL-21", "interleukin", "IL21R/IL2RG", ["Tfh", "Th17"], ["B_cell", "plasma_cell", "CD8_T"],
         "Germinal center cytokine. Drives B cell differentiation, plasma cell formation, Ig class switch.",
         "JAK-STAT", True, "IL21", 15.0),
        ("IL-23", "interleukin", "IL23R/IL12RB1", ["dendritic", "macrophage"], ["Th17", "ILC3"],
         "Th17 maintenance cytokine. Stabilizes Th17 phenotype. Key in autoimmunity and mucosal defense.",
         "JAK-STAT", True, "IL23A", 58.0),
        ("IL-33", "interleukin", "IL1RL1 (ST2)", ["epithelial", "endothelial"], ["ILC2", "Th2", "mast_cell"],
         "Alarmin cytokine. Released on tissue damage, activates type 2 immunity. DAMP function.",
         "NF-kB", False, "IL33", 30.0),
        ("IFN-gamma", "interferon", "IFNGR1/IFNGR2", ["Th1", "CD8_T", "NK_cell"], ["macrophage", "dendritic", "all_nucleated"],
         "Type II interferon. Activates macrophages (M1), upregulates MHC-I/II, antiviral state.",
         "JAK-STAT", True, "IFNG", 17.0),
        ("IFN-alpha", "interferon", "IFNAR1/IFNAR2", ["pDC", "macrophage"], ["all_nucleated"],
         "Type I interferon. Antiviral state induction, MHC-I upregulation, NK cell activation.",
         "JAK-STAT", True, "IFNA1", 19.0),
        ("IFN-beta", "interferon", "IFNAR1/IFNAR2", ["fibroblast", "dendritic", "macrophage"], ["all_nucleated"],
         "Type I interferon. Antiviral and immunomodulatory. Drives ISG expression.",
         "JAK-STAT", True, "IFNB1", 20.0),
        ("TNF-alpha", "TNF", "TNFR1/TNFR2", ["macrophage_M1", "T_cell", "NK_cell"], ["endothelial", "tumor", "macrophage"],
         "Master pro-inflammatory cytokine. Septic shock, cachexia, NF-kB activation, apoptosis.",
         "NF-kB", True, "TNF", 17.4),
        ("TGF-beta1", "TGF", "TGFBR1/TGFBR2", ["Treg", "macrophage", "platelet"], ["T_cell", "B_cell", "fibroblast"],
         "Pleiotropic immunosuppressive cytokine. Treg induction, fibrosis, wound healing.",
         "SMAD", False, "TGFB1", 25.0),
        ("CXCL8 (IL-8)", "chemokine", "CXCR1/CXCR2", ["macrophage", "epithelial", "neutrophil"], ["neutrophil"],
         "Neutrophil chemoattractant. Major driver of neutrophil recruitment to infection sites.",
         "NF-kB", True, "CXCL8", 8.4),
        ("CCL2 (MCP-1)", "chemokine", "CCR2", ["macrophage", "endothelial", "fibroblast"], ["monocyte", "macrophage"],
         "Monocyte chemoattractant protein 1. Recruits monocytes to inflammation sites.",
         "NF-kB", True, "CCL2", 11.0),
        ("CXCL10 (IP-10)", "chemokine", "CXCR3", ["macrophage", "dendritic", "endothelial"], ["Th1", "CD8_T", "NK_cell"],
         "IFNg-induced chemokine. Recruits Th1 and cytotoxic T cells to infection sites.",
         "JAK-STAT", True, "CXCL10", 8.7),
        ("GM-CSF", "CSF", "CSF2RA/CSF2RB", ["T_cell", "macrophage", "fibroblast"], ["neutrophil", "macrophage", "dendritic"],
         "Granulocyte-macrophage colony-stimulating factor. Emergency myelopoiesis during infection.",
         "JAK-STAT", True, "CSF2", 14.5),
        ("M-CSF", "CSF", "CSF1R", ["macrophage", "fibroblast", "endothelial"], ["monocyte", "macrophage"],
         "Macrophage colony-stimulating factor. Macrophage differentiation, survival, and proliferation.",
         "MAPK", False, "CSF1", 60.0),
    ]
    count = 0
    for name, family, receptor, producing, target, function, pathway, pro_inf, gene, mw in cytokines:
        entity = Cytokine(
            name=name,
            display_name=name,
            description=function,
            cytokine_family=family,
            receptor=receptor,
            producing_cells=producing,
            target_cells=target,
            function=function,
            signaling_pathway=pathway,
            pro_inflammatory=pro_inf,
            gene_symbol=gene,
            molecular_weight_kda=mw,
            source="curated",
            tags=["immunology", "cytokine", family],
        )
        store.add_entity(entity)
        count += 1
    return count


# ── Pathogens: Viruses ─────────────────────────────────────────────

def _seed_pathogens_viruses(store: EntityStore) -> int:
    viruses = [
        # (name, family, genome, capsid, envelope, size_kb, replication, tropism, receptors, evasion, incubation)
        ("SARS-CoV-2", "Coronaviridae", "ssRNA+", "helical", True, 29.9, 100.0,
         ["respiratory_epithelial"], ["ACE2", "TMPRSS2"],
         ["antigenic_drift", "interferon_antagonism", "ORF_immune_evasion"], 120.0),
        ("Influenza A", "Orthomyxoviridae", "ssRNA-", "helical", True, 13.5, 50.0,
         ["respiratory_epithelial"], ["sialic_acid"],
         ["antigenic_shift", "antigenic_drift", "NS1_interferon_block"], 48.0),
        ("HIV-1", "Retroviridae", "retro", "icosahedral", True, 9.7, 10.0,
         ["CD4_T_cell", "macrophage"], ["CD4", "CCR5", "CXCR4"],
         ["latency", "antigenic_variation", "Nef_MHC_downregulation", "Vpu_CD4_degradation"], 504.0),
        ("Epstein-Barr virus (EBV)", "Herpesviridae", "dsDNA", "icosahedral", True, 172.0, 1.0,
         ["B_cell", "epithelial"], ["CD21", "HLA-DQ"],
         ["latency", "IL10_homolog", "EBNA1_antigen_processing_escape"], 720.0),
        ("Hepatitis B virus (HBV)", "Hepadnaviridae", "dsDNA", "icosahedral", True, 3.2, 5.0,
         ["hepatocyte"], ["NTCP"],
         ["cccDNA_persistence", "HBx_immune_evasion", "decoy_subviral_particles"], 1440.0),
        ("Zika virus", "Flaviviridae", "ssRNA+", "icosahedral", True, 10.8, 20.0,
         ["neural_progenitor", "dendritic"], ["AXL", "DC-SIGN"],
         ["NS5_STAT2_degradation", "interferon_antagonism"], 168.0),
        ("Measles virus", "Paramyxoviridae", "ssRNA-", "helical", True, 15.9, 30.0,
         ["dendritic", "T_cell", "B_cell", "macrophage"], ["CD150", "nectin-4"],
         ["immune_amnesia", "STAT_signaling_block", "DC_suppression"], 240.0),
        ("Rabies virus", "Rhabdoviridae", "ssRNA-", "helical", True, 12.0, 5.0,
         ["neuron", "muscle"], ["nAChR", "NCAM", "p75NTR"],
         ["neurotropism", "BBB_evasion", "apoptosis_inhibition"], 2160.0),
    ]
    count = 0
    for name, family, genome, capsid, env, size, rep, tropism, receptors, evasion, incub in viruses:
        entity = Virus(
            name=name,
            display_name=name,
            description=f"{family} virus. Genome: {genome}, {size} kb. Targets: {', '.join(receptors)}.",
            virus_family=family,
            genome_type=genome,
            capsid_type=capsid,
            envelope=env,
            genome_size_kb=size,
            replication_rate=rep,
            host_tropism=tropism,
            target_receptors=receptors,
            evasion_mechanisms=evasion,
            incubation_hours=incub,
            source="curated",
            tags=["immunology", "pathogen", "virus", family],
        )
        store.add_entity(entity)
        count += 1
    return count


# ── Pathogens: Bacteria ────────────────────────────────────────────

def _seed_pathogens_bacteria(store: EntityStore) -> int:
    bacteria = [
        # (name, gram, shape, oxygen, pathogenicity, resistance, growth, toxins, niche, genome_mb)
        ("Staphylococcus aureus", "positive", "coccus", "facultative",
         ["protein_A", "coagulase", "PVL", "TSST-1"],
         ["methicillin", "vancomycin"],
         1.0, ["alpha-hemolysin", "TSST-1", "enterotoxin_B"],
         "extracellular", 2.8),
        ("Escherichia coli (UPEC)", "negative", "rod", "facultative",
         ["type_1_fimbriae", "P_fimbriae", "LPS", "capsule"],
         ["ampicillin", "trimethoprim"],
         2.0, ["endotoxin", "hemolysin"],
         "extracellular", 5.0),
        ("Mycobacterium tuberculosis", "acid-fast", "rod", "aerobic",
         ["cord_factor", "ESAT-6", "CFP-10", "mycolic_acid"],
         ["isoniazid", "rifampicin"],
         0.02, ["cord_factor"],
         "intracellular", 4.4),
        ("Streptococcus pyogenes", "positive", "coccus", "facultative",
         ["M_protein", "streptolysin_O", "hyaluronidase", "C5a_peptidase"],
         [],
         1.5, ["streptolysin_O", "streptolysin_S", "erythrogenic_toxin"],
         "extracellular", 1.8),
        ("Salmonella enterica", "negative", "rod", "facultative",
         ["type_III_secretion", "Vi_capsule", "flagella"],
         ["ciprofloxacin", "azithromycin"],
         1.5, ["endotoxin"],
         "intracellular", 4.8),
        ("Neisseria meningitidis", "negative", "coccus", "aerobic",
         ["polysaccharide_capsule", "pili", "LOS", "IgA_protease"],
         ["penicillin"],
         1.0, ["endotoxin"],
         "extracellular", 2.3),
        ("Clostridioides difficile", "positive", "rod", "anaerobic",
         ["toxin_A", "toxin_B", "binary_toxin", "spore_formation"],
         ["metronidazole", "vancomycin"],
         0.5, ["toxin_A_TcdA", "toxin_B_TcdB"],
         "extracellular", 4.3),
        ("Pseudomonas aeruginosa", "negative", "rod", "aerobic",
         ["type_III_secretion", "biofilm", "alginate", "pyocyanin"],
         ["carbapenem", "aminoglycoside", "fluoroquinolone"],
         1.5, ["exotoxin_A", "pyocyanin"],
         "extracellular", 6.3),
    ]
    count = 0
    for name, gram, shape, oxygen, pathogenicity, resistance, growth, toxins, niche, genome in bacteria:
        entity = Bacterium(
            name=name,
            display_name=name,
            description=f"Gram-{gram} {shape}. Niche: {niche}. Key factors: {', '.join(pathogenicity[:3])}.",
            gram_stain=gram,
            shape=shape,
            oxygen_requirement=oxygen,
            pathogenicity_factors=pathogenicity,
            antibiotic_resistance=resistance,
            growth_rate=growth,
            toxins=toxins,
            host_niche=niche,
            genome_size_mb=genome,
            source="curated",
            tags=["immunology", "pathogen", "bacterium"],
        )
        store.add_entity(entity)
        count += 1
    return count


# ── Antibody Isotypes ──────────────────────────────────────────────

def _seed_antibody_types(store: EntityStore) -> int:
    antibodies = [
        # (name, isotype, heavy, light, fc_functions, half_life, valency, complement)
        ("IgG1", "IgG1", "gamma1", "kappa/lambda",
         ["ADCC", "CDC", "opsonization", "neutralization", "placental_transfer"],
         21.0, 2, True),
        ("IgG2", "IgG2", "gamma2", "kappa/lambda",
         ["opsonization", "anti-polysaccharide"],
         21.0, 2, False),
        ("IgG3", "IgG3", "gamma3", "kappa/lambda",
         ["ADCC", "CDC", "opsonization", "neutralization"],
         7.0, 2, True),
        ("IgG4", "IgG4", "gamma4", "kappa/lambda",
         ["neutralization", "blocking_antibody"],
         21.0, 2, False),
        ("IgA1 (secretory)", "IgA1", "alpha1", "kappa/lambda",
         ["mucosal_neutralization", "immune_exclusion"],
         6.0, 4, False),
        ("IgE", "IgE", "epsilon", "kappa/lambda",
         ["mast_cell_degranulation", "anti-helminth", "ADCC_eosinophil"],
         2.5, 2, False),
        ("IgM (pentameric)", "IgM", "mu", "kappa/lambda",
         ["complement_activation", "agglutination", "first_response"],
         5.0, 10, True),
        ("IgD", "IgD", "delta", "kappa/lambda",
         ["B_cell_receptor", "basophil_activation"],
         3.0, 2, False),
    ]
    count = 0
    for name, isotype, heavy, light, fc, half_life, valency, complement in antibodies:
        entity = Antibody(
            name=name,
            display_name=name,
            description=f"{isotype} antibody. Functions: {', '.join(fc[:3])}. Half-life: {half_life} days.",
            isotype=isotype,
            heavy_chain=heavy,
            light_chain=light,
            fc_function=fc,
            half_life_days=half_life,
            valency=valency,
            complement_fixation=complement,
            source="curated",
            tags=["immunology", "antibody", isotype],
        )
        store.add_entity(entity)
        count += 1
    return count


# ── Complement Components ──────────────────────────────────────────

def _seed_complement_components(store: EntityStore) -> int:
    components = [
        # (name, pathway, step, products, function, deficiency, gene, conc_ug_ml)
        ("C1q", "classical", 1, ["C1q-Ab complex"],
         "Pattern recognition. Binds IgG/IgM Fc regions and apoptotic cells.",
         "SLE-like autoimmunity", "C1QA", 70.0),
        ("C3", "all", 3, ["C3a", "C3b"],
         "Central hub of complement. C3b opsonizes; C3a is anaphylatoxin. Amplification loop.",
         "Recurrent pyogenic infections", "C3", 1300.0),
        ("C3a", "all", 3, [],
         "Anaphylatoxin. Mast cell degranulation, smooth muscle contraction, chemotaxis.",
         "", "", 0.0),
        ("C3b", "all", 3, ["iC3b"],
         "Opsonin. Tags pathogens for phagocytosis via CR1. Also part of C3/C5 convertase.",
         "", "", 0.0),
        ("C5", "terminal", 5, ["C5a", "C5b"],
         "C5a is most potent anaphylatoxin. C5b initiates MAC assembly.",
         "Recurrent Neisseria infections", "C5", 75.0),
        ("C5a", "terminal", 5, [],
         "Most potent anaphylatoxin and chemotactic factor. Neutrophil activation.",
         "", "", 0.0),
        ("Factor B", "alternative", 2, ["Ba", "Bb"],
         "Alternative pathway serine protease. Forms C3bBb (alternative C3 convertase).",
         "Recurrent infections", "CFB", 200.0),
        ("Factor D", "alternative", 1, [],
         "Rate-limiting enzyme of alternative pathway. Cleaves Factor B.",
         "Recurrent infections", "CFD", 2.0),
        ("Properdin", "alternative", 2, [],
         "Only positive regulator of complement. Stabilizes alternative C3 convertase.",
         "Recurrent Neisseria infections", "CFP", 25.0),
        ("MBL", "lectin", 1, ["MBL-MASP complex"],
         "Mannose-binding lectin. Recognizes mannose/fucose on microbial surfaces.",
         "Increased susceptibility to infection in children", "MBL2", 1.5),
        ("MAC (C5b-9)", "terminal", 9, [],
         "Membrane attack complex. Pore-forming complex that lyses gram-negative bacteria.",
         "Recurrent Neisseria infections", "C9", 60.0),
    ]
    count = 0
    for name, pathway, step, products, function, deficiency, gene, conc in components:
        entity = ComplementComponent(
            name=name,
            display_name=name,
            description=function,
            pathway=pathway,
            activation_step=step,
            cleavage_products=products,
            function=function,
            deficiency_phenotype=deficiency,
            gene_symbol=gene,
            serum_concentration_ug_ml=conc,
            source="curated",
            tags=["immunology", "complement", pathway],
        )
        store.add_entity(entity)
        count += 1
    return count


# ── Pattern Recognition Receptors ──────────────────────────────────

def _seed_prrs(store: EntityStore) -> int:
    prrs = [
        # (name, type, ligands, pamp/damp, pathway, cells, effectors, location, gene)
        ("TLR4", "TLR", ["LPS", "HMGB1"], "both",
         "MyD88-dependent + TRIF-dependent", ["macrophage", "dendritic", "neutrophil"],
         ["NF-kB", "IRF3", "TNFa", "IFNb"], "cell_surface", "TLR4"),
        ("TLR3", "TLR", ["dsRNA"], "PAMP",
         "TRIF-dependent", ["dendritic", "macrophage"],
         ["IRF3", "IFNb", "IL12"], "endosomal", "TLR3"),
        ("TLR7", "TLR", ["ssRNA"], "PAMP",
         "MyD88-dependent", ["pDC", "B_cell"],
         ["IRF7", "IFNa", "NF-kB"], "endosomal", "TLR7"),
        ("TLR9", "TLR", ["CpG_DNA"], "PAMP",
         "MyD88-dependent", ["pDC", "B_cell"],
         ["IRF7", "IFNa", "NF-kB"], "endosomal", "TLR9"),
        ("TLR2/1", "TLR", ["triacyl_lipopeptide", "peptidoglycan"], "PAMP",
         "MyD88-dependent", ["macrophage", "dendritic"],
         ["NF-kB", "TNFa", "IL6"], "cell_surface", "TLR2"),
        ("TLR5", "TLR", ["flagellin"], "PAMP",
         "MyD88-dependent", ["macrophage", "dendritic", "epithelial"],
         ["NF-kB", "TNFa", "IL8"], "cell_surface", "TLR5"),
        ("NLRP3 inflammasome", "NLR", ["ATP", "uric_acid", "silica", "cholesterol_crystals"], "DAMP",
         "inflammasome", ["macrophage", "dendritic"],
         ["caspase-1", "IL1b", "IL18", "pyroptosis"], "cytoplasmic", "NLRP3"),
        ("NOD2", "NLR", ["muramyl_dipeptide"], "PAMP",
         "NF-kB + autophagy", ["macrophage", "dendritic", "Paneth_cell"],
         ["NF-kB", "autophagy", "defensins"], "cytoplasmic", "NOD2"),
        ("RIG-I", "RLR", ["5ppp_dsRNA", "short_dsRNA"], "PAMP",
         "MAVS-dependent", ["all_nucleated"],
         ["IRF3", "IRF7", "IFNb", "NF-kB"], "cytoplasmic", "DDX58"),
        ("MDA5", "RLR", ["long_dsRNA"], "PAMP",
         "MAVS-dependent", ["all_nucleated"],
         ["IRF3", "IFNb"], "cytoplasmic", "IFIH1"),
        ("cGAS-STING", "cGAS_STING", ["cytosolic_dsDNA"], "both",
         "STING-TBK1-IRF3", ["macrophage", "dendritic", "all_nucleated"],
         ["IRF3", "IFNb", "NF-kB", "autophagy"], "cytoplasmic", "MB21D1"),
        ("Dectin-1", "CLR", ["beta-glucan"], "PAMP",
         "Syk-CARD9", ["macrophage", "dendritic", "neutrophil"],
         ["NF-kB", "IL6", "TNFa", "ROS"], "cell_surface", "CLEC7A"),
    ]
    count = 0
    for name, ptype, ligands, pamp_damp, pathway, cells, effectors, loc, gene in prrs:
        entity = PatternRecognitionReceptor(
            name=name,
            display_name=name,
            description=f"{ptype} sensing {', '.join(ligands[:2])}. Pathway: {pathway}.",
            prr_type=ptype,
            ligands=ligands,
            pamp_or_damp=pamp_damp,
            signaling_pathway=pathway,
            cell_expression=cells,
            downstream_effectors=effectors,
            subcellular_location=loc,
            gene_symbol=gene,
            source="curated",
            tags=["immunology", "prr", ptype],
        )
        store.add_entity(entity)
        count += 1
    return count


# ── MHC Molecules ──────────────────────────────────────────────────

def _seed_mhc_molecules(store: EntityStore) -> int:
    mhc_molecules = [
        # (name, class, allele, gene, peptide_range, distribution, presenting, diseases)
        ("HLA-A*02:01", "I", "HLA-A*02:01", "HLA-A", [8, 10],
         ["all_nucleated"], [],
         ["Type 1 diabetes", "Ankylosing spondylitis"]),
        ("HLA-A*03:01", "I", "HLA-A*03:01", "HLA-A", [8, 10],
         ["all_nucleated"], [],
         ["Hemochromatosis"]),
        ("HLA-B*27:05", "I", "HLA-B*27:05", "HLA-B", [8, 10],
         ["all_nucleated"], [],
         ["Ankylosing spondylitis", "Reactive arthritis"]),
        ("HLA-B*57:01", "I", "HLA-B*57:01", "HLA-B", [8, 10],
         ["all_nucleated"], [],
         ["HIV elite controller", "Abacavir hypersensitivity"]),
        ("HLA-C*07:02", "I", "HLA-C*07:02", "HLA-C", [8, 10],
         ["all_nucleated"], [],
         []),
        ("HLA-DRB1*04:01", "II", "HLA-DRB1*04:01", "HLA-DRB1", [13, 25],
         ["thymic_epithelial"], ["dendritic", "macrophage", "B_cell"],
         ["Rheumatoid arthritis", "Type 1 diabetes"]),
        ("HLA-DRB1*15:01", "II", "HLA-DRB1*15:01", "HLA-DRB1", [13, 25],
         ["thymic_epithelial"], ["dendritic", "macrophage", "B_cell"],
         ["Multiple sclerosis", "SLE"]),
        ("HLA-DQ2.5", "II", "HLA-DQA1*05:01/DQB1*02:01", "HLA-DQB1", [13, 25],
         ["thymic_epithelial"], ["dendritic", "macrophage", "B_cell"],
         ["Celiac disease", "Type 1 diabetes"]),
        ("HLA-DQ8", "II", "HLA-DQA1*03:01/DQB1*03:02", "HLA-DQB1", [13, 25],
         ["thymic_epithelial"], ["dendritic", "macrophage", "B_cell"],
         ["Type 1 diabetes", "Celiac disease"]),
        ("B2M (beta-2 microglobulin)", "I", "", "B2M", [8, 10],
         ["all_nucleated"], [],
         ["MHC class I deficiency"]),
    ]
    count = 0
    for name, mclass, allele, gene, pep_range, dist, presenting, diseases in mhc_molecules:
        entity = MHCMolecule(
            name=name,
            display_name=name,
            description=f"MHC class {mclass}. Presents {pep_range[0]}-{pep_range[1]}mer peptides.",
            mhc_class=mclass,
            hla_allele=allele,
            gene_symbol=gene,
            peptide_length_range=pep_range,
            tissue_distribution=dist,
            presenting_cell_types=presenting,
            associated_diseases=diseases,
            source="curated",
            tags=["immunology", "mhc", f"class_{mclass}"],
        )
        store.add_entity(entity)
        count += 1
    return count


# ── Immunotherapy Drugs ────────────────────────────────────────────

def _seed_immune_drugs(store: EntityStore) -> int:
    drugs = [
        # (name, drug_class, mechanism, targets, status)
        ("Nivolumab", "checkpoint_inhibitor",
         "Anti-PD-1 monoclonal antibody. Blocks PD-1/PD-L1 interaction to restore T cell anti-tumor immunity.",
         ["PDCD1"], "approved"),
        ("Ipilimumab", "checkpoint_inhibitor",
         "Anti-CTLA-4 monoclonal antibody. Blocks inhibitory CTLA-4 signaling, enhancing T cell activation in lymph nodes.",
         ["CTLA4"], "approved"),
        ("Atezolizumab", "checkpoint_inhibitor",
         "Anti-PD-L1 monoclonal antibody. Blocks PD-L1 on tumor cells from engaging PD-1 on T cells.",
         ["CD274"], "approved"),
        ("Relatlimab", "checkpoint_inhibitor",
         "Anti-LAG-3 monoclonal antibody. Blocks LAG-3 to prevent T cell exhaustion.",
         ["LAG3"], "approved"),
        ("Rituximab", "anti_cd20",
         "Anti-CD20 monoclonal antibody. Depletes B cells via ADCC, CDC, and apoptosis.",
         ["MS4A1"], "approved"),
        ("Aldesleukin (IL-2)", "cytokine_therapy",
         "Recombinant IL-2. Stimulates T cell and NK cell proliferation and activation.",
         ["IL2RA", "IL2RB"], "approved"),
        ("Interferon alfa-2b", "cytokine_therapy",
         "Recombinant IFN-alpha. Antiviral and immunomodulatory. Upregulates MHC-I.",
         ["IFNAR1", "IFNAR2"], "approved"),
        ("Tofacitinib", "JAK_inhibitor",
         "JAK1/JAK3 inhibitor. Blocks cytokine signaling (IL-2, IL-6, IFNg). Immunosuppressive.",
         ["JAK1", "JAK3"], "approved"),
        ("Ruxolitinib", "JAK_inhibitor",
         "JAK1/JAK2 inhibitor. Reduces inflammatory cytokine production.",
         ["JAK1", "JAK2"], "approved"),
        ("Infliximab", "anti_TNF",
         "Anti-TNF chimeric monoclonal antibody. Neutralizes soluble and membrane TNF-alpha.",
         ["TNF"], "approved"),
        ("Adalimumab", "anti_TNF",
         "Fully human anti-TNF monoclonal antibody. Blocks TNF-alpha signaling.",
         ["TNF"], "approved"),
        ("Imiquimod", "TLR_agonist",
         "TLR7 agonist. Activates innate immunity via pDC and macrophage stimulation. Topical.",
         ["TLR7"], "approved"),
        ("Eculizumab", "complement_inhibitor",
         "Anti-C5 monoclonal antibody. Blocks terminal complement activation and MAC formation.",
         ["C5"], "approved"),
    ]
    count = 0
    for name, dclass, mech, targets, status in drugs:
        entity = Drug(
            name=name,
            display_name=name,
            description=mech,
            drug_class=dclass,
            mechanism=mech,
            targets=targets,
            approval_status=status,
            source="curated",
            tags=["immunology", "drug", dclass],
        )
        store.add_entity(entity)
        count += 1
    return count


# ── Immune Pathways ────────────────────────────────────────────────

def _seed_immune_pathways(store: EntityStore) -> int:
    pathways = [
        # (name, type, genes, description, kegg)
        ("TLR Signaling", "signaling",
         ["TLR4", "TLR3", "TLR7", "TLR9", "MYD88", "TIRAP", "TICAM1", "IRAK1", "IRAK4", "TRAF6", "NFKB1", "IRF3", "IRF7"],
         "Toll-like receptor signaling — innate immune recognition of PAMPs and activation of NF-kB and interferon responses.",
         "hsa04620"),
        ("TCR Signaling", "signaling",
         ["CD3E", "CD3D", "CD3G", "CD247", "LCK", "ZAP70", "LAT", "PLCG1", "NFATC1", "NFKB1", "AP1"],
         "T cell receptor signaling — antigen recognition triggers T cell activation, proliferation, and effector function.",
         "hsa04660"),
        ("BCR Signaling", "signaling",
         ["CD79A", "CD79B", "SYK", "BTK", "BLNK", "PLCG2", "NFKB1", "NFATC1", "PIK3CA"],
         "B cell receptor signaling — antigen binding activates B cells for antibody production and class switching.",
         "hsa04662"),
        ("Complement Cascade", "signaling",
         ["C1QA", "C2", "C3", "C4A", "C5", "C6", "C7", "C8A", "C9", "CFB", "CFD", "MBL2", "MASP1"],
         "Complement cascade — classical, lectin, and alternative pathways converge on C3/C5 for opsonization and MAC.",
         "hsa04610"),
        ("NF-kB Signaling", "signaling",
         ["NFKB1", "NFKB2", "RELA", "RELB", "NFKBIA", "IKBKB", "IKBKG", "CHUK", "TRAF2", "TRAF6"],
         "NF-kB pathway — master transcriptional regulator of innate immunity, inflammation, and cell survival.",
         "hsa04064"),
        ("JAK-STAT Signaling", "signaling",
         ["JAK1", "JAK2", "JAK3", "TYK2", "STAT1", "STAT2", "STAT3", "STAT4", "STAT5A", "STAT5B", "STAT6"],
         "JAK-STAT pathway — cytokine receptor signaling for immune cell differentiation and function.",
         "hsa04630"),
        ("Inflammasome Pathway", "signaling",
         ["NLRP3", "NLRC4", "PYCARD", "CASP1", "IL1B", "IL18", "GSDMD"],
         "Inflammasome assembly — intracellular danger sensing leading to IL-1b/IL-18 maturation and pyroptosis.",
         ""),
        ("PD-1/PD-L1 Checkpoint", "signaling",
         ["PDCD1", "CD274", "PDCD1LG2", "SHP2", "LCK", "ZAP70"],
         "PD-1 immune checkpoint — inhibitory signaling that dampens T cell effector function. Target of immunotherapy.",
         "hsa04514"),
    ]
    count = 0
    for name, ptype, genes, desc, kegg in pathways:
        entity = Pathway(
            name=name,
            display_name=name,
            description=desc,
            pathway_type=ptype,
            genes=genes,
            kegg_id=kegg,
            source="curated",
            tags=["immunology", "pathway", ptype],
        )
        store.add_entity(entity)
        count += 1
    return count


# ── Adhesion Molecules (Diapedesis) ────────────────────────────────

def _seed_adhesion_molecules(store: EntityStore) -> int:
    """Seed cell adhesion molecules involved in leukocyte diapedesis."""
    molecules = [
        # (name, family, expressed_on, ligands, kd_uM, on_rate, off_rate, regulation, pathway, step, gene, structure, description)
        ("E-selectin (CD62E)", "selectin", ["endothelial"],
         ["PSGL-1", "CD44", "ESL-1"], 100.0, 10.0, 5.0,
         "cytokine_induced", "NF-kB",
         "rolling", "SELE", "type_I_transmembrane",
         "Endothelial selectin. Induced by TNF-alpha and IL-1beta (4-6h). Mediates initial leukocyte rolling via low-affinity bonds with PSGL-1. Catch-bond behavior under shear stress."),
        ("P-selectin (CD62P)", "selectin", ["endothelial", "platelet"],
         ["PSGL-1"], 100.0, 15.0, 8.0,
         "histamine_induced", "Weibel-Palade body exocytosis",
         "rolling", "SELP", "type_I_transmembrane",
         "Platelet/endothelial selectin. Rapidly mobilized from Weibel-Palade bodies within minutes of histamine/thrombin stimulation. First selectin to appear at inflammation sites."),
        ("L-selectin (CD62L)", "selectin", ["leukocyte"],
         ["GlyCAM-1", "CD34", "MAdCAM-1"], 100.0, 12.0, 6.0,
         "constitutive", "ectodomain shedding (ADAM17)",
         "rolling", "SELL", "type_I_transmembrane",
         "Leukocyte selectin. Constitutively expressed on most leukocytes. Mediates lymphocyte homing to lymph nodes via HEV. Shed upon activation by ADAM17."),
        ("PSGL-1 (CD162)", "selectin_ligand", ["leukocyte"],
         ["E-selectin", "P-selectin", "L-selectin"], 100.0, 10.0, 5.0,
         "constitutive", "sialyl Lewis-x decoration",
         "rolling", "SELPLG", "type_I_transmembrane",
         "P-selectin glycoprotein ligand 1. Primary ligand for all three selectins. Requires sialyl Lewis-x (sLex) post-translational modification. LAD-2 patients lack sLex."),
        ("LFA-1 (CD11a/CD18)", "integrin", ["leukocyte"],
         ["ICAM-1", "ICAM-2", "JAM-A"], 1.0, 1.0, 0.01,
         "chemokine_activated", "Rap1/talin/kindlin inside-out signaling",
         "arrest", "ITGAL/ITGB2", "heterodimer",
         "Lymphocyte function-associated antigen 1. αLβ2 integrin on all leukocytes. Exists in bent (low-affinity) and extended (high-affinity) conformations. Chemokine-triggered inside-out signaling via Rap1/talin/kindlin-3 switches to high affinity. LAD-1: defective β2 (CD18)."),
        ("MAC-1 (CD11b/CD18)", "integrin", ["neutrophil", "monocyte", "macrophage"],
         ["ICAM-1", "iC3b", "fibrinogen", "factor X"], 5.0, 0.8, 0.02,
         "chemokine_activated", "Rap1/talin inside-out signaling",
         "arrest", "ITGAM/ITGB2", "heterodimer",
         "Macrophage-1 antigen. αMβ2 integrin, primarily on myeloid cells. Complement receptor 3 (CR3) — also binds iC3b for phagocytosis. Important for neutrophil firm adhesion and transmigration."),
        ("VLA-4 (α4β1)", "integrin", ["lymphocyte", "monocyte", "eosinophil"],
         ["VCAM-1", "fibronectin (CS1)"], 0.5, 1.5, 0.015,
         "chemokine_activated", "Rap1/talin inside-out signaling",
         "arrest", "ITGA4/ITGB1", "heterodimer",
         "Very late antigen 4. α4β1 integrin on lymphocytes and monocytes (NOT neutrophils). Binds VCAM-1 on endothelium. Important for lymphocyte and monocyte recruitment. Target of natalizumab (anti-α4) for MS."),
        ("ICAM-1 (CD54)", "immunoglobulin_superfamily", ["endothelial", "epithelial", "immune"],
         ["LFA-1", "MAC-1"], 1.0, 1.0, 0.01,
         "cytokine_induced", "NF-kB",
         "arrest", "ICAM1", "type_I_transmembrane",
         "Intercellular adhesion molecule 1. Ig superfamily member on endothelium. Low basal expression, strongly upregulated by TNF-alpha, IL-1beta, IFN-gamma (12-24h). Primary counter-receptor for LFA-1 and MAC-1."),
        ("ICAM-2 (CD102)", "immunoglobulin_superfamily", ["endothelial"],
         ["LFA-1"], 5.0, 0.5, 0.05,
         "constitutive", "none",
         "arrest", "ICAM2", "type_I_transmembrane",
         "Intercellular adhesion molecule 2. Constitutively expressed on endothelium. Provides baseline LFA-1 binding. NOT induced by inflammatory cytokines."),
        ("VCAM-1 (CD106)", "immunoglobulin_superfamily", ["endothelial"],
         ["VLA-4", "α4β7"], 0.5, 1.5, 0.01,
         "cytokine_induced", "NF-kB",
         "arrest", "VCAM1", "type_I_transmembrane",
         "Vascular cell adhesion molecule 1. Ig superfamily member. Strongly induced by TNF-alpha, IL-1beta, IL-4. Counter-receptor for VLA-4. Key for lymphocyte and monocyte recruitment."),
        ("CD31/PECAM-1", "immunoglobulin_superfamily", ["endothelial", "leukocyte", "platelet"],
         ["CD31 (homophilic)", "αvβ3 integrin"], 10.0, 0.3, 0.1,
         "constitutive", "ITIM signaling",
         "transmigration", "PECAM1", "type_I_transmembrane",
         "Platelet endothelial cell adhesion molecule. Concentrated at endothelial junctions. Homophilic binding between leukocyte PECAM and endothelial PECAM mediates paracellular transmigration. Required for efficient diapedesis."),
        ("VE-cadherin (CD144)", "cadherin", ["endothelial"],
         ["VE-cadherin (homophilic)"], 1.0, 0.5, 0.2,
         "constitutive", "β-catenin/p120-catenin",
         "transmigration", "CDH5", "type_I_transmembrane",
         "Vascular endothelial cadherin. THE adherens junction protein of endothelium. Homophilic binding maintains endothelial barrier. Must be transiently disrupted for leukocyte transmigration. Phosphorylation of VE-cadherin by SRC kinases disrupts complex."),
        ("JAM-A", "JAM", ["endothelial", "leukocyte"],
         ["LFA-1", "JAM-A (homophilic)"], 10.0, 0.2, 0.15,
         "constitutive", "tight junction signaling",
         "transmigration", "F11R", "type_I_transmembrane",
         "Junctional adhesion molecule A. At endothelial tight junctions. Homophilic binding between endothelial cells and heterophilic binding to LFA-1 on leukocytes. Guides leukocytes through paracellular pathway."),
        ("CD99", "transmembrane", ["endothelial", "leukocyte"],
         ["CD99 (homophilic)"], 5.0, 0.3, 0.1,
         "constitutive", "PKC signaling",
         "transmigration", "CD99", "type_I_transmembrane",
         "CD99 glycoprotein. Required for the FINAL step of diapedesis, downstream of PECAM-1. Homophilic interactions between leukocyte and endothelial CD99. Anti-CD99 antibodies block transmigration at a late stage."),
        ("MAdCAM-1", "immunoglobulin_superfamily", ["endothelial"],
         ["α4β7", "L-selectin"], 2.0, 1.0, 0.05,
         "constitutive", "mucosal homing",
         "rolling", "MADCAM1", "type_I_transmembrane",
         "Mucosal addressin cell adhesion molecule 1. Expressed on gut-associated endothelium. Dual function: mucin domain binds L-selectin (rolling), Ig domains bind α4β7 (arrest). Target of vedolizumab."),
    ]

    count = 0
    for (name, family, expressed_on, ligands, kd, on_r, off_r,
         regulation, pathway, step, gene, structure, desc) in molecules:
        entity = AdhesionMolecule(
            name=name,
            description=desc,
            molecule_family=family,
            expressed_on=expressed_on,
            ligands=ligands,
            binding_affinity_kd=kd,
            on_rate=on_r,
            off_rate=off_r,
            regulation=regulation,
            signaling_pathway=pathway,
            diapedesis_step=step,
            gene_symbol=gene,
            structure_type=structure,
        )
        store.add_entity(entity)
        count += 1

    log.info("  Seeded %d adhesion molecules", count)
    return count


# ── Endothelial Cell Types ─────────────────────────────────────────

def _seed_endothelial_cells(store: EntityStore) -> int:
    """Seed specialized endothelial cell subtypes relevant to diapedesis."""
    cells = [
        ("Postcapillary venule endothelial cell",
         "endothelial", "",
         ["CD31", "CD34", "VE-cadherin", "VEGFR2", "E-selectin_inducible", "ICAM1_inducible", "VCAM1_inducible"],
         [],
         "Primary site of leukocyte diapedesis. Postcapillary venules (10-50 μm diameter) have thinner walls and lower shear stress than arterioles, facilitating leukocyte adhesion and transmigration. Express inducible E-selectin, ICAM-1, VCAM-1 upon TNF/IL-1 stimulation."),
        ("High endothelial venule (HEV) cell",
         "endothelial", "",
         ["CD31", "CD34", "PNAd", "CCL21", "GlyCAM-1", "MAdCAM-1"],
         ["CCL21", "CCL19"],
         "Specialized cuboidal endothelial cells found in lymph node paracortex. Constitutively express PNAd and GlyCAM-1 for L-selectin-mediated lymphocyte homing. Key entry point for naive T and B cells into lymph nodes."),
        ("Inflamed endothelial cell",
         "endothelial", "",
         ["CD31", "CD34", "E-selectin", "P-selectin", "ICAM-1_high", "VCAM-1_high", "CXCL8_displayed"],
         ["CXCL8", "CCL2", "CXCL1"],
         "Activated endothelial cell at site of inflammation. Upregulated adhesion molecules: P-selectin (minutes), E-selectin (4-6h), ICAM-1 and VCAM-1 (12-24h). Displays chemokines on surface via glycosaminoglycans for leukocyte integrin activation."),
    ]

    count = 0
    for (name, immune_type, subtype, markers, cytokines, desc) in cells:
        entity = ImmuneCellEntity(
            name=name,
            description=desc,
            immune_type=immune_type,
            immune_subtype=subtype,
            surface_markers=markers,
            cytokines_secreting=cytokines,
        )
        store.add_entity(entity)
        count += 1

    log.info("  Seeded %d endothelial cell types", count)
    return count


# ── Diapedesis Relationships ──────────────────────────────────────

def _seed_diapedesis_relationships(store: EntityStore) -> int:
    """Create relationships between adhesion molecules, cells, and cytokines for diapedesis."""
    count = 0

    def _add_rel(source_name: str, target_name: str, rel_type: RelationshipType, evidence: str = "curated") -> bool:
        nonlocal count
        src = store.find_entity_by_name(source_name)
        tgt = store.find_entity_by_name(target_name)
        if src and tgt:
            store.add_relationship(Relationship(
                source_id=src.entity_id,
                target_id=tgt.entity_id,
                rel_type=rel_type,
                evidence=evidence,
            ))
            count += 1
            return True
        return False

    # Selectin → ligand binding (rolling)
    _add_rel("E-selectin (CD62E)", "PSGL-1 (CD162)", RelationshipType.BINDS_TO, "E-selectin/PSGL-1 rolling interaction, Kd ~100μM")
    _add_rel("P-selectin (CD62P)", "PSGL-1 (CD162)", RelationshipType.BINDS_TO, "P-selectin/PSGL-1 rolling interaction")
    _add_rel("L-selectin (CD62L)", "MAdCAM-1", RelationshipType.BINDS_TO, "L-selectin/MAdCAM-1 mucin domain interaction")

    # Integrin → ligand binding (arrest)
    _add_rel("LFA-1 (CD11a/CD18)", "ICAM-1 (CD54)", RelationshipType.BINDS_TO, "LFA-1/ICAM-1 firm adhesion, Kd ~1μM (high affinity)")
    _add_rel("LFA-1 (CD11a/CD18)", "ICAM-2 (CD102)", RelationshipType.BINDS_TO, "LFA-1/ICAM-2 constitutive adhesion")
    _add_rel("MAC-1 (CD11b/CD18)", "ICAM-1 (CD54)", RelationshipType.BINDS_TO, "MAC-1/ICAM-1 neutrophil adhesion")
    _add_rel("VLA-4 (α4β1)", "VCAM-1 (CD106)", RelationshipType.BINDS_TO, "VLA-4/VCAM-1 lymphocyte/monocyte adhesion, Kd ~0.5μM")

    # Junctional binding (transmigration)
    _add_rel("CD31/PECAM-1", "CD31/PECAM-1", RelationshipType.BINDS_TO, "PECAM homophilic binding at endothelial junctions")
    _add_rel("LFA-1 (CD11a/CD18)", "JAM-A", RelationshipType.BINDS_TO, "LFA-1/JAM-A guides transmigration")

    # Cytokine → adhesion molecule induction
    _add_rel("TNF-alpha", "E-selectin (CD62E)", RelationshipType.ACTIVATES, "TNF induces E-selectin expression (4-6h)")
    _add_rel("TNF-alpha", "ICAM-1 (CD54)", RelationshipType.ACTIVATES, "TNF upregulates ICAM-1 (12-24h)")
    _add_rel("TNF-alpha", "VCAM-1 (CD106)", RelationshipType.ACTIVATES, "TNF upregulates VCAM-1")
    _add_rel("IL-1beta", "E-selectin (CD62E)", RelationshipType.ACTIVATES, "IL-1β induces E-selectin")
    _add_rel("IL-1beta", "ICAM-1 (CD54)", RelationshipType.ACTIVATES, "IL-1β upregulates ICAM-1")
    _add_rel("IFN-gamma", "ICAM-1 (CD54)", RelationshipType.ACTIVATES, "IFNγ strongly induces ICAM-1")

    # Chemokine → integrin activation (inside-out signaling)
    _add_rel("CXCL8 (IL-8)", "LFA-1 (CD11a/CD18)", RelationshipType.ACTIVATES, "CXCL8 activates LFA-1 via CXCR1/2 → Rap1 → talin/kindlin")
    _add_rel("CXCL8 (IL-8)", "MAC-1 (CD11b/CD18)", RelationshipType.ACTIVATES, "CXCL8 activates MAC-1 on neutrophils")
    _add_rel("CCL2 (MCP-1)", "VLA-4 (α4β1)", RelationshipType.ACTIVATES, "CCL2 activates VLA-4 via CCR2 → Rap1 inside-out signaling")

    # Cell → adhesion molecule expression
    _add_rel("Neutrophil", "LFA-1 (CD11a/CD18)", RelationshipType.EXPRESSES_ON_SURFACE, "Neutrophils constitutively express LFA-1")
    _add_rel("Neutrophil", "MAC-1 (CD11b/CD18)", RelationshipType.EXPRESSES_ON_SURFACE, "Neutrophils express MAC-1, upregulated upon activation")
    _add_rel("Neutrophil", "PSGL-1 (CD162)", RelationshipType.EXPRESSES_ON_SURFACE, "Neutrophils constitutively express PSGL-1 with sLex")
    _add_rel("Neutrophil", "L-selectin (CD62L)", RelationshipType.EXPRESSES_ON_SURFACE, "Neutrophils express L-selectin, shed upon activation")
    _add_rel("Inflamed endothelial cell", "E-selectin (CD62E)", RelationshipType.EXPRESSES_ON_SURFACE, "Inflamed endothelium expresses E-selectin")
    _add_rel("Inflamed endothelial cell", "ICAM-1 (CD54)", RelationshipType.EXPRESSES_ON_SURFACE, "Inflamed endothelium upregulates ICAM-1")
    _add_rel("Inflamed endothelial cell", "VCAM-1 (CD106)", RelationshipType.EXPRESSES_ON_SURFACE, "Inflamed endothelium upregulates VCAM-1")
    _add_rel("Postcapillary venule endothelial cell", "CD31/PECAM-1", RelationshipType.EXPRESSES_ON_SURFACE, "Endothelial PECAM at junctions")
    _add_rel("Postcapillary venule endothelial cell", "VE-cadherin (CD144)", RelationshipType.EXPRESSES_ON_SURFACE, "VE-cadherin at adherens junctions")

    # Leukocyte → endothelium diapedesis relationships
    _add_rel("Neutrophil", "Postcapillary venule endothelial cell", RelationshipType.ADHERES_TO, "Neutrophil adhesion cascade at postcapillary venule")
    _add_rel("Neutrophil", "Postcapillary venule endothelial cell", RelationshipType.TRANSMIGRATES_THROUGH, "Neutrophil paracellular transmigration")

    log.info("  Seeded %d diapedesis relationships", count)
    return count


# ── Immune Relationships ───────────────────────────────────────────

def _seed_immune_relationships(store: EntityStore) -> int:
    """Create key immunological relationships between seeded entities."""
    count = 0

    def _add_rel(source_name: str, target_name: str, rel_type: RelationshipType, evidence: str = "curated") -> bool:
        nonlocal count
        src = store.find_entity_by_name(source_name)
        tgt = store.find_entity_by_name(target_name)
        if src and tgt:
            store.add_relationship(Relationship(
                source_id=src.entity_id,
                target_id=tgt.entity_id,
                rel_type=rel_type,
                evidence=evidence,
            ))
            count += 1
            return True
        return False

    # Cytokine → cell polarization
    _add_rel("IFN-gamma", "Macrophage M1", RelationshipType.POLARIZES, "IFNg drives M1 polarization")
    _add_rel("IL-4", "Macrophage M2", RelationshipType.POLARIZES, "IL-4 drives M2 polarization")

    # Cytokine → T cell differentiation
    _add_rel("IL-12", "CD4+ Th1 cell", RelationshipType.DIFFERENTIATES, "IL-12 drives Th1 differentiation")
    _add_rel("IL-4", "CD4+ Th2 cell", RelationshipType.DIFFERENTIATES, "IL-4 drives Th2 differentiation")
    _add_rel("IL-6", "CD4+ Th17 cell", RelationshipType.DIFFERENTIATES, "IL-6 + TGFb drives Th17")
    _add_rel("TGF-beta1", "Regulatory T cell (Treg)", RelationshipType.DIFFERENTIATES, "TGFb drives iTreg")
    _add_rel("IL-21", "T follicular helper (Tfh)", RelationshipType.DIFFERENTIATES, "IL-21 promotes Tfh")

    # Antibody → pathogen neutralization
    _add_rel("IgG1", "SARS-CoV-2", RelationshipType.NEUTRALIZES, "IgG1 neutralizing antibodies")
    _add_rel("IgG1", "Influenza A", RelationshipType.NEUTRALIZES, "Hemagglutinin-specific IgG1")
    _add_rel("IgM (pentameric)", "SARS-CoV-2", RelationshipType.NEUTRALIZES, "Early IgM response")

    # Antibody → complement activation
    _add_rel("IgG1", "C1q", RelationshipType.ACTIVATES_COMPLEMENT, "IgG1 Fc binds C1q")
    _add_rel("IgG3", "C1q", RelationshipType.ACTIVATES_COMPLEMENT, "IgG3 Fc binds C1q (strongest)")
    _add_rel("IgM (pentameric)", "C1q", RelationshipType.ACTIVATES_COMPLEMENT, "IgM pentamer binds C1q")

    # Antibody/complement → opsonization
    _add_rel("IgG1", "Staphylococcus aureus", RelationshipType.OPSONIZES, "IgG1 opsonization via FcgR")
    _add_rel("C3b", "Escherichia coli (UPEC)", RelationshipType.OPSONIZES, "C3b opsonization via CR1")

    # MHC → antigen presentation
    _add_rel("HLA-A*02:01", "CD8+ Effector T cell", RelationshipType.PRESENTS, "MHC-I presents to CD8+ T cells")
    _add_rel("HLA-DRB1*04:01", "CD4+ Th1 cell", RelationshipType.PRESENTS, "MHC-II presents to CD4+ T cells")

    # PRR → pathogen recognition
    _add_rel("TLR4", "Escherichia coli (UPEC)", RelationshipType.RECOGNIZES, "TLR4 senses LPS")
    _add_rel("TLR3", "SARS-CoV-2", RelationshipType.RECOGNIZES, "TLR3 senses dsRNA intermediate")
    _add_rel("TLR7", "HIV-1", RelationshipType.RECOGNIZES, "TLR7 senses ssRNA")
    _add_rel("NLRP3 inflammasome", "Staphylococcus aureus", RelationshipType.RECOGNIZES, "NLRP3 senses toxins")
    _add_rel("cGAS-STING", "Mycobacterium tuberculosis", RelationshipType.RECOGNIZES, "cGAS senses bacterial DNA")

    # Drug → target relationships (activates/inhibits)
    _add_rel("Nivolumab", "CD8+ Effector T cell", RelationshipType.ACTIVATES, "anti-PD-1 restores T cell function")
    _add_rel("Ipilimumab", "CD8+ Effector T cell", RelationshipType.ACTIVATES, "anti-CTLA-4 enhances T cell activation")
    _add_rel("Rituximab", "Naive B cell", RelationshipType.KILLS, "anti-CD20 depletes B cells")
    _add_rel("Eculizumab", "MAC (C5b-9)", RelationshipType.INHIBITS, "anti-C5 blocks MAC formation")
    _add_rel("Infliximab", "TNF-alpha", RelationshipType.INHIBITS, "anti-TNF neutralizes TNFa")

    return count
