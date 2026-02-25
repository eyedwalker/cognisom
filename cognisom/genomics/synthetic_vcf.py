"""
Synthetic Prostate Cancer VCF
==============================

Generate a realistic synthetic VCF file for demo purposes.
Contains ~50 variants including known prostate cancer driver mutations,
passenger variants, and germline polymorphisms.

This allows the platform to be fully demonstrable before
real patient data (e.g., from Mayo Clinic) is available.
"""

SYNTHETIC_PROSTATE_VCF = """##fileformat=VCFv4.2
##source=CognisomSyntheticGenerator
##reference=GRCh38
##INFO=<ID=GENE,Number=1,Type=String,Description="Gene symbol">
##INFO=<ID=CONSEQUENCE,Number=1,Type=String,Description="Variant consequence">
##INFO=<ID=AA_CHANGE,Number=1,Type=String,Description="Amino acid change">
##INFO=<ID=COSMIC_ID,Number=1,Type=String,Description="COSMIC identifier">
##INFO=<ID=CLNSIG,Number=1,Type=String,Description="ClinVar significance">
##INFO=<ID=NOTE,Number=1,Type=String,Description="Clinical note">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Read depth">
##FORMAT=<ID=AD,Number=R,Type=Integer,Description="Allelic depths">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE
chrX\t66943592\trs137852591\tA\tG\t99\tPASS\tGENE=AR;CONSEQUENCE=missense;AA_CHANGE=p.T877A;COSMIC_ID=COSM10687;CLNSIG=Pathogenic;NOTE=Broadens_ligand_specificity\tGT:DP:AD\t0/1:85:42,43
chr17\t7578406\trs28934578\tC\tT\t99\tPASS\tGENE=TP53;CONSEQUENCE=missense;AA_CHANGE=p.R248W;COSMIC_ID=COSM10656;CLNSIG=Pathogenic;NOTE=DNA_contact_mutant\tGT:DP:AD\t0/1:120:55,65
chr10\t87933147\trs121909219\tC\tT\t99\tPASS\tGENE=PTEN;CONSEQUENCE=nonsense;AA_CHANGE=p.R130*;COSMIC_ID=COSM5033;CLNSIG=Pathogenic;NOTE=Complete_loss_of_function\tGT:DP:AD\t1/1:95:2,93
chr13\t32914438\t.\tTGAA\tT\t98\tPASS\tGENE=BRCA2;CONSEQUENCE=frameshift;AA_CHANGE=p.E1143fs;CLNSIG=Pathogenic;NOTE=Frameshift_deletion_HRD\tGT:DP:AD\t0/1:78:40,38
chr17\t7674220\trs121912651\tC\tA\t95\tPASS\tGENE=SPOP;CONSEQUENCE=missense;AA_CHANGE=p.F133V;COSMIC_ID=COSM159871;CLNSIG=Pathogenic;NOTE=Substrate_binding_domain\tGT:DP:AD\t0/1:92:48,44
chr14\t37589552\t.\tA\tG\t90\tPASS\tGENE=FOXA1;CONSEQUENCE=missense;AA_CHANGE=p.I176M;CLNSIG=Likely_pathogenic;NOTE=Wing2_region_altered_DNA_binding\tGT:DP:AD\t0/1:88:45,43
chr17\t37687345\t.\tG\tA\t85\tPASS\tGENE=CDK12;CONSEQUENCE=nonsense;AA_CHANGE=p.Q756*;CLNSIG=Pathogenic;NOTE=Biallelic_loss_neoantigen_rich\tGT:DP:AD\t0/1:75:38,37
chr17\t37687400\t.\tCTG\tC\t82\tPASS\tGENE=CDK12;CONSEQUENCE=frameshift;AA_CHANGE=p.L780fs;CLNSIG=Pathogenic;NOTE=Second_hit_biallelic\tGT:DP:AD\t0/1:70:36,34
chr13\t48877883\t.\tG\tT\t88\tPASS\tGENE=RB1;CONSEQUENCE=missense;AA_CHANGE=p.R552Q;CLNSIG=Likely_pathogenic;NOTE=Pocket_domain_disruption\tGT:DP:AD\t0/1:82:42,40
chr8\t127736507\t.\tC\tT\t75\tPASS\tGENE=MYC;CONSEQUENCE=synonymous;AA_CHANGE=p.S99S;NOTE=MYC_region_amplified\tGT:DP:AD\t0/1:150:30,120
chr21\t38412670\t.\tG\tA\t92\tPASS\tGENE=ERG;CONSEQUENCE=missense;AA_CHANGE=p.R367Q;CLNSIG=Likely_pathogenic;NOTE=TMPRSS2-ERG_fusion_context\tGT:DP:AD\t0/1:95:47,48
chr11\t108236086\trs1801516\tG\tA\t99\tPASS\tGENE=ATM;CONSEQUENCE=missense;AA_CHANGE=p.D1853N;CLNSIG=Likely_pathogenic;NOTE=DNA_damage_response_impaired\tGT:DP:AD\t0/1:110:56,54
chr3\t178952085\trs121913279\tG\tA\t96\tPASS\tGENE=PIK3CA;CONSEQUENCE=missense;AA_CHANGE=p.E545K;COSMIC_ID=COSM763;CLNSIG=Pathogenic;NOTE=Helical_domain_constitutive_activation\tGT:DP:AD\t0/1:88:44,44
chr5\t112175770\t.\tG\tT\t80\tPASS\tGENE=APC;CONSEQUENCE=missense;AA_CHANGE=p.I1307K;CLNSIG=Risk_factor;NOTE=Wnt_pathway_activation\tGT:DP:AD\t0/1:72:37,35
chr1\t11288758\trs3218536\tC\tT\t65\tPASS\tGENE=MTOR;CONSEQUENCE=synonymous;AA_CHANGE=p.A1105A;NOTE=mTOR_pathway_variant\tGT:DP:AD\t0/1:85:44,41
chr2\t29443695\trs1801133\tC\tT\t70\tPASS\tGENE=MTHFR;CONSEQUENCE=missense;AA_CHANGE=p.A222V;CLNSIG=Drug_response;NOTE=Folate_metabolism\tGT:DP:AD\t0/1:90:46,44
chr7\t55249071\trs121913529\tG\tT\t99\tPASS\tGENE=EGFR;CONSEQUENCE=missense;AA_CHANGE=p.L858R;COSMIC_ID=COSM6224;NOTE=Incidental_EGFR_activating\tGT:DP:AD\t0/1:65:33,32
chr12\t25398284\trs121913529\tC\tA\t88\tPASS\tGENE=KRAS;CONSEQUENCE=missense;AA_CHANGE=p.G12V;COSMIC_ID=COSM520;NOTE=RAS_pathway_activation\tGT:DP:AD\t0/1:78:40,38
chr1\t115256530\t.\tG\tA\t55\tPASS\tGENE=NRAS;CONSEQUENCE=missense;AA_CHANGE=p.Q61R;NOTE=RAS_pathway\tGT:DP:AD\t0/1:60:31,29
chr6\t31382176\trs1050407\tA\tG\t99\tPASS\tGENE=HLA-B;CONSEQUENCE=synonymous;AA_CHANGE=p.S140S;NOTE=HLA_region\tGT:DP:AD\t0/1:200:102,98
chr6\t29944050\trs2308655\tG\tA\t99\tPASS\tGENE=HLA-A;CONSEQUENCE=missense;AA_CHANGE=p.R62Q;NOTE=HLA_region\tGT:DP:AD\t0/1:180:92,88
chr15\t90631934\trs1042522\tG\tC\t99\tPASS\tGENE=IDH2;CONSEQUENCE=synonymous;AA_CHANGE=p.P72R;NOTE=Metabolic_variant\tGT:DP:AD\t0/1:95:48,47
chr9\t21971120\t.\tG\tA\t78\tPASS\tGENE=CDKN2A;CONSEQUENCE=missense;AA_CHANGE=p.P114L;CLNSIG=Likely_pathogenic;NOTE=Cell_cycle_checkpoint_loss\tGT:DP:AD\t0/1:72:37,35
chr4\t1801997\t.\tC\tT\t60\tPASS\tGENE=FGFR3;CONSEQUENCE=missense;AA_CHANGE=p.S249C;COSMIC_ID=COSM715;NOTE=Activating_mutation\tGT:DP:AD\t0/1:68:35,33
chr17\t41276045\trs80357906\tCT\tC\t88\tPASS\tGENE=BRCA1;CONSEQUENCE=frameshift;AA_CHANGE=p.K519fs;CLNSIG=Pathogenic;NOTE=Pathogenic_frameshift\tGT:DP:AD\t0/1:82:42,40
chr2\t209113192\t.\tC\tA\t50\tPASS\tGENE=IDH1;CONSEQUENCE=missense;AA_CHANGE=p.R132H;COSMIC_ID=COSM28746;NOTE=Neomorphic_enzyme_activity\tGT:DP:AD\t0/1:55:28,27
chr1\t43815008\trs4148323\tG\tA\t99\tPASS\tGENE=UGT1A1;CONSEQUENCE=missense;AA_CHANGE=p.G71R;CLNSIG=Drug_response;NOTE=Irinotecan_toxicity_risk\tGT:DP:AD\t0/1:95:48,47
chr10\t96541616\trs1800562\tG\tA\t99\tPASS\tGENE=CYP2C19;CONSEQUENCE=missense;AA_CHANGE=p.P227L;CLNSIG=Drug_response;NOTE=Poor_metabolizer\tGT:DP:AD\t0/1:88:45,43
chr19\t1220321\t.\tC\tT\t45\tPASS\tGENE=STK11;CONSEQUENCE=missense;AA_CHANGE=p.F354L;NOTE=Kinase_domain\tGT:DP:AD\t0/1:50:26,24
chr7\t140753336\trs113488022\tA\tT\t99\tPASS\tGENE=BRAF;CONSEQUENCE=missense;AA_CHANGE=p.V600E;COSMIC_ID=COSM476;CLNSIG=Pathogenic;NOTE=Activating_kinase_mutation\tGT:DP:AD\t0/1:110:56,54
chr1\t114713881\trs1801131\tT\tG\t85\tPASS\tGENE=MTHFR;CONSEQUENCE=missense;AA_CHANGE=p.E429A;CLNSIG=Drug_response;NOTE=Folate_metabolism\tGT:DP:AD\t0/1:80:41,39
chr16\t3786742\t.\tG\tA\t40\t.\tGENE=CREBBP;CONSEQUENCE=missense;AA_CHANGE=p.R1446H;NOTE=HAT_domain\tGT:DP:AD\t0/1:45:23,22
chr3\t37034946\t.\tC\tT\t55\t.\tGENE=MLH1;CONSEQUENCE=synonymous;AA_CHANGE=p.I219V;NOTE=Mismatch_repair\tGT:DP:AD\t0/1:60:31,29
chr2\t47702181\t.\tA\tG\t50\t.\tGENE=MSH2;CONSEQUENCE=missense;AA_CHANGE=p.N596S;NOTE=Mismatch_repair_variant\tGT:DP:AD\t0/1:55:28,27
chr7\t6026775\t.\tG\tA\t65\tPASS\tGENE=PMS2;CONSEQUENCE=missense;AA_CHANGE=p.P246L;NOTE=Mismatch_repair\tGT:DP:AD\t0/1:70:36,34
chr2\t215645464\t.\tG\tC\t50\t.\tGENE=BARD1;CONSEQUENCE=missense;AA_CHANGE=p.R378S;NOTE=BRCA1_interactor\tGT:DP:AD\t0/1:55:28,27
chr16\t68771195\t.\tC\tT\t60\tPASS\tGENE=CDH1;CONSEQUENCE=missense;AA_CHANGE=p.A634V;NOTE=E-cadherin_EMT\tGT:DP:AD\t0/1:65:33,32
chr17\t7577539\t.\tG\tA\t70\tPASS\tGENE=TP53;CONSEQUENCE=missense;AA_CHANGE=p.R175H;COSMIC_ID=COSM10648;CLNSIG=Pathogenic;NOTE=Structural_hotspot\tGT:DP:AD\t0/1:75:38,37
chr10\t87952143\t.\tA\tG\t80\tPASS\tGENE=PTEN;CONSEQUENCE=missense;AA_CHANGE=p.C124S;COSMIC_ID=COSM5152;CLNSIG=Pathogenic;NOTE=Catalytic_site_phosphatase_dead\tGT:DP:AD\t0/1:82:42,40
chrX\t66943700\t.\tC\tT\t75\tPASS\tGENE=AR;CONSEQUENCE=missense;AA_CHANGE=p.L702H;COSMIC_ID=COSM10690;CLNSIG=Pathogenic;NOTE=Glucocorticoid_responsive\tGT:DP:AD\t0/1:78:40,38
chr1\t27100000\t.\tG\tA\t30\t.\tCONSEQUENCE=intronic;NOTE=Intronic_variant\tGT:DP:AD\t0/1:35:18,17
chr2\t48010000\t.\tC\tT\t25\t.\tCONSEQUENCE=intronic;NOTE=Intronic_variant\tGT:DP:AD\t0/1:30:15,15
chr3\t120000000\t.\tA\tG\t35\t.\tCONSEQUENCE=intergenic;NOTE=Intergenic_variant\tGT:DP:AD\t0/1:40:20,20
chr5\t180000000\t.\tT\tC\t28\t.\tCONSEQUENCE=intronic;NOTE=Intronic_variant\tGT:DP:AD\t0/1:32:16,16
chr7\t100000000\t.\tG\tT\t22\t.\tCONSEQUENCE=intergenic;NOTE=Intergenic_variant\tGT:DP:AD\t0/1:25:13,12
chr9\t130000000\t.\tA\tC\t30\t.\tCONSEQUENCE=intronic;NOTE=Intronic_variant\tGT:DP:AD\t0/1:35:18,17
chr11\t70000000\t.\tC\tG\t27\t.\tCONSEQUENCE=intergenic;NOTE=Intergenic_variant\tGT:DP:AD\t0/1:30:15,15
chr14\t50000000\t.\tT\tA\t32\t.\tCONSEQUENCE=intronic;NOTE=Intronic_variant\tGT:DP:AD\t0/1:36:18,18
chr16\t90000000\t.\tG\tC\t20\t.\tCONSEQUENCE=intergenic;NOTE=Intergenic_variant\tGT:DP:AD\t0/1:24:12,12
chr19\t50000000\t.\tA\tG\t35\t.\tCONSEQUENCE=intronic;NOTE=Intronic_variant\tGT:DP:AD\t0/1:40:20,20
chr22\t30000000\t.\tC\tT\t28\t.\tCONSEQUENCE=intergenic;NOTE=Intergenic_variant\tGT:DP:AD\t0/1:32:16,16
""".strip()


def get_synthetic_vcf() -> str:
    """Return the synthetic prostate cancer VCF text."""
    return SYNTHETIC_PROSTATE_VCF


def get_synthetic_profile_description() -> str:
    """Description of the synthetic patient for the dashboard."""
    return (
        "**Synthetic Patient: COGNISOM-DEMO-001**\n\n"
        "Simulated 65-year-old male with metastatic castration-resistant "
        "prostate cancer (mCRPC). This synthetic VCF contains ~50 variants "
        "including:\n\n"
        "- **AR T877A + L702H**: Dual androgen receptor mutations — broadened "
        "ligand specificity and glucocorticoid responsiveness\n"
        "- **TP53 R248W + R175H**: Biallelic p53 loss — DNA contact and "
        "structural hotspot mutations\n"
        "- **PTEN R130* + C124S**: Biallelic PTEN loss — PI3K/AKT pathway "
        "hyperactivation\n"
        "- **BRCA2 E1143fs**: Frameshift — homologous recombination deficiency "
        "(PARP inhibitor candidate)\n"
        "- **BRCA1 K519fs**: Additional DNA repair defect\n"
        "- **CDK12 Q756* + L780fs**: Biallelic CDK12 loss — neoantigen-rich, "
        "potential immunotherapy responder\n"
        "- **SPOP F133V**: Substrate binding mutation — impaired AR degradation\n"
        "- **PIK3CA E545K**: Constitutive PI3K activation\n"
        "- **ERG R367Q**: In context of TMPRSS2-ERG fusion\n"
        "- **ATM D1853N**: DNA damage response impairment\n\n"
        "This profile represents an aggressive, heavily mutated tumor "
        "with multiple therapeutic vulnerabilities (PARP inhibitors, "
        "checkpoint inhibitors, PI3K/AKT inhibitors)."
    )
