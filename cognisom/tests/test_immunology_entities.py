"""Tests for Immunology Entity System.

Covers:
- 8 new entity types (Virus, Bacterium, Antibody, Antigen, Cytokine, PRR, Complement, MHC)
- 7 new RelationshipType values
- ImmuneCellEntity expanded fields
- Entity registry integration
- Seed immunology catalog
- Gene set lists
- Bio-USD ImmuneCellType enum expansion
- Immune visualization prototypes
"""

import pytest

from cognisom.library.models import (
    Antibody,
    Antigen,
    Bacterium,
    BioEntity,
    ComplementComponent,
    Cytokine,
    EntityType,
    ENTITY_CLASS_MAP,
    ImmuneCellEntity,
    MHCMolecule,
    PatternRecognitionReceptor,
    Relationship,
    RelationshipType,
    Virus,
    entity_registry,
)


# ── Entity Type Tests ──────────────────────────────────────────────


class TestVirusEntity:

    def test_create_virus(self):
        v = Virus(
            name="SARS-CoV-2",
            virus_family="Coronaviridae",
            genome_type="ssRNA+",
            envelope=True,
            target_receptors=["ACE2", "TMPRSS2"],
            evasion_mechanisms=["interferon_antagonism"],
        )
        assert v.entity_type == EntityType.VIRUS
        assert v.virus_family == "Coronaviridae"
        assert v.genome_type == "ssRNA+"
        assert v.envelope is True
        assert "ACE2" in v.target_receptors

    def test_virus_roundtrip(self):
        v = Virus(
            name="HIV-1", virus_family="Retroviridae",
            genome_type="retro", capsid_type="icosahedral",
            envelope=True, genome_size_kb=9.7,
            replication_rate=10.0,
            host_tropism=["CD4_T_cell"],
            target_receptors=["CD4", "CCR5"],
            incubation_hours=504.0,
            taxonomy_id="11676",
        )
        d = v.to_dict()
        v2 = BioEntity.from_dict(d)
        assert isinstance(v2, Virus)
        assert v2.virus_family == "Retroviridae"
        assert v2.genome_type == "retro"
        assert v2.envelope is True
        assert v2.genome_size_kb == 9.7
        assert v2.target_receptors == ["CD4", "CCR5"]
        assert v2.taxonomy_id == "11676"


class TestBacteriumEntity:

    def test_create_bacterium(self):
        b = Bacterium(
            name="E. coli",
            gram_stain="negative",
            shape="rod",
            oxygen_requirement="facultative",
            toxins=["endotoxin"],
        )
        assert b.entity_type == EntityType.BACTERIUM
        assert b.gram_stain == "negative"
        assert b.shape == "rod"

    def test_bacterium_roundtrip(self):
        b = Bacterium(
            name="S. aureus",
            gram_stain="positive",
            shape="coccus",
            pathogenicity_factors=["protein_A", "coagulase"],
            antibiotic_resistance=["methicillin"],
            growth_rate=1.0,
            host_niche="extracellular",
            genome_size_mb=2.8,
        )
        d = b.to_dict()
        b2 = BioEntity.from_dict(d)
        assert isinstance(b2, Bacterium)
        assert b2.gram_stain == "positive"
        assert b2.antibiotic_resistance == ["methicillin"]
        assert b2.genome_size_mb == 2.8


class TestAntibodyEntity:

    def test_create_antibody(self):
        ab = Antibody(
            name="Anti-PD-1 IgG4",
            isotype="IgG4",
            target_antigen="PDCD1",
            valency=2,
            complement_fixation=False,
        )
        assert ab.entity_type == EntityType.ANTIBODY
        assert ab.isotype == "IgG4"
        assert ab.valency == 2

    def test_antibody_roundtrip(self):
        ab = Antibody(
            name="IgM", isotype="IgM", heavy_chain="mu",
            valency=10, complement_fixation=True,
            fc_function=["complement_activation", "agglutination"],
            half_life_days=5.0,
        )
        d = ab.to_dict()
        ab2 = BioEntity.from_dict(d)
        assert isinstance(ab2, Antibody)
        assert ab2.isotype == "IgM"
        assert ab2.valency == 10
        assert ab2.complement_fixation is True
        assert ab2.half_life_days == 5.0


class TestAntigenEntity:

    def test_create_antigen(self):
        ag = Antigen(
            name="Spike protein",
            antigen_type="protein",
            mhc_restriction="both",
            immunogenicity=0.9,
        )
        assert ag.entity_type == EntityType.ANTIGEN
        assert ag.antigen_type == "protein"

    def test_antigen_roundtrip(self):
        ag = Antigen(
            name="LPS", antigen_type="lipid",
            mhc_restriction="none",
            source_organism="E. coli",
            b_cell_epitope=True,
            peptide_length=0,
            processing_pathway="endosomal",
        )
        d = ag.to_dict()
        ag2 = BioEntity.from_dict(d)
        assert isinstance(ag2, Antigen)
        assert ag2.source_organism == "E. coli"
        assert ag2.b_cell_epitope is True


class TestCytokineEntity:

    def test_create_cytokine(self):
        ck = Cytokine(
            name="IL-2",
            cytokine_family="interleukin",
            receptor="IL2RA/IL2RB/IL2RG",
            producing_cells=["Th1", "CD8_effector"],
            target_cells=["T_cell", "NK_cell"],
            pro_inflammatory=True,
            gene_symbol="IL2",
        )
        assert ck.entity_type == EntityType.CYTOKINE
        assert ck.cytokine_family == "interleukin"
        assert ck.pro_inflammatory is True

    def test_cytokine_roundtrip(self):
        ck = Cytokine(
            name="TGF-b1", cytokine_family="TGF",
            signaling_pathway="SMAD",
            pro_inflammatory=False,
            half_life_hours=2.0,
            molecular_weight_kda=25.0,
            gene_symbol="TGFB1",
        )
        d = ck.to_dict()
        ck2 = BioEntity.from_dict(d)
        assert isinstance(ck2, Cytokine)
        assert ck2.pro_inflammatory is False
        assert ck2.molecular_weight_kda == 25.0


class TestPatternRecognitionReceptor:

    def test_create_prr(self):
        prr = PatternRecognitionReceptor(
            name="TLR4",
            prr_type="TLR",
            ligands=["LPS", "HMGB1"],
            pamp_or_damp="both",
            gene_symbol="TLR4",
        )
        assert prr.entity_type == EntityType.PATTERN_RECOGNITION_RECEPTOR
        assert prr.prr_type == "TLR"
        assert "LPS" in prr.ligands

    def test_prr_roundtrip(self):
        prr = PatternRecognitionReceptor(
            name="NLRP3", prr_type="NLR",
            ligands=["ATP", "uric_acid"],
            pamp_or_damp="DAMP",
            signaling_pathway="inflammasome",
            cell_expression=["macrophage"],
            downstream_effectors=["caspase-1", "IL1b"],
            subcellular_location="cytoplasmic",
            gene_symbol="NLRP3",
        )
        d = prr.to_dict()
        prr2 = BioEntity.from_dict(d)
        assert isinstance(prr2, PatternRecognitionReceptor)
        assert prr2.subcellular_location == "cytoplasmic"
        assert prr2.downstream_effectors == ["caspase-1", "IL1b"]


class TestComplementComponent:

    def test_create_complement(self):
        cc = ComplementComponent(
            name="C3",
            pathway="all",
            activation_step=3,
            cleavage_products=["C3a", "C3b"],
            gene_symbol="C3",
        )
        assert cc.entity_type == EntityType.COMPLEMENT_COMPONENT
        assert cc.pathway == "all"
        assert cc.activation_step == 3

    def test_complement_roundtrip(self):
        cc = ComplementComponent(
            name="C5", pathway="terminal",
            activation_step=5,
            cleavage_products=["C5a", "C5b"],
            function="MAC initiation",
            deficiency_phenotype="Recurrent Neisseria",
            serum_concentration_ug_ml=75.0,
            gene_symbol="C5",
        )
        d = cc.to_dict()
        cc2 = BioEntity.from_dict(d)
        assert isinstance(cc2, ComplementComponent)
        assert cc2.serum_concentration_ug_ml == 75.0
        assert cc2.deficiency_phenotype == "Recurrent Neisseria"


class TestMHCMolecule:

    def test_create_mhc(self):
        mhc = MHCMolecule(
            name="HLA-A*02:01",
            mhc_class="I",
            hla_allele="HLA-A*02:01",
            gene_symbol="HLA-A",
        )
        assert mhc.entity_type == EntityType.MHC_MOLECULE
        assert mhc.mhc_class == "I"

    def test_mhc_roundtrip(self):
        mhc = MHCMolecule(
            name="HLA-DRB1", mhc_class="II",
            hla_allele="HLA-DRB1*04:01",
            gene_symbol="HLA-DRB1",
            peptide_length_range=[13, 25],
            presenting_cell_types=["dendritic", "macrophage", "B_cell"],
            associated_diseases=["Rheumatoid arthritis"],
            population_frequency=0.12,
        )
        d = mhc.to_dict()
        mhc2 = BioEntity.from_dict(d)
        assert isinstance(mhc2, MHCMolecule)
        assert mhc2.mhc_class == "II"
        assert mhc2.peptide_length_range == [13, 25]
        assert mhc2.population_frequency == 0.12


# ── ImmuneCellEntity Expanded Fields ────────────────────────────────

class TestImmuneCellEntityExpanded:

    def test_new_fields(self):
        ic = ImmuneCellEntity(
            name="CD8+ effector",
            immune_type="T_cell",
            immune_subtype="effector",
            polarization_state="",
            cytokines_secreting=["IFNg", "TNFa"],
            surface_markers=["CD8", "CD45"],
            exhaustion_level=0.3,
        )
        assert ic.immune_subtype == "effector"
        assert ic.exhaustion_level == 0.3
        assert "IFNg" in ic.cytokines_secreting
        assert "CD8" in ic.surface_markers

    def test_expanded_roundtrip(self):
        ic = ImmuneCellEntity(
            name="Treg",
            immune_type="T_cell",
            immune_subtype="Treg",
            polarization_state="regulatory",
            cytokines_secreting=["IL10", "TGFb"],
            surface_markers=["CD4", "CD25", "FOXP3"],
            exhaustion_level=0.0,
        )
        d = ic.to_dict()
        ic2 = BioEntity.from_dict(d)
        assert isinstance(ic2, ImmuneCellEntity)
        assert ic2.immune_subtype == "Treg"
        assert ic2.cytokines_secreting == ["IL10", "TGFb"]
        assert ic2.surface_markers == ["CD4", "CD25", "FOXP3"]


# ── Entity Registry ────────────────────────────────────────────────

class TestEntityRegistry:

    def test_new_types_in_registry(self):
        """All 8 new immunology types are registered."""
        new_types = [
            "virus", "bacterium", "antibody", "antigen",
            "cytokine", "prr", "complement", "mhc",
        ]
        for t in new_types:
            assert t in entity_registry, f"{t} not in registry"

    def test_new_types_in_class_map(self):
        """All 8 new types are in ENTITY_CLASS_MAP."""
        expected = {
            "virus": Virus,
            "bacterium": Bacterium,
            "antibody": Antibody,
            "antigen": Antigen,
            "cytokine": Cytokine,
            "prr": PatternRecognitionReceptor,
            "complement": ComplementComponent,
            "mhc": MHCMolecule,
        }
        for name, cls in expected.items():
            assert ENTITY_CLASS_MAP[name] is cls

    def test_entity_type_enum_has_new_values(self):
        """EntityType enum contains all 8 new values."""
        new_values = [
            "virus", "bacterium", "antibody", "antigen",
            "cytokine", "prr", "complement", "mhc",
        ]
        enum_values = [e.value for e in EntityType]
        for v in new_values:
            assert v in enum_values, f"{v} not in EntityType enum"


# ── RelationshipType ───────────────────────────────────────────────

class TestNewRelationshipTypes:

    def test_new_relationship_types_exist(self):
        """7 new immunology RelationshipType values exist."""
        new_types = [
            "presents", "recognizes", "neutralizes", "opsonizes",
            "activates_complement", "polarizes", "differentiates",
        ]
        for t in new_types:
            rt = RelationshipType(t)
            assert rt.value == t

    def test_create_relationship_with_new_types(self):
        for rt in [RelationshipType.PRESENTS, RelationshipType.NEUTRALIZES,
                    RelationshipType.OPSONIZES, RelationshipType.POLARIZES]:
            r = Relationship(source_id="a", target_id="b", rel_type=rt)
            d = r.to_dict()
            r2 = Relationship.from_dict(d)
            assert r2.rel_type == rt


# ── Seed Immunology Catalog ────────────────────────────────────────

class MockStore:
    """Minimal store mock for seed testing."""

    def __init__(self):
        self._entities = {}
        self._rels = []

    def add_entity(self, entity):
        self._entities[entity.entity_id] = entity
        return True

    def add_relationship(self, rel):
        self._rels.append(rel)
        return True

    def find_entity_by_name(self, name, entity_type=None):
        for e in self._entities.values():
            if e.name == name:
                return e
        return None


class TestSeedImmunologyCatalog:

    def test_seed_creates_entities(self):
        from cognisom.library.seed_immunology import seed_immunology_catalog

        store = MockStore()
        counts = seed_immunology_catalog(store)

        assert counts["immune_cells"] >= 20
        assert counts["cytokines"] >= 15
        assert counts["viruses"] >= 5
        assert counts["bacteria"] >= 5
        assert counts["antibodies"] >= 5
        assert counts["complement"] >= 5
        assert counts["prrs"] >= 10
        assert counts["mhc"] >= 5
        assert counts["drugs"] >= 10
        assert counts["pathways"] >= 5
        assert counts["relationships"] >= 15
        assert len(store._entities) >= 100

    def test_seed_entity_types_correct(self):
        from cognisom.library.seed_immunology import seed_immunology_catalog

        store = MockStore()
        seed_immunology_catalog(store)

        types_seen = set()
        for e in store._entities.values():
            types_seen.add(e.entity_type.value)

        # Should see all the new types
        assert "immune_cell" in types_seen
        assert "cytokine" in types_seen
        assert "virus" in types_seen
        assert "bacterium" in types_seen
        assert "antibody" in types_seen
        assert "complement" in types_seen
        assert "prr" in types_seen
        assert "mhc" in types_seen


# ── Gene Sets ──────────────────────────────────────────────────────

class TestImmunologyGeneSets:

    def test_gene_sets_importable(self):
        from cognisom.library.gene_sets import (
            INNATE_IMMUNITY_GENES,
            ADAPTIVE_IMMUNITY_GENES,
            CYTOKINE_CHEMOKINE_GENES,
            T_CELL_SUBSET_MARKERS,
            B_CELL_MARKERS,
            COMPLEMENT_GENES,
            ANTIGEN_PRESENTATION_GENES,
        )
        assert len(INNATE_IMMUNITY_GENES) >= 40
        assert len(ADAPTIVE_IMMUNITY_GENES) >= 40
        assert len(CYTOKINE_CHEMOKINE_GENES) >= 60
        assert len(T_CELL_SUBSET_MARKERS) >= 30
        assert len(B_CELL_MARKERS) >= 20
        assert len(COMPLEMENT_GENES) >= 20
        assert len(ANTIGEN_PRESENTATION_GENES) >= 20

    def test_gene_sets_no_empty_strings(self):
        from cognisom.library.gene_sets import (
            INNATE_IMMUNITY_GENES,
            ADAPTIVE_IMMUNITY_GENES,
            CYTOKINE_CHEMOKINE_GENES,
            T_CELL_SUBSET_MARKERS,
            B_CELL_MARKERS,
            COMPLEMENT_GENES,
            ANTIGEN_PRESENTATION_GENES,
        )
        all_genes = (INNATE_IMMUNITY_GENES + ADAPTIVE_IMMUNITY_GENES +
                     CYTOKINE_CHEMOKINE_GENES + T_CELL_SUBSET_MARKERS +
                     B_CELL_MARKERS + COMPLEMENT_GENES + ANTIGEN_PRESENTATION_GENES)
        for g in all_genes:
            assert g.strip() != "", f"Empty gene symbol found"

    def test_import_sets_contain_immunology(self):
        from cognisom.library.gene_sets import IMPORT_SETS

        immuno_sets = [
            "Immunology Foundations (~130 genes)",
            "Innate Immunity (~80 genes)",
            "Adaptive Immunity (~90 genes)",
            "Cytokine Network (~70 genes)",
            "Immune Therapeutic Targets (~50 genes + 17 drugs)",
            "Complete Immunology (~300 genes + 17 drugs)",
        ]
        for name in immuno_sets:
            assert name in IMPORT_SETS, f"'{name}' not in IMPORT_SETS"
            s = IMPORT_SETS[name]
            assert "genes" in s
            assert "drugs" in s
            assert "description" in s

    def test_complete_immunology_gene_count(self):
        from cognisom.library.gene_sets import IMPORT_SETS
        s = IMPORT_SETS["Complete Immunology (~300 genes + 17 drugs)"]
        assert len(s["genes"]) >= 250


# ── Bio-USD ImmuneCellType Enum ────────────────────────────────────

class TestImmuneCellTypeEnum:

    def test_enum_has_25_plus_members(self):
        from cognisom.biousd.schema import ImmuneCellType
        assert len(ImmuneCellType) >= 25

    def test_enum_has_t_cell_subtypes(self):
        from cognisom.biousd.schema import ImmuneCellType
        subtypes = [
            ImmuneCellType.T_CELL_CD8_NAIVE,
            ImmuneCellType.T_CELL_CD8_EFFECTOR,
            ImmuneCellType.T_CELL_CD8_MEMORY,
            ImmuneCellType.T_CELL_CD4_TH1,
            ImmuneCellType.T_CELL_CD4_TH2,
            ImmuneCellType.T_CELL_CD4_TH17,
            ImmuneCellType.T_CELL_TREG,
            ImmuneCellType.T_CELL_TFH,
            ImmuneCellType.T_CELL_GAMMA_DELTA,
        ]
        for st in subtypes:
            assert st is not None

    def test_enum_has_granulocytes(self):
        from cognisom.biousd.schema import ImmuneCellType
        for name in ["NEUTROPHIL", "MAST_CELL", "BASOPHIL", "EOSINOPHIL"]:
            assert hasattr(ImmuneCellType, name)

    def test_enum_has_ilcs(self):
        from cognisom.biousd.schema import ImmuneCellType
        for name in ["ILC1", "ILC2", "ILC3"]:
            assert hasattr(ImmuneCellType, name)

    def test_backward_compat_legacy_aliases(self):
        from cognisom.biousd.schema import ImmuneCellType
        # Original 5 values still work
        assert ImmuneCellType.T_CELL.value == "T_cell"
        assert ImmuneCellType.NK_CELL.value == "NK_cell"
        assert ImmuneCellType.MACROPHAGE.value == "macrophage"
        assert ImmuneCellType.DENDRITIC.value == "dendritic"
        assert ImmuneCellType.B_CELL.value == "B_cell"


# ── Bio-USD New Prim Types ─────────────────────────────────────────

class TestBioUSDNewPrimTypes:

    def test_bio_antibody(self):
        from cognisom.biousd.schema import BioAntibody
        ab = BioAntibody(isotype="IgG1", target_antigen="spike", bound=True)
        assert ab.isotype == "IgG1"
        assert ab.bound is True

    def test_bio_virus_particle(self):
        from cognisom.biousd.schema import BioVirusParticle
        vp = BioVirusParticle(virus_type="SARS-CoV-2", genome_type="ssRNA+")
        assert vp.virus_type == "SARS-CoV-2"
        assert vp.capsid_intact is True

    def test_bio_cytokine_field(self):
        from cognisom.biousd.schema import BioCytokineField
        cf = BioCytokineField(cytokine_name="IFNg", concentration=150.0)
        assert cf.cytokine_name == "IFNg"
        assert cf.concentration == 150.0

    def test_new_prims_in_registry(self):
        from cognisom.biousd.schema import prim_registry
        assert "bio_antibody" in prim_registry
        assert "bio_virus_particle" in prim_registry
        assert "bio_cytokine_field" in prim_registry


# ── Immune Visualization Prototypes ────────────────────────────────

class TestImmuneVisualizationPrototypes:

    def test_all_17_prototypes_exist(self):
        from cognisom.omniverse.prototype_library import ALL_PROTOTYPES

        immune_prototypes = [
            "t_cell_cd8", "t_cell_cd4", "b_cell", "plasma_cell",
            "dendritic_cell", "macrophage_m1", "macrophage_m2",
            "neutrophil", "mast_cell", "nk_cell",
            "antibody_igg", "antibody_igm", "virus_capsid",
            "bacterium_rod", "bacterium_coccus",
            "cytokine_particle", "complement_complex",
        ]
        for name in immune_prototypes:
            assert name in ALL_PROTOTYPES, f"'{name}' not in ALL_PROTOTYPES"

    def test_immune_cell_sizes_reasonable(self):
        from cognisom.omniverse.prototype_library import IMMUNE_CELL_PROTOTYPES

        for name, spec in IMMUNE_CELL_PROTOTYPES.items():
            assert 5.0 <= spec.default_size <= 25.0, \
                f"{name} size {spec.default_size} outside 5-25 um range"

    def test_immune_particles_are_small(self):
        from cognisom.omniverse.prototype_library import IMMUNE_PARTICLE_PROTOTYPES

        for name, spec in IMMUNE_PARTICLE_PROTOTYPES.items():
            # Bacteria can be up to 2um, molecules < 0.1um
            assert spec.default_size <= 2.5, \
                f"{name} size {spec.default_size} too large for particle"

    def test_total_prototype_count(self):
        from cognisom.omniverse.prototype_library import ALL_PROTOTYPES
        assert len(ALL_PROTOTYPES) >= 47
