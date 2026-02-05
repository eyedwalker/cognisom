# Grant & Funding Targets for cognisom Platform

## Immediate Actions (Apply This Week)

### 1. NVIDIA Inception Program
**Amount**: GPU credits + technical support  
**URL**: https://www.nvidia.com/inception/  
**Application**: Use `/funding/NVIDIA_APPLICATION.md` content  
**Timeline**: Rolling applications, 2-4 week review  
**Status**: ⏳ Ready to submit

**What to include**:
- Mission statement (cellular + humanoid platforms)
- NVIDIA technology usage (CUDA, H100, Jetson)
- Competitive differentiators
- Team & roadmap

---

### 2. AWS Cloud Credit for Research
**Amount**: $5k-$50k in compute credits  
**URL**: https://aws.amazon.com/research-credits/  
**Requirements**: Research project description, institution affiliation (optional)  
**Timeline**: 4-6 weeks  
**Status**: ⏳ Ready to submit

**Application Summary** (copy/paste):
```
Project: GPU-Accelerated Cellular Simulation for Cancer Research

We are developing cognisom, a mechanistic simulation platform that models 
how cancer cells evade immune detection. We need H100/A100 instances to 
run multi-million cell tumor-immune simulations for prostate and pancreatic 
cancer research. Our platform will be open-source and enable predictive 
treatment response modeling.

Compute needs: 500-1000 GPU-hours/month for parameter sweeps and validation.
```

---

### 3. Google Cloud Research Credits
**Amount**: $5k-$25k in credits  
**URL**: https://cloud.google.com/edu/researchers  
**Timeline**: 3-4 weeks  
**Status**: ⏳ Ready to submit

---

### 4. Microsoft Azure Research Sponsorship
**Amount**: Variable compute credits  
**URL**: https://azure.microsoft.com/free/research/  
**Timeline**: 4-6 weeks  
**Status**: ⏳ Ready to submit

---

## Near-Term Grants (6-12 Months)

### 5. NIH NCI ITCR (Informatics Technology for Cancer Research)
**Amount**: $100k-$400k  
**Deadline**: Check https://itcr.cancer.gov/funding-opportunities  
**FOA**: RFA-CA-XX-XXX (varies by year)  
**Timeline**: 6-9 month review cycle  
**Status**: 🔍 Monitor for next RFA

**Key Points**:
- Funds development of cancer research software/tools
- Must be open-source and community-accessible
- Strong fit for cognisom platform
- Need preliminary data (single-cell validation)

**Pre-Application Strategy**:
1. Contact program officer (Dr. Juli Klemm, klemmmj@mail.nih.gov)
2. Send 1-page concept summary
3. Ask: "Does this fit current ITCR priorities?"
4. Request feedback before full application

---

### 6. NSF CSSI (Cyberinfrastructure for Sustained Scientific Innovation)
**Amount**: $80k-$300k (Elements track)  
**Deadline**: Check https://www.nsf.gov/funding/  
**Program**: OAC-2103836 (or current year equivalent)  
**Timeline**: 6-9 months  
**Status**: 🔍 Monitor for next call

**Key Points**:
- Funds reusable scientific software infrastructure
- GPU acceleration is a priority area
- Open-source requirement
- Community engagement plan needed

**Pre-Application Email Template**:
```
Subject: CSSI Elements Inquiry - GPU-Accelerated Cellular Simulation

Dear Program Officer,

I am developing cognisom, a GPU-accelerated platform for mechanistic 
cellular and immune simulation targeting cancer research. The platform 
will be open-source and provide:

1. CUDA kernels for stochastic biochemical simulation (SSA/tau-leap)
2. Multi-GPU spatial reaction-diffusion solvers
3. Immune recognition models (MHC-I, T-cell, NK cell)
4. Standards-based I/O (SBML, Zarr, HDF5)

Would this align with current CSSI Elements priorities? I would 
appreciate guidance on positioning this for the next call.

Best regards,
David Walker
eyentelligence
```

---

### 7. DoD PCRP (Prostate Cancer Research Program)
**Amount**: $400k-$1.2M  
**Program**: Idea Development Award  
**Deadline**: Check https://cdmrp.health.mil/pcrp/  
**Timeline**: Annual cycle (typically June-August)  
**Status**: 🔍 Monitor for FY2026 announcement

**Key Points**:
- Funds innovative prostate cancer research
- Computational/modeling approaches encouraged
- Must address clinically relevant questions
- Consumer advocate involvement required

**Focus Areas for cognisom**:
- Castration-resistant prostate cancer (CRPC)
- Immune evasion mechanisms
- Treatment resistance prediction
- Combination therapy optimization

---

### 8. NIH R21 (Exploratory/Developmental Research Grant)
**Amount**: $275k over 2 years  
**Deadline**: Standard dates (Feb 16, June 16, Oct 16)  
**Timeline**: 6-9 months  
**Status**: ⏳ Can apply anytime

**Key Points**:
- For high-risk, high-reward projects
- No preliminary data required
- Perfect for novel computational approaches
- Can apply to multiple institutes (NCI, NIGMS, NIBIB)

**Specific Aims Template**:
```
Title: GPU-Accelerated Mechanistic Simulation of Immune Evasion 
       in Prostate Cancer

Aim 1: Develop and validate a GPU-accelerated intracellular simulation 
       engine with MHC-I antigen presentation

Aim 2: Model NK and CD8 T-cell recognition of prostate cancer cells 
       under oncogenic stress

Aim 3: Simulate immune evasion strategies and predict checkpoint 
       blockade response
```

---

## Long-Term / High-Impact (12-24 Months)

### 9. Cancer Grand Challenges
**Amount**: $1M-$25M over 5 years  
**URL**: https://cancergrandchallenges.org/  
**Timeline**: Multi-year, team-based  
**Status**: 🎯 Future target (need team + preliminary data)

**Requirements**:
- Multi-institutional team
- Addresses one of the Grand Challenges
- Significant preliminary data
- International collaboration encouraged

**Relevant Challenges**:
- "Tumour evolution and ecosystem"
- "Lethal vs non-lethal cancers"
- "Therapeutic resistance"

---

### 10. NIH SBIR Phase I
**Amount**: $150k-$250k  
**Requirements**: For-profit entity (need to form company)  
**Timeline**: 6-9 months  
**Status**: 🏢 Requires business entity

**Commercialization Path**:
1. Form LLC or C-corp
2. Apply for SBIR Phase I (feasibility)
3. Phase II ($1M-$2M) for development
4. Phase IIB (up to $4M) for commercialization

---

## Free Compute Access

### 11. NIH STRIDES Initiative
**Benefit**: Discounted cloud compute (AWS, Google, Azure)  
**Requirements**: Active NIH grant or collaboration with NIH-funded lab  
**URL**: https://datascience.nih.gov/strides  
**Status**: 🤝 Need NIH collaborator

**Strategy**: Partner with a cancer biology lab that has NIH funding

---

### 12. XSEDE/ACCESS Allocations
**Benefit**: Free HPC time on national supercomputers  
**URL**: https://allocations.access-ci.org/  
**Timeline**: Startup allocations (quick), Research allocations (competitive)  
**Status**: ⏳ Can apply now

**Allocation Types**:
- **Explore ACCESS**: Up to 400k SUs (free, fast approval)
- **Discover ACCESS**: Up to 1.5M SUs (requires proposal)
- **Accelerate ACCESS**: Up to 3M SUs (competitive)

---

## Application Priority Order

### Week 1-2 (Immediate)
1. ✅ NVIDIA Inception
2. ✅ AWS Cloud Credits
3. ✅ Google Cloud Credits
4. ✅ Azure Research Credits

### Month 1-3 (Build Prototype)
5. ⏳ XSEDE/ACCESS Explore (startup allocation)
6. ⏳ Develop single-cell validation data
7. ⏳ Create demo videos/visualizations

### Month 3-6 (Pre-Application Outreach)
8. 📧 Contact NIH NCI ITCR program officer
9. 📧 Contact NSF CSSI program officer
10. 📧 Contact DoD PCRP scientific contact
11. 🔬 Publish preprint (bioRxiv) with preliminary results

### Month 6-12 (Submit Grants)
12. 📝 NIH R21 application
13. 📝 NSF CSSI Elements application
14. 📝 DoD PCRP Idea Development Award (if cycle open)

---

## Program Officer Contact Templates

### Template 1: Initial Inquiry
```
Subject: Pre-Application Inquiry - [Grant Program Name]

Dear [Program Officer Name],

I am developing a GPU-accelerated cellular simulation platform for 
cancer research and would like to inquire whether this aligns with 
current [Program] priorities.

Project Summary:
cognisom is a mechanistic simulation engine that models immune recognition 
and cancer immune evasion at cellular resolution. Key innovations include:
- GPU-accelerated stochastic biochemistry (1M+ cells)
- MHC-I antigen presentation from first principles
- Immune cell recognition (NK, CD8 T-cells)
- Treatment response prediction (ADT, checkpoint inhibitors)

Target Applications:
- Prostate cancer (normal → CRPC progression)
- Pancreatic cancer (immune-excluded microenvironment)
- Open-source platform for cancer research community

Would you be available for a brief call to discuss fit and provide 
guidance on application strategy?

Best regards,
David Walker
Founder, eyentelligence
research@eyentelligence.ai
```

---

### Template 2: Follow-Up After Initial Contact
```
Subject: Re: Pre-Application Inquiry - [Grant Program Name]

Dear [Program Officer Name],

Thank you for the helpful feedback on [date]. Based on our conversation, 
I have refined the project scope to emphasize [specific points they mentioned].

I have attached:
1. One-page project summary
2. Preliminary validation results
3. Technical architecture diagram

I plan to submit for the [deadline] cycle. Would you be willing to 
review a draft Specific Aims page before submission?

Thank you for your time and guidance.

Best regards,
David Walker
```

---

## Budget Justification Template

### For Grant Applications

**Personnel** ($180k-$260k/year)
- PI/Technical Lead (0.5 FTE): $90k-$130k
- Computational Biologist (1.0 FTE): $90k-$130k

**Equipment** ($0 if using cloud credits)
- Workstation (RTX 4090): $5k (one-time, if needed)

**Cloud Compute** ($20k-$60k/year)
- H100 GPU time: 500-1000 hrs/month @ $6-8/hr
- Storage (S3/equivalent): $2k-$5k/year

**Travel** ($5k-$10k/year)
- Conference presentations (GTC, AACR, ISMB)
- Collaborator meetings

**Publication/Dissemination** ($3k-$8k/year)
- Open-access publication fees
- Preprint servers (free)
- GitHub/documentation hosting (free)

**Total Direct Costs**: $230k-$340k/year

---

## Success Metrics for Grant Applications

### Technical Milestones
- ✓ Single-cell model validated against literature
- ✓ GPU SSA kernel achieving 10-100× speedup
- ✓ Tumor spheroid growth curves match experimental data
- ✓ Immune surveillance → escape dynamics reproduced

### Community Impact
- ✓ Open-source release (GitHub, MIT license)
- ✓ Documentation and tutorials
- ✓ User community (5-10 early adopters)
- ✓ Published validation study (preprint or peer-reviewed)

### Scientific Output
- ✓ 1-2 peer-reviewed publications
- ✓ Conference presentations (GTC, AACR, ISMB)
- ✓ Collaborations with cancer biology labs
- ✓ Integration with existing tools (PhysiCell, VCell)

---

## Tracking Spreadsheet

Create a simple tracking sheet:

| Grant | Amount | Deadline | Status | Contact | Notes |
|-------|--------|----------|--------|---------|-------|
| NVIDIA Inception | Credits | Rolling | Submitted | - | Waiting |
| AWS Credits | $10k | Rolling | Submitted | - | Waiting |
| NIH R21 | $275k | Feb 16 | Planning | - | Need prelim data |
| NSF CSSI | $200k | TBD | Monitoring | - | Next cycle |
| DoD PCRP | $600k | Jun 2026 | Future | - | FY26 cycle |

---

## Key Contacts to Cultivate

### NIH Program Officers
- **NCI ITCR**: Dr. Juli Klemm (klemmmj@mail.nih.gov)
- **NCI Cancer Systems Biology**: Check current roster
- **NIBIB**: Computational modeling program officers

### NSF Program Officers
- **OAC (Cyberinfrastructure)**: Check current roster
- **MCB (Molecular & Cellular Biosciences)**: Systems biology

### DoD CDMRP
- **PCRP**: Scientific Program Manager (changes yearly)

### Academic Collaborators (to identify)
- Prostate cancer biologists
- Tumor immunologists
- Computational biology groups

---

## Final Checklist Before Submitting

- [ ] Read the full FOA (Funding Opportunity Announcement)
- [ ] Check eligibility requirements
- [ ] Contact program officer (if allowed)
- [ ] Prepare biosketches/CVs
- [ ] Get letters of support (if required)
- [ ] Budget matches allowed categories
- [ ] Data management plan (if required)
- [ ] Human subjects / animal protocols (N/A for computational)
- [ ] Submit 48 hours before deadline (not last minute!)

---

**Remember**: Grant writing is a skill. First applications often don't get funded. Use reviewer feedback to improve. Persistence pays off.

**Timeline Reality**: From application to funding is typically 9-12 months. Apply to multiple programs simultaneously to increase odds.

**Success Rate**: NIH R21 ~15-20%, NSF CSSI ~20-25%, DoD PCRP ~10-15%. You need to apply to 5-10 programs to secure 1-2 awards.

---

*Good luck! The cognisom platform has strong potential. Focus on clear communication of the biological impact and technical innovation.*
