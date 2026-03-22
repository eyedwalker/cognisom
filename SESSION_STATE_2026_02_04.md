# Session State - February 4, 2026

## Completed Today

### 1. AWS Migration (COMPLETE)
- Migrated from NVIDIA Brev to AWS ECS
- Dashboard live at: **https://cognisom.com**
- ECS Cluster: `cognisom-production-cluster`
- ECR Image: `780457123717.dkr.ecr.us-east-1.amazonaws.com/cognisom:latest`
- Brev instance (`eyentelligence`) is STOPPED

### 2. Realistic Molecular 3D Visualization (COMPLETE)
Added realistic molecular shapes to VCell Solvers page:

| Molecule Type | Shape |
|---------------|-------|
| Receptors (EGFR, TCR, Fas) | Y-shaped: extracellular domain + TM helix + intracellular kinase |
| Ligands (EGF, FasL) | Compact globular + binding loop |
| Kinases (Raf, MEK, ERK) | Bilobed (N-lobe + C-lobe) + active site cleft |
| G-proteins (Ras) | Sphere + switch I/II regions |
| Transcription factors | Helix-turn-helix + dimerization domain |
| Caspases | Heterodimer + yellow cysteine active site |
| Cytochrome C | Sphere + red heme disk + iron center |
| Cell membrane | Lipid bilayer (1600 phospholipid heads) |
| Nucleus | Sphere + 30 nuclear pore complexes |
| DNA | Double helix with colored base pairs |
| Mitochondria | Outer + inner membrane + cristae |

### 3. Bug Fix: Three.js r128 Compatibility
- Fixed `CapsuleGeometry` not available in r128
- Created `createCapsule()` helper using sphere+cylinder
- All geometries now r128 compatible

## Key Files Modified
- `dashboard/pages/20_vcell_solvers.py` - Molecular 3D visualization
- `cognisom/dashboard/pages/20_vcell_solvers.py` - Synced copy

## AWS Infrastructure
```
ECS Cluster: cognisom-production-cluster
Service: cognisom-production-cpu (Fargate, 1 task)
ALB: cognisom-production-alb-1273102351.us-east-1.elb.amazonaws.com
Domain: cognisom.com (Route53 → ALB)
ECR: 780457123717.dkr.ecr.us-east-1.amazonaws.com/cognisom
```

## Deployment Commands
```bash
# Login to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 780457123717.dkr.ecr.us-east-1.amazonaws.com

# Build
docker build -f Dockerfile.prod -t 780457123717.dkr.ecr.us-east-1.amazonaws.com/cognisom:latest .

# Push
docker push 780457123717.dkr.ecr.us-east-1.amazonaws.com/cognisom:latest

# Deploy
aws ecs update-service --cluster cognisom-production-cluster --service cognisom-production-cpu --force-new-deployment
```

## Infrastructure Additions (Created - Not Yet Applied)

### RDS PostgreSQL (`infrastructure/terraform/rds.tf`)
- HIPAA-eligible PostgreSQL 15.4
- Encryption at rest (AES-256) + in transit (TLS)
- 35-day backup retention for compliance
- Secrets Manager for credentials
- db.t3.micro (~$15/mo, free tier eligible)

### AWS Cognito (`infrastructure/terraform/cognito.tf`)
- User Pool with MFA support
- SAML/OIDC federation ready (for university SSO)
- Research custom attributes: organization, role, research_area
- User groups: admin, researcher, viewer
- Identity Pool for AWS resource access (S3 user folders)
- HIPAA-eligible with BAA

### Infrastructure Applied ✅
```bash
# RDS PostgreSQL 16.11 - HIPAA-eligible
Endpoint: cognisom-production-postgres.cs1cieu8w59o.us-east-1.rds.amazonaws.com:5432
Database: cognisom
Credentials: AWS Secrets Manager (cognisom-production/db-credentials)

# AWS Cognito - Research-grade auth
User Pool ID: us-east-1_xDPTQFusL
Web Client ID: 7977nudcem6jnublqc5mgaqr3a
Auth Domain: https://cognisom-production-auth.auth.us-east-1.amazoncognito.com
Identity Pool: us-east-1:36ca38bc-2307-47f0-88c8-fae1bb95afbd
```

### Completed Tasks
- [x] Start Docker Desktop
- [x] Rebuild with cellxgene-census
- [x] Apply Terraform for RDS + Cognito
- [x] Update EntityStore to use PostgreSQL
  - Updated `library/store.py` to support PostgreSQL (production) + SQLite (fallback)
  - Added psycopg2-binary and boto3 to requirements
  - Updated ECS task definition with DATABASE_URL secret
  - Updated IAM policy to allow reading DB credentials

### All Infrastructure Tasks Complete ✅
- [x] Integrate Cognito with dashboard auth
  - Created `auth/cognito_provider.py` with full Cognito integration
  - Updated `auth/middleware.py` for dual auth (Cognito + local)
  - Added Cognito environment variables to ECS task
  - Supports: registration, login, MFA, password reset, token refresh
  - University SSO via Cognito hosted UI

## Research-Grade Infrastructure Summary

| Component | Service | Status |
|-----------|---------|--------|
| Dashboard | ECS Fargate | ✅ Live at https://cognisom.com |
| Database | RDS PostgreSQL 16.11 | ✅ HIPAA-compliant |
| Auth | AWS Cognito | ✅ MFA + SSO ready |
| Storage | S3 + Secrets Manager | ✅ Encrypted |
| DNS | Route53 | ✅ cognisom.com |

## Next Steps (from VCell Parity Plan)
1. Phase 1: Full ODE Solver (GPU batched CVODE-style)
2. Phase 2: Smoldyn Spatial (particle Brownian dynamics)
3. Phase 3: Hybrid ODE/SSA coupling
4. Phase 4: BNGL rule-based modeling
5. Phase 5: Imaging pipeline (image-to-geometry)

## URLs
- Dashboard: https://cognisom.com
- VCell Solvers: https://cognisom.com → Page 20
- Plan file: ~/.claude/plans/piped-bouncing-mountain.md
