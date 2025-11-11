#!/bin/bash
# Rename bioSim to cognisom
# Run this script to update all references

echo "=========================================="
echo "Renaming bioSim → cognisom"
echo "=========================================="
echo ""

# Confirm
read -p "This will rename all 'bioSim' to 'cognisom'. Continue? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Cancelled."
    exit 1
fi

echo "Starting rename..."
echo ""

# Count occurrences first
echo "Current occurrences of 'bioSim':"
grep -r "bioSim" --include="*.md" --include="*.py" --exclude-dir=PhysiCell --exclude-dir=.git . | wc -l
echo ""

# Rename in all markdown files (excluding PhysiCell)
echo "Updating .md files..."
find . -type f -name "*.md" -not -path "./PhysiCell/*" -not -path "./.git/*" -exec sed -i '' 's/bioSim/cognisom/g' {} +

# Rename in all Python files
echo "Updating .py files..."
find . -type f -name "*.py" -not -path "./PhysiCell/*" -not -path "./.git/*" -exec sed -i '' 's/bioSim/cognisom/g' {} +

# Rename in requirements.txt
if [ -f "requirements.txt" ]; then
    echo "Updating requirements.txt..."
    sed -i '' 's/bioSim/cognisom/g' requirements.txt
fi

# Rename in setup.py if exists
if [ -f "setup.py" ]; then
    echo "Updating setup.py..."
    sed -i '' 's/bioSim/cognisom/g' setup.py
fi

echo ""
echo "=========================================="
echo "✓ Rename complete!"
echo "=========================================="
echo ""

# Verify
echo "Remaining occurrences of 'bioSim':"
grep -r "bioSim" --include="*.md" --include="*.py" --exclude-dir=PhysiCell --exclude-dir=.git . | wc -l
echo ""

echo "Next steps:"
echo "1. Rename directory: cd .. && mv bioSim cognisom"
echo "2. Update git remote if needed"
echo "3. Review changes: git diff"
echo "4. Commit: git add -A && git commit -m 'Rename project to cognisom'"
echo ""
