#!/usr/bin/env python3
"""
Publication Export System
=========================

Generate publication-ready reports and figures.

Formats:
- HTML reports
- PDF reports (via weasyprint)
- Markdown documentation
- LaTeX manuscripts
"""

import sys
sys.path.insert(0, '..')

from datetime import datetime
import json


class Publisher:
    """
    Generate publication-ready outputs
    """
    
    def __init__(self, engine):
        self.engine = engine
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def generate_html_report(self, filename='report.html'):
        """Generate HTML report"""
        state = self.engine.get_state()
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>cognisom Simulation Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }}
        h1 {{ color: #667eea; }}
        h2 {{ color: #764ba2; border-bottom: 2px solid #667eea; padding-bottom: 10px; }}
        .summary {{ background: #f3f4f6; padding: 20px; border-radius: 10px; margin: 20px 0; }}
        .stat {{ display: inline-block; margin: 10px 20px; }}
        .stat-label {{ font-weight: bold; color: #666; }}
        .stat-value {{ font-size: 1.5em; color: #667eea; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #667eea; color: white; }}
        .footer {{ margin-top: 40px; padding-top: 20px; border-top: 2px solid #ddd; color: #666; }}
    </style>
</head>
<body>
    <h1>🧬 cognisom Simulation Report</h1>
    <p><strong>Generated:</strong> {self.timestamp}</p>
    
    <div class="summary">
        <h2>Executive Summary</h2>
        <div class="stat">
            <div class="stat-label">Duration</div>
            <div class="stat-value">{state['time']:.2f} hours</div>
        </div>
        <div class="stat">
            <div class="stat-label">Total Steps</div>
            <div class="stat-value">{state['step_count']}</div>
        </div>
        <div class="stat">
            <div class="stat-label">Modules Active</div>
            <div class="stat-value">{len([m for m in state.keys() if m not in ['time', 'step_count', 'running']])}</div>
        </div>
    </div>
    
    <h2>📊 Results by Module</h2>
"""
        
        # Add module results
        for module_name, module_state in state.items():
            if module_name not in ['time', 'step_count', 'running']:
                html += f"""
    <h3>{module_name.upper()}</h3>
    <table>
        <tr><th>Parameter</th><th>Value</th></tr>
"""
                for key, value in module_state.items():
                    if isinstance(value, (int, float, str)):
                        html += f"        <tr><td>{key}</td><td>{value}</td></tr>\n"
                
                html += "    </table>\n"
        
        html += f"""
    <div class="footer">
        <p><strong>cognisom</strong> - Multi-Scale Cellular Simulation Platform</p>
        <p>Report generated: {self.timestamp}</p>
    </div>
</body>
</html>
"""
        
        with open(filename, 'w') as f:
            f.write(html)
        
        print(f"✓ HTML report generated: {filename}")
        return filename
    
    def generate_markdown_report(self, filename='report.md'):
        """Generate Markdown report"""
        state = self.engine.get_state()
        
        md = f"""# cognisom Simulation Report

**Generated:** {self.timestamp}

## Executive Summary

- **Duration:** {state['time']:.2f} hours
- **Total Steps:** {state['step_count']}
- **Modules Active:** {len([m for m in state.keys() if m not in ['time', 'step_count', 'running']])}

## Results by Module

"""
        
        for module_name, module_state in state.items():
            if module_name not in ['time', 'step_count', 'running']:
                md += f"### {module_name.upper()}\n\n"
                md += "| Parameter | Value |\n"
                md += "|-----------|-------|\n"
                
                for key, value in module_state.items():
                    if isinstance(value, (int, float, str)):
                        md += f"| {key} | {value} |\n"
                
                md += "\n"
        
        md += f"""
---

**cognisom** - Multi-Scale Cellular Simulation Platform  
Report generated: {self.timestamp}
"""
        
        with open(filename, 'w') as f:
            f.write(md)
        
        print(f"✓ Markdown report generated: {filename}")
        return filename
    
    def generate_latex_manuscript(self, filename='manuscript.tex'):
        """Generate LaTeX manuscript template"""
        state = self.engine.get_state()
        
        latex = r"""\documentclass[12pt]{article}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{hyperref}

\title{Multi-Scale Cellular Simulation Results}
\author{cognisom Platform}
\date{""" + self.timestamp + r"""}

\begin{document}

\maketitle

\begin{abstract}
This report presents results from a multi-scale cellular simulation using the cognisom platform, 
integrating molecular, cellular, immune, vascular, lymphatic, spatial, epigenetic, circadian, 
and morphogen modules.
\end{abstract}

\section{Methods}

The simulation was performed using the cognisom platform with the following configuration:
\begin{itemize}
    \item Duration: """ + f"{state['time']:.2f}" + r""" hours
    \item Time step: 0.01 hours
    \item Total steps: """ + str(state['step_count']) + r"""
\end{itemize}

\section{Results}

"""
        
        for module_name, module_state in state.items():
            if module_name not in ['time', 'step_count', 'running']:
                latex += f"\\subsection{{{module_name.capitalize()}}}\n\n"
                latex += "\\begin{table}[h]\n"
                latex += "\\centering\n"
                latex += "\\begin{tabular}{ll}\n"
                latex += "\\toprule\n"
                latex += "Parameter & Value \\\\\n"
                latex += "\\midrule\n"
                
                for key, value in module_state.items():
                    if isinstance(value, (int, float, str)):
                        latex += f"{key.replace('_', ' ')} & {value} \\\\\n"
                
                latex += "\\bottomrule\n"
                latex += "\\end{tabular}\n"
                latex += f"\\caption{{{module_name.capitalize()} module results}}\n"
                latex += "\\end{table}\n\n"
        
        latex += r"""
\section{Discussion}

[Add your discussion here]

\section{Conclusions}

[Add your conclusions here]

\end{document}
"""
        
        with open(filename, 'w') as f:
            f.write(latex)
        
        print(f"✓ LaTeX manuscript generated: {filename}")
        return filename
    
    def generate_all_formats(self, prefix='report'):
        """Generate all report formats"""
        html_file = self.generate_html_report(f'{prefix}.html')
        md_file = self.generate_markdown_report(f'{prefix}.md')
        tex_file = self.generate_latex_manuscript(f'{prefix}.tex')
        
        return {
            'html': html_file,
            'markdown': md_file,
            'latex': tex_file
        }


# Test
if __name__ == '__main__':
    from core import SimulationEngine, SimulationConfig
    from modules import CellularModule, ImmuneModule
    
    print("=" * 70)
    print("Publication Export System Test")
    print("=" * 70)
    print()
    
    # Create and run simulation
    engine = SimulationEngine(SimulationConfig(dt=0.01, duration=1.0))
    engine.register_module('cellular', CellularModule)
    engine.register_module('immune', ImmuneModule)
    engine.initialize()
    
    immune = engine.modules['immune']
    cellular = engine.modules['cellular']
    immune.set_cellular_module(cellular)
    
    engine.run()
    
    # Generate reports
    publisher = Publisher(engine)
    files = publisher.generate_all_formats('test_report')
    
    print()
    print("✓ All reports generated:")
    for format_type, filename in files.items():
        print(f"  - {format_type}: {filename}")
    
    print()
    print("=" * 70)
    print("✓ Publisher working!")
    print("=" * 70)
