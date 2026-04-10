"""
CMIFES Module Architecture Diagram
Drawn based on CMIFES.py v7 source code.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe

FW, FH = 20, 26
fig, ax = plt.subplots(figsize=(FW, FH))
ax.set_xlim(0, FW)
ax.set_ylim(0, FH)
ax.axis('off')
fig.patch.set_facecolor('white')

# ── Palette ──
FC = {
    'input':    '#EBF5FB',
    'gate':     '#FAD7A0',   # CLG gate
    'se':        '#A9DFBF',   # SE attention
    'spatial':  '#D2B4DE',   # Spatial attention
    'align':     '#AED6F1',   # Align conv
    'proj':      '#F9E79F',   # Projection
    'add':       '#FDEBD0',   # Addition/element-wise
    'mul':       '#FDEDEC',   # Multiplication
    'concat':    '#D5DBDB',   # Concat stack
    'output':    '#EAFAF1',
    'bdr':       '#1B2631',
    'arrow':     '#2C3E50',
    'pe':        '#922B21',   # Path edge / emphasis
    'txt':       '#1A1A1A',
    'sub':       '#555555',
    'dim':       '#7F8C8D',
    'sig':       '#1A5276',   # Sigmoid
    'silu':      '#1D8348',   # SiLU
    'softmax':   '#784212',   # Softmax
    'sum':       '#C0392B',   # Summation
}

# ── Helpers ──
BW, BH = 3.0, 1.0

def box(x, y, w, h, label, subs, fc, ec=None, lw=1.4, ls='-', shadow=False, z=3):
    if ec is None: ec = FC['bdr']
    if shadow:
        ax.add_patch(FancyBboxPatch((x+.06, y-.06), w, h,
            boxstyle="round,pad=.06", facecolor='#C0B8C0',
            edgecolor='none', lw=0, zorder=z-1))
    ax.add_patch(FancyBboxPatch((x, y), w, h,
        boxstyle="round,pad=.06", facecolor=fc,
        edgecolor=ec, linewidth=lw, linestyle=ls, zorder=z))
    ax.text(x+w/2, y+h-.13, label,
            ha='center', va='top', fontsize=9, weight='bold',
            color=FC['txt'], zorder=z+1)
    for i, s in enumerate(subs):
        ax.text(x+w/2, y+h-.27-i*.16, s,
                ha='center', va='top', fontsize=7,
                color=FC['sub'], zorder=z+1)

def arrow(x1, y1, x2, y2, col=None, lw=1.5, ls='-', rad=0.,
          label=None, lo=(0, .12)):
    if col is None: col = FC['arrow']
    ax.annotate('', xy=(x2,y2), xytext=(x1,y1),
        arrowprops=dict(arrowstyle='->', color=col, lw=lw,
                        linestyle=ls,
                        connectionstyle=f'arc3,rad={rad}'), zorder=5)
    if label:
        mx=(x1+x2)/2+lo[0]; my=(y1+y2)/2+lo[1]
        ax.text(mx, my, label, ha='center', va='bottom',
                fontsize=7, color=col, weight='bold',
                bbox=dict(facecolor='white', edgecolor=col,
                          boxstyle='round,pad=.10', lw=.8, alpha=.93), zorder=6)

def hline(x1, x2, y, col=None, lw=1.3, ls='-'):
    if col is None: col = FC['arrow']
    ax.plot([x1, x2], [y, y], color=col, lw=lw, linestyle=ls, zorder=4)

def vline(x, y1, y2, col=None, lw=1.3, ls='-'):
    if col is None: col = FC['arrow']
    ax.plot([x, x], [y1, y2], color=col, lw=lw, linestyle=ls, zorder=4)

# ── Layout anchors ──
TITLE_Y = FH - 0.6
SEC_Y   = FH - 1.1
TOP     = FH - 1.4

# ===== SECTION 1: Input → CLG =====
SEC1_Y = 23.5

ax.text(FW/2, TITLE_Y,
        'CMIFES Module Architecture (CMIFES.py v7)',
        ha='center', va='top', fontsize=13, weight='bold', color='#1A1A1A')
ax.text(FW/2, TITLE_Y - 0.4,
        'Cross-scale Multi-level Information Fusion Enhancement — Simplified',
        ha='center', va='top', fontsize=9.5, color='#444444', style='italic')

ax.text(FW/2, SEC_Y,
        'Section 1: Multi-Input Fusion — CrossLayerGating (CLG)',
        ha='center', va='bottom', fontsize=11, weight='bold', color='#784212')

# ── Multi-input branches ──
# F1, F2, F3 branches
BX = 2.0
BY = SEC1_Y

# Input feature maps
arrow_x = 1.5
for i, (lbl, ch, sz) in enumerate([('F₁ (Backbone)', '256ch', 'H×W'),
                                      ('F₂ (Neck)',    '512ch', 'H/2×W/2'),
                                      ('F₃ (Extra)',   '256ch', 'H×W')]):
    iy = BY - 2.2 * i
    box(BX, iy-0.5, 2.5, 1.0, lbl,
        [f'{ch}, {sz}'],
        FC['input'], ec=FC['bdr'], lw=1.3)
    # Arrow down
    arrow(BX+1.25, iy-0.5, BX+1.25, iy-0.85, col=FC['arrow'])

# Align convs
AC_X = 5.5
for i in range(3):
    iy = BY - 2.2 * i - 1.0
    box(AC_X, iy-0.5, 2.5, 1.0, '1×1 Conv + BN',
        ['C→512, bias=False'],
        FC['align'], ec=FC['bdr'])
    arrow(BX+1.25+2.5, BY-2.2*i-0.5, AC_X, iy-0.05,
          col=FC['arrow'])

# Spatial interpolation
SI_X = 9.0
for i in range(3):
    iy = BY - 2.2 * i - 1.5
    box(SI_X, iy-0.5, 2.5, 1.0, 'F.interpolate',
        ['bilinear, align_corners=False'],
        FC['align'], ec=FC['dim'], lw=1.0, ls='--')
    arrow(AC_X+2.5, BY-2.2*i-1.0, SI_X, iy-0.05,
          col=FC['arrow'])

# Stacked concat
ST_X = 12.5
ST_Y = BY - 2.2 * 3 + 0.5
box(ST_X, ST_Y-1.0, 2.5, 2.8, 'Stacked Features',
    ['Concat([F1\',F2\',F3\'])', 'dim=1: Bx(3C)xHxW'],
    FC['concat'], ec=FC['dim'], lw=1.2, ls='--')
arrow(SI_X+1.25, BY-2.2*0-1.5, ST_X, ST_Y+1.5, col=FC['arrow'])
arrow(SI_X+1.25, BY-2.2*1-1.5, ST_X, ST_Y+0.5, col=FC['arrow'])
arrow(SI_X+1.25, BY-2.2*2-1.5, ST_X, ST_Y-0.5, col=FC['arrow'])

# GAP
GP_X = 12.5
GP_Y = ST_Y - 2.2
box(GP_X, GP_Y-0.6, 2.5, 0.6, 'GAP (1×1)',
    ['AdaptiveAvgPool2d(1)'],
    FC['gate'], ec=FC['bdr'], lw=1.0)
arrow(ST_X+1.25, ST_Y-1.0, GP_X+1.25, GP_Y+0.0, col=FC['arrow'])

# Flatten
FL_X = 12.5
FL_Y = GP_Y - 1.0
box(FL_X, FL_Y-0.6, 2.5, 0.6, 'Flatten',
    ['view(batch, -1)'],
    FC['gate'], ec=FC['dim'], lw=1.0, ls='--')
arrow(GP_X+1.25, GP_Y-0.6, FL_X+1.25, FL_Y+0.0, col=FC['arrow'])

# Gate FC layers
FC1_X = 12.5
FC1_Y = FL_Y - 1.2
box(FC1_X, FC1_Y-0.6, 2.5, 0.6, 'FC₁',
    ['Lin(3C → 12), SiLU'],
    FC['gate'], ec=FC['silu'], lw=1.8, ls='-')
arrow(FL_X+1.25, FL_Y-0.6, FC1_X+1.25, FC1_Y+0.0, col=FC['arrow'])

FC2_X = 12.5
FC2_Y = FC1_Y - 1.2
box(FC2_X, FC2_Y-0.6, 2.5, 0.6, 'FC₂',
    ['Lin(12 → 3), Softmax'],
    FC['gate'], ec=FC['softmax'], lw=1.8, ls='-')
arrow(FC1_X+1.25, FC1_Y-0.6, FC2_X+1.25, FC2_Y+0.0, col=FC['arrow'])

# Gate weights
GW_X = 16.5
GW_Y = FC2_Y - 0.5
box(GW_X, GW_Y-0.5, 2.5, 1.0, 'Gate Weights',
    ['w = softmax(FC₂(GAP))', '[w₁,w₂,w₃], Σw=1'],
    FC['gate'], ec=FC['softmax'], lw=1.8, ls='-')
arrow(FC2_X+2.5, FC2_Y-0.3, GW_X, GW_Y, col=FC['arrow'])

# Weighted sum
WS_X = 12.5
WS_Y = GW_Y - 1.5
box(WS_X, WS_Y-0.6, 2.5, 1.2, 'Weighted Sum Σ',
    ['w₁·F₁\' + w₂·F₂\' + w₃·F₃\'', '-> BxCxHxW'],
    FC['gate'], ec=FC['sum'], lw=2.0, ls='-')
arrow(FC2_X+1.25, FC2_Y-0.6, WS_X+1.25, WS_Y+0.3, col=FC['arrow'])
# w arrows back to features
for i in range(3):
    iy = WS_Y + 0.3 - 0.3
    pass

# Connect features to weighted sum
for i in range(3):
    iy_feat = BY - 2.2 * i - 1.5
    arrow(SI_X+1.25, iy_feat, WS_X-0.1, WS_Y+0.2,
          col=FC['arrow'], lw=1.0, ls='--')
    ax.text(WS_X-0.1, (iy_feat + WS_Y+0.2)/2,
            f'w{i+1}',
            ha='right', va='center', fontsize=7,
            color=FC['arrow'], style='italic')

# Projection
PR_X = 12.5
PR_Y = WS_Y - 1.8
box(PR_X, PR_Y-0.5, 2.5, 1.0, 'Proj Conv + BN + SiLU',
    ['Conv1×1, C→C, bias=False', 'BatchNorm2d, SiLU(inplace)'],
    FC['proj'], ec=FC['bdr'], lw=1.4)
arrow(WS_X+1.25, WS_Y-0.6, PR_X+1.25, PR_Y+0.5, col=FC['arrow'])

# Output of CLG
CLG_OUT_X = PR_X
CLG_OUT_Y = PR_Y - 1.8
box(CLG_OUT_X, CLG_OUT_Y-0.5, 2.5, 1.0, 'F_gated',
    ['BxCxHxW', 'CLGfused'],
    FC['output'], ec=FC['pe'], lw=2.0, ls='--')
arrow(PR_X+1.25, PR_Y-0.5, CLG_OUT_X+1.25, CLG_OUT_Y+0.5, col=FC['arrow'])

# ============================
# SECTION 2: SE Channel Attention
# ============================
SEC2_Y = CLG_OUT_Y - 2.0
ax.text(FW/2, SEC2_Y + 0.8,
        'Section 2: SE Channel Attention (Squeeze-and-Excitation)',
        ha='center', va='bottom', fontsize=11, weight='bold', color='#1D8348')

SE_Y = SEC2_Y

# GAP branch
GAP2_X = 2.0
box(GAP2_X, SE_Y-0.6, 2.5, 0.6, 'GAP',
    ['AdaptiveAvgPool2d(1)', '-> BxCx1x1'],
    FC['se'], ec=FC['bdr'], lw=1.4)

# GMP branch
GMP_X = 5.5
box(GMP_X, SE_Y-0.6, 2.5, 0.6, 'GMP',
    ['AdaptiveMaxPool2d(1)', '-> BxCx1x1'],
    FC['se'], ec=FC['bdr'], lw=1.4)

# Arrows from input
arrow(CLG_OUT_X+1.25, CLG_OUT_Y-0.5, GAP2_X+1.25, SE_Y+0.0, col=FC['arrow'])
arrow(CLG_OUT_X+1.25, CLG_OUT_Y-0.5, GMP_X+1.25, SE_Y+0.0, col=FC['arrow'], lw=1.0)

# Input label
box(CLG_OUT_X, CLG_OUT_Y-0.5, 2.5, 1.0, 'F_gated',
    ['BxCxHxW'],
    FC['output'], ec=FC['pe'], lw=2.0, ls='--')

# FC1 for each branch
FC1_SE_X = GAP2_X + 3.5
box(FC1_SE_X, SE_Y-0.6, 2.5, 0.6, 'FC₁',
    ['Lin(C → C/r), SiLU', 'r=16, min=8'],
    FC['se'], ec=FC['silu'], lw=1.8, ls='-')
box(FC1_SE_X, SE_Y-2.2, 2.5, 0.6, 'FC₁',
    ['Lin(C → C/r), SiLU'],
    FC['se'], ec=FC['silu'], lw=1.8, ls='-')
arrow(GAP2_X+2.5, SE_Y-0.3, FC1_SE_X, SE_Y-0.3, col=FC['arrow'])
arrow(GMP_X+2.5, SE_Y-0.3, FC1_SE_X, SE_Y-2.0, col=FC['arrow'])

# FC2
FC2_SE_X = FC1_SE_X + 3.5
box(FC2_SE_X, SE_Y-1.4, 2.5, 0.6, 'FC₂',
    ['Lin(C/r → C)'],
    FC['se'], ec=FC['bdr'], lw=1.4)
# Two inputs merge
vline(GAP2_X+2.5, SE_Y-0.3, SE_Y-1.4, col=FC['arrow'])
vline(GMP_X+2.5, SE_Y-0.3, SE_Y-1.4, col=FC['arrow'])
hline(GAP2_X+2.5, FC1_SE_X, SE_Y-0.3, col=FC['arrow'])
hline(GMP_X+2.5, FC1_SE_X, SE_Y-0.3, col=FC['arrow'])
hline(GAP2_X+2.5, FC1_SE_X, SE_Y-1.4, col=FC['arrow'])
hline(GMP_X+2.5, FC1_SE_X, SE_Y-1.4, col=FC['arrow'])
arrow(FC1_SE_X, SE_Y-0.6, FC2_SE_X, SE_Y-1.1, col=FC['arrow'])
arrow(FC1_SE_X, SE_Y-2.2, FC2_SE_X, SE_Y-1.1, col=FC['arrow'])

# Sigmoid
SIG_X = FC2_SE_X + 3.5
box(SIG_X, SE_Y-1.4, 2.0, 0.6, 'σ (Sigmoid)',
    ['-> Mc, BxCx1x1'],
    FC['se'], ec=FC['sig'], lw=1.8, ls='-')
arrow(FC2_SE_X+2.5, SE_Y-1.1, SIG_X, SE_Y-1.1, col=FC['arrow'])

# Element-wise multiplication
MUL_SE_X = SIG_X + 2.8
box(MUL_SE_X, SE_Y-0.9, 2.0, 1.8, '⊙ (Mul)',
    ['M_c ⊙ F_gated', '-> F_ca'],
    FC['mul'], ec=FC['bdr'], lw=1.4)
arrow(SIG_X+1.0, SE_Y-1.4, MUL_SE_X+1.0, SE_Y-0.2, col=FC['arrow'])
# F_gated connects too
arrow(CLG_OUT_X+1.25, CLG_OUT_Y, MUL_SE_X+0.2, SE_Y-0.0,
      col=FC['arrow'], lw=1.0, ls='--')

# ============================
# SECTION 3: Simple Spatial Attention
# ============================
SSA_Y = SE_Y - 4.0
ax.text(FW/2, SSA_Y + 3.8,
        'Section 3: Simple Spatial Attention (Lightweight)',
        ha='center', va='bottom', fontsize=11, weight='bold', color='#6C3483')

# Input label
box(1.5, SSA_Y+1.0, 2.5, 1.0, 'F_ca',
    ['BxCxHxW'],
    FC['output'], ec=FC['pe'], lw=2.0, ls='--')

# Conv1
CV1_X = 5.0
box(CV1_X, SSA_Y+0.5, 2.5, 1.0, 'Conv 3×3',
    ['C → C/4, bias=False'],
    FC['spatial'], ec=FC['bdr'], lw=1.4)
arrow(1.5+2.5, SSA_Y+1.5, CV1_X, SSA_Y+1.1, col=FC['arrow'])

# BN
BN_X = CV1_X + 3.0
box(BN_X, SSA_Y+0.5, 2.5, 1.0, 'BN',
    ['BatchNorm2d(C/4)'],
    FC['spatial'], ec=FC['bdr'], lw=1.4)
arrow(CV1_X+2.5, SSA_Y+1.0, BN_X, SSA_Y+1.1, col=FC['arrow'])

# SiLU
SL_X = BN_X + 3.0
box(SL_X, SSA_Y+0.5, 2.5, 1.0, 'SiLU',
    ['SiLU (inplace)'],
    FC['spatial'], ec=FC['silu'], lw=1.8, ls='-')
arrow(BN_X+2.5, SSA_Y+1.0, SL_X, SSA_Y+1.1, col=FC['arrow'])

# Conv2 (1×1)
CV2_X = SL_X + 3.0
box(CV2_X, SSA_Y+0.5, 2.5, 1.0, 'Conv 1×1',
    ['C/4 → 1'],
    FC['spatial'], ec=FC['bdr'], lw=1.4)
arrow(SL_X+2.5, SSA_Y+1.0, CV2_X, SSA_Y+1.1, col=FC['arrow'])

# Residual +1.0
RS_X = CV2_X + 3.0
box(RS_X, SSA_Y+0.5, 2.5, 1.0, '+ 1.0 (Residual)',
    ['identity init', 'σ(output + 1.0)'],
    FC['spatial'], ec=FC['sum'], lw=1.6, ls='-')
# Connect conv output
arrow(CV2_X+1.25, SSA_Y+0.5, CV2_X+1.25, SSA_Y+0.2, col=FC['dim'], lw=1.0, ls='--')
# F_ca connects for residual
arrow(1.5+2.5, SSA_Y+1.0, RS_X+2.5, SSA_Y+0.8,
      col=FC['arrow'], lw=1.0, ls='--')

# Sigmoid
SG_X = RS_X + 3.0
box(SG_X, SSA_Y+0.5, 2.5, 1.0, 'σ (Sigmoid)',
    ['-> Ms, Bx1xHxW'],
    FC['spatial'], ec=FC['sig'], lw=1.8, ls='-')
arrow(RS_X+2.5, SSA_Y+1.0, SG_X, SSA_Y+1.1, col=FC['arrow'])

# Final multiplication
FM_X = SG_X + 3.2
box(FM_X, SSA_Y+0.3, 2.5, 1.4, '⊙ (Mul)',
    ['M_s ⊙ F_ca', '-> F_out'],
    FC['mul'], ec=FC['bdr'], lw=1.4)
arrow(SG_X+1.25, SSA_Y+0.5, FM_X+1.25, SSA_Y+0.9, col=FC['arrow'])
arrow(1.5+2.5, SSA_Y+0.8, FM_X+0.3, SSA_Y+0.6,
      col=FC['arrow'], lw=1.0, ls='--')

# ============================
# FINAL OUTPUT
# ============================
OUT_Y = SSA_Y - 1.5
box(OUT_Y+5.0, OUT_Y-0.5, 3.0, 1.0, 'CMIFES Output',
    ['F_out = M_s ⊙ (M_c ⊙ F_gated)', 'BxCxHxW'],
    FC['output'], ec='#1A5276', lw=2.5, ls='-')
arrow(FM_X+1.25, SSA_Y+0.3, OUT_Y+5.0+1.5, OUT_Y+0.0, col=FC['arrow'])

# ============================
# PARAM COMPARISON
# ============================
COMP_X = 0.5
COMP_Y = OUT_Y - 2.5
box(COMP_X, COMP_Y-1.0, 5.0, 2.0, 'Parameter Comparison',
    ['Deformable SA (CMIFE v6):', '  ~400K params at 256ch (DCNv2 offset)',
     'SE Attention (CMIFES v7):', '  ~4K params at 256ch (90× reduction)',
     'Simple Spatial Attention:', '  ~6K params at 256ch',
     'Total CMIFES v7 overhead:', '  ~14K params (vs 400K)'],
    '#F2F3F4', ec=FC['bdr'], lw=1.2)

# ============================
# FORMULA BOX
# ============================
FORM_X = 6.0
FORM_Y = COMP_Y - 0.5
box(FORM_X, FORM_Y-0.5, 13.5, 1.8, 'Forward Pass Equations',
    ['F_gated = Proj(Σᵢ wᵢ·Fᵢ\')   where wᵢ = softmax(FC₂(SiLU(FC₁(GAP(F)))))',
     'F_ca = σ(W₂·δ(W₁·(F_gap+F_gmp)))) ⊙ F_gated         (SE Channel Attention)',
     'F_out = σ(Conv₁ₓ₁(BN(δ(Conv₃ₓ₃(F_ca)))) + 1.0) ⊙ F_ca  (Simple Spatial Attention)'],
    '#FDFEFE', ec=FC['pe'], lw=2.0, ls='--', z=3)
ax.text(FORM_X, FORM_Y+1.0, 'Forward Pass Equations',
        ha='center', va='top', fontsize=9, weight='bold', color='#1A1A1A')

# ============================
# LEGEND
# ============================
LG_X = 0.5; LG_Y = 1.2
ax.add_patch(FancyBboxPatch((LG_X, LG_Y-0.2), 9.5, 1.2,
             boxstyle="round,pad=.10", facecolor='#F8F9F9',
             edgecolor='#B0B0B0', lw=1.0, zorder=2))
ax.text(LG_X+4.75, LG_Y+0.85, 'Legend',
        ha='center', va='bottom', fontsize=8, weight='bold', color='#1A1A1A')
items = [
    ('Feature Map', FC['input'], FC['bdr'], '-'),
    ('CLG Gate', FC['gate'], FC['bdr'], '-'),
    ('SE / Spatial Attn', FC['se'], FC['bdr'], '-'),
    ('Conv / Linear', FC['align'], FC['bdr'], '-'),
    ('Element-wise Op', FC['mul'], FC['bdr'], '-'),
    ('Output', FC['output'], FC['pe'], '--'),
]
for i, (lbl, fc, ec, ls) in enumerate(items):
    lx = LG_X + (i % 3) * 3.2
    ly = LG_Y + 0.45 if i < 3 else LG_Y + 0.05
    ax.add_patch(FancyBboxPatch((lx, ly-0.15), 0.4, 0.30,
                 boxstyle="round,pad=.03", facecolor=fc,
                 edgecolor=ec, lw=1.2, linestyle=ls, zorder=4))
    ax.text(lx+0.5, ly, lbl, ha='left', va='center',
            fontsize=6.5, color=FC['txt'], zorder=4)

# Save
OUT = r'C:/Users/Administrator/Desktop/drone_paper/paper_v7/figures0331/fig_cmifes_module.png'
plt.tight_layout(pad=0.3)
plt.savefig(OUT, dpi=180, bbox_inches='tight', facecolor='white')
plt.close()
print(f'Saved → {OUT}')
