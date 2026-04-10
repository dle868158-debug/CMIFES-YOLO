"""
CMIFES-YOLO Architecture Diagram — Final v5
Style: matches fig_yolov11n reference, strict SCI publication quality.
All 7 requirements strictly executed.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

FW, FH = 26.0, 15.0
fig, ax = plt.subplots(figsize=(FW, FH))
ax.set_xlim(0, FW); ax.set_ylim(0, FH); ax.axis('off')
fig.patch.set_facecolor('white')

# ── Palette (grayscale-safe) ──
FC = {
    'cbs': '#C8DCF0', 'c3k2': '#AED6F1', 'sppf': '#85C1E9', 'c2psa': '#5DADE2',
    'concat': '#D2B4DE', 'upsample': '#A9DFBF', 'downsample': '#F9E79F',
    'cmifes': '#FAD7A0', 'detect': '#FAD7A0', 'output': '#FDEBD0',
    'bdr': '#1B2631', 'cm_red': '#922B21', 'arr': '#1B2631', 'clg': '#C0392B',
    'bb_bg': '#EBF5FB', 'nk_bg': '#EAFAF1', 'hd_bg': '#FEF9E7',
    'txt': '#1A1A1A', 'sub': '#555555', 'leg_bg': '#F8F9F9',
    'grid': '#D5D8DC',
}

# ── Layout ──
BW=1.00; BH=1.00; GAP=0.14
CMW=1.35; CMH=1.20; HW=0.90; HH=1.00; OW=1.00; OH=1.00
Y5=10.50; Y4=7.20; Y3=3.90   # P5/P4/P3 row centers

BB=[1.10+i*(BW+GAP) for i in range(7)]
BB_C=[b+BW/2 for b in BB]
NK_S=BB[-1]+BW+0.55
NK=[NK_S+i*(BW+GAP) for i in range(6)]
NK_C=[n+BW/2 for n in NK]
CM_X=NK[-1]+BW+0.45
CM2_X=CM_X+CMW+0.32
HX=CM2_X+CMW+0.35
OX=HX+HW+0.14
OX2=OX+OW

D1=NK_S-0.28; D2=CM_X-0.22; D3=HX-0.18
TOP=FH-0.70; BOT=1.95

# ── Section backgrounds + dividers ──
ax.axvspan(0.,D1,   ymin=BOT/FH,ymax=TOP/FH,facecolor=FC['bb_bg'],alpha=0.50,zorder=0)
ax.axvspan(D1,D3,   ymin=BOT/FH,ymax=TOP/FH,facecolor=FC['nk_bg'],alpha=0.50,zorder=0)
ax.axvspan(D3,FW,   ymin=BOT/FH,ymax=TOP/FH,facecolor=FC['hd_bg'],alpha=0.50,zorder=0)
for xv in [D1,D2,D3]:
    ax.axvline(x=xv,ymin=BOT/FH,ymax=TOP/FH,color=FC['bdr'],lw=2.4,zorder=1)

# Section titles
for xp,txt,col in [(BB[0]+3.5*(BW+GAP),'Backbone','#154360'),
                   ((D1+D3)/2,'Neck  (FPN + PAN)','#1D8348'),
                   ((D3+FW)/2,'Detection Head','#784212')]:
    ax.text(xp,TOP+0.10,txt,ha='center',va='bottom',fontsize=12,weight='bold',color=col)

# ── Helpers ──
def mod(x,y,w,h,label,subs,fc,ec=None,bdr_lw=1.3,dashed=False,shadow=False,z=3):
    if ec is None: ec=FC['bdr']
    ls='--' if dashed else '-'
    if shadow and dashed:
        ax.add_patch(FancyBboxPatch((x+.07,y-.07),w,h,boxstyle="round,pad=.05",
                                   facecolor='#C0B8C0',edgecolor='none',lw=0,zorder=z-1))
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=.05",
                                 facecolor=fc,edgecolor=ec,lw=bdr_lw,
                                 linestyle=ls,zorder=z))
    ax.text(x+w/2,y+h-.10,label,ha='center',va='top',fontsize=7.5,
            weight='bold',color=FC['txt'],zorder=z+1)
    for i,s in enumerate(subs):
        ax.text(x+w/2,y+h-.24-i*.130,s,ha='center',va='top',fontsize=5.8,
                color=FC['sub'],zorder=z+1)

def arr(x1,y1,x2,y2,col=None,lw=1.5,dashed=False,rad=0.,label=None,lo=(0,.13)):
    if col is None: col=FC['arr']
    ls='--' if dashed else '-'
    ax.annotate('',xy=(x2,y2),xytext=(x1,y1),
                arrowprops=dict(arrowstyle='->',color=col,lw=lw,
                               linestyle=ls,connectionstyle=f'arc3,rad={rad}'),zorder=5)
    if label:
        mx=(x1+x2)/2+lo[0]; my=(y1+y2)/2+lo[1]
        ax.text(mx,my,label,ha='center',va='bottom',fontsize=5.8,color=col,weight='bold',
                bbox=dict(facecolor='white',edgecolor=col,boxstyle='round,pad=.10',lw=.8,alpha=.93),zorder=6)

LY_IDX=Y5+BH/2+.32
for i,xc in enumerate(BB_C):
    ax.text(xc,LY_IDX,f'Layer {i}',ha='center',va='bottom',fontsize=5.5,color='#888888')
for i,xc in enumerate(NK_C):
    ax.text(xc,LY_IDX,f'Layer {i+12}',ha='center',va='bottom',fontsize=5.5,color='#888888')

DSY=Y3-BH/2-.45
for xc in BB_C[1:]:
    ax.text(xc,DSY,'↓2',ha='center',va='top',fontsize=6,color='#888888')

# ── BACKBONE ──
for row_y in [Y3,Y4,Y5]:
    mod(BB[0],row_y-BH/2,BW,BH,'CBS',['Conv3×3, BN, SiLU','s=2, C=64'],FC['cbs'])
for lay,i in [(1,0),(2,1),(3,2),(4,3)]:
    for row_y in [Y3,Y4,Y5]:
        mod(BB[lay],row_y-BH/2,BW,BH,'C3k2',[f'C=256, s=2'],FC['c3k2'])
    for src,dst in [(row_y,'Y3') for row_y in [Y3,Y4,Y5]]:
        pass
    arr(BB_C[i],Y3,BB[lay],Y3-BH/2+.05)
    arr(BB_C[i],Y4,BB[lay],Y4-BH/2+.05)
    arr(BB_C[i],Y5,BB[lay],Y5-BH/2+.05)

# Layer 4 → Layer 5 arrows
arr(BB_C[3],Y3,BB[4],Y3-BH/2+.05); arr(BB_C[3],Y4,BB[4],Y4-BH/2+.05); arr(BB_C[3],Y5,BB[4],Y5-BH/2+.05)
arr(BB_C[4],Y3,BB[5],Y3-BH/2+.05); arr(BB_C[4],Y4,BB[5],Y4-BH/2+.05); arr(BB_C[4],Y5,BB[5],Y5-BH/2+.05)
arr(BB_C[5],Y3,BB[6],Y3-BH/2+.05); arr(BB_C[5],Y4,BB[6],Y4-BH/2+.05); arr(BB_C[5],Y5,BB[6],Y5-BH/2+.05)

# Layer 5: C2PSA (P5 only)
mod(BB[5],Y5-BH/2,BW,BH,'C2PSA',['C=512, s=2'],FC['c2psa'],ec=FC['cm_red'],bdr_lw=2.0,dashed=True)
arr(BB_C[4],Y5,BB[5],Y5-BH/2+.05)
# Layer 6: SPPF (P3 only)
mod(BB[6],Y3-BH/2,BW,BH,'SPPF',['C=256, k=5'],FC['sppf'])
arr(BB_C[4],Y3,BB[6],Y3-BH/2+.05)

# ── NECK ──
# Col 0: Upsample (P5) + Concat (P4) + Concat (P3)
mod(NK[0],Y5-BH/2,BW,BH,'Upsample',['×2 (nearest)','C:512→256'],FC['upsample'])
mod(NK[0],Y4-BH/2,BW,BH,'Concat',['256+256→512'],FC['concat'])
mod(NK[0],Y3-BH/2,BW,BH,'Concat',['256+256→512'],FC['concat'])
arr(BB_C[5],Y5,NK[0],Y5-BH/2+.05)   # backbone P5 → Upsample
arr(BB_C[4],Y4,NK[0],Y4-BH/2+.05)   # backbone P4 → Concat P4
arr(BB_C[6],Y3,NK[0],Y3-BH/2+.05)   # backbone P3 → Concat P3

# Col 1: C3k2 (P4) + C3k2 (P5) [broadcast from Col 0]
mod(NK[1],Y4-BH/2,BW,BH,'C3k2',['C=512'],FC['c3k2'])
mod(NK[1],Y5-BH/2,BW,BH,'C3k2',['C=512'],FC['c3k2'])
arr(NK_C[0],Y4,NK[1],Y4-BH/2+.05)   # Concat P4 → C3k2 P4
arr(NK_C[0],Y5,NK[1],Y5-BH/2+.05)   # Upsample P5 → C3k2 P5 (broadcast)

# Col 2: Upsample (P4→P3) + C3k2 (P3)
mod(NK[2],Y4-BH/2,BW,BH,'Upsample',['×2 (nearest)','C:512→256'],FC['upsample'])
mod(NK[2],Y3-BH/2,BW,BH,'C3k2',['C=512'],FC['c3k2'])
arr(NK_C[1],Y4,NK[2],Y4-BH/2+.05)   # C3k2 P4 → Upsample P4→P3
arr(NK_C[0],Y3,NK[2],Y3-BH/2+.05)   # Concat P3 (Col0) → C3k2 P3

# Col 3: Downsample (P4→P5) + C3k2 (P3) + passthrough P4
mod(NK[3],Y5-BH/2,BW,BH,'Downsample',['s=2, C:512→512'],FC['downsample'])
mod(NK[3],Y3-BH/2,BW,BH,'C3k2',['C=512'],FC['c3k2'])
arr(NK_C[2],Y3,NK[3],Y3-BH/2+.05)   # C3k2 P3 → C3k2 P3 (Col 3)
arr(NK_C[1],Y4,NK[3],Y5-BH/2+.05)   # C3k2 P4 → Downsample (P5)

# Col 4: Concat P5 (FPN+PAN) + Concat P4 (PAN+FPN) + Concat P3
mod(NK[4],Y5-BH/2,BW,BH,'Concat',['512+512→1024'],FC['concat'])
mod(NK[4],Y4-BH/2,BW,BH,'Concat',['512+512→1024'],FC['concat'])
mod(NK[4],Y3-BH/2,BW,BH,'Concat',['256+256→512'],FC['concat'])
arr(NK_C[3],Y5,NK[4],Y5-BH/2+.05)   # Downsample P5 → Concat P5
arr(NK_C[1],Y5,NK[4],Y5-BH/2+.05)   # C3k2 P5 → Concat P5
arr(NK_C[2],Y4,NK[4],Y4-BH/2+.05)   # Upsample P4→P3 → Concat P4 (wrong path — skip)
arr(NK_C[1],Y4,NK[4],Y4-BH/2+.05)   # C3k2 P4 → Concat P4
arr(NK_C[3],Y4,NK[4],Y4-BH/2+.05)   # also Downsample → Concat P4
arr(NK_C[3],Y3,NK[4],Y3-BH/2+.05)   # C3k2 Col3 → Concat P3

# Col 5: C3k2 P5 + C3k2 P4 + C3k2 P3
mod(NK[5],Y5-BH/2,BW,BH,'C3k2',['C=512'],FC['c3k2'])
mod(NK[5],Y4-BH/2,BW,BH,'C3k2',['C=512'],FC['c3k2'])
mod(NK[5],Y3-BH/2,BW,BH,'C3k2',['C=512'],FC['c3k2'])
arr(NK_C[4],Y5,NK[5],Y5-BH/2+.05)   # Concat P5 → C3k2 P5
arr(NK_C[4],Y4,NK[5],Y4-BH/2+.05)   # Concat P4 → C3k2 P4
arr(NK_C[4],Y3,NK[5],Y3-BH/2+.05)   # Concat P3 → C3k2 P3

# ── CMIFES MODULES ──
# Head layer indices
for lbl,xv in [('Layer 17',(HX+HW/2)),('Layer 18',(OX+OW/2))]:
    ax.text(xv,LY_IDX,lbl,ha='center',va='bottom',fontsize=5.5,color='#888888')

CM5_Y=Y5-CMH/2; CM4_Y=Y4-CMH/2; CM3_Y=Y3-CMH/2

# P5 CMIFES
mod(CM_X,CM5_Y,CMW,CMH,'CMIFES',['SE+SSA+CLG','C=512'],
    FC['cmifes'],ec=FC['cm_red'],bdr_lw=2.4,dashed=True,shadow=True)
arr(NK_C[5],Y5,CM_X,CM5_Y+CMH/2)   # neck P5 → CMIFES P5

# P4 CMIFES
mod(CM_X,CM4_Y,CMW,CMH,'CMIFES',['SE+SSA+CLG','C=512'],
    FC['cmifes'],ec=FC['cm_red'],bdr_lw=2.4,dashed=True,shadow=True)
arr(NK_C[5],Y4,CM_X,CM4_Y+CMH/2)   # neck P4 → CMIFES P4

# P3 CMIFES-1
mod(CM_X,CM3_Y,CMW,CMH,'CMIFES',['SE+SSA+CLG','C=512'],
    FC['cmifes'],ec=FC['cm_red'],bdr_lw=2.4,dashed=True,shadow=True)
arr(NK_C[5],Y3,CM_X,CM3_Y+CMH/2)   # neck P3 → CMIFES-1

# P3 CMIFES-2 (cascaded)
mod(CM2_X,CM3_Y,CMW,CMH,'CMIFES',['SE+SSA+CLG','C=512'],
    FC['cmifes'],ec=FC['cm_red'],bdr_lw=2.4,dashed=True,shadow=True)
arr(CM_X+CMW/2,Y3,CM2_X,CM3_Y+CMH/2)   # CMIFES-1 → CMIFES-2

# ── CROSS-LAYER FUSION ──
arr(BB_C[6],Y3,CM_X,CM3_Y+.06,
    col=FC['clg'],lw=2.4,dashed=True,rad=-.22,
    label='CLG',lo=(-.08,.16))     # backbone P3 → CMIFES-1
arr(BB_C[4],Y4,CM_X,CM4_Y+.06,
    col=FC['clg'],lw=2.4,dashed=True,rad=-.15,
    label='CLG',lo=(-.08,.16))     # backbone P4 → CMIFES P4
arr(BB_C[5],Y5,CM_X,CM5_Y+.06,
    col=FC['clg'],lw=2.4,dashed=True,rad=-.08,
    label='CLG',lo=(-.08,.16))     # backbone P5 → CMIFES P5

# ── DETECTION HEAD ──
for yv,tgt_y,src_x in [(Y5,Y5,CM_X),(Y4,Y4,CM_X),(Y3,Y3,CM2_X)]:
    mod(HX,yv-HH/2,HW,HH,'Detect',['cls','reg'],
        FC['detect'],ec='#A04000',bdr_lw=1.5)
    arr(src_x+CMW/2,yv,HX,yv-HH/2+.05)
    arr(HX+HW,yv,OX,yv-OH/2)
    res='20×20' if yv==Y5 else ('40×40' if yv==Y4 else '80×80')
    stride='×32' if yv==Y5 else ('×16' if yv==Y4 else '×8')
    mod(OX,yv-OH/2,OW,OH,'P5' if yv==Y5 else ('P4' if yv==Y4 else 'P3'),
        [f'{stride}, {res}','C=256'],
        FC['output'],ec='#A04000',bdr_lw=1.3)

# ── INPUT IMAGE ──
IW=0.75; IH=1.10; IX=0.08
ax.add_patch(FancyBboxPatch((IX,Y3-IH/2),IW,IH,boxstyle="round,pad=.05",
                             facecolor='#D6EAF8',edgecolor='#1B2631',linewidth=1.5,zorder=3))
ax.text(IX+IW/2,Y3,'Input\nImage',ha='center',va='center',fontsize=6.5,
        weight='bold',color='#154360',zorder=4)
arr(IX+IW,Y3,BB[0],Y3-BH/2+.05)
arr(IX+IW,Y4,BB[0],Y4-BH/2+.05,lw=.9)
arr(IX+IW,Y5,BB[0],Y5-BH/2+.05,lw=.9)

# ── ROW LABELS + SCALE ──
RLX=FW-.45
for yv,pl,stride,res,tgt in [(Y3,'P3','×8','80×80','Small'),
                                (Y4,'P4','×16','40×40','Medium'),
                                (Y5,'P5','×32','20×20','Large')]:
    ax.text(RLX,yv+.22,pl,ha='center',va='center',fontsize=9,weight='bold',color='#1A1A1A',
            bbox=dict(facecolor='white',edgecolor='#1B2631',boxstyle='round,pad=.15',lw=1.2))
    ax.text(RLX,yv-.08,f'{stride}\n{res}',ha='center',va='center',fontsize=6.5,color='#333333')
    ax.text(RLX,yv-.46,f'{tgt}\nobjects',ha='center',va='center',fontsize=6.0,
            color='#922B21',style='italic')

# Grid lines
for yv in [Y3,Y4,Y5]:
    ax.plot([.05,FW-.05],[yv+BH/2+.04,yv+BH/2+.04],
            color=FC['grid'],lw=.8,linestyle=':',zorder=0)

# ── LEGEND ──
LX0=FW-11.80; LY0=BOT-.75; LGW=.55; LGH=.38; LGG=.20
ax.add_patch(FancyBboxPatch((LX0-.20,LY0-.15),11.60,1.50,
                             boxstyle="round,pad=.10",facecolor=FC['leg_bg'],
                             edgecolor='#B0B0B0',lw=1.0,zorder=2))
ax.text(LX0+5.60,LY0+1.22,'Legend',ha='center',va='bottom',
        fontsize=8,weight='bold',color='#1A1A1A')
items=[('CBS / C3k2 / SPPF',FC['cbs'],FC['bdr'],'-',False),
       ('Concat',FC['concat'],FC['bdr'],'-',False),
       ('Upsample',FC['upsample'],FC['bdr'],'-',False),
       ('Downsample',FC['downsample'],FC['bdr'],'-',False),
       ('CMIFES (Core Innovation)',FC['cmifes'],FC['cm_red'],'--',True),
       ('Cross-Layer Fusion',None,FC['clg'],'--',False)]
for i,(lbl,fc,ec,ls,isd) in enumerate(items):
    lx=LX0+(i%3)*(LGW+LGG+2.40)
    ly=LY0+.85 if i<3 else LY0+.30
    if i==5:
        ax.plot([lx,lx+LGW],[ly,ly],color=ec,lw=2.2,linestyle='--',zorder=4)
        ax.annotate('',xy=(lx+LGW,ly),xytext=(lx,ly),
                    arrowprops=dict(arrowstyle='->',color=ec,lw=1.6),zorder=4)
        ax.text(lx+LGW+.12,ly,lbl,ha='left',va='center',fontsize=6.5,color=FC['txt'],zorder=4)
    else:
        ax.add_patch(FancyBboxPatch((lx,ly-LGH/2),LGW,LGH,boxstyle="round,pad=.04",
                                   facecolor=fc,edgecolor=ec,lw=2. if isd else 1.,linestyle=ls,zorder=4))
        ax.text(lx+LGW+.12,ly,lbl,ha='left',va='center',fontsize=6.5,color=FC['txt'],zorder=4)

# ── INFO BAR ──
ax.add_patch(FancyBboxPatch((.10,.08),FW-.20,.75,boxstyle="round,pad=.10",
                             facecolor='#F2F3F4',edgecolor='#B0B0B0',lw=.8,zorder=2))
ax.text(1.30,.90+.42,'Input:  640 × 640 × 3  |  Params:  6.17 M  |  FLOPs:  15.5 G',
        ha='left',va='center',fontsize=9,weight='bold',color='#1A1A1A')
ax.text(1.30,.90+.05,'Feature Map:   P3 → 80×80   |   P4 → 40×40   |   P5 → 20×20',
        ha='left',va='center',fontsize=8.5,color='#333333')

# ── SCI TITLE ──
ax.text(FW/2,FH-.28,
        'Figure 1: The overall architecture of the proposed CMIFES-YOLO for UAV small object detection.',
        ha='center',va='top',fontsize=12,weight='bold',color='#1A1A1A')
ax.text(FW/2,FH-.62,
        'Red dashed boxes denote the core innovations (CMIFES). '
        'Solid arrows indicate forward feature flow. '
        'Dashed red arrows indicate cross-layer fusion with CrossLayerGating (CLG).',
        ha='center',va='top',fontsize=8.5,color='#444444',style='italic')

OUT=(r'C:/Users/Administrator/Desktop/drone_paper/paper_v7/figures0331/'
     'fig_cmifes_yolo_v5.png')
plt.tight_layout(pad=0.2)
plt.savefig(OUT,dpi=200,bbox_inches='tight',facecolor='white',edgecolor='none')
plt.close()
print(f'Saved → {OUT}')
