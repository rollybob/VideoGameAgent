"""Filter harvested uniq crops to TEXT-LIKE ones, dropping EAST false-fires on game
textures/sprites. Heuristic (calibrated on realcap 2026-06-29): real text line-crops are
high-contrast + DESATURATED (white/light glyphs on dark dialogue boxes), with a sane ink
fraction; grass/sprite strips are colorful (high HSV saturation) and/or tiny fragments.
Writes <dir>/clean_manifest.json (subset of uniq_manifest) + prints kept/dropped.

PER-GAME sat_max override (2026-06-30): the default sat>62 "colorful" gate assumes
white-on-dark text (Zelda/Pokemon/FRLG). FFTA -- and to a lesser extent Baldur's Gate --
render COLORED text on COLORED menu/parchment backgrounds, so real dialogue/menu text trips
the saturation gate and gets dropped (audited: FFTA lost ~900 clean text crops, the sat[62,90)
band is ~95% real text). Saturation does NOT separate text from texture in those games, so we
raise the ceiling per game and rely on the VLM labeler to skip residual non-text crops
(empty label = reviewed-skip). Pass "game:SATMAX", e.g. "ffta0630:120".
Usage:  filter_textlike.py pkmn0630 ffta0630:120 baldur0630:100"""
import cv2, numpy as np, glob, os, json, sys

def feats(im):
    hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
    s_mean=float(hsv[...,1].mean())
    g=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    _,th=cv2.threshold(g,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    ink=float((th<128).mean()); ink=min(ink,1-ink)  # polarity-agnostic minority frac
    contrast=float(g.std())
    h,w=g.shape[:2]
    return s_mean,ink,contrast,w,h

def textlike(im, sat_max=62):
    s,ink,c,w,h=feats(im)
    if w<26 or h<7: return False,"tiny"
    if w/float(h)<2.2: return False,"aspect"
    if s>sat_max: return False,"colorful"      # textured/sprite (per-game ceiling)
    if c<32: return False,"flat"               # blank/low-contrast
    if ink<0.02 or ink>0.55: return False,"ink"
    return True,"ok"

for tok in sys.argv[1:]:
    game,_,sm=tok.partition(":")
    sat_max=int(sm) if sm else 62
    d=f"train/ocr/data/realcap_{game}"
    man=json.load(open(f"{d}/uniq_manifest.json"))
    keep=[]; drop=0; reasons={}
    for e in man:
        im=cv2.imread(f"{d}/uniq_crops/{e['id']}.png")
        if im is None: drop+=1; continue
        ok,why=textlike(im,sat_max=sat_max)
        if ok: keep.append(e)
        else: drop+=1; reasons[why]=reasons.get(why,0)+1
    json.dump(keep,open(f"{d}/clean_manifest.json","w"),indent=2)
    print(f"{game:12s} (sat_max={sat_max}) kept {len(keep):4d} / {len(man):4d}  dropped {drop}  {reasons}")
