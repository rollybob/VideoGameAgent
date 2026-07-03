"""Dedicated wild-battle capture: title -> overworld (NO menu) -> walk grass until an
encounter -> spam A to push through intro/FIGHT/move/damage dialogue. Captures every
step. The tour's battle failed only because a menu was open during the 'walk'; here we
go straight to the overworld and move the PLAYER. Frames -> captures/battle/."""
from __future__ import annotations
import os, sys, time
import cv2
sys.path.insert(0, os.path.expanduser("~/projects/VGA"))
from vga.core.contract import Action, Button
from vga.core.emulator import GbaEmulator
OUT=os.path.expanduser("~/projects/VGA/captures/battle"); NAMES={b.name:b for b in Button}; _n=[0]
def shot(emu,tag): cv2.imwrite(os.path.join(OUT,f"{_n[0]:03d}_{tag}.png"),emu.capture()); _n[0]+=1
def press(emu,k,hold=0.10,settle=0.30):
    for t in k.split(): emu.send(Action.press(NAMES[t],duration=hold)); time.sleep(settle)
def step(emu,k,tag,pre=0.35):
    press(emu,k); time.sleep(pre); shot(emu,tag)
def main():
    os.makedirs(OUT,exist_ok=True); emu=GbaEmulator(); time.sleep(1)
    # title -> CONTINUE -> overworld (no menus)
    step(emu,"START","title",pre=1.2)
    step(emu,"A","recap",pre=1.5)
    for i in range(7): step(emu,"A",f"recap{i}",pre=1.1)
    step(emu,"","overworld",pre=1.0)
    # walk grass: cycle all 4 directions so the PLAYER actually moves onto grass tiles
    cyc=["DOWN","UP","LEFT","RIGHT"]
    for i in range(72):
        step(emu,cyc[i%4],f"walk{i}",pre=0.18)
    # push through whatever battle/dialogue triggered
    for i in range(30):
        step(emu,"A",f"fight{i}",pre=0.5)
    print(f"[battle] captured {_n[0]} frames -> {OUT}",flush=True)
if __name__=="__main__": main()
