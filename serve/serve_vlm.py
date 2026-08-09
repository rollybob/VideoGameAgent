"""
VGA local VLM server - the System-2 reasoner served on Thor (no API, no data leaves
the box). Runs INSIDE the thor-vlm container (GPU); the host reasoner
(vga/reason/local_reasoner.py) POSTs a frame and gets back a structured action.

Contract (mirrors the API path's `act` tool, but as plain JSON since local generate
has no forced tool_choice):
    POST /act  {image_b64, goal, step, last_action, last_changed, history, looping,
                 dialog_text, already_read, subgoal, progress}
                 -> {button, repeats, reason, subgoal, progress, latency_s}
        last_changed (bool|null): did the screen change after the last executed action?
        history (list, oldest first): [{button, reason, changed}] recent executed steps.
        looping (bool): recent screens are dominated by a small recurring set (spinning in
                 place). A one-off hub revisit does NOT set this - only genuine loops do.
        dialog_text (str): OCR of the current bottom dialogue/menu strip (advisory).
        already_read (bool): that text was already read recently (only surfaced when looping).
        subgoal/progress (str): the model's own scratchpad from the PREVIOUS step, fed back;
                 the model returns updated ones (layer b, INCLUDE_GOALS). Empty on step 0.
        These are the Phase-A working memory (2026-07-02) that lets the stateless VLM see
        its own recent behavior; omitting them falls back to the old 1-button context.
    GET  /health -> {ok, model_dir, device}

Env:
    MODEL_DIR       (default /models)
    PORT            (default 8077)
    MAX_NEW_TOKENS  (default 64) - safety ceiling only; generation stops at EOS well
                    before this, so it is NOT a latency lever (measured 2026-07-01).
    MAX_PIXELS      (optional) - per-platform vision-token budget, in raw pixels, passed
    MIN_PIXELS      (optional)   to the Qwen3-VL processor (smart-resizes into
                    [min,max]). Left UNSET by default: a GBA frame already tokenizes to
                    ~150 vision tokens, which is cheap, and capping it lower does NOT cut
                    latency (decode dominates) while it CAN cost read accuracy. The knob
                    exists for HD platforms later (cap a 1080p frame down); GBA wants the
                    full budget. Set e.g. MAX_PIXELS=50176 (=64*28*28, ~GBA-native) only
                    if you deliberately want to trade read detail for tokens.
    INCLUDE_REASON  (default 1) - when 1 the model emits a one-sentence "reason" with each
                    action; this chain-of-thought is the training signal we log for
                    distillation, but it roughly DOUBLES output tokens and thus latency
                    (~2.7s vs ~1.2s, measured). Set INCLUDE_REASON=0 for latency-sensitive
                    play once we no longer need to harvest reasons.
Kept self-contained (prompt + button list inline) so the container needs no vga import.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import re
import threading
import time

import torch
from flask import Flask, jsonify, request
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL_DIR = os.environ.get("MODEL_DIR", "/models")
PORT = int(os.environ.get("PORT", "8077"))
# Raised from 64 -> 128 with the goal scratchpad (2026-07-02): subgoal + progress + reason +
# button + repeats is a longer JSON object; 64 could truncate it mid-object -> invalid JSON.
# Still only a ceiling - generation stops at EOS (~40-55 tok here), so it is not a latency lever.
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "128"))
INCLUDE_REASON = os.environ.get("INCLUDE_REASON", "1") not in ("0", "false", "False", "")
# Goal/task-state scratchpad (layer b): when on, the model maintains its own sub-goal +
# progress across steps (fed back each call), making play goal-directed. Toggle for A/B.
INCLUDE_GOALS = os.environ.get("INCLUDE_GOALS", "1") not in ("0", "false", "False", "")
# Tutorial-learning (docs/TUTORIAL_LEARNING_PLAN.md): when on, the /act prompt carries the
# facts the agent has LEARNED from this game's tutorials (host-side KnowledgeStore), and the
# model flags instructional screens via is_tutorial so the host can read+distill+store them.
INCLUDE_KNOWLEDGE = os.environ.get("INCLUDE_KNOWLEDGE", "1") not in ("0", "false", "False", "")
# Sharpened, always-on MENU-vs-DIALOGUE perception rule (2026-07-02). The gated skill store
# failed because the model mislabels menus as dialogue (calls the pub Rumors menu 'dialog'
# and mashes A), so the menu-cursor skill never fired. This puts the decision-linked rule in
# the SYSTEM prompt UNCONDITIONALLY - present no matter what mode the model thinks it is in.
# Toggle for A/B (baseline vs treatment) so the change is measured, not assumed.
SHARP_MODE = os.environ.get("SHARP_MODE", "0") not in ("0", "false", "False", "")
# Decoding: greedy by default (deterministic, reproducible). For a ROBUSTNESS/variance
# baseline over multiple episodes from a FIXED savestate, greedy gives identical runs, so
# enable sampling (DO_SAMPLE=1, TEMP) to measure how often the agent actually completes the
# task under stochasticity - the mash<->paralysis instability we are trying to quantify.
DO_SAMPLE = os.environ.get("DO_SAMPLE", "0") not in ("0", "false", "False", "")
TEMP = float(os.environ.get("TEMP", "0.7"))

# --- Over-current mitigation: gradual GPU load ramp (2026-08-07) ----------------------------
# The Blackwell SoC over-current comparator (soctherm_oc, oc3 rail) trips on fast current
# TRANSIENTS (di/dt), not average power: thor-oc-watch caught ~110 oc3 trips in ~84s at only
# ~17W average during a VLM cold-start (2026-08-06), and dropping nvpmodel to 120W did NOT stop
# them -- the budget caps sustained watts, the wrong axis. The fault is the idle->full occupancy
# slam at a pinned max GPU clock. WARMUP runs a graduated ramp of dummy generations at startup so
# the first real request is not that cold slam (and front-loads the cuBLAS/cuDNN autotune storm,
# itself part of the cold-start current). Set WARMUP=0 to reproduce the burst for an A/B. KEEPWARM
# (opt-in, default off) runs a light idle trickle while serving so mid-session idle->request steps
# stay small -- targets the mid-session trips, which the startup ramp alone does not cover.
WARMUP = os.environ.get("WARMUP", "1") not in ("0", "false", "False", "")
WARMUP_GAP_S = float(os.environ.get("WARMUP_GAP_S", "0.4"))   # settle pause between ramp steps
WARMUP_HOLD = int(os.environ.get("WARMUP_HOLD", "2"))         # full-size back-to-back holds at top
KEEPWARM = os.environ.get("KEEPWARM", "0") not in ("0", "false", "False", "")
KEEPWARM_IDLE_S = float(os.environ.get("KEEPWARM_IDLE_S", "1.5"))  # idle gap before a trickle fires


def _pixel_kwargs() -> dict:
    """Optional per-platform vision-token budget for the processor (see module docstring).
    Only pass a bound if its env var is set, so the default is the model's own generous
    default (best read accuracy for GBA)."""
    kw = {}
    for env, key in (("MIN_PIXELS", "min_pixels"), ("MAX_PIXELS", "max_pixels")):
        val = os.environ.get(env)
        if val:
            kw[key] = int(val)
    return kw

# Keep in sync with vga/reason/base.BUTTON_CHOICES (the abstract controller).
BUTTONS = ["up", "down", "left", "right", "A", "B", "start", "select", "L", "R", "wait"]
# Case-insensitive lookup -> canonical button. The model sometimes emits "a"/"START"
# etc.; without this, a lowercase "a" failed the `in BUTTONS` test and silently became
# "wait" (dropping a real A press). Observed 2026-07-01.
_BUTTON_BY_LOWER = {b.lower(): b for b in BUTTONS}
# Keep in sync with vga/reason/skills.MODES. The screen-mode the model self-reports each
# step; keys which persistent skills the host feeds back next step (skill store, 2026-07-02).
MODES = ["dialog", "menu", "battle", "overworld", "shop", "title", "cutscene", "unknown"]
_MODE_SET = set(MODES)

SYSTEM_PROMPT = (
    "You are VGA, an agent that plays video games from the screen alone, like a human. "
    "You are given one screenshot of a Game Boy Advance game and must choose the single "
    "best controller input to make progress right now. Read on-screen text, menus, and "
    "tutorials and follow them. Advance dialogue and confirm menus with A; cancel or go back "
    "with B; move the cursor or character with the D-pad; open the menu (or pause) with start. "
    "ALL buttons are available, not just A/B/D-pad - use them when the screen calls for it: "
    "select often toggles a secondary view or mode; L and R usually switch tabs/pages, cycle "
    "targets, or rotate the camera. To move several tiles or advance repeated prompts in one "
    "go, set repeats > 1 rather than tapping once at a time. If an animation or transition is "
    "playing and nothing needs doing, choose \"wait\". Pick exactly ONE button; you will "
    "see the result on the next screenshot. You see pixels only - there is no hidden game "
    "state. Base every decision on what is visible."
)

# Appended to SYSTEM_PROMPT when SHARP_MODE is on. The single most important perceptual
# distinction the agent keeps getting wrong, stated operationally and tied to the action.
MENU_RULE = (
    " CRITICAL - tell a MENU from DIALOGUE, because it decides your input. A MENU is any list "
    "of selectable options with ONE highlighted (a cursor, arrow, or highlight bar on one "
    "item) - this INCLUDES service menus like a pub's Rumors/Missions/Quit/Leave list, and it "
    "is STILL a menu even when a text box (a greeting, a title) is also on screen. In a menu, A "
    "acts ONLY on the HIGHLIGHTED option; to choose a DIFFERENT option you MUST first press a "
    "D-pad direction (up/down/left/right) to move the cursor onto it, THEN press A - pressing A "
    "again without moving just re-selects the same option and will NOT make progress. A "
    "DIALOGUE is a text box you advance and has NO list of selectable options; press A to "
    "advance it. Rule of thumb: if you keep pressing A and the screen is not changing, you are "
    "in a MENU sitting on an option that does nothing here - MOVE the cursor to a different "
    "option instead of pressing A again."
    " COMMIT, do not dither: in a menu, decide the ONE option that best serves your objective, "
    "move the cursor onto THAT option, and press A to SELECT it. Moving the cursor is only "
    "setup - the selection does not happen until you press A. Do NOT scroll the cursor back and "
    "forth (down then up then down): alternating directions makes zero progress. If you have "
    "already read the options, stop moving and press A on your chosen one."
    " OBSERVE FIRST (this is the most common mistake): fill the \"highlighted\" field with the "
    "EXACT option that is highlighted RIGHT NOW (the one in a different color / with the cursor), "
    "BEFORE you choose a button. Then act on it: if that highlighted option is ALREADY the one "
    "your goal needs, press A immediately - do NOT press any D-pad direction. Press a D-pad "
    "direction ONLY when the option you want is NOT the one currently highlighted. Never keep "
    "pressing down toward an option that is already highlighted."
)


def _history_lines(history: list) -> list:
    """Render the rolling log of recent executed actions into compact prompt lines.
    Each entry is {button, reason, changed}; we show button + effect (and a short
    reason) so the model can see e.g. "down (no change) x5" and stop re-deciding it."""
    if not history:
        return []
    lines = ["Your recent inputs (oldest first) and whether each changed the screen:"]
    for h in history[-6:]:
        btn = h.get("button", "?")
        effect = "screen CHANGED" if h.get("changed") else "NO change"
        reason = str(h.get("reason", ""))[:60]
        note = f" - {reason}" if reason else ""
        lines.append(f'  {btn}: {effect}{note}')
    return lines


def _skill_lines(skills: list) -> list:
    """Render the persistent, mode-keyed procedural skills into prompt lines. These are
    hand-authored or learned how-to rules for THIS kind of screen (e.g. the menu-cursor
    rule) - advisory, but they encode procedures the model should not have to re-derive."""
    if not skills:
        return []
    lines = ["Known tactics for this kind of screen (apply if they fit what you see):"]
    for s in skills[:4]:
        lines.append(f"  - {str(s)}")
    return lines


def _knowledge_lines(knowledge: list) -> list:
    """Render the facts learned from this game's tutorials. FRAMING DISCIPLINE (echo/anchor
    bug, burned twice): present these as BACKGROUND REFERENCE that may not apply to the current
    screen - NEVER as facts about what is on screen now - so the model uses them as priors but
    still trusts the pixels. They are declarative world-facts, which are safer to feed back than
    goals, but the framing rule still holds."""
    if not knowledge:
        return []
    lines = ["What you have LEARNED about this game from its tutorials (background knowledge - "
             "may not apply to the current screen; trust what you actually see first):"]
    for k in knowledge[:4]:
        lines.append(f"  - {str(k)}")
    return lines


def _instruction(goal: str, step: int, last_action: str,
                 last_changed=None, history=None, looping=False,
                 dialog_text="", already_read=False, subgoal="", progress="",
                 mode="", skills=None, knowledge=None, committed="", task_phase="") -> str:
    lines = [f"Step {step}. This is the current screen."]
    if goal:
        lines.append(f"Objective: {goal}")
    if task_phase:
        # Durable, RAM-taught task-state phase directive (Task 07, 2026-07-06). The host has
        # OBJECTIVELY determined (from a pixels-only cue it validated against the RAM oracle at
        # train time -- e.g. the info-fee dialog that means the mission fee was paid == accepted)
        # WHERE the agent is in the multi-step task, and asserts it AUTHORITATIVELY here. Unlike
        # the sub-goal below (the model's own note, framed "may be stale"), this is GROUND TRUTH
        # about task progress that the model cannot see from one frame: after accepting a mission
        # the stateless agent forgets and re-tries to accept / wanders back into the pub. Placed
        # ABOVE the sub-goal so it governs it: if the sub-goal contradicts the phase, the phase
        # wins. It is durable by design (persists across scene changes / the scratchpad wipe),
        # which is safe here precisely because it is set by an objective cue, not the model's
        # belief. Empty (mechanism off / phase not yet advanced) -> this block is absent.
        lines.append("TASK STATE (authoritative - the game state confirms this; trust it over "
                     "your own sub-goal note below): " + str(task_phase))
    if INCLUDE_GOALS and subgoal:
        # PERCEPTION-FIRST framing. Feeding the model's own prior sub-goal back as fact
        # anchored it: it kept "confirming English" for 158 steps while the screen had long
        # moved to the pub (echo-bug class, 2026-07-02). So present the sub-goal as possibly
        # STALE and make the screen the source of truth.
        lines.append(f"Your sub-goal from the previous step (it MAY BE OUT OF DATE): {subgoal}")
        if progress:
            lines.append(f"Progress you noted before (also may be stale): {progress}")
        lines.append("FIRST, describe what is actually on the screen RIGHT NOW. If the screen "
                     "no longer matches that sub-goal, you have already moved past it - set a "
                     "NEW sub-goal from what you currently see. Trust the screen over the old "
                     "sub-goal.")
    elif INCLUDE_GOALS:
        lines.append("You have no sub-goal yet - decide one from the objective and what you see.")
    if committed:
        # Intent-persistence commitment (Task 01, 2026-07-04). The agent has relevant learned
        # knowledge that implies a specific move, but on the previous step it reflexively fell
        # back on the per-frame visual prior (e.g. walking the map) instead. The host has locked
        # this directive as FOREGROUND for a few steps so the plan survives the frames that do
        # not immediately confirm progress. This is the DELIBERATE anti-reflex override; unlike
        # the sub-goal above it is NOT hedged with "trust the screen". It is still self-limiting:
        # the host clears it the moment the screen materially changes (a menu opens) or the short
        # window expires, so it cannot zombie past its purpose.
        lines.append(
            "COMMITTED PLAN (do this NOW - it OVERRIDES the sub-goal note above): you recently "
            "learned something about THIS game that applies right here, and you have committed to "
            "acting on it for the next few steps: " + str(committed) + ". Choose the button that "
            "carries out this plan THIS step. Do NOT fall back to walking around or repeating your "
            "reflex action just because the screen looks unchanged - following the plan is exactly "
            "what will change it. Only stop if a menu or new screen is already open (then the plan "
            "worked - act on what you now see).")
    lines += _history_lines(history or [])
    if last_action:
        if last_changed is None:
            lines.append(f"Your previous input was: {last_action}.")
        elif last_changed:
            lines.append(f"Your previous input was: {last_action}, and the screen changed after it.")
        else:
            lines.append(
                f"Your previous input was: {last_action}, and the screen did NOT change after it. "
                "Repeating it is unlikely to help - try a different button or direction.")
    # "Going in circles": recent screens keep recurring even though each step changes the
    # screen (re-opening a menu entry, an A/B bounce). last_changed can't catch this. Note the
    # nudge does NOT say "leave" - repeating and leaving are BOTH failures at a hub you need to
    # act in; the fix is to do something you have NOT tried, which may be to finally act here.
    if looping:
        note = ("WARNING: you are going in circles - the last several steps keep returning to "
                "the same few screens without making progress. What you have been doing here is "
                "NOT working. Choose a DIFFERENT action than your recent inputs to actually make "
                "progress (e.g. a menu option you have not selected yet, or the task you came "
                "here to do).")
        if already_read and dialog_text:
            note += " (You are also re-reading text you have already seen - do not read it again.)"
        lines.append(note)
    lines += _skill_lines(skills or [])
    if INCLUDE_KNOWLEDGE:
        lines += _knowledge_lines(knowledge or [])
    # Compose the JSON schema. "mode" goes first so the model CLASSIFIES the screen before
    # planning (perception-first, and it keys which skills it gets next step); then the goal
    # fields so it settles its plan before it commits to a button (a light chain-of-thought).
    fields = ['"mode": "<one of: ' + " ".join(MODES) + ' - what kind of screen this is>"']
    # Perception-first for menus (2026-07-04): force the model to NAME the currently-highlighted
    # option BEFORE it picks a button. The model CAN read the highlight when asked (verified via
    # /read: 3/3), but in the act-loop it executed its carried plan without looking and pressed
    # down while the target was already highlighted. Placing this early in the ordered schema
    # makes it a chain-of-thought observation the button choice is then conditioned on.
    fields.append('"highlighted": "<if this screen is a MENU or option-list, the EXACT text of '
                  'the option highlighted RIGHT NOW - the one row drawn in a DIFFERENT COLOR or '
                  'with the cursor/arrow on it. Report ONLY what you literally SEE highlighted, '
                  'NOT the option you want or plan to pick; the highlighted row is often NOT your '
                  'target yet. A freshly opened list usually starts highlighting its TOP item. '
                  'Otherwise empty string>"')
    # Objective scene descriptor (Task 03B, 2026-07-05). A short PERCEPTION-first phrase naming
    # what is on screen (setting, any menu/window and its visible options, key entities). It is a
    # RETRIEVAL-ROUTING key: the host carries it forward one step and folds it into the context
    # that selects which learned facts/skills reach the next prompt, so routing keys on scene
    # CONTENT instead of a bare one-word mode. Placed early (after mode/highlighted) so it is a
    # genuine observation the action is conditioned on. STRICTLY objective - NOT the goal/plan -
    # to avoid the echo/anchor bug: it is a host-side key, never shown back as an authoritative
    # fact. Parsed defensively (capped; missing -> "").
    fields.append('"scene": "<up to ~12 words: OBJECTIVELY describe what is on screen RIGHT NOW - '
                  'the setting, any menu/window name and its visible options, and the key entities '
                  'or text. Describe only what you SEE; do NOT state your goal or plan>"')
    if INCLUDE_KNOWLEDGE:
        # Detection flag for tutorial-learning: the host reads+distills+stores when true.
        # Judge the CURRENT screen ONLY (ignore your goal/history - priming makes the model
        # over-flag the next screen), and exclude menus/option-lists explicitly (they read as
        # "learning" but are navigation, not instruction). Observed over-flag 2026-07-02.
        fields.append('"is_tutorial": <judge ONLY what is on THIS screen, ignoring your goal and '
                      'history: true if THIS screen is an explanatory tutorial/help/rules/info '
                      'screen whose text TEACHES a game mechanic or rule; false for normal '
                      'gameplay, story/flavor dialogue, and any MENU or list of selectable '
                      'options (e.g. Rumors/Missions/Items) - a menu is NEVER a tutorial even '
                      'when you are trying to learn something>')
    if INCLUDE_GOALS:
        fields.append('"subgoal": "<one short phrase, grounded in the CURRENT screen; if you '
                      'have clearly moved past the previous sub-goal, replace it with a new one>"')
        fields.append('"progress": "<one short phrase describing what the CURRENT screen shows '
                      'about your progress toward the objective>"')
    if INCLUDE_REASON:
        fields.append('"reason": "<one short sentence>"')
    fields.append('"button": "<one of: ' + " ".join(BUTTONS) + '>"')
    fields.append('"repeats": <integer 1-6: how many times to tap the button this decision; '
                  'use more to move several tiles or clear repeated prompts at once>')
    lines.append("Respond with ONLY a compact JSON object and nothing else, of the form: {"
                 + ", ".join(fields) + "}")
    return "\n".join(lines)


app = Flask(__name__)
_model = None
_processor = None
# All GPU generation is serialized behind this lock so the optional keep-warm idle trickle can
# never run a forward pass concurrently with a real request -- two generate() calls on one model
# at once is unsafe and would DOUBLE the instantaneous current, the opposite of the goal. With
# Flask threaded=False and keep-warm off it is uncontended, so serving behavior is unchanged.
_infer_lock = threading.Lock()
_last_infer = 0.0  # time.monotonic() of the last completed generation (keep-warm idle gating)


def _load():
    global _model, _processor
    if _model is not None:
        return
    print(f"[serve] loading {MODEL_DIR} ...", flush=True)
    t0 = time.time()
    _processor = AutoProcessor.from_pretrained(MODEL_DIR, **_pixel_kwargs())
    _model = AutoModelForImageTextToText.from_pretrained(
        MODEL_DIR, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    _model.eval()
    print(f"[serve] model ready in {time.time() - t0:.1f}s", flush=True)


def _locked_generate(**kwargs):
    """Every GPU generation goes through here so the optional keep-warm trickle can never overlap
    a real request's forward pass (see _infer_lock). Records completion time so keep-warm can tell
    whether the GPU has genuinely been idle. Free (uncontended) when keep-warm is off."""
    global _last_infer
    with _infer_lock:
        out = _model.generate(**kwargs)
    _last_infer = time.monotonic()
    return out


def _parse_action(text: str) -> dict:
    """Extract {button, repeats, reason, subgoal, progress, mode} from the model's text."""
    out = {"button": "wait", "repeats": 1, "reason": "", "subgoal": "", "progress": "",
           "mode": "", "is_tutorial": False, "highlighted": "", "scene": ""}
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            b = str(obj.get("button", "wait")).strip()
            out["button"] = _BUTTON_BY_LOWER.get(b.lower(), "wait")
            try:
                out["repeats"] = max(1, min(6, int(obj.get("repeats", 1))))
            except (TypeError, ValueError):
                out["repeats"] = 1
            out["reason"] = str(obj.get("reason", ""))[:200]
            out["highlighted"] = str(obj.get("highlighted", ""))[:60]
            # Objective scene descriptor (Task 03B): retrieval-routing key, capped defensively.
            out["scene"] = str(obj.get("scene", ""))[:120]
            out["subgoal"] = str(obj.get("subgoal", ""))[:120]
            out["progress"] = str(obj.get("progress", ""))[:160]
            md = str(obj.get("mode", "")).strip().lower()
            out["mode"] = md if md in _MODE_SET else ("unknown" if md else "")
            out["is_tutorial"] = bool(obj.get("is_tutorial", False))
            return out
        except json.JSONDecodeError:
            pass
    # Fallback: find any button word in the text (case-insensitive).
    for b in BUTTONS:
        if re.search(rf"\b{re.escape(b)}\b", text, re.IGNORECASE):
            out["button"] = b
            break
    out["reason"] = "(unparsed) " + text[:120]
    return out


@app.post("/read")
def read():
    """Free-form reading/distillation over a frame, with NO action-tool forcing (unlike /act).
    This is the tutorial-learning text channel: transcribe on-screen text, or distill it into a
    fact. Always GREEDY (do_sample=False) regardless of DO_SAMPLE - reading must be faithful and
    reproducible, not sampled. See docs/TUTORIAL_LEARNING_PLAN.md."""
    data = request.get_json(force=True)
    img = Image.open(io.BytesIO(base64.b64decode(data["image_b64"]))).convert("RGB")
    prompt = data.get("prompt", "Transcribe ALL on-screen text exactly as written. "
                       "Output only the text, nothing else.")
    system_text = data.get("system", "You are a careful reader of Game Boy Advance game "
                           "screens. Read text exactly as shown; never invent words you cannot "
                           "see.")
    max_tok = int(data.get("max_new_tokens", 256))
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_text}]},
        {"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": prompt},
        ]},
    ]
    chat = _processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = _processor(text=[chat], images=[img], return_tensors="pt").to(_model.device)
    t0 = time.time()
    with torch.no_grad():
        gen = _locked_generate(**inputs, max_new_tokens=max_tok, do_sample=False)
    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, gen)]
    text = _processor.batch_decode(trimmed, skip_special_tokens=True)[0]
    return jsonify({"text": text, "latency_s": round(time.time() - t0, 3)})


@app.post("/complete")
def complete():
    """Text-only completion over the local VLM - NO image, NO action-tool forcing. This is the
    self-reflection channel for the experiential loop (Task 03, 2026-07-05): mine_insights.py
    posts the objective success/fail contrast (pure text) and the SAME model that plays the game
    extracts transferable facts from it (ExpeL-style self-improvement), instead of an external
    Claude teacher. Qwen3-VL is an image-text-to-text model but generates fine with no image in
    the message, so the processor is called text-only here.

    Always GREEDY (do_sample=False) regardless of DO_SAMPLE - reflection must be reproducible,
    like /read. Keep the caller's prompt simple and parse defensively: the 8B has failed
    structured-output schemas before, so we do not force JSON here; the caller salvages it.
        POST /complete {prompt, system?, max_new_tokens?} -> {text, latency_s}"""
    data = request.get_json(force=True)
    prompt = data.get("prompt", "")
    system_text = data.get("system", "You are a careful, literal assistant. Answer exactly what "
                           "is asked and output nothing else.")
    max_tok = int(data.get("max_new_tokens", 256))
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_text}]},
        {"role": "user", "content": [{"type": "text", "text": prompt}]},
    ]
    chat = _processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = _processor(text=[chat], return_tensors="pt").to(_model.device)
    t0 = time.time()
    with torch.no_grad():
        gen = _locked_generate(**inputs, max_new_tokens=max_tok, do_sample=False)
    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, gen)]
    text = _processor.batch_decode(trimmed, skip_special_tokens=True)[0]
    return jsonify({"text": text, "latency_s": round(time.time() - t0, 3)})


@app.get("/health")
def health():
    return jsonify({
        "ok": _model is not None,
        "model_dir": MODEL_DIR,
        "device": "cuda",
        "include_reason": INCLUDE_REASON,
        "include_goals": INCLUDE_GOALS,
        "include_knowledge": INCLUDE_KNOWLEDGE,
        "sharp_mode": SHARP_MODE,
        "do_sample": DO_SAMPLE,
        "temp": TEMP,
        "max_new_tokens": MAX_NEW_TOKENS,
        "pixel_budget": _pixel_kwargs() or "model-default",
        "warmup": WARMUP,
        "keepwarm": KEEPWARM,
    })


@app.post("/act")
def act():
    data = request.get_json(force=True)
    img = Image.open(io.BytesIO(base64.b64decode(data["image_b64"]))).convert("RGB")
    instruction = _instruction(
        data.get("goal", ""), int(data.get("step", 0)), data.get("last_action", ""),
        last_changed=data.get("last_changed"), history=data.get("history"),
        looping=data.get("looping", False),
        dialog_text=data.get("dialog_text", ""), already_read=data.get("already_read", False),
        subgoal=data.get("subgoal", ""), progress=data.get("progress", ""),
        mode=data.get("mode", ""), skills=data.get("skills"),
        knowledge=data.get("knowledge"), committed=data.get("committed", ""),
        task_phase=data.get("task_phase", ""))

    system_text = SYSTEM_PROMPT + (MENU_RULE if SHARP_MODE else "")
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_text}]},
        {"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": instruction},
        ]},
    ]
    chat = _processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = _processor(text=[chat], images=[img], return_tensors="pt").to(_model.device)

    t0 = time.time()
    with torch.no_grad():
        if DO_SAMPLE:
            # REPRODUCIBLE sampling (2026-07-04): seed the RNG from the exact input (image +
            # full prompt) so the same state yields the same action and whole episodes replay
            # identically -> clean A/B, while temp>0 keeps the exploration a weak policy needs.
            # (Pure greedy is reproducible too but gets DETERMINISTICALLY STUCK -- observed the
            # same day: greedy failed the mission-accept rung that sampling clears.) Determinism
            # of the model is thus a function of its inputs, which is exactly what eval wants.
            # seed_salt (default "") lets a caller draw N reproducible-but-distinct trajectories
            # from the same state -> multi-seed ladder A/B. "" is a no-op, preserving the exact
            # un-salted seed for every existing caller.
            seed = int(hashlib.sha256((data["image_b64"] + chat + data.get("seed_salt", "")).encode()).hexdigest()[:15], 16)
            torch.manual_seed(seed)
            gen_kw = {"do_sample": True, "temperature": TEMP}
        else:
            gen_kw = {"do_sample": False}
        gen = _locked_generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, **gen_kw)
    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, gen)]
    text = _processor.batch_decode(trimmed, skip_special_tokens=True)[0]
    latency = round(time.time() - t0, 3)

    action = _parse_action(text)
    action["latency_s"] = latency
    action["raw"] = text[:300]
    return jsonify(action)


# Neutral gray frames of increasing size + increasing decode length: current approaches its peak
# in bounded steps instead of one idle->full slam. Ends at ~2x GBA size and the full token ceiling;
# nominal sizes may be resized by the processor, but max_new_tokens rises monotonically so the
# decode-side current ramps regardless. Runs the REAL /act path so the exact kernels initialize here.
_WARMUP_RAMP = [
    ((48, 32), 2), ((96, 64), 4), ((160, 108), 8), ((240, 160), 16),
    ((320, 214), 32), ((480, 320), 64), ((480, 320), 128),
]


def _warmup_once(size, max_new_tokens):
    w, h = size
    img = Image.new("RGB", (w, h), (128, 128, 128))
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": "Warm-up; reply with one word."},
        ]},
    ]
    chat = _processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = _processor(text=[chat], images=[img], return_tensors="pt").to(_model.device)
    with torch.no_grad():
        _locked_generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    torch.cuda.synchronize()  # force each step to finish before the pause, so the ramp is real


def _warmup():
    """Gradual GPU load ramp to avoid the cold-start over-current trip (see WARMUP note). Runs the
    real /act code path at increasing sizes so cuBLAS/cuDNN init and the first dense matmuls happen
    here, not on the first live request. Best-effort: a step failure is logged, never fatal."""
    if not WARMUP:
        print("[serve] warmup DISABLED (WARMUP=0)", flush=True)
        return
    t0 = time.time()
    print(f"[serve] warmup: {len(_WARMUP_RAMP)}-step ramp + {WARMUP_HOLD} hold(s), gap={WARMUP_GAP_S}s",
          flush=True)
    for i, (size, mnt) in enumerate(_WARMUP_RAMP):
        try:
            _warmup_once(size, mnt)
            print(f"[serve]   warmup {i + 1}/{len(_WARMUP_RAMP)}: {size[0]}x{size[1]}, {mnt} tok",
                  flush=True)
        except Exception as e:  # noqa: BLE001 - warm-up must never block serving
            print(f"[serve]   warmup {i + 1} skipped: {e}", flush=True)
        time.sleep(WARMUP_GAP_S)
    # Trailing sustained holds at full size, no gap: keep the top of the ramp contiguous so the
    # regulator settles at peak rather than relaxing and re-slamming on the first real request.
    top_size, top_tok = _WARMUP_RAMP[-1]
    for _ in range(max(0, WARMUP_HOLD)):
        try:
            _warmup_once(top_size, top_tok)
        except Exception as e:  # noqa: BLE001
            print(f"[serve]   warmup hold skipped: {e}", flush=True)
    print(f"[serve] warmup done in {time.time() - t0:.1f}s", flush=True)


def _keepwarm_loop():
    """Opt-in idle trickle (KEEPWARM=1): when no generation has run for KEEPWARM_IDLE_S, run one
    tiny generation so the GPU never fully collapses to idle, keeping the next real request's
    idle->full step (and its di/dt) small. Serialized behind _infer_lock; daemon thread; errors
    are logged and the loop continues. Targets the mid-session trips the startup ramp cannot."""
    print(f"[serve] keep-warm idle trickle ON (idle > {KEEPWARM_IDLE_S}s)", flush=True)
    while True:
        try:
            time.sleep(KEEPWARM_IDLE_S)
            if time.monotonic() - _last_infer < KEEPWARM_IDLE_S:
                continue  # a real request ran recently -- no trickle needed
            _warmup_once((160, 108), 2)
        except Exception as e:  # noqa: BLE001 - keep-warm must never crash serving
            print(f"[serve] keepwarm error (continuing): {e}", flush=True)
            time.sleep(1.0)


if __name__ == "__main__":
    _load()
    _warmup()
    if KEEPWARM:
        threading.Thread(target=_keepwarm_loop, daemon=True, name="keepwarm").start()
    app.run(host="127.0.0.1", port=PORT, threaded=False)
