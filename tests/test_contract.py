from vga.core import contract
import numpy as np

# Test Button Enum
assert contract.Button.UP.value == "up"
assert contract.Button.DOWN.value == "down"
assert contract.Button.LEFT.value == "left"
assert contract.Button.RIGHT.value == "right"
assert contract.Button.A.value == "A"
assert contract.Button.B.value == "B"
assert contract.Button.START.value == "start"
assert contract.Button.SELECT.value == "select"
assert contract.Button.L.value == "L"
assert contract.Button.R.value == "R"

# Test Action class
action_wait = contract.Action.wait(note="Waiting")
assert action_wait.button is None
assert action_wait.duration == 0.12
assert action_wait.repeats == 1
assert action_wait.gap == 0.10
assert action_wait.note == "Waiting"
assert str(action_wait) == "wait(Waiting)"

action_press = contract.Action.press(button=contract.Button.A, duration=0.2, note="Press A")
assert action_press.button == contract.Button.A
assert action_press.duration == 0.2
assert action_press.repeats == 1
assert action_press.gap == 0.10
assert action_press.note == "Press A"
assert str(action_press) == "A(Press A)"

action_mash = contract.Action.mash(button=contract.Button.B, repeats=5, duration=0.03, gap=0.07, note="Mash B")
assert action_mash.button == contract.Button.B
assert action_mash.duration == 0.03
assert action_mash.repeats == 5
assert action_mash.gap == 0.07
assert action_mash.note == "Mash B"
assert str(action_mash) == "Bx5(Mash B)"

# Test Detection class
detection = contract.Detection(label="enemy", bbox=(10, 20, 30, 40), confidence=0.95)
assert detection.label == "enemy"
assert detection.bbox == (10, 20, 30, 40)
assert detection.confidence == 0.95

# Test GameState class
game_state = contract.GameState(frame_index=1, timestamp=1633072800.0, width=240, height=160)
assert game_state.frame_index == 1
assert game_state.timestamp == 1633072800.0
assert game_state.width == 240
assert game_state.height == 160
assert game_state.objects == []
assert game_state.text == {}
assert game_state.regions == {}
assert game_state.frame is None

game_state_with_data = contract.GameState(
    frame_index=2,
    timestamp=1633072860.0,
    width=240,
    height=160,
    objects=[detection],
    text={"health": "100"},
    regions={"battle_area": (50, 50, 190, 110)},
    frame=np.zeros((160, 240), dtype=np.uint8)
)
assert game_state_with_data.frame_index == 2
assert game_state_with_data.timestamp == 1633072860.0
assert game_state_with_data.width == 240
assert game_state_with_data.height == 160
assert game_state_with_data.objects == [detection]
assert game_state_with_data.text == {"health": "100"}
assert game_state_with_data.regions == {"battle_area": (50, 50, 190, 110)}
assert np.array_equal(game_state_with_data.frame, np.zeros((160, 240), dtype=np.uint8))

print('OK')
