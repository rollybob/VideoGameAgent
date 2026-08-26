import oracle

# Test Oracle class
assert oracle.Oracle.name == "generic"
assert oracle.Oracle.addrs == {}

emu_mock = lambda: None
emu_mock.u8 = lambda addr: 1 if addr == 0 else 2
emu_mock.u16 = lambda addr: 3 if addr == 0 else 4
emu_mock.u32 = lambda addr: 5 if addr == 0 else 6
emu_mock.run = lambda frames: None

oracle_instance = oracle.Oracle()
oracle_instance.addrs = {"field1": (0, "u8"), "field2": (1, "u16")}
state = oracle_instance.read_state(emu_mock)
assert state == {"field1": 1, "field2": 4}

oracle_instance.boot_to_play(emu_mock)  # No assertion needed for this method in base class

reward = oracle_instance.reward(None, {})
assert reward == 0.0

summary = oracle_instance.progress_summary({})
assert summary == ""

checkpoints = oracle_instance.checkpoints({})
assert checkpoints == []

# Test PokemonAIRed class
pokemon_ai_red = oracle.PokemonAIRed()
assert pokemon_ai_red.name == "pokemon_ai_red"
assert pokemon_ai_red.addrs == oracle.POKEMON_AI_RED

emu_mock.tap = lambda button, hold=0, then=0: None
pokemon_ai_red.boot_to_play(emu_mock)  # No assertion needed for this method in subclass

reward = pokemon_ai_red.reward(None, {"player_x": 10, "player_y": 20})
assert reward == 1.0
assert (10, 20) in pokemon_ai_red._visited

reward = pokemon_ai_red.reward({"player_x": 10, "player_y": 20}, {"player_x": 10, "player_y": 20})
assert reward == -0.05

summary = pokemon_ai_red.progress_summary({"player_x": 10, "player_y": 20})
assert summary == "pos=(10,20) tiles_explored=1"

# Test Ffta class
ffta = oracle.Ffta()
assert ffta.name == "ffta"
assert ffta.addrs == oracle.FFTA

ffta.boot_to_play(emu_mock)  # No assertion needed for this method in subclass

reward = ffta.reward(None, {"worldmap_cursor": 10})
assert reward == 1.0
assert 10 in ffta._cursor_history

reward = ffta.reward({"worldmap_cursor": 10}, {"worldmap_cursor": 10})
assert reward == 0.0

summary = ffta.progress_summary({"worldmap_cursor": 10, "clan_funds": 5000, "mode_overlay": 208})
assert summary == "worldmap_cursor=10 visited=1 funds=5000 mode_overlay=208"

checkpoints = ffta.checkpoints({"clan_funds": 4999, "mode_overlay": 208, "scene": 7, "clan_pos": 20})
assert checkpoints == [
    ("pub_open", True),
    ("mission_list", True),
    ("mission_accepted", True),
    ("worldmap_regained", True),
    ("at_giza", True),
    ("battle_entered", False),
]

ffta.task = "naming"
checkpoints = ffta.checkpoints({"naming_sig": oracle.Ffta.NAMING_SIG_ON, "dialog_sig": oracle.Ffta.DIALOG_SIG_ON})
assert checkpoints == [
    ("naming_open", True),
    ("confirm_reached", True),
    ("name_committed", False),
]

# Test for_rom function
rom_path = "/path/to/pokemon ai red.gba"
oracle_instance = oracle.for_rom(rom_path)
assert isinstance(oracle_instance, oracle.PokemonAIRed)

rom_path = "/path/to/ffta.nds"
oracle_instance = oracle.for_rom(rom_path)
assert isinstance(oracle_instance, oracle.Ffta)

rom_path = "/path/to/final fantasy tactics.iso"
oracle_instance = oracle.for_rom(rom_path)
assert isinstance(oracle_instance, oracle.Ffta)

try:
    rom_path = "/path/to/unknown.gba"
    oracle_instance = oracle.for_rom(rom_path)
except ValueError as e:
    assert str(e) == "No RAM oracle registered for ROM 'unknown.gba'. Add addresses + a registry entry in train/ram/oracle.py."

print('OK')
