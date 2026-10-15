"""Static checks on the mod's runtime Lua (control.lua and what it requires),
for mistakes that only surface as a crash in game: calling a library function
or reading a global that exists while prototypes load but not at runtime, such
as table.deepcopy, which only util.lua defines.
"""

import re
import shutil
import subprocess
from pathlib import Path

import pytest
from lupa import lua52

MOD = Path(__file__).resolve().parent.parent / "factorion-mod" / "mod"
RUNTIME_FILES = ["control.lua", "gui.lua", "inference.lua", "parity.lua"]
LIBRARIES = ["table", "string", "math", "bit32", "coroutine"]

# Factorio's runtime globals, beside Lua's base library.
FACTORIO_GLOBALS = {
    "game", "script", "storage", "defines", "rendering", "remote", "helpers", "prototypes",
    "settings", "commands", "rcon", "serpent", "log", "localised_print", "table_size",
}
# Base functions Factorio removes from mods.
REMOVED = {"dofile", "loadfile", "io", "os", "module"}


def code(path):
    """The source without comments and string literals."""
    text = path.read_text()
    text = re.sub(r"--\[(=*)\[.*?\]\1\]", "", text, flags=re.S)
    text = re.sub(r"\[(=*)\[.*?\]\1\]", '""', text, flags=re.S)
    text = re.sub(r'"(\\.|[^"\\\n])*"|\'(\\.|[^\'\\\n])*\'', '""', text)
    return re.sub(r"--[^\n]*", "", text)


def lua_members(name):
    runtime = lua52.LuaRuntime()
    return set(runtime.eval(name).keys())


def test_runtime_only_calls_library_functions_that_exist():
    members = {library: lua_members(library) for library in LIBRARIES}
    unknown = []
    for name in RUNTIME_FILES:
        for library, field in re.findall(r"(?<![\w.])(" + "|".join(LIBRARIES) + r")\.(\w+)", code(MOD / name)):
            if field not in members[library]:
                unknown.append(f"{name}: {library}.{field}")
    assert not unknown, f"Not in Lua 5.2's libraries at runtime: {unknown}"


@pytest.mark.skipif(not shutil.which("luac"), reason="needs luac")
def test_runtime_only_reads_globals_that_exist():
    runtime = lua52.LuaRuntime()
    allowed = (set(runtime.globals().keys()) - REMOVED) | FACTORIO_GLOBALS
    unknown = []
    for name in RUNTIME_FILES:
        listing = subprocess.run(
            ["luac", "-p", "-l", "-l", str(MOD / name)], capture_output=True, text=True, check=True
        ).stdout
        for global_name in set(re.findall(r'GETTABUP.*_ENV "(\w+)"', listing)):
            if global_name not in allowed:
                unknown.append(f"{name}: {global_name}")
    assert not unknown, f"Globals Factorio's runtime doesn't provide: {unknown}"
