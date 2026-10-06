"""Exercise real game integration. Run after building terrain_renderer.

python3 tests/monk_dialogue_replay.py [path/to/terrain_renderer]
Requires the game's usual display/Vulkan support and local Alps dataset.
"""
import os
from pathlib import Path
import subprocess
import sys
import tempfile

root = Path(__file__).resolve().parents[1]
binary = Path(sys.argv[1]).resolve() if len(sys.argv)>1 else root/'build/terrain_renderer'
with tempfile.TemporaryDirectory(prefix='gameport-monk-replay-') as output:
    script = [
        '0 teleport monk', '1 interact', '2 expect_screen dialogue', '2 expect_dialogue mumble_1',
        '2 remember_world', '3 confirm', '4 down', '5 confirm', '6 expect_dialogue mumble_2',
        '7 confirm', '8 down', '9 confirm', '10 expect_dialogue mumble_3',
        '11 confirm', '12 down', '13 confirm', '14 expect_dialogue warning',
        '15 confirm', '16 confirm', '17 expect_phase edit', '17 text wow im sorry',
        '18 expect_reply wow im sorry', '18 submit', '19 expect_phase interrupt',
        '19 expect_fragment wow im so--', f'19 capture {output}/interrupt.png',
        '20 restart', '20 walk 1 1', '35 expect_phase interrupt',
        '40 expect_dialogue confession', '40 expect_frozen', '40 walk 0 0',
        '41 confirm', f'42 capture {output}/story.png', '43 escape',
        '44 expect_screen playing', '45 interact', '46 expect_dialogue confession',
        '47 confirm', '48 escape', '49 restart', '55 teleport monk', '56 interact',
        '57 expect_dialogue confession', '58 expect_screen dialogue', '59 quit'
    ]
    env = dict(os.environ, TERRAIN_SCENE='dungeon', DUNGEON_START_PLAYING='1',
               DUNGEON_SESSION_SEED='17', DUNGEON_SEED='17', DUNGEON_GAME_SCRIPT=';'.join(script))
    subprocess.run([str(binary)], env=env, cwd=root, check=True, timeout=180)
    assert all((Path(output)/name).is_file() for name in ('interrupt.png','story.png'))
print('Monk game replay passed: progression, text, cutoff, frozen world, exit, resume and reload')
