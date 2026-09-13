#pragma once

#include "dungeon_camera.h"
#include "dungeon_scene.h"
#include "input.h"
#include "renderer.h"

#include <stdbool.h>

/* Scripted driver for the dungeon scene, so lock picking can be exercised
 * end-to-end in the real renderer without a person at the keyboard. Enabled by
 * DUNGEON_SCRIPT=<path>; absent, nothing in this file runs and the main loop is
 * unchanged.
 *
 * It exists because the interesting claims -- "the camera turns to face the
 * lock", "solving it drops the collider so the player can walk through" -- are
 * about a running frame loop, not about anything the headless unit tests can
 * reach. Assertions are numeric (`expect`) and evidence is a PNG the renderer
 * writes itself (`capture`); neither involves screenshotting a desktop.
 *
 * Script format: one command per line, `#` starts a comment.
 *
 *   teleport <x> <z>              put the player at a world XZ position
 *   teleport_door <n> [far]       put them in reach of door n, on the side its
 *                                 lock hardware is on (or the far side)
 *   set_yaw <degrees>             aim the camera somewhere deliberately wrong
 *   set_pitch <degrees>           orbit to a pitch (clamped to -35..-82), to check
 *                                 a view at an angle a player can actually reach
 *   facing_of <n>                 print the yaw that would face door n
 *   wait <seconds>                let the loop run
 *   press <key>                   one frame of interact/left/right/up/down/
 *                                 confirm/cancel
 *   walk <forward> <right> <sec>  drive movement for a duration
 *   walk_to <x> <z> <seconds>     steer toward a world point
 *   walk_through <n> <seconds>    steer through door n to its far side
 *   sweep_pins                    drive the selection across every pin, checking
 *                                 the pick's clearance on every animation frame
 *   drive_safe_pins <n>           walk the selection to the pin the order wants
 *                                 next and press it, until n more stand driven
 *   select_wrong_pin              move the selection onto a pin the order does
 *                                 NOT want next, ready for a real confirm
 *   drive_prisms                  solve the prism lock through real Input flags
 *   expect prism_selected <n> | prism_rotating <0|1> | prism_solved
 *   solve                         fill the focused lock with its solution --
 *                                 proves the gate, not the input path
 *   capture <path.png>            settle two frames, then write the frame
 *   report                        print one line of state
 *   expect phase <exploring|pin|safe|prism>
 *   expect facing <tolerance_deg> camera yaw faces the focused lock
 *   expect focus <minimum>        eased focus weight at least this
 *   expect aimed <tolerance_deg>  the camera is actually pointing at the lock
 *   expect pins_left_to_right     pin 0 is left of the last pin on screen
 *   expect pick_clear             the pick is not intersecting the mechanism
 *   expect pins_left_to_right     pin 0 is left of the last pin on screen
 *   expect pin_selected <n>       which pin the keys are currently driving
 *   expect pin_height <n> <value> a pin's slot, so key input is proved not eyeballed
 *   expect pin_locked <n> | pin_free <n>
 *                                 whether a pin has set and stopped responding
 *   expect all_pins_free          no pin is set yet -- every bore needs working
 *   expect safe_progress <n>      safe pins driven so far
 *   expect safe_selected <n>      which face pin the keys are on
 *   expect safe_pin_next | safe_pin_wrong
 *                                 whether the selected pin is the one the order
 *                                 wants next -- nothing on the face reports it
 *   expect safe_pins_out <n>      how many pins the RENDERER will draw fully
 *                                 driven, so the animation is proved not assumed
 *   expect safe_pins_flush        no pin is standing proud of the face
 *   expect door_open <n> | door_shut <n>
 *   expect past_door <n>          player is on the far side of door n
 *   quit
 *
 * Any failed `expect` is reported and makes the process exit non-zero. */

typedef struct DungeonHarness DungeonHarness;

/* Returns NULL when DUNGEON_SCRIPT is unset, or on a script that cannot be
 * read (which is reported and treated as a failure by the caller). */
DungeonHarness *dungeon_harness_create(bool *out_script_error);
void dungeon_harness_destroy(DungeonHarness *harness);

/* Fixed timestep the harness forces so `wait` durations are exact and a run
 * reproduces regardless of frame rate. */
float dungeon_harness_timestep(const DungeonHarness *harness);

/* Runs script commands up to the next one that consumes a frame, writing any
 * synthesised key presses and movement into `input`. Returns false when the
 * script is finished and the loop should exit. */
bool dungeon_harness_pre_frame(DungeonHarness *harness, DungeonScene *scene,
							   DungeonCamera *camera, Input *input);

/* Flushes a pending capture once a frame has actually been presented. */
void dungeon_harness_post_frame(DungeonHarness *harness, Renderer *renderer);

/* 0 when every expectation held. */
int dungeon_harness_exit_code(const DungeonHarness *harness);
