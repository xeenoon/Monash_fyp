# The padlock

A pin-tumbler door wears `assets/dungeons/padlock/runtime/padlock.gltf`: an
imported, animated model of a torn industrial padlock, posed every frame from
the puzzle's own state. The source (`assets/dungeons/padlock/padlock_animated.blend`,
22 MB) is a render asset -- 275 bone-parented objects, 964k triangles, 21 fully
procedural materials, no textures at all, and ten actions -- and none of that
can be loaded by this renderer. `tools/export_padlock_runtime.py` is the whole
distance between the two.

This extends two claims in `dungeon_architecture.md`, which describes the
*generated* cutaway mechanism that is now the fallback:

- **Pin count is fixed at four.** `dungeon_session` used to seed three or four
  pins from the door's seed. The padlock has four pin stacks, and a fifth bore
  the player can select with nothing on the lock to show for it is not a harder
  puzzle, it is a broken one.
- **`expect pick_clear` means something different on a padlock door.** The
  generated pick is *posed* from puzzle state, so what has to be proved is that
  it does not cut through the lock, and `dungeon_lock_layout` is the authority.
  The padlock's pick is *authored*: the animator decided where it goes. The only
  part of its motion this engine chooses is which pin it has been driven to, so
  that is what `pick_clear` and `sweep_pins` assert there --
  `dungeon_scene_padlock_pick_on_selection` poses the model twice, once with the
  clips alone and once with the selection offset on top, and resolves the
  difference to a pin index. Measuring against the rest pose instead would be
  reading the animator's insertion travel, which runs along the same axis.
  Passing the layout check on a door that is not wearing the layout would be an
  assertion about geometry nobody is drawing.

## The nearest door wears it

Doors are placed in whatever order the hallway scan finds them, which has
nothing to do with the order a player walks into them -- and the lock a level
opens with is the one that teaches it. `dungeon_cave_generate` therefore sorts
the doors by **walking** distance from the spawn (4-connected steps over open
field) before handing out kinds, so door 0 is both the first one met and the pin
tumbler, with the safe and the prism cycling out from there. The mix is
unchanged; only the order is.

Walking distance rather than straight-line is not a detail: seed 3's prism door
is 12.9 m from the spawn as the crow flies and 48.8 m to walk, so a straight-line
sort would still open that level with the light puzzle.

Sorting the array rather than assigning kinds out of order also keeps door 0
meaning "the first one you meet", which is what every harness script's
`teleport_door 0` is written against. `the_nearest_door_is_the_pin_tumbler` in
`tests/dungeon_cave_tests.c` holds it across twelve seeds.

## The importer keeps a hierarchy only when there is animation

`gltf_scene` has always flattened a file's node transforms into its vertices at
load, which is right for the wall torch and the quarry and impossible for a lock
whose pins move. So an imported scene is now in one of two shapes, decided by
whether the file has clips: static files are flattened exactly as before, byte
for byte, and animated ones keep their nodes, their clips, and their vertices in
node space. Flattening an animated file would be discarding the very thing the
animation addresses; keeping a hierarchy for a static one would charge every
existing caller a matrix chain for an identity.

Skinning, morph targets, morph-weight channels and CUBICSPLINE interpolation are
all refused with a message naming the file, rather than loaded into something
that silently never moves.

## Posing is three calls, not one

`gltf_scene_rest_pose` fills a pose, `gltf_clip_sample` writes *only* the nodes a
clip animates, and `gltf_scene_world_matrices` chains the result down the
hierarchy (over a topological order built at load, since glTF does not require a
parent to precede its children).

The gap between the second and third call is deliberate, and is where game state
that the animation does not carry goes: how far an unset pin has been raised,
and which pin the pick is on. The importer never learns that pins or picks
exist.

Sampling **clamps** at both ends of a channel. That is what lets a finished
one-shot hold its last pose by parking its time past the end, so a set pin stays
set with no second "finished" pose to author or store.

## Clips layer because the exporter makes them disjoint

Blender bakes every bone into every action, so each of the eight clips arrives
with all 28 bones keyed whether the animator touched them or not -- and a pin
popping would then snap the pick, the core and the shackle back to rest
underneath it. The export tool drops every channel that never leaves its node's
rest pose, which leaves `Pin_02_Pop` touching one pin stack and `Pick_Insert`
touching the pick. A pin setting while the pick works is then two clips running
at once rather than a clip authored per combination.
`tests/padlock_asset_tests.py` asserts exactly that property of the baked asset,
because it is the contract the whole presentation rests on.

That in turn depends on the rest pose being *rest*. The source is saved part-way
into its tear-off action with the pick scaled to zero, and clearing the action
does not clear the pose -- so the first export collapsed the entire lockpick to
a point and gave every clip a PICK channel that "differed from rest" because
rest itself was wrong. The tool clears the pose transforms, which is what the
source's own README says to do before changing actions.

`dungeon_padlock` (`src/dungeon_padlock.c`) owns which clips are playing and how
far into each, as a pure state machine with no renderer and no glTF dependency,
so `tests/dungeon_padlock_tests.c` runs it headlessly the way
`tests/dungeon_lock_tests.c` runs the puzzle. It never hardcodes a clip length:
durations are handed in from the asset.

## The window, because a closed padlock does not show its pins

Casting rays at the four key pins from 2000 directions, every direction that
reaches all four is more than 14 degrees below the lock: they are visible up
through the keyway and nowhere else -- in the intact state and with the front
plate peeled alike, since the pins live inside the plug and the tear does not
open the plug. A lock whose pin heights cannot be read is not the puzzle this
replaces.

The first attempt cut the lock in half down its width. That did show the pins,
and left **an open shoebox with half a shackle on top**, because the half it
removed included the entire face a padlock is recognised by. So the exporter
cuts a **window** instead: a box taken out of the face and the front of the
plug, over the mechanism, leaving the casing's frame, the shackle and the
silhouette intact. The lock is 72 mm wide again rather than the 52 mm a
half-section left.

Which wall the window goes in follows from the mount angle, and is not
obvious. The pin row runs along the lock's **depth**, so the sightline to each
pin leaves through the **face**, not the side: at the mounted angle a ray
reaching the deepest pin crosses the front plane about half a unit off centre,
well inside the face and nowhere near the side wall. A window cut in the side
left every pin buried behind the rear one. There is half a unit of clearance
between the front wall and the first pin, which is where the cut lands.

It is a boolean against a box, not a plane bisect -- a bisect can only take a
whole half -- and only objects that actually reach into the window are cut, so
the solver runs over the casing, the face and the plug rather than over 200
screws. The pick is excluded by name: it tunnels down the keyway straight
through the window box, and cutting it would leave the player holding a stub.

None of that is trusted. `check_pins_are_visible` ray-casts at each pin **from
the angle the lock is actually mounted at**, because a window that clears a
head-on view can still be blocked by its own edge at fifty degrees, and the
export fails if any pin is behind anything.

## The model's frame, and why the lock is turned

Measured from the asset, and relied on by `padlock_placement`:

| axis | meaning |
| --- | --- |
| +Y | up: the shackle stands above the body |
| +X | across the lock's width; the window is cut toward this side |
| +Z | the keyway axis, and the line the four pin stacks are spaced along |

Those last two want opposite things and the model does not let you have both.
The pins are spaced along the **keyway**, front to back, the way a real pin
tumbler is built -- so they only line up across the screen when the lock is
viewed from the side of the lock rather than its face. Mounted that way the
padlock is edge-on: a tall thin slab that does not read as a padlock at all.
Mounted the other way round it is unmistakably a padlock with its mechanism
hidden behind itself.

The gameplay view now puts the broad lock face square to the player with
`DUNGEON_PADLOCK_FACE_TURN_DEGREES` set to 90 degrees:

```
model +X = cos t * normal + sin t * across    (the lock's width)
model +Y = up
model +Z = sin t * normal - cos t * across    (keyway, and the pin row)
```

At `t` = 0 the lock is seen from its narrow side. At 90 the broad outer face
points squarely at the player. The latter is the requested presentation; mouse
look supplies small inspection adjustments around that centred starting view.

For any turn short of 90 degrees, +Z keeps a negative `across` component, so
PIN_01 -- the puzzle's pin 0 -- stays on the left of the screen. `across` is the
camera's right (see `lock_frame`), and the harness's `pins_left_to_right`
projects the model's own posed pin positions onto it rather than trusting any of
this.

Which pin the
keys are driving is shown by the pick's **depth** in the keyway, not a sideways
slide: the four stacks are spaced along the keyway axis, so reaching pin 3 means
reaching further in, which is what picking a real lock looks like.

An unset pin is shown at a fraction of the travel its own pop clip authored --
read off the clip at load, not written down in the renderer -- capped at
`DUNGEON_PADLOCK_PIN_HEIGHT_SPAN` so a pin at the top of its range is still
visibly short of a set one.

`tests/padlock_pose_tests.c` measures both readouts against the baked asset,
without a device, because "the pick tells you which pin you are on" is a
question about millimetres and should not be settled by looking at the screen:

| readout | movement at the drawn scale |
| --- | --- |
| pick, one pin of selection | 17.3 mm |
| pick, across the whole row | 52.0 mm |
| pin, one key press | 6.0 mm |
| pin, slot 0 to slot 3 | 18.0 mm |
| pin, once it pops | 13.5 mm (the authored set travel) |

The pin readout is **exaggerated on purpose**. Held to the lock's own travel it
is physically honest and unplayable: a real pin tumbler moves its pins about
3 mm, which at the old framing put a whole key press inside two pixels, and a
puzzle cannot be solved from a readout nobody can see.
`DUNGEON_PADLOCK_PIN_HEIGHT_SPAN` spreads the four slots over 1.3x the authored
travel, so a pin near the top of its range stands about where a set one does.
That is the cost and it is a small one: a pin SETTING is announced by its own
pop clip -- the driver dropping, the spring compressing above it -- not by
height alone. The test bounds the exaggeration rather than forbidding it.

The alternative was to keep the span under 1.0 and grow `DUNGEON_PADLOCK_SCALE`
until the millimetres survived, which reaches a lock taller than the doorway
long before it reaches a legible one.

The focus camera was retuned to match: 1.0 m back at 38 degrees rather than
1.45 m at 42, and lower, so it looks **across** the mechanism instead of down
onto it -- pin height is the thing being read, and a steep angle foreshortens
exactly that. A metre of frame used to span the view; now 0.69 m does, which
turns a key press from two pixels into nine.

The test also pins the things that would break silently: the pick must not
follow the selection while the lock is not being worked (the player is not
holding it), and the lock must stand upright with the shackle at the top. At
the requested 90-degree starting pose the keyway points into screen depth; a
small mouse adjustment reveals the spacing between its pins.

## Lighting it, and the trap half a metre in front of the camera

Two separate things made the mechanism hard to read, and only one of them was
lighting.

The first is asset exposure. The source padlock was authored for a bright
Blender studio, and its baked atlas is several stops darker than the generated
lock material the safe wears -- so under identical torchlight the padlock read
as a black hole beside a legible safe. `padlock_push` applies a 2.4x linear
albedo gain to this model alone. Values above one are deliberate: base colour is
linear HDR by the time `mesh.frag` multiplies by it. Correcting it here rather
than raising the room lights is the whole point -- the door in front of the lock
is already correctly lit, and brightening the room blows it out to fix a
material that was never the room's fault.

The second is where the light is. Picking a pin-tumbler lock moves the carried
torch's **point light** beside the mechanism, where it has a clear oblique path
into the cutaway. The imported torch mesh and animated flame billboard are
suppressed for this close-up, so they cannot cover the mechanism or distract
from the pin readout. The light still hangs off `player_torch_transform`, and
its intensity drops from 16 to 6 while focused because it is much closer to the
lock than it is during exploration.

The lockpick script holds both halves of that presentation contract:

| check | what it refuses to let happen |
| --- | --- |
| `expect torch_hidden` | the pin-tumbler close-up exposing the carried mesh or flame billboard |
| `expect torch_lit` | hiding the visuals accidentally removing the point light too |

## Working it: which key does what

| input | while exploring | while a lock is up |
| --- | --- | --- |
| WASD | walk | nudge the inspection view |
| arrow keys | orbit the camera | drive the lock |
| mouse | -- | nudge the inspection view, faster |
| E | enter a lock in reach | leave the lock |
| Q / Enter / Space | -- | cancel / confirm |
| Ctrl | release or recapture the pointer | as exploring |

WASD and the mouse write the same clamped offsets (`focus_yaw_offset`,
`focus_pitch_offset`, +/-24 and +/-15 degrees), so they cannot fight each other
and neither can leave the framing. WASD was originally bound to the puzzle as an
alias for the arrow keys, which left a player at a lock with no way to look
around it without reaching for the mouse; `dungeon_camera_focus_pan` is the
keyboard's way into the same offsets, at a rate per second rather than per pixel
because a held key has no magnitude of its own.

`expect view_panned` / `expect view_centred` drive this through the harness's
real `move_forward`/`move_right` -- the same fields a held key writes -- and
check that panning out and back lands on centre again, and that the view keys
never touched the puzzle on the way past.

## One material, one atlas

The source's 21 procedural materials are baked into a single 4096 atlas --
albedo, ORM, and a tangent-space normal map -- with every object unwrapped and
packed into one UV square by surface area, so the housing gets the resolution
and a retaining screw does not. A shader cannot evaluate Blender's noise and
voronoi trees, and a per-part material set would cost a descriptor set and a
pipeline bind per part. Occlusion rides in the ORM's red channel, which is what
`GltfLoadOptions.use_metallic_roughness_red_as_occlusion` already existed for.

Every object that shares a bone is joined, which turns 241 draws into one per
animated bone.

## Scale

The model is exported at true size, because that is what the asset is: a 72 mm
padlock, 120 mm over the shackle. The game multiplies by
`DUNGEON_PADLOCK_SCALE` when it places it -- at true size on a 2.4 m door it is
a thumbnail, and the focus camera, the pick stance and the lock's reserved light
are all framed for the half-metre mechanism it replaces. At 4.5 the lock is
0.54 m over the shackle, which is what a 6 mm-per-press pin readout costs.

The lock's back sits on the standoff plane the rest of the door hardware hangs
off, at whatever depth the model turns out to have, rather than at a
written-down offset -- and that depth is measured from the *turned* box, since
turning the lock swings its keyway depth into the door normal as well. A
re-export that changes the casing therefore cannot bury it in the leaf.

## A missing asset is not an error

The runtime glTF is baked out of a Blender file that a fresh checkout has no
reason to have processed, so `dungeon_scene_create` falls back to the generated
cutaway mechanism and says so on stdout; CMake says the same at configure time.
A *broken* asset is an error: silently falling back would hide an export having
gone wrong.

To bake it:

```sh
blender -b assets/dungeons/padlock/padlock_animated.blend \
        --python tools/export_padlock_runtime.py -- --out assets/dungeons/padlock/runtime
```

## Phase 2: the tear-off

`Tear_Front_Plate` -- seven shape-keyed peel regions, a driven `front_tear`
property, a detached plate centre, a pry bar, four torn rivets, and a visibility
swap from the intact front to the fractured one at `front_tear` 0.16 -- is **not
exported and not supported**. It needs morph targets, animated morph weights,
and a per-node visibility concept that glTF cannot express at all; the importer
refuses the first two loudly rather than loading a lock that silently never
tears. The exporter drops the tear collection, the peeled and removed plates,
the pry bar and the rivet bones, and keeps the intact front.

The cheapest shape for it, when it comes, is two models swapped at the moment of
rupture rather than a morph pipeline in the renderer.
