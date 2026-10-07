#pragma once

/* One campaign slot corresponds to one overworld entrance, persistent cache
 * file, procedural layout seed, and set of completion stars. Keep the count in
 * this shared header so the surface and dungeon halves cannot drift apart. */
#define DUNGEON_CAMPAIGN_LEVELS 20u
