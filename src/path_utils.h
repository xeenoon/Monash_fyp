#pragma once

/* Returns a newly allocated path formed from left and right with exactly one
   separator between them. The caller owns the result and releases it with
   free(). */
char *path_join(const char *left, const char *right);
