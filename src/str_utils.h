#pragma once

#include <stddef.h>

/* C11 has no bounded string duplication function. Returns a newly allocated,
   NUL-terminated copy of exactly length bytes, or NULL on allocation failure. */
char *str_dup_n(const char *source, size_t length);
