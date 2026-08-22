#pragma once

#include <stdbool.h>
#include <stddef.h>

/* Adds amount to *total when representable; leaves *total unchanged on
   overflow. C11 has no standard checked-arithmetic API. */
bool size_add_checked(size_t *total, size_t amount);
