#include "size_utils.h"

#include <stdint.h>

bool size_add_checked(size_t *total, size_t amount) {
    if (!total || amount > SIZE_MAX - *total) return false;
    *total += amount;
    return true;
}
