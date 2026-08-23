#include "str_utils.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

char *str_dup_n(const char *source, size_t length)
{
	if (!source || length == SIZE_MAX)
		return NULL;
	char *result = malloc(length + 1u);
	if (!result)
		return NULL;
	memcpy(result, source, length);
	result[length] = '\0';
	return result;
}
