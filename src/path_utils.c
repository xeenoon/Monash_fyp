#include "path_utils.h"

#include "size_utils.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

char *path_join(const char *left, const char *right)
{
	if (!left || !right)
		return NULL;
	size_t left_length = strlen(left);
	size_t right_offset = right[0] == '/' || right[0] == '\\' ? 1u : 0u;
	size_t right_length = strlen(right + right_offset);
	bool has_separator =
		left_length && (left[left_length - 1u] == '/' || left[left_length - 1u] == '\\');
	bool needs_separator = left_length && right_length && !has_separator;

	size_t size = left_length;
	if (needs_separator && !size_add_checked(&size, 1u))
		return NULL;
	if (!size_add_checked(&size, right_length) || !size_add_checked(&size, 1u))
		return NULL;
	char *result = malloc(size);
	if (!result)
		return NULL;
	memcpy(result, left, left_length);
	size_t cursor = left_length;
	if (needs_separator)
		result[cursor++] = '/';
	memcpy(result + cursor, right + right_offset, right_length);
	result[cursor + right_length] = '\0';
	return result;
}
