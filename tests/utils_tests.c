#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "byte_utils.h"
#include "checksum_utils.h"
#include "file_utils.h"
#include "path_utils.h"
#include "size_utils.h"
#include "str_utils.h"

static void require(int condition, const char *message)
{
	if (!condition)
	{
		fprintf(stderr, "utils: %s\n", message);
		exit(EXIT_FAILURE);
	}
}

static void test_byte_utils(void)
{
	const uint8_t u16[] = {0x34, 0x12};
	const uint8_t u32[] = {0x78, 0x56, 0x34, 0x12};
	const uint8_t u64[] = {0xef, 0xcd, 0xab, 0x89, 0x67, 0x45, 0x23, 0x01};
	const uint8_t f32[] = {0x00, 0x00, 0x80, 0x3f};
	const uint8_t f64[] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0xc0};
	require(byte_read_u16_le(u16) == UINT16_C(0x1234), "u16 decode failed");
	require(byte_read_u32_le(u32) == UINT32_C(0x12345678), "u32 decode failed");
	require(byte_read_u64_le(u64) == UINT64_C(0x0123456789abcdef), "u64 decode failed");
	require(byte_read_f32_le(f32) == 1.0f, "f32 decode failed");
	require(byte_read_f64_le(f64) == -2.5, "f64 decode failed");
}

static void test_string_and_size_utils(void)
{
	char *copy = str_dup_n("terrain-extra", 7);
	require(copy && strcmp(copy, "terrain") == 0, "bounded string copy failed");
	free(copy);

	size_t total = 10;
	require(size_add_checked(&total, 20) && total == 30, "checked add failed");
	total = SIZE_MAX - 1u;
	require(!size_add_checked(&total, 2) && total == SIZE_MAX - 1u,
			"checked add did not detect overflow");

	char *path = path_join("terrain/", "/tiles/0.trn");
	require(path && strcmp(path, "terrain/tiles/0.trn") == 0, "path join failed");
	free(path);
}

static void test_checksum(void)
{
	static const char input[] = "123456789";
	require(checksum_crc32(input, sizeof(input) - 1u) == UINT32_C(0xcbf43926),
			"CRC-32 check vector failed");
}

static void test_file_utils(const char *executable_path)
{
	uint8_t *data;
	size_t size;
	require(file_read_all(executable_path, &data, &size) == FILE_READ_OK && size > 0,
			"could not read test executable");
	free(data);
	require(file_read_all("this-file-must-not-exist", &data, &size) == FILE_READ_OPEN_FAILED &&
				data == NULL && size == 0,
			"missing file result was incorrect");
}

int main(int argc, char **argv)
{
	require(argc > 0, "missing executable path");
	test_byte_utils();
	test_string_and_size_utils();
	test_checksum();
	test_file_utils(argv[0]);
	puts("reusable utility tests passed");
	return EXIT_SUCCESS;
}
