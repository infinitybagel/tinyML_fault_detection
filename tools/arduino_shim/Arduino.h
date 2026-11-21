#pragma once
#include <cstdint>
#include <cstring>
#define PROGMEM
#define pgm_read_word(p) (*(const uint16_t *)(p))
#define memcpy_P memcpy
