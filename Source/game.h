#pragma once

#include <stdint.h>

#define PROFILE_MODE

#define GAME_MAX_INSTANCE_COUNT 50000000

typedef struct
{
	float spriteIndices[GAME_MAX_INSTANCE_COUNT];
	float scales[GAME_MAX_INSTANCE_COUNT];
	float positions[GAME_MAX_INSTANCE_COUNT * 2];
} Game_InstanceBuffer;

void game_init(Game_InstanceBuffer* buffer);
void game_tick(float delta);
void game_kill(void);

uint32_t game_gen_instance_buffer(Game_InstanceBuffer* buffer);
