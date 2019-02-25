#pragma once

#include <stdint.h>

#define PROFILE_MODE

#define GAME_MAX_INSTANCE_COUNT 50000000

typedef struct
{
	int16_t spriteIndicesAndScales[GAME_MAX_INSTANCE_COUNT * 2];
	uint16_t positionX[GAME_MAX_INSTANCE_COUNT];
	uint16_t positionY[GAME_MAX_INSTANCE_COUNT];
} Game_InstanceBuffer;

void game_init(Game_InstanceBuffer* buffer);
void game_tick(float delta);
void game_kill(void);

uint32_t game_gen_instance_buffer(Game_InstanceBuffer* buffer);
