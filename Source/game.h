#pragma once

#include <stdint.h>

void game_init(void);
void game_tick(float delta);
void game_kill(void);

#define GAME_MAX_INSTANCE_COUNT 50000000
typedef struct
{
    float spriteIndex;
    float scale;
    float pos[2];
} Game_Instance;

typedef struct
{
    Game_Instance instances[GAME_MAX_INSTANCE_COUNT];
} Game_InstanceBuffer;

uint32_t game_gen_instance_buffer(Game_InstanceBuffer* buffer);
