#include "game.h"

#define MIST_PROFILE_ENABLED
#include "3rd/Mist_Profiler.h"

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define SWAP(type, a, b) do{ type t = b; b = a; a = t; }while(0);

// Random
uint32_t rand_range(uint32_t min, uint32_t max)
{
	uint32_t randMax = (RAND_MAX << 15) | RAND_MAX;
	uint32_t randVal = (rand() << 15) | rand();
	return (uint32_t)(((float)randVal / randMax) * (max - min) + min);
}

float rand_rangef(float min, float max)
{
	uint32_t randMax = (RAND_MAX << 15) | RAND_MAX;
	uint32_t randVal = (rand() << 15) | rand();
	return ((float)randVal / randMax) * (max - min) + min;
}

// Math
typedef struct
{
	float x, y;
} Vec2;

Vec2 vec2_add(Vec2 l, Vec2 r)
{
	return (Vec2) { .x = l.x + r.x, .y = l.y + r.y };
}

Vec2 vec2_sub(Vec2 l, Vec2 r)
{
	return (Vec2) { .x = l.x - r.x, .y = l.y - r.y };
}

Vec2 vec2_mul(Vec2 v, float s)
{
	return (Vec2) { .x = v.x * s, .y = v.y * s };
}

float vec2_mag(Vec2 v)
{
	return sqrtf(v.x * v.x + v.y * v.y);
}

Vec2 vec2_norm(Vec2 v)
{
	float mag = vec2_mag(v);
	if (mag == 0.0f) return (Vec2) { .x = 0.0f, .y = 0.0f };

	return vec2_mul(v, 1.0f / mag);
}

float math_maxf(float l, float r)
{
	return l > r ? l : r;
}

float math_minf(float l, float r)
{
	return l < r ? l : r;
}

uint32_t math_max(uint32_t l, uint32_t r)
{
	return l > r ? l : r;
}

// Crops
typedef enum
{
	FieldStage_Arable = 0,
	FieldStage_Fallow,
	FieldStage_Planted,
	FieldStage_Grown,
	FieldState_Max
} Field_Stage;

typedef struct
{
	float grown;
	float lifetime;
	uint32_t cropType;
} Field_Crop;

typedef struct
{
	Field_Crop* crop;
	Field_Stage stage;
	Vec2 pos;
} Field_Tile;

const uint32_t Field_Width = 1000;
const uint32_t Field_Height = 1000;
static Field_Tile* Field_Tiles = NULL;

const float Crop_MinLifetime = 1.0f;
const float Crop_MaxLifetime = 10.0f;
const uint32_t Crop_MaxCropType = 4;

void field_tick(float delta)
{
	MIST_PROFILE_BEGIN("Game", "Field-Tick");

	for (uint32_t i = 0; i < Field_Width * Field_Height; ++i)
	{
		if (Field_Tiles[i].stage == FieldStage_Planted)
		{
			Field_Crop* crop = Field_Tiles[i].crop;
			crop->lifetime += delta;

			if (crop->lifetime >= crop->grown)
			{
				Field_Tiles[i].stage = FieldStage_Grown;
			}
		}
	}
	MIST_PROFILE_END("Game", "Field-Tick");
}


// AI

typedef struct
{
	Vec2 pos;
	Vec2 vel;
	Vec2 tilePos;
	Field_Tile* tile;
} AI_FarmerMoveState;

typedef struct
{
	Vec2 pos;
	float farmTimer;
	Field_Tile* tile;
} AI_FarmerFarmState;

typedef struct
{
	Vec2 pos;
	float searchTimer;
} AI_FarmerSearchState;

const float AI_FarmerSpeed = 0.5f;
const float AI_FarmerCropRadius = 0.005f;
const float AI_FarmerSearchSpeedMin = 0.0f;
const float AI_FarmerSearchSpeedMax = 1.0f;
const float AI_FarmerFarmSpeedMin = 3.0f;
const float AI_FarmerFarmSpeedMax = 5.0f;
const uint32_t AI_FarmerCount = 1000000;

static uint32_t AI_FarmerMoveCount = 0;
static AI_FarmerMoveState* AI_FarmersMove = NULL;
static uint32_t AI_FarmerFarmCount = 0;
static AI_FarmerFarmState* AI_FarmersFarm = NULL;
static uint32_t AI_FarmerSearchCount = 0;
static AI_FarmerSearchState* AI_FarmersSearch = NULL;

void ai_tick(float delta)
{
	MIST_PROFILE_BEGIN("Game", "AI-Tick");

	uint32_t previousFarmerMoveCount = AI_FarmerMoveCount;
	{
		for (uint32_t i = 0; i < AI_FarmerSearchCount; i++)
		{
			AI_FarmerSearchState* farmer = &AI_FarmersSearch[i];

			farmer->searchTimer -= delta;
			farmer->searchTimer = math_maxf(farmer->searchTimer, 0.0f);

			if (farmer->searchTimer <= 0.0f)
			{
				uint32_t tileIndex = rand_range(0U, Field_Width * Field_Height);
				Field_Tile* tile = &Field_Tiles[tileIndex];

				if (tile->stage != FieldStage_Planted)
				{
					AI_FarmerMoveState* moveFarmer = &AI_FarmersMove[AI_FarmerMoveCount++];
					moveFarmer->tile = tile;
					moveFarmer->tilePos = tile->pos;
					moveFarmer->pos = farmer->pos;

					moveFarmer->vel = vec2_mul(vec2_norm(vec2_sub(tile->pos, farmer->pos)), AI_FarmerSpeed);

					SWAP(AI_FarmerSearchState, *farmer, AI_FarmersSearch[AI_FarmerSearchCount - 1]);
					AI_FarmerSearchCount--;
				}
				else
				{
					farmer->searchTimer = rand_rangef(AI_FarmerSearchSpeedMin, AI_FarmerSearchSpeedMax);
				}
			}
		}
	}

	uint32_t previousFarmerFarmCount = AI_FarmerFarmCount;
	{
		// We only do the farmers that were already in the move array before this tick
		for (uint32_t i = 0; i < previousFarmerMoveCount; i++)
		{
			AI_FarmerMoveState* farmer = &AI_FarmersMove[i];

			Vec2 tilePos = farmer->tilePos;
			float currentDist = vec2_mag(vec2_sub(tilePos, farmer->pos));
			Vec2 vel = vec2_mul(farmer->vel, delta);
			vel = vec2_mul(vec2_norm(vel), math_minf(vec2_mag(vel), currentDist));
			farmer->pos = vec2_add(farmer->pos, vel);

			float dist = vec2_mag(vec2_sub(tilePos, farmer->pos));
			if (dist < AI_FarmerCropRadius)
			{
				Field_Tile* tile = farmer->tile;

				AI_FarmerFarmState* farmFarmer = &AI_FarmersFarm[AI_FarmerFarmCount++];
				farmFarmer->farmTimer = rand_rangef(AI_FarmerFarmSpeedMin, AI_FarmerFarmSpeedMax);
				farmFarmer->tile = tile;
				farmFarmer->pos = farmer->pos;

				SWAP(AI_FarmerMoveState, *farmer, AI_FarmersMove[AI_FarmerMoveCount - 1]);
				AI_FarmerMoveCount--;
			}
		}
	}

	{
		// We only do the farmers that were already in the farm array before this tick
		for (uint32_t i = 0; i < previousFarmerFarmCount; i++)
		{
			AI_FarmerFarmState* farmer = &AI_FarmersFarm[i];

			farmer->farmTimer -= delta;
			if (farmer->farmTimer <= 0.0f)
			{
				Field_Tile* tile = farmer->tile;

				if (tile->stage == FieldStage_Grown)
				{
					free(tile->crop);
					tile->crop = NULL;
				}

				tile->stage = math_max((tile->stage + 1) % FieldState_Max, FieldStage_Fallow);

				if (tile->stage == FieldStage_Planted)
				{
					tile->crop = (Field_Crop*)malloc(sizeof(Field_Crop));
					tile->crop->grown = rand_rangef(Crop_MinLifetime, Crop_MaxLifetime);
					tile->crop->cropType = rand_range(0, Crop_MaxCropType);
				}

				AI_FarmerSearchState* searchFarmer = &AI_FarmersSearch[AI_FarmerSearchCount++];
				searchFarmer->searchTimer = rand_rangef(AI_FarmerSearchSpeedMin, AI_FarmerSearchSpeedMax);
				searchFarmer->pos = farmer->pos;

				SWAP(AI_FarmerFarmState, *farmer, AI_FarmersFarm[AI_FarmerFarmCount - 1]);
				AI_FarmerFarmCount--;
			}
		}
	}

	MIST_PROFILE_END("Game", "AI-Tick");
}

// Game

const float FarmerState_Search = 0.0f;
const float FarmerState_Move = 1.0f;
const float FarmerState_Farm = 2.0f;

float Game_FieldImageTable[] =
{
	[FieldStage_Arable] = 3.0f,
	[FieldStage_Fallow] = 4.0f,
	[FieldStage_Planted] = 5.0f,
	[FieldStage_Grown] = 6.0f
};

void game_init(void)
{
	MIST_PROFILE_BEGIN("Game", "Game-Init");

#ifndef PROFILE_MODE
	srand((unsigned int)time(NULL));
#endif // PROFILE_MODE

	Field_Tiles = (Field_Tile*)malloc(sizeof(Field_Tile) * Field_Width * Field_Height);
	memset(Field_Tiles, 0, sizeof(Field_Tile) * Field_Width * Field_Height);

	for (uint32_t y = 0; y < Field_Height; ++y)
	{
		for (uint32_t x = 0; x < Field_Width; ++x)
		{
			Field_Tiles[y * Field_Width + x].pos = (Vec2) { .x = (float)x / Field_Width, .y = (float)y / Field_Height };
			Field_Tiles[y * Field_Width + x].pos = vec2_sub(vec2_mul(Field_Tiles[y * Field_Width + x].pos, 2.0f), (Vec2) { .x = 1.0f, .y = 1.0f });
		}
	}

	AI_FarmersMove = (AI_FarmerMoveState*)malloc(sizeof(AI_FarmerMoveState) * AI_FarmerCount);
	AI_FarmersFarm = (AI_FarmerFarmState*)malloc(sizeof(AI_FarmerFarmState) * AI_FarmerCount);
	AI_FarmersSearch = (AI_FarmerSearchState*)malloc(sizeof(AI_FarmerSearchState) * AI_FarmerCount);
	AI_FarmerSearchCount = AI_FarmerCount;

	for (uint32_t ai = 0; ai < AI_FarmerCount; ++ai)
	{
		AI_FarmerSearchState* farmer = &AI_FarmersSearch[ai];
		farmer->searchTimer = rand_rangef(AI_FarmerSearchSpeedMin, AI_FarmerSearchSpeedMax);
		farmer->pos = (Vec2) { .x = 0.0f, .y = 0.0f };
	}

	MIST_PROFILE_END("Game", "Game-Init");
}

void game_tick(float delta)
{
	MIST_PROFILE_BEGIN("Game", "Game-Tick");

	field_tick(delta);
	ai_tick(delta);

	MIST_PROFILE_END("Game", "Game-Tick");
}

void game_kill(void)
{
	MIST_PROFILE_BEGIN("Game", "Game-Kill");

	for (uint32_t i = 0; i < Field_Width * Field_Height; ++i)
	{
		free(Field_Tiles[i].crop);
		Field_Tiles[i].crop = NULL;
	}

	free(Field_Tiles);
	Field_Tiles = NULL;
	free(AI_FarmersMove);
	AI_FarmersMove = NULL;
	free(AI_FarmersFarm);
	AI_FarmersFarm = NULL;
	free(AI_FarmersSearch);
	AI_FarmersSearch = NULL;

	MIST_PROFILE_END("Game", "Game-Kill");
}

uint32_t game_gen_instance_buffer(Game_InstanceBuffer* buffer)
{
	MIST_PROFILE_BEGIN("Game", "Game-GenInstanceBuffer");

	uint32_t writeIndex = 0;
	for (uint32_t i = 0; i < Field_Width * Field_Height; ++i)
	{
		uint32_t writeLoc = writeIndex++;
		buffer->instances[writeLoc].spriteIndex = Game_FieldImageTable[Field_Tiles[i].stage];
		buffer->instances[writeLoc].scale = 2.0f / Field_Width;
		buffer->instances[writeLoc].pos[0] = Field_Tiles[i].pos.x;
		buffer->instances[writeLoc].pos[1] = Field_Tiles[i].pos.y;

		if (Field_Tiles[i].crop != NULL)
		{
			uint32_t cropWriteIndex = writeIndex++;
			buffer->instances[cropWriteIndex].spriteIndex = 7.0f + Field_Tiles[i].crop->cropType;
			buffer->instances[cropWriteIndex].scale = 2.0f / Field_Width;

			buffer->instances[cropWriteIndex].pos[0] = Field_Tiles[i].pos.x;
			buffer->instances[cropWriteIndex].pos[1] = Field_Tiles[i].pos.y;
		}
	}

	for (uint32_t i = 0; i < AI_FarmerSearchCount; ++i)
	{
		uint32_t writeLoc = writeIndex++;
		buffer->instances[writeLoc].spriteIndex = FarmerState_Search;
		buffer->instances[writeLoc].scale = 0.025f;
		buffer->instances[writeLoc].pos[0] = AI_FarmersSearch[i].pos.x;
		buffer->instances[writeLoc].pos[1] = AI_FarmersSearch[i].pos.y;
	}

	for (uint32_t i = 0; i < AI_FarmerMoveCount; ++i)
	{
		uint32_t writeLoc = writeIndex++;
		buffer->instances[writeLoc].spriteIndex = FarmerState_Move;
		buffer->instances[writeLoc].scale = 0.025f;
		buffer->instances[writeLoc].pos[0] = AI_FarmersMove[i].pos.x;
		buffer->instances[writeLoc].pos[1] = AI_FarmersMove[i].pos.y;
	}

	for (uint32_t i = 0; i < AI_FarmerFarmCount; ++i)
	{
		uint32_t writeLoc = writeIndex++;
		buffer->instances[writeLoc].spriteIndex = FarmerState_Farm;
		buffer->instances[writeLoc].scale = 0.025f;
		buffer->instances[writeLoc].pos[0] = AI_FarmersFarm[i].pos.x;
		buffer->instances[writeLoc].pos[1] = AI_FarmersFarm[i].pos.y;
	}

	MIST_PROFILE_END("Game", "Game-GenInstanceBuffer");

	return writeIndex;
}
