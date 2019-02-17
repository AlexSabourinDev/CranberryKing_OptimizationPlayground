#include "game.h"

#define MIST_PROFILE_ENABLED
#include "3rd/Mist_Profiler.h"

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include <xmmintrin.h>
#include <emmintrin.h>

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
	uint32_t tileIndex;
	Vec2 pos;
} Field_Crop;

typedef struct
{
	uint32_t writeIndex;
	int16_t spriteIndex;
	Vec2 pos;
} Field_CropDrawCommand;

static uint32_t Field_CropDrawCommandCount = 0;
static Field_CropDrawCommand* Field_CropDrawCommands = NULL;

static uint32_t Field_CropCount = 0;
static Field_Crop* Field_Crops = NULL;

typedef struct
{
	Field_Stage stage;
	Vec2 pos;
} Field_Tile;

typedef struct
{
	uint32_t writeIndex;
	int16_t spriteIndex;
} Field_TileDrawCommand;

int16_t Field_ImageTable[] =
{
	[FieldStage_Arable] = 3.0f / 11.0f * INT16_MAX,
	[FieldStage_Fallow] = 4.0f / 11.0f * INT16_MAX,
	[FieldStage_Planted] = 5.0f / 11.0f * INT16_MAX,
	[FieldStage_Grown] = 6.0f / 11.0f * INT16_MAX
};

static uint32_t Field_TileDrawCommandCount = 0;
static Field_TileDrawCommand* Field_TileDrawCommands = NULL;

#define Field_Width 1000
#define Field_Height 1000
static Field_Tile* Field_Tiles = NULL;

const int16_t Field_TileScale = (2.0f / Field_Width) * INT16_MAX;

const float Crop_MinLifetime = 1.0f;
const float Crop_MaxLifetime = 10.0f;
const uint32_t Crop_MaxCropType = 4;

void field_tick(float delta)
{
	MIST_PROFILE_BEGIN("Game", "Field-Tick");

	for (uint32_t i = 0; i < Field_CropCount; ++i)
	{
		Field_Crop* crop = &Field_Crops[i];
		crop->lifetime += delta;

		if (crop->lifetime >= crop->grown)
		{
			Field_Tile* tile = &Field_Tiles[crop->tileIndex];
			tile->stage = FieldStage_Grown;

			Field_TileDrawCommands[Field_TileDrawCommandCount] = (Field_TileDrawCommand)
				{
					.writeIndex = crop->tileIndex,
					.spriteIndex = Field_ImageTable[FieldStage_Grown]
				};
			Field_TileDrawCommandCount++;

			SWAP(Field_Crop, *crop, Field_Crops[Field_CropCount - 1]);
			Field_CropCount--;

			Field_CropDrawCommands[Field_CropDrawCommandCount] = (Field_CropDrawCommand)
				{
					.writeIndex = i,
					.spriteIndex = (7.0f + crop->cropType) / 11.0f * INT16_MAX,
					.pos = crop->pos
				};
			Field_CropDrawCommandCount++;

			i--;
		}
	}
	MIST_PROFILE_END("Game", "Field-Tick");
}


// AI

typedef struct
{
	Vec2 tilePos;
} AI_FarmerMoveStateHot;

typedef struct
{
	uint32_t tileIndex;
} AI_FarmerMoveStateCold;

typedef struct
{
	Vec2 pos;
} AI_FarmerMoveStateGen;

typedef struct
{
	float farmTimer;
} AI_FarmerFarmStateHot;

typedef struct
{
	uint32_t tileIndex;
} AI_FarmerFarmStateCold;

typedef struct
{
	Vec2 pos;
} AI_FarmerFarmStateGen;

typedef struct
{
	float searchTimer;
} AI_FarmerSearchStateHot;

typedef struct
{
	Vec2 pos;
} AI_FarmerSearchStateGen;

const int16_t AI_FarmerScale = 0.025f * INT16_MAX;
const int16_t FarmerState_Search = 0;
const int16_t FarmerState_Move = 1.0f / 11.0f * INT16_MAX;
const int16_t FarmerState_Farm = 2.0f / 11.0f * INT16_MAX;

const float AI_FarmerSpeed = 0.5f;
const float AI_FarmerCropRadius = 0.005f;
const float AI_FarmerSearchSpeedMin = 0.0f;
const float AI_FarmerSearchSpeedMax = 1.0f;
const float AI_FarmerFarmSpeedMin = 3.0f;
const float AI_FarmerFarmSpeedMax = 5.0f;
const uint32_t AI_FarmerCount = 1000000;

static uint32_t AI_FarmerMoveCount = 0;
static AI_FarmerMoveStateHot* AI_FarmersMoveHot = NULL;
static AI_FarmerMoveStateCold* AI_FarmersMoveCold = NULL;
static AI_FarmerMoveStateGen* AI_FarmersMoveGen = NULL;
static uint32_t AI_FarmerFarmCount = 0;
static AI_FarmerFarmStateHot* AI_FarmersFarmHot = NULL;
static AI_FarmerFarmStateCold* AI_FarmersFarmCold = NULL;
static AI_FarmerFarmStateGen* AI_FarmersFarmGen = NULL;
static uint32_t AI_FarmerSearchCount = 0;
static AI_FarmerSearchStateHot* AI_FarmersSearchHot = NULL;
static AI_FarmerSearchStateGen* AI_FarmersSearchGen = NULL;

void ai_tick(float delta)
{
	MIST_PROFILE_BEGIN("Game", "AI-Tick");

	uint32_t previousFarmerMoveCount = AI_FarmerMoveCount;
	{
		for (uint32_t i = 0; i < AI_FarmerSearchCount; i++)
		{
			AI_FarmerSearchStateHot* farmer = &AI_FarmersSearchHot[i];

			farmer->searchTimer -= delta;
			farmer->searchTimer = math_maxf(farmer->searchTimer, 0.0f);

			if (farmer->searchTimer <= 0.0f)
			{
				uint32_t tileIndex = rand_range(0U, Field_Width * Field_Height);
				Field_Tile* tile = &Field_Tiles[tileIndex];

				if (tile->stage != FieldStage_Planted)
				{
					AI_FarmerSearchStateGen* genFarmer = &AI_FarmersSearchGen[i];

					AI_FarmerMoveStateHot* moveFarmerHot = &AI_FarmersMoveHot[AI_FarmerMoveCount];
					AI_FarmerMoveStateCold* moveFarmerCold = &AI_FarmersMoveCold[AI_FarmerMoveCount];
					AI_FarmerMoveStateGen* moveFarmerGen = &AI_FarmersMoveGen[AI_FarmerMoveCount];
					AI_FarmerMoveCount++;
					moveFarmerCold->tileIndex = tileIndex;
					moveFarmerHot->tilePos = tile->pos;
					moveFarmerGen->pos = genFarmer->pos;

					SWAP(AI_FarmerSearchStateHot, *farmer, AI_FarmersSearchHot[AI_FarmerSearchCount - 1]);
					SWAP(AI_FarmerSearchStateGen, *genFarmer, AI_FarmersSearchGen[AI_FarmerSearchCount - 1]);
					AI_FarmerSearchCount--;
					i--;
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
			AI_FarmerMoveStateHot* farmer = &AI_FarmersMoveHot[i];
			AI_FarmerMoveStateGen* genFarmer = &AI_FarmersMoveGen[i];

			Vec2 tilePos = farmer->tilePos;

			float velMag = AI_FarmerSpeed * delta;
			Vec2 dirVec = vec2_sub(tilePos, genFarmer->pos);
			float mag = vec2_mag(dirVec);

			Vec2 vel = vec2_mul(dirVec, (1.0f / mag) * velMag);
			genFarmer->pos = vec2_add(genFarmer->pos, vel);

			if (velMag > mag)
			{
				genFarmer->pos = farmer->tilePos;

				AI_FarmerMoveStateCold* coldFarmer = &AI_FarmersMoveCold[i];

				AI_FarmerFarmStateHot* farmFarmerHot = &AI_FarmersFarmHot[AI_FarmerFarmCount];
				AI_FarmerFarmStateCold* farmFarmerCold = &AI_FarmersFarmCold[AI_FarmerFarmCount];
				AI_FarmerFarmStateGen* farmFarmerGen = &AI_FarmersFarmGen[AI_FarmerFarmCount];
				AI_FarmerFarmCount++;
				farmFarmerHot->farmTimer = rand_rangef(AI_FarmerFarmSpeedMin, AI_FarmerFarmSpeedMax);
				farmFarmerCold->tileIndex = coldFarmer->tileIndex;
				farmFarmerGen->pos = genFarmer->pos;

				SWAP(AI_FarmerMoveStateHot, *farmer, AI_FarmersMoveHot[AI_FarmerMoveCount - 1]);
				SWAP(AI_FarmerMoveStateCold, *coldFarmer, AI_FarmersMoveCold[AI_FarmerMoveCount - 1]);
				SWAP(AI_FarmerMoveStateGen, *genFarmer, AI_FarmersMoveGen[AI_FarmerMoveCount - 1]);
				AI_FarmerMoveCount--;
				i--;
			}
		}
	}

	{
		// We only do the farmers that were already in the farm array before this tick
		for (uint32_t i = 0; i < previousFarmerFarmCount; i++)
		{
			AI_FarmerFarmStateHot* farmer = &AI_FarmersFarmHot[i];

			farmer->farmTimer -= delta;
			if (farmer->farmTimer <= 0.0f)
			{
				AI_FarmerFarmStateCold* coldFarmer = &AI_FarmersFarmCold[i];
				Field_Tile* tile = &Field_Tiles[coldFarmer->tileIndex];
				tile->stage = math_max((tile->stage + 1) % FieldState_Max, FieldStage_Fallow);

				Field_TileDrawCommands[Field_TileDrawCommandCount] = (Field_TileDrawCommand)
					{
						.writeIndex = coldFarmer->tileIndex,
						.spriteIndex = Field_ImageTable[tile->stage]
					};
				Field_TileDrawCommandCount++;

				if (tile->stage == FieldStage_Planted)
				{
					Field_Crop* crop = &Field_Crops[Field_CropCount];
					crop->grown = rand_rangef(Crop_MinLifetime, Crop_MaxLifetime);
					crop->lifetime = 0.0f;
					crop->cropType = rand_range(0, Crop_MaxCropType);
					crop->tileIndex = coldFarmer->tileIndex;
					crop->pos = tile->pos;

					Field_CropDrawCommands[Field_CropDrawCommandCount] = (Field_CropDrawCommand)
						{
							.writeIndex = Field_CropCount,
							.spriteIndex = (7.0f + crop->cropType) / 11.0f * INT16_MAX,
							.pos = crop->pos
						};
					Field_CropDrawCommandCount++;

					Field_CropCount++;
				}

				AI_FarmerSearchStateHot* searchFarmerHot = &AI_FarmersSearchHot[AI_FarmerSearchCount];
				AI_FarmerSearchStateGen* searchFarmerGen = &AI_FarmersSearchGen[AI_FarmerSearchCount];
				AI_FarmerSearchCount++;

				AI_FarmerFarmStateGen* genFarmer = &AI_FarmersFarmGen[i];
				searchFarmerHot->searchTimer = rand_rangef(AI_FarmerSearchSpeedMin, AI_FarmerSearchSpeedMax);
				searchFarmerGen->pos = genFarmer->pos;

				SWAP(AI_FarmerFarmStateHot, *farmer, AI_FarmersFarmHot[AI_FarmerFarmCount - 1]);
				SWAP(AI_FarmerFarmStateCold, *coldFarmer, AI_FarmersFarmCold[AI_FarmerFarmCount - 1]);
				SWAP(AI_FarmerFarmStateGen, *genFarmer, AI_FarmersFarmGen[AI_FarmerFarmCount - 1]);
				AI_FarmerFarmCount--;
				i--;
			}
		}
	}

	MIST_PROFILE_END("Game", "AI-Tick");
}

// Game

void game_init(Game_InstanceBuffer* buffer)
{
	MIST_PROFILE_BEGIN("Game", "Game-Init");

#ifndef PROFILE_MODE
	srand((unsigned int)time(NULL));
#endif // PROFILE_MODE

	Field_Tiles = (Field_Tile*)malloc(sizeof(Field_Tile) * Field_Width * Field_Height);
	memset(Field_Tiles, 0, sizeof(Field_Tile) * Field_Width * Field_Height);
	Field_TileDrawCommands = (Field_TileDrawCommand*)malloc(sizeof(Field_TileDrawCommand) * Field_Width * Field_Height);

	Field_Crops = (Field_Crop*)malloc(sizeof(Field_Crop) * Field_Width * Field_Height);
	Field_CropDrawCommands = (Field_CropDrawCommand*)malloc(sizeof(Field_CropDrawCommand) * Field_Width * Field_Height);

	for (uint32_t y = 0; y < Field_Height; ++y)
	{
		for (uint32_t x = 0; x < Field_Width; ++x)
		{
			uint32_t writeLoc = y * Field_Width + x;
			Vec2 pos = (Vec2){ .x = (float)x / Field_Width, .y = (float)y / Field_Height };
			Field_Tiles[writeLoc].pos = vec2_sub(vec2_mul(pos, 2.0f), (Vec2) { .x = 1.0f, .y = 1.0f });

			buffer->spriteIndicesAndScales[writeLoc * 2] = Field_ImageTable[0];
			buffer->spriteIndicesAndScales[writeLoc * 2 + 1] = Field_TileScale;
			memcpy(&buffer->positions[writeLoc * 2], &Field_Tiles[writeLoc].pos, sizeof(int16_t) * 2);
		}
	}

	AI_FarmersMoveHot = (AI_FarmerMoveStateHot*)malloc(sizeof(AI_FarmerMoveStateHot) * AI_FarmerCount);
	AI_FarmersMoveCold = (AI_FarmerMoveStateCold*)malloc(sizeof(AI_FarmerMoveStateCold) * AI_FarmerCount);
	AI_FarmersMoveGen = (AI_FarmerMoveStateGen*)malloc(sizeof(AI_FarmerMoveStateGen) * AI_FarmerCount);

	AI_FarmersFarmHot = (AI_FarmerFarmStateHot*)malloc(sizeof(AI_FarmerFarmStateHot) * AI_FarmerCount);
	AI_FarmersFarmCold = (AI_FarmerFarmStateCold*)malloc(sizeof(AI_FarmerFarmStateCold) * AI_FarmerCount);
	AI_FarmersFarmGen = (AI_FarmerFarmStateGen*)malloc(sizeof(AI_FarmerFarmStateGen) * AI_FarmerCount);

	AI_FarmersSearchHot = (AI_FarmerSearchStateHot*)malloc(sizeof(AI_FarmerSearchStateHot) * AI_FarmerCount);
	AI_FarmersSearchGen = (AI_FarmerSearchStateGen*)malloc(sizeof(AI_FarmerSearchStateGen) * AI_FarmerCount);

	AI_FarmerSearchCount = AI_FarmerCount;

	for (uint32_t ai = 0; ai < AI_FarmerCount; ++ai)
	{
		AI_FarmersSearchHot[ai].searchTimer = rand_rangef(AI_FarmerSearchSpeedMin, AI_FarmerSearchSpeedMax);
		AI_FarmersSearchGen[ai].pos = (Vec2) { .x = 0.0f, .y = 0.0f };
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

	free(Field_Crops);
	Field_Crops = NULL;

	free(Field_CropDrawCommands);
	Field_CropDrawCommands = NULL;

	free(Field_Tiles);
	Field_Tiles = NULL;

	free(Field_TileDrawCommands);
	Field_TileDrawCommands = NULL;

	free(AI_FarmersMoveHot);
	AI_FarmersMoveHot = NULL;
	free(AI_FarmersMoveCold);
	AI_FarmersMoveCold = NULL;
	free(AI_FarmersMoveGen);
	AI_FarmersMoveGen = NULL;

	free(AI_FarmersFarmHot);
	AI_FarmersFarmHot = NULL;
	free(AI_FarmersFarmCold);
	AI_FarmersFarmCold = NULL;
	free(AI_FarmersFarmGen);
	AI_FarmersFarmCold = NULL;

	free(AI_FarmersSearchHot);
	AI_FarmersSearchHot = NULL;
	free(AI_FarmersSearchGen);
	AI_FarmersSearchGen = NULL;

	MIST_PROFILE_END("Game", "Game-Kill");
}

uint32_t game_gen_instance_buffer(Game_InstanceBuffer* buffer)
{
	MIST_PROFILE_BEGIN("Game", "Game-GenInstanceBuffer");

	for (uint32_t i = 0; i < Field_TileDrawCommandCount; ++i)
	{
		Field_TileDrawCommand* command = &Field_TileDrawCommands[i];
		buffer->spriteIndicesAndScales[command->writeIndex * 2] = command->spriteIndex;
	}
	Field_TileDrawCommandCount = 0;

	for (uint32_t i = 0; i < Field_CropDrawCommandCount; i++)
	{
		Field_CropDrawCommand* command = &Field_CropDrawCommands[i];
		buffer->spriteIndicesAndScales[command->writeIndex * 2] = command->spriteIndex;
		buffer->spriteIndicesAndScales[command->writeIndex * 2 + 1] = Field_TileScale;
		memcpy(&buffer->positions[command->writeIndex * 2], &command->pos, sizeof(float) * 2);
	}
	Field_CropDrawCommandCount = 0;

	uint32_t writeIndex = Field_Width * Field_Height + Field_CropCount;

	memcpy(&buffer->positions[writeIndex * 2], AI_FarmersSearchGen, sizeof(float) * 2 * AI_FarmerSearchCount);
	__m128i searchAndScale = _mm_set_epi16(FarmerState_Search, AI_FarmerScale, FarmerState_Search, AI_FarmerScale, FarmerState_Search, AI_FarmerScale, FarmerState_Search, AI_FarmerScale);
	for (uint32_t i = 0; i < AI_FarmerSearchCount; i+=4)
	{
		_mm_storeu_si128((__m128i*)&buffer->spriteIndicesAndScales[(writeIndex + i) * 2], searchAndScale);
	}
	writeIndex += AI_FarmerSearchCount;

	memcpy(&buffer->positions[writeIndex * 2], AI_FarmersMoveGen, sizeof(float) * 2 * AI_FarmerMoveCount);
	__m128i moveAndScale = _mm_set_epi16(FarmerState_Move, AI_FarmerScale, FarmerState_Move, AI_FarmerScale, FarmerState_Move, AI_FarmerScale, FarmerState_Move, AI_FarmerScale);
	for (uint32_t i = 0; i < AI_FarmerMoveCount; i+=4)
	{
		_mm_storeu_si128((__m128i*)&buffer->spriteIndicesAndScales[(writeIndex + i) * 2], moveAndScale);
	}
	writeIndex += AI_FarmerMoveCount;

	memcpy(&buffer->positions[writeIndex * 2], AI_FarmersFarmGen, sizeof(float) * 2 * AI_FarmerFarmCount);
	__m128i farmAndScale = _mm_set_epi16(FarmerState_Farm, AI_FarmerScale, FarmerState_Farm, AI_FarmerScale, FarmerState_Farm, AI_FarmerScale, FarmerState_Farm, AI_FarmerScale);
	for (uint32_t i = 0; i < AI_FarmerFarmCount; i+=4)
	{
		_mm_storeu_si128((__m128i*)&buffer->spriteIndicesAndScales[(writeIndex + i) * 2], farmAndScale);
	}
	writeIndex += AI_FarmerFarmCount;

	MIST_PROFILE_END("Game", "Game-GenInstanceBuffer");

	return writeIndex;
}
