#include "game.h"

#include "3rd/Mist_Profiler.h"

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

// Util
#define swap(type, a, b) do { type t = a; a = b; b = t;}while(0)

// Random
uint32_t rand_range(uint32_t min, uint32_t max)
{
    return (uint32_t)(((float)rand() / RAND_MAX) * (max - min) + min);
}

float rand_rangef(float min, float max)
{
    return ((float)rand() / RAND_MAX) * (max - min) + min;
}

// Math
typedef struct
{
    float x, y;
} Vec2;

Vec2 vec2_add(Vec2 l, Vec2 r)
{
    return (Vec2){ .x = l.x + r.x, .y = l.y + r.y };
}

Vec2 vec2_sub(Vec2 l, Vec2 r)
{
    return (Vec2) { .x = l.x - r.x, .y = l.y - r.y };
}

Vec2 vec2_mul(Vec2 v, float s)
{
    return (Vec2){ .x = v.x * s, .y = v.y * s };
}

float vec2_mag(Vec2 v)
{
    return sqrtf(v.x * v.x + v.y * v.y);
}

float vec2_magSqr(Vec2 v)
{
	return (v.x * v.x + v.y * v.y);
}

Vec2 vec2_norm(Vec2 v)
{
    float mag = vec2_mag(v);
    if (mag == 0.0f) return (Vec2) {.x = 0.0f, .y = 0.0f};

    return vec2_mul(v, 1.0f / mag);
}

typedef struct
{
	int16_t x, y;
} QuantizedVec2;

Vec2 quantizedVec2_Read(QuantizedVec2 v)
{
	const float r = 1.0f / INT16_MAX;
	return (Vec2){ ((float)v.x * r), ((float)v.y * r) };
}

QuantizedVec2 quantizedVec2_Write(Vec2 v)
{
	return (QuantizedVec2) { (int16_t)(v.x * INT16_MAX), (int16_t)(v.y * INT16_MAX) };
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
} Field_CropHot;

typedef struct
{
	uint32_t fieldTile;
} Field_CropCold;

typedef struct
{
	QuantizedVec2 pos;
	uint8_t cropType;
} Field_CropGen;

typedef struct
{
	QuantizedVec2 pos;
	uint8_t stage;
} Field_TileGen;

typedef struct
{
	uint32_t fieldCrop;
} Field_TileCold;

const uint32_t Field_Width  = 500;
const uint32_t Field_Height = 500;
static Field_TileCold* Field_TilesCold = NULL;
static Field_TileGen* Field_TilesGen = NULL;

static Field_CropHot* Field_CropsHot = NULL;
static Field_CropCold* Field_CropsCold = NULL;
static Field_CropGen* Field_CropsGen = NULL;
static uint32_t Field_CropCount = 0;

const float Crop_MinLifetime = 1.0f;
const float Crop_MaxLifetime = 10.0f;
const uint32_t Crop_MaxCropType = 4;

void field_tick(float delta)
{
	MIST_PROFILE_BEGIN("Game", "Field_Tick");
    for (uint32_t i = 0; i < Field_CropCount; ++i)
    {
        Field_CropHot* crop = &Field_CropsHot[i];
        crop->lifetime += delta;

        if (crop->lifetime >= crop->grown)
        {
			Field_CropCold* cropCold = &Field_CropsCold[i];
            Field_TilesGen[cropCold->fieldTile].stage = FieldStage_Grown;
        }
    }
	MIST_PROFILE_END("Game", "Field_Tick");
}


// AI
typedef enum
{
	FarmerState_Search = 0,
	FarmerState_Move,
	FarmerState_Farm
} FarmerState;

typedef struct
{
	Vec2 vel;
	Vec2 targetPos;
} AI_FarmerMoveHot;

typedef struct
{
	Vec2 pos;
} AI_FarmerMovePos;

typedef struct
{
	uint32_t tileIndex;
} AI_FarmerMoveCold;

typedef struct
{
	float farmTimer;
} AI_FarmerFarmHot;

typedef struct
{
	uint32_t tileIndex;
} AI_FarmerFarmCold;

typedef struct
{
	float searchTimer;
} AI_FarmerSearchHot;

typedef struct
{
	QuantizedVec2 pos;
} AI_FarmerSearchCold;

const float AI_FarmerSpeed = 0.5f;
const float AI_FarmerCropRadius = 0.005f;
const float AI_FarmerSearchSpeedMin = 0.0f;
const float AI_FarmerSearchSpeedMax = 1.0f;
const float AI_FarmerFarmSpeedMin = 3.0f;
const float AI_FarmerFarmSpeedMax = 5.0f;
const uint32_t AI_FarmerCount = 1000000;

static AI_FarmerSearchHot* AI_FarmersSearchHot = NULL;
static AI_FarmerSearchCold* AI_FarmersSearchCold = NULL;
static uint32_t AI_FarmerSearchCount = 0;

static AI_FarmerMoveHot* AI_FarmersMoveHot = NULL;
static AI_FarmerMoveCold* AI_FarmersMoveCold = NULL;
static AI_FarmerMovePos* AI_FarmersMovePos = NULL;
static uint32_t AI_FarmerMoveCount = 0;

static AI_FarmerFarmHot* AI_FarmersFarmHot = NULL;
static AI_FarmerFarmCold* AI_FarmersFarmCold = NULL;
static QuantizedVec2* AI_FarmersFarmPos = NULL;
static uint32_t AI_FarmerFarmCount = 0;

uint32_t ai_tick_search(float delta)
{
	MIST_PROFILE_BEGIN("Game", "Ai_Search");
	uint32_t previousMoveCount = AI_FarmerMoveCount;
	for (uint32_t ai = 0; ai < AI_FarmerSearchCount; ++ai)
	{
		AI_FarmerSearchHot* farmer = &AI_FarmersSearchHot[ai];
		farmer->searchTimer -= delta;
		farmer->searchTimer = math_maxf(farmer->searchTimer, 0.0f);

		if (farmer->searchTimer <= 0.0f)
		{
			uint32_t tileIndex = rand_range(0U, Field_Width * Field_Height);
			Field_TileGen* tile = &Field_TilesGen[tileIndex];

			if (tile->stage != FieldStage_Planted)
			{
				AI_FarmerSearchCold* farmerCold = &AI_FarmersSearchCold[ai];

				AI_FarmerMoveHot* moveFarmerHot = &AI_FarmersMoveHot[AI_FarmerMoveCount];
				AI_FarmerMoveCold* moveFarmerCold = &AI_FarmersMoveCold[AI_FarmerMoveCount];
				AI_FarmerMovePos* moveFarmerPos = &AI_FarmersMovePos[AI_FarmerMoveCount];
				AI_FarmerMoveCount++;
				moveFarmerCold->tileIndex = tileIndex;
				Vec2 farmerPos = quantizedVec2_Read(farmerCold->pos);
				moveFarmerPos->pos = farmerPos;

				Vec2 tilePos = quantizedVec2_Read(tile->pos);
				moveFarmerHot->vel = vec2_mul(vec2_norm(vec2_sub(tilePos, farmerPos)), AI_FarmerSpeed);
				moveFarmerHot->targetPos = tilePos;

				swap(AI_FarmerSearchHot, *farmer, AI_FarmersSearchHot[AI_FarmerSearchCount - 1]);
				swap(AI_FarmerSearchCold, *farmerCold, AI_FarmersSearchCold[AI_FarmerSearchCount - 1]);
				AI_FarmerSearchCount--;
				ai--;
			}
		}
	}
	MIST_PROFILE_END("Game", "Ai_Search");

	return previousMoveCount;
}

uint32_t ai_tick_move(float delta, uint32_t previousMoveCount)
{
	MIST_PROFILE_BEGIN("Game", "Ai_Move");
	uint32_t previousFarmCount = AI_FarmerFarmCount;
	for (uint32_t ai = 0; ai < previousMoveCount; ++ai)
	{
		AI_FarmerMoveHot* farmer = &AI_FarmersMoveHot[ai];
		AI_FarmerMovePos* farmerPosInstance = &AI_FarmersMovePos[ai];
		Vec2 vel = vec2_mul(farmer->vel, delta);

		Vec2 farmerPos = farmerPosInstance->pos;

		float velMag = delta * AI_FarmerSpeed;
		float dist = vec2_magSqr(vec2_sub(farmer->targetPos, farmerPos));

		farmerPosInstance->pos = vec2_add(farmerPos, vel);
		if (velMag * velMag > dist)
		{
			AI_FarmerMoveCold* farmerCold = &AI_FarmersMoveCold[ai];

			AI_FarmerFarmHot* farmFarmerHot = &AI_FarmersFarmHot[AI_FarmerFarmCount];
			AI_FarmerFarmCold* farmFarmerCold = &AI_FarmersFarmCold[AI_FarmerFarmCount];
			QuantizedVec2* farmFarmerPos = &AI_FarmersFarmPos[AI_FarmerFarmCount++];

			farmFarmerHot->farmTimer = rand_rangef(AI_FarmerFarmSpeedMin, AI_FarmerFarmSpeedMax);
			farmFarmerCold->tileIndex = farmerCold->tileIndex;
			*farmFarmerPos = quantizedVec2_Write(farmer->targetPos);

			// Swap with an item we want to process
			swap(AI_FarmerMoveHot, *farmer, AI_FarmersMoveHot[previousMoveCount - 1]);
			swap(AI_FarmerMoveCold, *farmerCold, AI_FarmersMoveCold[previousMoveCount - 1]);
			swap(AI_FarmerMovePos, *farmerPosInstance, AI_FarmersMovePos[previousMoveCount - 1]);
			previousMoveCount--;

			// Swap with the end to remove it from the list
			swap(AI_FarmerMoveHot, AI_FarmersMoveHot[previousMoveCount], AI_FarmersMoveHot[AI_FarmerMoveCount - 1]);
			swap(AI_FarmerMoveCold, AI_FarmersMoveCold[previousMoveCount], AI_FarmersMoveCold[AI_FarmerMoveCount - 1]);
			swap(AI_FarmerMovePos, AI_FarmersMovePos[previousMoveCount], AI_FarmersMovePos[AI_FarmerMoveCount - 1]);
			AI_FarmerMoveCount--;

			ai--;
		}
	}
	MIST_PROFILE_END("Game", "Ai_Move");
	return previousFarmCount;
}

void ai_tick_farm(float delta, uint32_t previousFarmCount)
{
	MIST_PROFILE_BEGIN("Game", "Ai_Farm");
	for (uint32_t ai = 0; ai < previousFarmCount; ++ai)
	{
		AI_FarmerFarmHot* farmer = &AI_FarmersFarmHot[ai];

		farmer->farmTimer -= delta;
		if (farmer->farmTimer <= 0.0f)
		{
			AI_FarmerFarmCold* farmerCold = &AI_FarmersFarmCold[ai];
			Field_TileGen* tile = &Field_TilesGen[farmerCold->tileIndex];
			Field_TileCold* tileCold = &Field_TilesCold[farmerCold->tileIndex];

			if (tile->stage == FieldStage_Grown)
			{
				Field_CropHot* crop = &Field_CropsHot[tileCold->fieldCrop];
				swap(Field_CropHot, *crop, Field_CropsHot[Field_CropCount - 1]);

				Field_CropCold* cropCold = &Field_CropsCold[tileCold->fieldCrop];
				swap(Field_CropCold, *cropCold, Field_CropsCold[Field_CropCount - 1]);

				Field_CropGen* cropGen = &Field_CropsGen[tileCold->fieldCrop];
				swap(Field_CropGen, *cropGen, Field_CropsGen[Field_CropCount - 1]);

				Field_TilesCold[cropCold->fieldTile].fieldCrop = tileCold->fieldCrop;

				Field_CropCount--;
			}

			tile->stage = math_max((tile->stage + 1) % FieldState_Max, FieldStage_Fallow);

			if (tile->stage == FieldStage_Planted)
			{
				Field_CropHot* crop = &Field_CropsHot[Field_CropCount];
				crop->grown = rand_rangef(Crop_MinLifetime, Crop_MaxLifetime);

				Field_CropCold* cropCold = &Field_CropsCold[Field_CropCount];
				cropCold->fieldTile = farmerCold->tileIndex;

				Field_CropGen* cropGen = &Field_CropsGen[Field_CropCount];
				cropGen->cropType = rand_range(0, Crop_MaxCropType);
				cropGen->pos = tile->pos;

				tileCold->fieldCrop = Field_CropCount++;
			}

			AI_FarmerSearchHot* searchFarmer = &AI_FarmersSearchHot[AI_FarmerSearchCount];
			AI_FarmerSearchCold* searchFarmerCold = &AI_FarmersSearchCold[AI_FarmerSearchCount];
			searchFarmer->searchTimer = rand_rangef(AI_FarmerSearchSpeedMin, AI_FarmerSearchSpeedMax);
			searchFarmerCold->pos = AI_FarmersFarmPos[ai];

			AI_FarmerSearchCount++;

			swap(AI_FarmerFarmHot, *farmer, AI_FarmersFarmHot[previousFarmCount - 1]);
			swap(AI_FarmerFarmCold, *farmerCold, AI_FarmersFarmCold[previousFarmCount - 1]);
			swap(QuantizedVec2, AI_FarmersFarmPos[ai], AI_FarmersFarmPos[previousFarmCount - 1]);
			previousFarmCount--;

			swap(AI_FarmerFarmHot, AI_FarmersFarmHot[previousFarmCount], AI_FarmersFarmHot[AI_FarmerFarmCount - 1]);
			swap(AI_FarmerFarmCold, AI_FarmersFarmCold[previousFarmCount], AI_FarmersFarmCold[AI_FarmerFarmCount - 1]);
			swap(QuantizedVec2, AI_FarmersFarmPos[previousFarmCount], AI_FarmersFarmPos[AI_FarmerFarmCount - 1]);
			AI_FarmerFarmCount--;
			ai--;
		}
	}
	MIST_PROFILE_END("Game", "Ai_Farm");
}

void ai_tick(float delta)
{
	MIST_PROFILE_BEGIN("Game", "Ai_Tick");
	uint32_t previousMoveCount = ai_tick_search(delta);
	uint32_t previousFarmCount = ai_tick_move(delta, previousMoveCount);
	ai_tick_farm(delta, previousFarmCount);
	MIST_PROFILE_END("Game", "Ai_Tick");
}

// Game

float Game_FarmerImageTable[] =
    {
        [FarmerState_Search] = 0.0f,
        [FarmerState_Move] = 1.0f,
        [FarmerState_Farm] = 2.0f
    };

float Game_FieldImageTable[] =
    {
        [FieldStage_Arable] = 3.0f,
        [FieldStage_Fallow] = 4.0f,
        [FieldStage_Planted] = 5.0f,
        [FieldStage_Grown] = 6.0f
    };

void game_init(void)
{
    srand((unsigned int)time(NULL));

    Field_TilesGen = (Field_TileGen*)malloc(sizeof(Field_TileGen) * Field_Width * Field_Height);
	Field_TilesCold = (Field_TileCold*)malloc(sizeof(Field_TileCold) * Field_Width * Field_Height);
    memset(Field_TilesGen, 0, sizeof(Field_TileGen) * Field_Width * Field_Height);

	Field_CropsHot = (Field_CropHot*)malloc(sizeof(Field_CropHot) * Field_Width * Field_Height);
	Field_CropsCold = (Field_CropCold*)malloc(sizeof(Field_CropCold) * Field_Width * Field_Height);
	Field_CropsGen = (Field_CropGen*)malloc(sizeof(Field_CropGen) * Field_Width * Field_Height);

    for (uint32_t y = 0; y < Field_Height; ++y)
    {
        for (uint32_t x = 0; x < Field_Width; ++x)
        {
            Vec2 tilePos = (Vec2) { .x = (float)x / Field_Width, .y = (float)y / Field_Height };
            Field_TilesGen[y * Field_Width + x].pos = quantizedVec2_Write(vec2_sub(vec2_mul(tilePos, 2.0f), (Vec2) { .x = 1.0f, .y = 1.0f }));
        }
    }

    AI_FarmersSearchHot = (AI_FarmerSearchHot*)malloc(sizeof(AI_FarmerSearchHot) * AI_FarmerCount);
	AI_FarmersSearchCold = (AI_FarmerSearchCold*)malloc(sizeof(AI_FarmerSearchCold) * AI_FarmerCount);
	AI_FarmersMoveHot = (AI_FarmerMoveHot*)malloc(sizeof(AI_FarmerMoveHot) * AI_FarmerCount);
	AI_FarmersMoveCold = (AI_FarmerMoveCold*)malloc(sizeof(AI_FarmerMoveCold) * AI_FarmerCount);
	AI_FarmersMovePos = (AI_FarmerMovePos*)malloc(sizeof(AI_FarmerMovePos) * AI_FarmerCount);
	AI_FarmersFarmHot = (AI_FarmerFarmHot*)malloc(sizeof(AI_FarmerFarmHot) * AI_FarmerCount);
	AI_FarmersFarmCold = (AI_FarmerFarmCold*)malloc(sizeof(AI_FarmerFarmCold) * AI_FarmerCount);
	AI_FarmersFarmPos = (QuantizedVec2*)malloc(sizeof(QuantizedVec2) * AI_FarmerCount);
    for (uint32_t ai = 0; ai < AI_FarmerCount; ++ai)
    {
		AI_FarmerSearchHot* searchFarmer = &AI_FarmersSearchHot[ai];
		searchFarmer->searchTimer = rand_rangef(AI_FarmerSearchSpeedMin, AI_FarmerSearchSpeedMax);
    }

	AI_FarmerSearchCount = AI_FarmerCount;
}

void game_tick(float delta)
{
    delta = math_minf(delta, 0.02f);

    field_tick(delta);
    ai_tick(delta);
}

void game_kill(void)
{
	free(Field_CropsHot);
	Field_CropsHot = NULL;
	free(Field_CropsCold);
	Field_CropsCold = NULL;
	free(Field_CropsGen);
	Field_CropsGen = NULL;
    free(Field_TilesGen);
	Field_TilesGen = NULL;
	free(Field_TilesCold);
	Field_TilesCold = NULL;
    free(AI_FarmersSearchHot);
    AI_FarmersSearchHot = NULL;
	free(AI_FarmersSearchCold);
	AI_FarmersSearchCold = NULL;
	free(AI_FarmersMoveHot);
	AI_FarmersMoveHot = NULL;
	free(AI_FarmersMoveCold);
	AI_FarmersMoveCold = NULL;
	free(AI_FarmersFarmHot);
	AI_FarmersFarmHot = NULL;
	free(AI_FarmersFarmCold);
	AI_FarmersFarmCold = NULL;
	free(AI_FarmersFarmPos);
	AI_FarmersFarmPos = NULL;
}

uint32_t game_gen_instance_buffer(Game_InstanceBuffer* buffer)
{
    uint32_t writeIndex = 0;
    for (uint32_t i = 0; i < Field_Width * Field_Height; ++i)
    {
        uint32_t writeLoc = writeIndex++;
        buffer->instances[writeLoc].spriteIndex = 3.0f + Field_TilesGen[i].stage;
		memcpy(buffer->instances[writeLoc].pos, &Field_TilesGen[i].pos, sizeof(int16_t) * 2);
    }

	for (uint32_t i = 0; i < Field_CropCount; ++i)
	{
		uint32_t cropWriteIndex = writeIndex++;
		buffer->instances[cropWriteIndex].spriteIndex = 7.0f + Field_CropsGen[i].cropType;

		memcpy(buffer->instances[cropWriteIndex].pos, &Field_CropsGen[i].pos, sizeof(int16_t) * 2);
	}

	for (uint32_t i = 0; i < AI_FarmerSearchCount; ++i)
	{
		uint32_t writeLoc = writeIndex++;
		buffer->instances[writeLoc].spriteIndex = (float)FarmerState_Search;
		memcpy(buffer->instances[writeLoc].pos, &AI_FarmersSearchCold[i].pos, sizeof(int16_t) * 2);
	}

	for (uint32_t i = 0; i < AI_FarmerMoveCount; ++i)
	{
		uint32_t writeLoc = writeIndex++;
		buffer->instances[writeLoc].spriteIndex = (float)FarmerState_Move;
		QuantizedVec2 v = quantizedVec2_Write(AI_FarmersMovePos[i].pos);
		memcpy(buffer->instances[writeLoc].pos, &v, sizeof(int16_t) * 2);
	}

	for (uint32_t i = 0; i < AI_FarmerFarmCount; ++i)
	{
		uint32_t writeLoc = writeIndex++;
		buffer->instances[writeLoc].spriteIndex = (float)FarmerState_Farm;

		memcpy(buffer->instances[writeLoc].pos, &AI_FarmersFarmPos[i], sizeof(int16_t) * 2);
	}

    return writeIndex;
}
