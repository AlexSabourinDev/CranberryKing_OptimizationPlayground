#include "game.h"

#include "3rd/Mist_Profiler.h"

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define _M_IX86
#include <xmmintrin.h>
#include <emmintrin.h>
#include <tmmintrin.h>

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

uint32_t math_min(uint32_t l, uint32_t r)
{
	return l < r ? l : r;
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
} Field_CropGenPos;

typedef struct
{
	float cropType;
} Field_CropGenType;

typedef struct
{
	QuantizedVec2 pos;
} Field_TileGenPos;

typedef struct
{
	float stage;
} Field_TileGenStage;

typedef struct
{
	uint8_t stage;
} Field_TileStage;

typedef struct
{
	uint32_t fieldCrop;
} Field_TileCold;

const uint32_t Field_Width  = 500;
const uint32_t Field_Height = 500;
static Field_TileCold* Field_TilesCold = NULL;
static Field_TileGenPos* Field_TilesGenPos = NULL;
static Field_TileGenStage* Field_TilesGenStage = NULL;
static Field_TileStage* Field_TilesStage = NULL;

static Field_CropHot* Field_CropsHot = NULL;
static Field_CropCold* Field_CropsCold = NULL;
static Field_CropGenPos* Field_CropsGenPos = NULL;
static Field_CropGenType* Field_CropsGenType = NULL;
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
            Field_TilesStage[cropCold->fieldTile].stage = FieldStage_Grown;
			Field_TilesGenStage[cropCold->fieldTile].stage = 3.0f + FieldStage_Grown;
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
	float velX[4];
	float velY[4];
	float targetX[4];
	float targetY[4];
} AI_FarmerMoveHot;

typedef struct
{
	float posX[4];
	float posY[4];
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

static uint32_t* AI_RemovedIndices = NULL;

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
			Field_TileGenPos* tile = &Field_TilesGenPos[tileIndex];
			Field_TileStage* stage = &Field_TilesStage[tileIndex];

			if (stage->stage != FieldStage_Planted)
			{
				AI_FarmerSearchCold* farmerCold = &AI_FarmersSearchCold[ai];

				uint32_t blockIndex = AI_FarmerMoveCount / 4;
				uint32_t itemIndex = AI_FarmerMoveCount % 4;
				AI_FarmerMoveHot* moveFarmerHot = &AI_FarmersMoveHot[blockIndex];
				AI_FarmerMoveCold* moveFarmerCold = &AI_FarmersMoveCold[AI_FarmerMoveCount];
				AI_FarmerMovePos* moveFarmerPos = &AI_FarmersMovePos[blockIndex];
				AI_FarmerMoveCount++;
				moveFarmerCold->tileIndex = tileIndex;
				Vec2 farmerPos = quantizedVec2_Read(farmerCold->pos);
				moveFarmerPos->posX[itemIndex] = farmerPos.x;
				moveFarmerPos->posY[itemIndex] = farmerPos.y;

				Vec2 tilePos = quantizedVec2_Read(tile->pos);
				Vec2 vel = vec2_mul(vec2_norm(vec2_sub(tilePos, farmerPos)), AI_FarmerSpeed);
				moveFarmerHot->velX[itemIndex] = vel.x;
				moveFarmerHot->velY[itemIndex] = vel.y;

				moveFarmerHot->targetX[itemIndex] = tilePos.x;
				moveFarmerHot->targetY[itemIndex] = tilePos.y;

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

#define MASK_ELEMENT(a) (a) | (a + 1) << (8) | (a + 2) << (16) | (a + 3) << (24)
#define MASK_GEN(a, b, c, d) MASK_ELEMENT((a * 4)), MASK_ELEMENT((b * 4)), MASK_ELEMENT((c * 4)), MASK_ELEMENT((d * 4))
uint32_t ai_tick_move(float delta, uint32_t previousMoveCount)
{
	MIST_PROFILE_BEGIN("Game", "Ai_Move");

	uint32_t removeIndex = 0;

	__m128 d = _mm_load_ps1(&delta);
	for (uint32_t ai = 0; ai < previousMoveCount; ai += 4)
	{
		AI_FarmerMoveHot* farmer = &AI_FarmersMoveHot[ai / 4];
		AI_FarmerMovePos* farmerPosInstance = &AI_FarmersMovePos[ai / 4];

		__m128 velX = _mm_load_ps(farmer->velX);
		__m128 velY = _mm_load_ps(farmer->velY);
		velX = _mm_mul_ps(velX, d);
		velY = _mm_mul_ps(velY, d);

		__m128 posX = _mm_load_ps(farmerPosInstance->posX);
		__m128 posY = _mm_load_ps(farmerPosInstance->posY);

		float velMag = delta * AI_FarmerSpeed;
		velMag *= velMag;
		__m128 vMag = _mm_load_ps1(&velMag);

		__m128 tX = _mm_load_ps(farmer->targetX);
		__m128 tY = _mm_load_ps(farmer->targetY);
		__m128 iX = _mm_sub_ps(tX, posX);
		__m128 iY = _mm_sub_ps(tY, posY);
		__m128 dist = _mm_add_ps(_mm_mul_ps(iX, iX), _mm_mul_ps(iY, iY));

		_mm_store_ps(farmerPosInstance->posX, _mm_add_ps(posX, velX));
		_mm_store_ps(farmerPosInstance->posY, _mm_add_ps(posY, velY));

		const uint32_t invalidMaskTable[] =
		{
			0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF,
			0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0x00,
			0xFFFFFFFF, 0xFFFFFFFF, 0x00, 0x00,
			0xFFFFFFFF, 0x00, 0x00, 0x00
		};

		__m128 invalidMask = _mm_load_ps((float*)&invalidMaskTable[math_min(ai + 4, AI_FarmerMoveCount) % 4 * 4]);
		__m128 cmp = _mm_and_ps(_mm_cmpgt_ps(vMag, dist), invalidMask);
		int moveMask = _mm_movemask_ps(cmp);

		const uint8_t bitCountTable[] = { 0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4 };
		uint8_t bitCount = bitCountTable[moveMask];

		const uint32_t shuffleTable[] =
		{
			MASK_GEN(3, 3, 3, 3), MASK_GEN(0, 3, 3, 3), MASK_GEN(1, 3, 3, 3), MASK_GEN(0, 1, 3, 3), 
			MASK_GEN(2, 3, 3, 3), MASK_GEN(0, 2, 3, 3), MASK_GEN(1, 2, 3, 3), MASK_GEN(0, 1, 2, 3),
			MASK_GEN(3, 3, 3, 3), MASK_GEN(0, 3, 3, 3), MASK_GEN(1, 3, 3, 3), MASK_GEN(0, 1, 3, 3),
			MASK_GEN(2, 3, 3, 3), MASK_GEN(0, 2, 3, 3), MASK_GEN(1, 2, 3, 3), MASK_GEN(0, 1, 2, 3)
		};

		uint32_t indexArray[] = { ai, ai + 1, ai + 2, ai + 3 };
		__m128i indices = _mm_load_si128((const __m128i*)indexArray);
		__m128i removedIndices = _mm_shuffle_epi8(indices, _mm_load_si128((const __m128i*)&shuffleTable[moveMask * 4]));

		if (bitCount > 0)
		{
			_mm_storeu_si128((__m128i*)&AI_RemovedIndices[removeIndex], removedIndices);
			removeIndex += bitCount;
		}
	}

	uint32_t previousFarmCount = AI_FarmerFarmCount;
	for(uint32_t i = 0; i < removeIndex; ++i)
	{
		uint32_t removal = AI_RemovedIndices[i];
		uint32_t blockIndex = removal / 4;
		uint32_t itemIndex = removal % 4;

		AI_FarmerMoveHot* farmerHot = &AI_FarmersMoveHot[blockIndex];
		AI_FarmerMoveCold* farmerCold = &AI_FarmersMoveCold[blockIndex];

		AI_FarmerFarmHot* farmFarmerHot = &AI_FarmersFarmHot[AI_FarmerFarmCount];
		AI_FarmerFarmCold* farmFarmerCold = &AI_FarmersFarmCold[AI_FarmerFarmCount];
		QuantizedVec2* farmFarmerPos = &AI_FarmersFarmPos[AI_FarmerFarmCount];
		AI_FarmerFarmCount++;

		farmFarmerHot->farmTimer = rand_rangef(AI_FarmerFarmSpeedMin, AI_FarmerFarmSpeedMax);
		farmFarmerCold->tileIndex = farmerCold->tileIndex;
		*farmFarmerPos = quantizedVec2_Write((Vec2) { farmerHot->targetX[itemIndex], farmerHot->targetY[itemIndex] });


		// Swap with the end to remove it from the list
		AI_FarmerMoveHot* endHotBlock = &AI_FarmersMoveHot[AI_FarmerMoveCount / 4];
		AI_FarmerMovePos* endPosBlock = &AI_FarmersMovePos[AI_FarmerMoveCount / 4];

		AI_FarmerMovePos* currentPosBlock = &AI_FarmersMovePos[blockIndex];

		uint32_t swapIndex = AI_FarmerMoveCount % 4;
		swap(float, farmerHot->velX[itemIndex], endHotBlock->velX[swapIndex]);
		swap(float, farmerHot->velY[itemIndex], endHotBlock->velY[swapIndex]);
		swap(float, farmerHot->targetX[itemIndex], endHotBlock->targetX[swapIndex]);
		swap(float, farmerHot->targetY[itemIndex], endHotBlock->targetY[swapIndex]);

		swap(float, currentPosBlock->posX[itemIndex], endPosBlock->posX[swapIndex]);
		swap(float, currentPosBlock->posY[itemIndex], endPosBlock->posY[swapIndex]);

		swap(AI_FarmerMoveCold, *farmerCold, AI_FarmersMoveCold[AI_FarmerMoveCount - 1]);
		AI_FarmerMoveCount--;
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
			Field_TileGenPos* tilePos = &Field_TilesGenPos[farmerCold->tileIndex];
			Field_TileStage* tileStage = &Field_TilesStage[farmerCold->tileIndex];
			Field_TileCold* tileCold = &Field_TilesCold[farmerCold->tileIndex];

			if (tileStage->stage == FieldStage_Grown)
			{
				Field_CropHot* crop = &Field_CropsHot[tileCold->fieldCrop];
				swap(Field_CropHot, *crop, Field_CropsHot[Field_CropCount - 1]);

				Field_CropCold* cropCold = &Field_CropsCold[tileCold->fieldCrop];
				swap(Field_CropCold, *cropCold, Field_CropsCold[Field_CropCount - 1]);

				Field_CropGenPos* cropGen = &Field_CropsGenPos[tileCold->fieldCrop];
				swap(Field_CropGenPos, *cropGen, Field_CropsGenPos[Field_CropCount - 1]);

				Field_CropGenType* cropGenType = &Field_CropsGenType[tileCold->fieldCrop];
				swap(Field_CropGenType, *cropGenType, Field_CropsGenType[Field_CropCount - 1]);

				Field_TilesCold[cropCold->fieldTile].fieldCrop = tileCold->fieldCrop;

				Field_CropCount--;
			}

			tileStage->stage = math_max((tileStage->stage + 1) % FieldState_Max, FieldStage_Fallow);
			Field_TilesGenStage[farmerCold->tileIndex].stage = 3.0f + tileStage->stage;

			if (tileStage->stage == FieldStage_Planted)
			{
				Field_CropHot* crop = &Field_CropsHot[Field_CropCount];
				crop->grown = rand_rangef(Crop_MinLifetime, Crop_MaxLifetime);

				Field_CropCold* cropCold = &Field_CropsCold[Field_CropCount];
				cropCold->fieldTile = farmerCold->tileIndex;

				Field_CropGenType* cropGen = &Field_CropsGenType[Field_CropCount];
				cropGen->cropType = 7.0f + rand_range(0, Crop_MaxCropType);

				Field_CropGenPos* cropGenPos = &Field_CropsGenPos[Field_CropCount];
				cropGenPos->pos = tilePos->pos;

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

    Field_TilesGenPos = (Field_TileGenPos*)_mm_malloc(sizeof(Field_TileGenPos) * Field_Width * Field_Height, 16);
	Field_TilesGenStage = (Field_TileGenStage*)_mm_malloc(sizeof(Field_TileGenStage) * Field_Width * Field_Height, 16);
	Field_TilesStage = (Field_TileStage*)malloc(sizeof(Field_TileStage) * Field_Width * Field_Height);
	Field_TilesCold = (Field_TileCold*)malloc(sizeof(Field_TileCold) * Field_Width * Field_Height);
	memset(Field_TilesStage, 0, sizeof(Field_TileStage) * Field_Width * Field_Height);

	Field_CropsHot = (Field_CropHot*)malloc(sizeof(Field_CropHot) * Field_Width * Field_Height);
	Field_CropsCold = (Field_CropCold*)malloc(sizeof(Field_CropCold) * Field_Width * Field_Height);
	Field_CropsGenPos = (Field_CropGenPos*)_mm_malloc(sizeof(Field_CropGenPos) * Field_Width * Field_Height, 16);
	Field_CropsGenType = (Field_CropGenType*)_mm_malloc(sizeof(Field_CropGenType) * Field_Width * Field_Height, 16);

    for (uint32_t y = 0; y < Field_Height; ++y)
    {
        for (uint32_t x = 0; x < Field_Width; ++x)
        {
            Vec2 tilePos = (Vec2) { .x = (float)x / Field_Width, .y = (float)y / Field_Height };
            Field_TilesGenPos[y * Field_Width + x].pos = quantizedVec2_Write(vec2_sub(vec2_mul(tilePos, 2.0f), (Vec2) { .x = 1.0f, .y = 1.0f }));

			Field_TilesGenStage[y * Field_Width + x].stage = 3.0f;
        }
    }

    AI_FarmersSearchHot = (AI_FarmerSearchHot*)malloc(sizeof(AI_FarmerSearchHot) * AI_FarmerCount);
	AI_FarmersSearchCold = (AI_FarmerSearchCold*)malloc(sizeof(AI_FarmerSearchCold) * AI_FarmerCount);
	AI_FarmersMoveHot = (AI_FarmerMoveHot*)_mm_malloc(sizeof(AI_FarmerMoveHot) / 4 * AI_FarmerCount, 16);
	AI_FarmersMoveCold = (AI_FarmerMoveCold*)malloc(sizeof(AI_FarmerMoveCold) * AI_FarmerCount);
	AI_FarmersMovePos = (AI_FarmerMovePos*)_mm_malloc(sizeof(AI_FarmerMovePos) / 4 * AI_FarmerCount, 16);
	AI_FarmersFarmHot = (AI_FarmerFarmHot*)malloc(sizeof(AI_FarmerFarmHot) * AI_FarmerCount);
	AI_FarmersFarmCold = (AI_FarmerFarmCold*)malloc(sizeof(AI_FarmerFarmCold) * AI_FarmerCount);
	AI_FarmersFarmPos = (QuantizedVec2*)malloc(sizeof(QuantizedVec2) * AI_FarmerCount);

	AI_RemovedIndices = (uint32_t*)_mm_malloc(sizeof(uint32_t) * AI_FarmerCount, 16);
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
	_mm_free(Field_CropsGenPos);
	Field_CropsGenPos = NULL;
	_mm_free(Field_CropsGenType);
	Field_CropsGenType = NULL;
	_mm_free(Field_TilesGenPos);
	Field_TilesGenPos = NULL;
	_mm_free(Field_TilesGenStage);
	Field_TilesGenStage = NULL;
	free(Field_TilesStage);
	Field_TilesStage = NULL;
	free(Field_TilesCold);
	Field_TilesCold = NULL;
    free(AI_FarmersSearchHot);
    AI_FarmersSearchHot = NULL;
	free(AI_FarmersSearchCold);
	AI_FarmersSearchCold = NULL;
	_mm_free(AI_FarmersMoveHot);
	AI_FarmersMoveHot = NULL;
	free(AI_FarmersMoveCold);
	AI_FarmersMoveCold = NULL;
	_mm_free(AI_FarmersMovePos);
	AI_FarmersMovePos = NULL;
	free(AI_FarmersFarmHot);
	AI_FarmersFarmHot = NULL;
	free(AI_FarmersFarmCold);
	AI_FarmersFarmCold = NULL;
	free(AI_FarmersFarmPos);
	AI_FarmersFarmPos = NULL;
	_mm_free(AI_RemovedIndices);
	AI_RemovedIndices = NULL;
}

uint32_t game_gen_instance_buffer(Game_InstanceBuffer* buffer)
{
    uint32_t writeIndex = 0;
    for (uint32_t i = 0; i < Field_Width * Field_Height; i+=4)
    {
        _mm_store_ps(&buffer->spriteIndices[writeIndex + i], _mm_load_ps((float*)&Field_TilesGenStage[i]));
    }
	memcpy(&buffer->pos[writeIndex], Field_TilesGenPos, Field_Width * Field_Height * sizeof(int16_t) * 2);
	writeIndex += Field_Width * Field_Height;

	for (uint32_t i = 0; i < Field_CropCount; i += 4)
	{
		_mm_store_ps(&buffer->spriteIndices[writeIndex + i], _mm_load_ps((float*)&Field_CropsGenType[i]));
	}
	memcpy(&buffer->pos[writeIndex], Field_CropsGenPos, Field_CropCount * sizeof(int16_t) * 2);
	writeIndex += Field_CropCount;

	__m128 searchIndex = _mm_set_ps1((float)FarmerState_Search);
	for (uint32_t i = 0; i < AI_FarmerSearchCount; i += 4)
	{
		_mm_storeu_ps(&buffer->spriteIndices[writeIndex + i], searchIndex);
	}

	memcpy(&buffer->pos[writeIndex], AI_FarmersSearchCold, AI_FarmerSearchCount * sizeof(int16_t) * 2);
	writeIndex += AI_FarmerSearchCount;

	__m128 max = _mm_set_ps1((float)INT16_MAX);
	__m128 moveIndex = _mm_set_ps1((float)FarmerState_Move);
	for (uint32_t i = 0; i < AI_FarmerMoveCount; i += 4)
	{
		_mm_storeu_ps(&buffer->spriteIndices[writeIndex + i], moveIndex);

		__m128 posX = _mm_load_ps(AI_FarmersMovePos[i / 4].posX);
		__m128 posY = _mm_load_ps(AI_FarmersMovePos[i / 4].posY);

		const uint32_t xMaskI[] = { 0x00, 0xFFFFFFFF, 0x00, 0xFFFFFFFF };
		__m128 xMask = _mm_load_ps((float*)xMaskI);

		const uint32_t yMaskI[] = { 0xFFFFFFFF, 0x00, 0xFFFFFFFF, 0x00 };
		__m128 yMask = _mm_load_ps((float*)yMaskI);

		__m128 shufflePos1 = _mm_or_ps(_mm_and_ps(_mm_shuffle_ps(posY, posY, _MM_SHUFFLE(1, 0, 0, 0)), yMask), _mm_and_ps(_mm_shuffle_ps(posX, posX, _MM_SHUFFLE(0, 1, 0, 0)), xMask));
		__m128 shufflePos2 = _mm_or_ps(_mm_and_ps(_mm_shuffle_ps(posY, posY, _MM_SHUFFLE(3, 0, 2, 0)), yMask), _mm_and_ps(_mm_shuffle_ps(posX, posX, _MM_SHUFFLE(0, 3, 0, 2)), xMask));

		__m64 shufflePosi1 = _mm_cvtps_pi16(_mm_mul_ps(shufflePos1, max));
		__m64 shufflePosi2 = _mm_cvtps_pi16(_mm_mul_ps(shufflePos2, max));

		memcpy(&buffer->pos[writeIndex + i], &shufflePosi1, sizeof(__m64));
		memcpy(&buffer->pos[writeIndex + i + 2], &shufflePosi2, sizeof(__m64));
	}
	writeIndex += AI_FarmerMoveCount;

	__m128 farmIndex = _mm_set_ps1((float)FarmerState_Farm);
	for (uint32_t i = 0; i < AI_FarmerFarmCount; i += 4)
	{
		_mm_storeu_ps(&buffer->spriteIndices[writeIndex + i], farmIndex);
	}

	memcpy(&buffer->pos[writeIndex], AI_FarmersFarmPos, AI_FarmerFarmCount * sizeof(int16_t) * 2);
	writeIndex += AI_FarmerFarmCount;

    return writeIndex;
}
