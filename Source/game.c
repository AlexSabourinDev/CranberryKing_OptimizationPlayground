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
#include <smmintrin.h>
#include <immintrin.h>

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

// SIMD

// Thanks to https://stackoverflow.com/questions/36932240/avx2-what-is-the-most-efficient-way-to-pack-left-based-on-a-mask for this great algorithm
// Uses 64bit pdep / pext to save a step in unpacking.
__m256i simd_moveMaskToIndexMask(unsigned int mask /* from movmskps */)
{
	uint64_t expanded_mask = _pdep_u64(mask, 0x0101010101010101);  // unpack each bit to a byte
	expanded_mask *= 0xFF;    // mask |= mask<<1 | mask<<2 | ... | mask<<7;
	// ABC... -> AAAAAAAABBBBBBBBCCCCCCCC...: replicate each bit to fill its byte

	const uint64_t identity_indices = 0x0706050403020100;    // the identity shuffle for vpermps, packed to one index per byte
	uint64_t wanted_indices = _pext_u64(identity_indices, expanded_mask);

	__m128i bytevec = _mm_cvtsi64_si128(wanted_indices);
	__m256i shufmask = _mm256_cvtepu8_epi32(bytevec);

	return shufmask;
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
	uint32_t cropType;
	uint32_t tileIndex;
} Field_Crop;

typedef struct
{
	uint32_t writeIndex;
	int16_t spriteIndex;
	float posX;
	float posY;
} Field_CropDrawCommand;

static uint32_t Field_CropDrawCommandCount = 0;
static Field_CropDrawCommand* Field_CropDrawCommands = NULL;

static uint32_t Field_CropCount = 0;
static Field_Crop* Field_Crops = NULL;
static float* Field_CropLifetimes = NULL;

typedef struct
{
	Field_Stage stage;
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
		float* lifetime = &Field_CropLifetimes[i];
		*lifetime -= delta;

		if (*lifetime <= 0.0f)
		{
			Field_Crop* crop = &Field_Crops[i];

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


			float div = (float)crop->tileIndex / Field_Width;
			float f = floorf(div);
			Field_CropDrawCommands[Field_CropDrawCommandCount] = (Field_CropDrawCommand)
				{
					.writeIndex = i,
					.spriteIndex = (7.0f + crop->cropType) / 11.0f * INT16_MAX,
					.posX = (div - f) * 2.0f - 1.0f,
					.posY = f / Field_Height * 2.0f - 1.0f
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
	uint32_t tileIndex;
} AI_FarmerMoveStateCold;

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
	float searchTimer;
} AI_FarmerSearchStateHot;

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
#define AI_FarmerCount 1000000

static uint32_t AI_FarmerMoveCount = 0;
static float* AI_FarmersMoveHotX = NULL;
static float* AI_FarmersMoveHotY = NULL;
static AI_FarmerMoveStateCold* AI_FarmersMoveCold = NULL;
static float* AI_FarmersMoveGenX = NULL;
static float* AI_FarmersMoveGenY = NULL;
static uint32_t AI_FarmerFarmCount = 0;
static AI_FarmerFarmStateHot* AI_FarmersFarmHot = NULL;
static AI_FarmerFarmStateCold* AI_FarmersFarmCold = NULL;
static float* AI_FarmersFarmGenX = NULL;
static float* AI_FarmersFarmGenY = NULL;
static uint32_t AI_FarmerSearchCount = 0;
static AI_FarmerSearchStateHot* AI_FarmersSearchHot = NULL;
static float* AI_FarmersSearchGenX = NULL;
static float* AI_FarmersSearchGenY = NULL;

static int* AI_FarmerRemovalIndices = NULL;

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
					float* genFarmerX = &AI_FarmersSearchGenX[i];
					float* genFarmerY = &AI_FarmersSearchGenY[i];


					float div = (float)tileIndex / Field_Width;
					float f = floorf(div);
					AI_FarmersMoveHotX[AI_FarmerMoveCount] = (div - f) * 2.0f - 1.0f;
					AI_FarmersMoveHotY[AI_FarmerMoveCount] = f / Field_Height * 2.0f - 1.0f;

					AI_FarmersMoveCold[AI_FarmerMoveCount].tileIndex = tileIndex;
					
					AI_FarmersMoveGenX[AI_FarmerMoveCount] = *genFarmerX;
					AI_FarmersMoveGenY[AI_FarmerMoveCount] = *genFarmerY;
					AI_FarmerMoveCount++;

					SWAP(AI_FarmerSearchStateHot, *farmer, AI_FarmersSearchHot[AI_FarmerSearchCount - 1]);
					SWAP(float, *genFarmerX, AI_FarmersSearchGenX[AI_FarmerSearchCount - 1]);
					SWAP(float, *genFarmerY, AI_FarmersSearchGenY[AI_FarmerSearchCount - 1]);
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
		float v = AI_FarmerSpeed * delta;
		float rv = 1.0f / v;
		__m256 velMag = _mm256_set_ps(v, v, v, v, v, v, v, v);
		__m256 rvelMag = _mm256_set_ps(rv, rv, rv, rv, rv, rv, rv, rv);

		uint32_t removedFarmerCount = 0;

		// We only do the farmers that were already in the move array before this tick
		for (uint32_t i = 0; i < previousFarmerMoveCount; i+=8)
		{
			__m256 farmerX = _mm256_load_ps(AI_FarmersMoveHotX + i);
			__m256 farmerY = _mm256_load_ps(AI_FarmersMoveHotY + i);
			__m256 genFarmerX = _mm256_load_ps(AI_FarmersMoveGenX + i);
			__m256 genFarmerY = _mm256_load_ps(AI_FarmersMoveGenY + i);

			__m256 dirVecX = _mm256_sub_ps(farmerX, genFarmerX);
			__m256 dirVecY = _mm256_sub_ps(farmerY, genFarmerY);
			__m256 rmag = _mm256_rsqrt_ps(_mm256_add_ps(_mm256_mul_ps(dirVecX, dirVecX), _mm256_mul_ps(dirVecY, dirVecY)));

			__m256 velX = _mm256_mul_ps(dirVecX, _mm256_mul_ps(velMag, rmag));
			__m256 velY = _mm256_mul_ps(dirVecY, _mm256_mul_ps(velMag, rmag));
			_mm256_store_ps(AI_FarmersMoveGenX + i, _mm256_add_ps(genFarmerX, velX));
			_mm256_store_ps(AI_FarmersMoveGenY + i, _mm256_add_ps(genFarmerY, velY));

			int bitMask = (1 << math_min(previousFarmerMoveCount - i, 8)) - 1;

			// Since we're calculating reciprocals, the larger number is smaller
			// So instead of whether our velocity is larger than our distance,
			// we check to see if our r velocity is smaller than our r distance
			// vel = 1, dist = 0.5 vel > dist ? we passed it!
			// rvel = 1/1, dist = 1/0.5 rvel < rdist ? we passed it!
			__m256 cmpRes = _mm256_cmp_ps(rvelMag, rmag, _CMP_LT_OQ);
			int indexMask = _mm256_movemask_ps(cmpRes) & bitMask;

			__m256i indices = _mm256_set_epi32(i, i, i, i, i, i, i, i);
			__m256i indexAdd = simd_moveMaskToIndexMask(indexMask);
			indices = _mm256_add_epi32(indices, indexAdd);

			_mm256_storeu_si256((__m256i*)(AI_FarmerRemovalIndices + removedFarmerCount), indices);
			removedFarmerCount += _mm_popcnt_u32(indexMask);
		}


		for(uint32_t i = 0; i < removedFarmerCount; i++)
		{
			int r = AI_FarmerRemovalIndices[i];
			AI_FarmerMoveStateCold* coldFarmer = &AI_FarmersMoveCold[r];

			AI_FarmersFarmHot[AI_FarmerFarmCount + i].farmTimer = rand_rangef(AI_FarmerFarmSpeedMin, AI_FarmerFarmSpeedMax);
			AI_FarmersFarmCold[AI_FarmerFarmCount + i].tileIndex = coldFarmer->tileIndex;
			AI_FarmersFarmGenX[AI_FarmerFarmCount + i] = AI_FarmersMoveGenX[r];
			AI_FarmersFarmGenY[AI_FarmerFarmCount + i] = AI_FarmersMoveGenY[r];

			SWAP(float, AI_FarmersMoveHotX[r], AI_FarmersMoveHotX[AI_FarmerMoveCount - 1 - i]);
			SWAP(float, AI_FarmersMoveHotY[r], AI_FarmersMoveHotY[AI_FarmerMoveCount - 1 - i]);
			SWAP(AI_FarmerMoveStateCold, *coldFarmer, AI_FarmersMoveCold[AI_FarmerMoveCount - 1 - i]);
			SWAP(float, AI_FarmersMoveGenX[r], AI_FarmersMoveGenX[AI_FarmerMoveCount - 1 - i]);
			SWAP(float, AI_FarmersMoveGenY[r], AI_FarmersMoveGenY[AI_FarmerMoveCount - 1 - i]);
		}

		AI_FarmerMoveCount -= removedFarmerCount;
		AI_FarmerFarmCount += removedFarmerCount;
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
					Field_CropLifetimes[Field_CropCount] = rand_rangef(Crop_MinLifetime, Crop_MaxLifetime);
					crop->cropType = rand_range(0, Crop_MaxCropType);
					crop->tileIndex = coldFarmer->tileIndex;

					float div = (float)crop->tileIndex / Field_Width;
					float f = floorf(div);
					Field_CropDrawCommands[Field_CropDrawCommandCount] = (Field_CropDrawCommand)
						{
							.writeIndex = Field_CropCount,
							.spriteIndex = (7.0f + crop->cropType) / 11.0f * INT16_MAX,
							.posX = (div - f) * 2.0f - 1.0f,
							.posY = f / Field_Height * 2.0f - 1.0f
						};
					Field_CropDrawCommandCount++;

					Field_CropCount++;
				}

				AI_FarmersSearchHot[AI_FarmerSearchCount].searchTimer = rand_rangef(AI_FarmerSearchSpeedMin, AI_FarmerSearchSpeedMax);
				AI_FarmersSearchGenX[AI_FarmerSearchCount] = AI_FarmersFarmGenX[i];
				AI_FarmersSearchGenY[AI_FarmerSearchCount] = AI_FarmersFarmGenY[i];
				AI_FarmerSearchCount++;

				SWAP(AI_FarmerFarmStateHot, *farmer, AI_FarmersFarmHot[AI_FarmerFarmCount - 1]);
				SWAP(AI_FarmerFarmStateCold, *coldFarmer, AI_FarmersFarmCold[AI_FarmerFarmCount - 1]);
				SWAP(float, AI_FarmersFarmGenX[i], AI_FarmersFarmGenX[AI_FarmerFarmCount - 1]);
				SWAP(float, AI_FarmersFarmGenY[i], AI_FarmersFarmGenY[AI_FarmerFarmCount - 1]);
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
	Field_CropLifetimes = (float*)_mm_malloc(sizeof(float) * Field_Width * Field_Height, 64);
	Field_CropDrawCommands = (Field_CropDrawCommand*)malloc(sizeof(Field_CropDrawCommand) * Field_Width * Field_Height);

	for (uint32_t y = 0; y < Field_Height; ++y)
	{
		for (uint32_t x = 0; x < Field_Width; ++x)
		{
			uint32_t writeLoc = y * Field_Width + x;
			buffer->spriteIndicesAndScales[writeLoc * 2] = Field_ImageTable[0];
			buffer->spriteIndicesAndScales[writeLoc * 2 + 1] = Field_TileScale;
			buffer->positionX[writeLoc] = (float)x / Field_Width * 2.0f - 1.0f;
			buffer->positionY[writeLoc] = (float)y / Field_Height * 2.0f - 1.0f;
		}
	}

	AI_FarmersMoveHotX = (float*)_mm_malloc(sizeof(float) * AI_FarmerCount, 64);
	AI_FarmersMoveHotY = (float*)_mm_malloc(sizeof(float) * AI_FarmerCount, 64);
	AI_FarmersMoveCold = (AI_FarmerMoveStateCold*)malloc(sizeof(AI_FarmerMoveStateCold) * AI_FarmerCount);
	AI_FarmersMoveGenX = (float*)_mm_malloc(sizeof(float) * AI_FarmerCount, 64);
	AI_FarmersMoveGenY = (float*)_mm_malloc(sizeof(float) * AI_FarmerCount, 64);

	AI_FarmersFarmHot = (AI_FarmerFarmStateHot*)malloc(sizeof(AI_FarmerFarmStateHot) * AI_FarmerCount);
	AI_FarmersFarmCold = (AI_FarmerFarmStateCold*)malloc(sizeof(AI_FarmerFarmStateCold) * AI_FarmerCount);
	AI_FarmersFarmGenX = (float*)_mm_malloc(sizeof(float) * AI_FarmerCount, 64);
	AI_FarmersFarmGenY = (float*)_mm_malloc(sizeof(float) * AI_FarmerCount, 64);

	AI_FarmersSearchHot = (AI_FarmerSearchStateHot*)malloc(sizeof(AI_FarmerSearchStateHot) * AI_FarmerCount);
	AI_FarmersSearchGenX = (float*)_mm_malloc(sizeof(float) * AI_FarmerCount, 64);
	AI_FarmersSearchGenY = (float*)_mm_malloc(sizeof(float) * AI_FarmerCount, 64);

	AI_FarmerRemovalIndices = (int*)_mm_malloc(sizeof(int) * AI_FarmerCount, 64);

	AI_FarmerSearchCount = AI_FarmerCount;

	memset(AI_FarmersSearchGenX, 0, sizeof(float) * AI_FarmerCount);
	memset(AI_FarmersSearchGenY, 0, sizeof(float) * AI_FarmerCount);
	for (uint32_t ai = 0; ai < AI_FarmerCount; ++ai)
	{
		AI_FarmersSearchHot[ai].searchTimer = rand_rangef(AI_FarmerSearchSpeedMin, AI_FarmerSearchSpeedMax);
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
	_mm_free(Field_CropLifetimes);
	free(Field_CropDrawCommands);

	free(Field_Tiles);
	free(Field_TileDrawCommands);

	_mm_free(AI_FarmersMoveHotX);
	_mm_free(AI_FarmersMoveHotY);
	free(AI_FarmersMoveCold);
	_mm_free(AI_FarmersMoveGenX);
	_mm_free(AI_FarmersMoveGenY);

	free(AI_FarmersFarmHot);
	free(AI_FarmersFarmCold);
	_mm_free(AI_FarmersFarmGenX);
	_mm_free(AI_FarmersFarmGenY);

	free(AI_FarmersSearchHot);
	_mm_free(AI_FarmersSearchGenX);
	_mm_free(AI_FarmersSearchGenY);

	_mm_free(AI_FarmerRemovalIndices);

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
		buffer->positionX[command->writeIndex] = command->posX;
		buffer->positionY[command->writeIndex] = command->posY;
	}
	Field_CropDrawCommandCount = 0;

	uint32_t writeIndex = Field_Width * Field_Height + Field_CropCount;

	memcpy(&buffer->positionX[writeIndex], AI_FarmersSearchGenX, sizeof(float) * AI_FarmerSearchCount);
	memcpy(&buffer->positionY[writeIndex], AI_FarmersSearchGenY, sizeof(float) * AI_FarmerSearchCount);
	__m128i searchAndScale = _mm_set_epi16(FarmerState_Search, AI_FarmerScale, FarmerState_Search, AI_FarmerScale, FarmerState_Search, AI_FarmerScale, FarmerState_Search, AI_FarmerScale);
	_mm_storeu_si128((__m128i*)&buffer->spriteIndicesAndScales[writeIndex * 2], searchAndScale);
	for (uint32_t i = (4 - writeIndex % 4); i < AI_FarmerSearchCount; i+=4)
	{
		_mm_stream_si128((__m128i*)&buffer->spriteIndicesAndScales[(writeIndex + i) * 2], searchAndScale);
	}
	writeIndex += AI_FarmerSearchCount;

	memcpy(&buffer->positionX[writeIndex], AI_FarmersMoveGenX, sizeof(float) * AI_FarmerMoveCount);
	memcpy(&buffer->positionY[writeIndex], AI_FarmersMoveGenY, sizeof(float) * AI_FarmerMoveCount);
	__m128i moveAndScale = _mm_set_epi16(FarmerState_Move, AI_FarmerScale, FarmerState_Move, AI_FarmerScale, FarmerState_Move, AI_FarmerScale, FarmerState_Move, AI_FarmerScale);
	_mm_storeu_si128((__m128i*)&buffer->spriteIndicesAndScales[writeIndex * 2], moveAndScale);
	for (uint32_t i = (4 - writeIndex % 4); i < AI_FarmerMoveCount; i+=4)
	{
		_mm_stream_si128((__m128i*)&buffer->spriteIndicesAndScales[(writeIndex + i) * 2], moveAndScale);
	}
	writeIndex += AI_FarmerMoveCount;

	memcpy(&buffer->positionX[writeIndex], AI_FarmersFarmGenX, sizeof(float) * AI_FarmerFarmCount);
	memcpy(&buffer->positionY[writeIndex], AI_FarmersFarmGenY, sizeof(float) * AI_FarmerFarmCount);
	__m128i farmAndScale = _mm_set_epi16(FarmerState_Farm, AI_FarmerScale, FarmerState_Farm, AI_FarmerScale, FarmerState_Farm, AI_FarmerScale, FarmerState_Farm, AI_FarmerScale);
	_mm_storeu_si128((__m128i*)&buffer->spriteIndicesAndScales[writeIndex * 2], farmAndScale);
	for (uint32_t i = (4 - writeIndex % 4); i < AI_FarmerFarmCount; i+=4)
	{
		_mm_stream_si128((__m128i*)&buffer->spriteIndicesAndScales[(writeIndex + i) * 2], farmAndScale);
	}
	writeIndex += AI_FarmerFarmCount;

	MIST_PROFILE_END("Game", "Game-GenInstanceBuffer");

	return writeIndex;
}
