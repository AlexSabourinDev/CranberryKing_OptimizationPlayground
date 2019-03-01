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

// Random
uint32_t rand_range(uint32_t min, uint32_t max)
{
	uint32_t randVal = (rand() << 15) | rand();
	return randVal % (max - min) + min;
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

#define SIMD_FLOAT_TO_HALF(f) _mm_extract_epi16(_mm_cvtps_ph(_mm_set_ss(f), _MM_FROUND_NO_EXC), 0)
#define SIMD_LOAD_PH_TO_PS(a) _mm256_cvtph_ps(_mm_load_si128((__m128i*)(a)))

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

static int* SIMD_FarmerRemovalIndices = NULL;

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
	uint16_t posX;
	uint16_t posY;
} Field_CropDrawCommand;

static uint32_t Field_CropDrawCommandCount = 0;
static Field_CropDrawCommand* Field_CropDrawCommands = NULL;

static uint32_t Field_CropCount = 0;
static Field_Crop* Field_Crops = NULL;
static uint16_t* Field_CropLifetimes = NULL;

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

#define Field_TimePrecision 1000
const float Crop_MinLifetime = 1 * Field_TimePrecision;
const float Crop_MaxLifetime = 10 * Field_TimePrecision;
const uint32_t Crop_MaxCropType = 4;

void field_tick(float delta)
{
	MIST_PROFILE_BEGIN("Game", "Field-Tick");

	int16_t deltaI = (int16_t)(delta * Field_TimePrecision);
	__m256i delta256 = _mm256_set_epi16(
		deltaI, deltaI, deltaI, deltaI, deltaI, deltaI, deltaI, deltaI,
		deltaI, deltaI, deltaI, deltaI, deltaI, deltaI, deltaI, deltaI);
	__m256i zeroI = _mm256_setzero_si256();

	uint32_t grownCropCount = 0;
	for (uint32_t i = 0; i < Field_CropCount; i+=16)
	{
		__m256i lifetime = _mm256_load_si256((__m256i*)(Field_CropLifetimes + i));
		lifetime = _mm256_sub_epi16(lifetime, delta256);
		_mm256_store_si256((__m256i*)(Field_CropLifetimes + i), lifetime);

		int bitMask = (1 << math_min(Field_CropCount - i, 16)) - 1;

		__m256i cmpRes = _mm256_cmpeq_epi16(_mm256_max_epi16(lifetime, zeroI), zeroI);
		int indexMask = _mm256_movemask_epi8(_mm256_packs_epi16(cmpRes, zeroI)) & bitMask;

		__m256i indices = _mm256_set_epi32(i, i, i, i, i, i, i, i);
		__m256i indexAdd = simd_moveMaskToIndexMask(indexMask);
		indices = _mm256_add_epi32(indices, indexAdd);

		_mm256_storeu_si256((__m256i*)(SIMD_FarmerRemovalIndices + grownCropCount), indices);
		grownCropCount += _mm_popcnt_u32(indexMask);
	}

	for (uint32_t i = 0; i < grownCropCount; i++)
	{
		int r = SIMD_FarmerRemovalIndices[i];
		Field_Crop* crop = &Field_Crops[r];

		Field_Tile* tile = &Field_Tiles[crop->tileIndex];
		tile->stage = FieldStage_Grown;

		Field_TileDrawCommands[Field_TileDrawCommandCount + i] = (Field_TileDrawCommand)
		{
			.writeIndex = crop->tileIndex,
			.spriteIndex = Field_ImageTable[FieldStage_Grown]
		};

		*crop = Field_Crops[Field_CropCount - 1];

		float div = (float)crop->tileIndex / Field_Width;
		float f = floorf(div);
		Field_CropDrawCommands[Field_CropDrawCommandCount + i] = (Field_CropDrawCommand)
		{
			.writeIndex = r,
			.spriteIndex = (7.0f + crop->cropType) / 11.0f * INT16_MAX,
			.posX = SIMD_FLOAT_TO_HALF((div - f) * 2.0f - 1.0f),
			.posY = SIMD_FLOAT_TO_HALF(f / Field_Height * 2.0f - 1.0f)
		};
	}
	Field_TileDrawCommandCount += grownCropCount;
	Field_CropDrawCommandCount += grownCropCount;
	Field_CropCount -= grownCropCount;

	MIST_PROFILE_END("Game", "Field-Tick");
}


// AI

const int16_t AI_FarmerScale = 0.025f * INT16_MAX;
const int16_t FarmerState_Search = 0;
const int16_t FarmerState_Move = 1.0f / 11.0f * INT16_MAX;
const int16_t FarmerState_Farm = 2.0f / 11.0f * INT16_MAX;

#define AI_TimePrecision 1000

const float AI_FarmerSpeed = 0.5f;
const float AI_FarmerCropRadius = 0.005f;
const int16_t AI_FarmerSearchSpeedMin = 0;
const int16_t AI_FarmerSearchSpeedMax = 1 * AI_TimePrecision;
const int16_t AI_FarmerFarmSpeedMin = 3 * AI_TimePrecision;
const int16_t AI_FarmerFarmSpeedMax = 5 * AI_TimePrecision;
#define AI_FarmerCount 1000000

static uint32_t AI_FarmerMoveCount = 0;
static uint16_t* AI_FarmersMoveHotX = NULL;
static uint16_t* AI_FarmersMoveHotY = NULL;
static uint32_t* AI_FarmersMoveCold = NULL;
static uint16_t* AI_FarmersMoveGenX = NULL;
static uint16_t* AI_FarmersMoveGenY = NULL;
static uint32_t AI_FarmerFarmCount = 0;
static int16_t* AI_FarmersFarmHot = NULL;
static uint32_t* AI_FarmersFarmCold = NULL;
static uint16_t* AI_FarmersFarmGenX = NULL;
static uint16_t* AI_FarmersFarmGenY = NULL;
static uint32_t AI_FarmerSearchCount = 0;
static int16_t* AI_FarmersSearchHot = NULL;
static uint16_t* AI_FarmersSearchGenX = NULL;
static uint16_t* AI_FarmersSearchGenY = NULL;

void ai_tick(float delta)
{
	MIST_PROFILE_BEGIN("Game", "AI-Tick");

	uint32_t previousFarmerMoveCount = AI_FarmerMoveCount;
	{
		int16_t deltaI = (int16_t)(delta * AI_TimePrecision);
		__m256i delta256 = _mm256_set_epi16(
			deltaI, deltaI, deltaI, deltaI, deltaI, deltaI, deltaI, deltaI,
			deltaI, deltaI, deltaI, deltaI , deltaI, deltaI, deltaI , deltaI);
		__m256i zeroI = _mm256_setzero_si256();

		uint32_t removedFarmerCount = 0;
		for (uint32_t i = 0; i < AI_FarmerSearchCount; i+=16)
		{
			__m256i farmerSearchTimer = _mm256_load_si256((__m256i*)(AI_FarmersSearchHot + i));
			farmerSearchTimer = _mm256_sub_epi16(farmerSearchTimer, delta256);
			_mm256_store_si256((__m256i*)(AI_FarmersSearchHot + i), farmerSearchTimer);

			int bitMask = (1 << math_min(AI_FarmerSearchCount - i, 16)) - 1;

			__m256i cmpRes = _mm256_cmpeq_epi16(_mm256_max_epi16(farmerSearchTimer, zeroI), zeroI);
			int indexMask = _mm256_movemask_epi8(_mm256_packs_epi16(cmpRes, zeroI)) & bitMask;

			int count = _mm_popcnt_u32(indexMask);
			if (count > 0)
			{
				__m256i indices = _mm256_set_epi32(i, i, i, i, i, i, i, i);
				__m256i indexAdd = simd_moveMaskToIndexMask(indexMask);
				indices = _mm256_add_epi32(indices, indexAdd);

				_mm256_storeu_si256((__m256i*)(SIMD_FarmerRemovalIndices + removedFarmerCount), indices);
				removedFarmerCount += count;
			}
		}

		uint32_t tileIndex1 = rand_range(0U, Field_Width * Field_Height);
		uint32_t tileIndex2 = rand_range(0U, Field_Width * Field_Height);
		uint32_t tileIndex3 = rand_range(0U, Field_Width * Field_Height);
		_mm_prefetch((const char*)(Field_Tiles + tileIndex3), _MM_HINT_T0);
		for(uint32_t i = 0; i < removedFarmerCount; i++)
		{
			uint32_t r = SIMD_FarmerRemovalIndices[i];

			uint32_t nextRemoval = SIMD_FarmerRemovalIndices[i + 2];
			_mm_prefetch((const char*)(AI_FarmersSearchHot + nextRemoval), _MM_HINT_T0);
			_mm_prefetch((const char*)(AI_FarmersSearchGenX + nextRemoval), _MM_HINT_T0);
			_mm_prefetch((const char*)(AI_FarmersSearchGenY + nextRemoval), _MM_HINT_T0);

			uint32_t tileIndex = tileIndex1;
			Field_Tile* tile = &Field_Tiles[tileIndex];

			tileIndex1 = tileIndex2;
			tileIndex2 = tileIndex3;
			tileIndex3 = rand_range(0U, Field_Width * Field_Height);
			_mm_prefetch((const char*)(Field_Tiles + tileIndex3), _MM_HINT_T0);

			if (tile->stage != FieldStage_Planted)
			{
				float div = (float)tileIndex / Field_Width;
				float f = floorf(div);
				AI_FarmersMoveHotX[AI_FarmerMoveCount] = SIMD_FLOAT_TO_HALF((div - f) * 2.0f - 1.0f);
				AI_FarmersMoveHotY[AI_FarmerMoveCount] = SIMD_FLOAT_TO_HALF(f / Field_Height * 2.0f - 1.0f);

				AI_FarmersMoveCold[AI_FarmerMoveCount] = tileIndex | (tile->stage << 28);

				AI_FarmersMoveGenX[AI_FarmerMoveCount] = AI_FarmersSearchGenX[r];
				AI_FarmersMoveGenY[AI_FarmerMoveCount] = AI_FarmersSearchGenY[r];
				AI_FarmerMoveCount++;

				AI_FarmersSearchHot[r] = AI_FarmersSearchHot[AI_FarmerSearchCount - 1];
				AI_FarmersSearchGenX[r] = AI_FarmersSearchGenX[AI_FarmerSearchCount - 1];
				AI_FarmersSearchGenY[r] = AI_FarmersSearchGenY[AI_FarmerSearchCount - 1];

				AI_FarmerSearchCount--;
			}
			else
			{
				AI_FarmersSearchHot[r] = rand_range(AI_FarmerSearchSpeedMin, AI_FarmerSearchSpeedMax);
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
			__m256 farmerX = SIMD_LOAD_PH_TO_PS(AI_FarmersMoveHotX + i);
			__m256 farmerY = SIMD_LOAD_PH_TO_PS(AI_FarmersMoveHotY + i);
			__m256 genFarmerX = SIMD_LOAD_PH_TO_PS(AI_FarmersMoveGenX + i);
			__m256 genFarmerY = SIMD_LOAD_PH_TO_PS(AI_FarmersMoveGenY + i);

			__m256 dirVecX = _mm256_sub_ps(farmerX, genFarmerX);
			__m256 dirVecY = _mm256_sub_ps(farmerY, genFarmerY);
			__m256 rmag = _mm256_rsqrt_ps(_mm256_add_ps(_mm256_mul_ps(dirVecX, dirVecX), _mm256_mul_ps(dirVecY, dirVecY)));

			__m256 magnitude = _mm256_mul_ps(velMag, rmag);
			genFarmerX = _mm256_fmadd_ps(dirVecX, magnitude, genFarmerX);
			genFarmerY = _mm256_fmadd_ps(dirVecY, magnitude, genFarmerY);
			_mm_store_si128((__m128i*)(AI_FarmersMoveGenX + i), _mm256_cvtps_ph(genFarmerX, _MM_FROUND_NO_EXC));
			_mm_store_si128((__m128i*)(AI_FarmersMoveGenY + i), _mm256_cvtps_ph(genFarmerY, _MM_FROUND_NO_EXC));

			int bitMask = (1 << math_min(previousFarmerMoveCount - i, 8)) - 1;

			// Since we're calculating reciprocals, the larger number is smaller
			// So instead of whether our velocity is larger than our distance,
			// we check to see if our r velocity is smaller than our r distance
			// vel = 1, dist = 0.5 vel > dist ? we passed it!
			// rvel = 1/1, dist = 1/0.5 rvel < rdist ? we passed it!
			__m256 cmpRes = _mm256_cmp_ps(rvelMag, rmag, _CMP_LT_OQ);
			int indexMask = _mm256_movemask_ps(cmpRes) & bitMask;

			int count = _mm_popcnt_u32(indexMask);
			if (count > 0)
			{
				__m256i indices = _mm256_set_epi32(i, i, i, i, i, i, i, i);
				__m256i indexAdd = simd_moveMaskToIndexMask(indexMask);
				indices = _mm256_add_epi32(indices, indexAdd);

				_mm256_storeu_si256((__m256i*)(SIMD_FarmerRemovalIndices + removedFarmerCount), indices);
				removedFarmerCount += count;
			}
		}


		for(uint32_t i = 0; i < removedFarmerCount; i++)
		{
			int r = SIMD_FarmerRemovalIndices[i];

			int nextRemoval = SIMD_FarmerRemovalIndices[i + 2];
			_mm_prefetch((const char*)(AI_FarmersMoveCold + nextRemoval), _MM_HINT_T0);
			_mm_prefetch((const char*)(AI_FarmersMoveGenX + nextRemoval), _MM_HINT_T0);
			_mm_prefetch((const char*)(AI_FarmersMoveGenY + nextRemoval), _MM_HINT_T0);

			AI_FarmersFarmHot[AI_FarmerFarmCount + i] = rand_range(AI_FarmerFarmSpeedMin, AI_FarmerFarmSpeedMax);
			AI_FarmersFarmCold[AI_FarmerFarmCount + i] = AI_FarmersMoveCold[r];
			AI_FarmersFarmGenX[AI_FarmerFarmCount + i] = AI_FarmersMoveGenX[r];
			AI_FarmersFarmGenY[AI_FarmerFarmCount + i] = AI_FarmersMoveGenY[r];

			AI_FarmersMoveHotX[r] = AI_FarmersMoveHotX[AI_FarmerMoveCount - 1 - i];
			AI_FarmersMoveHotY[r] = AI_FarmersMoveHotY[AI_FarmerMoveCount - 1 - i];
			AI_FarmersMoveCold[r] = AI_FarmersMoveCold[AI_FarmerMoveCount - 1 - i];
			AI_FarmersMoveGenX[r] = AI_FarmersMoveGenX[AI_FarmerMoveCount - 1 - i];
			AI_FarmersMoveGenY[r] = AI_FarmersMoveGenY[AI_FarmerMoveCount - 1 - i];
		}

		AI_FarmerMoveCount -= removedFarmerCount;
		AI_FarmerFarmCount += removedFarmerCount;
	}

	{
		uint32_t removedFarmerCount = 0;

		int16_t deltaI = (int16_t)(delta * AI_TimePrecision);
		__m256i delta256 = _mm256_set_epi16(
			deltaI, deltaI, deltaI, deltaI, deltaI, deltaI, deltaI, deltaI,
			deltaI, deltaI, deltaI, deltaI, deltaI, deltaI, deltaI, deltaI);
		__m256i zeroI = _mm256_setzero_si256();

		// We only do the farmers that were already in the farm array before this tick
		for (uint32_t i = 0; i < previousFarmerFarmCount; i+=16)
		{
			__m256i farmerFarmTimer = _mm256_load_si256((__m256i*)(AI_FarmersFarmHot + i));
			farmerFarmTimer = _mm256_sub_epi16(farmerFarmTimer, delta256);
			_mm256_store_si256((__m256i*)(AI_FarmersFarmHot + i), farmerFarmTimer);

			int bitMask = (1 << math_min(previousFarmerFarmCount - i, 16)) - 1;

			__m256i cmpRes = _mm256_cmpeq_epi16(_mm256_max_epi16(farmerFarmTimer, zeroI), zeroI);
			int indexMask = _mm256_movemask_epi8(_mm256_packs_epi16(cmpRes, zeroI)) & bitMask;

			int count = _mm_popcnt_u32(indexMask);
			if (count > 0)
			{
				__m256i indices = _mm256_set_epi32(i, i, i, i, i, i, i, i);
				__m256i indexAdd = simd_moveMaskToIndexMask(indexMask);
				indices = _mm256_add_epi32(indices, indexAdd);

				_mm256_storeu_si256((__m256i*)(SIMD_FarmerRemovalIndices + removedFarmerCount), indices);
				removedFarmerCount += count;
			}
		}

		for(uint32_t i = 0; i < removedFarmerCount; i++)
		{
			int r = SIMD_FarmerRemovalIndices[i];

			int nextRemoval = SIMD_FarmerRemovalIndices[i + 2];
			_mm_prefetch((const char*)(AI_FarmersFarmCold + nextRemoval), _MM_HINT_T0);
			_mm_prefetch((const char*)(AI_FarmersFarmGenX + nextRemoval), _MM_HINT_T0);
			_mm_prefetch((const char*)(AI_FarmersFarmGenY + nextRemoval), _MM_HINT_T0);
			
			uint32_t indexAndStage = AI_FarmersFarmCold[r];
			uint32_t index = indexAndStage & ((1 << 28) - 1);
			uint32_t stage = indexAndStage >> 28;
			uint32_t newStage = math_max((stage + 1) % FieldState_Max, FieldStage_Fallow);

			Field_Tile* tile = &Field_Tiles[index];
			tile->stage = newStage;

			Field_TileDrawCommands[Field_TileDrawCommandCount + i] = (Field_TileDrawCommand)
				{
					.writeIndex = index,
					.spriteIndex = Field_ImageTable[newStage]
				};

			if (newStage == FieldStage_Planted)
			{
				Field_Crop* crop = &Field_Crops[Field_CropCount];
				Field_CropLifetimes[Field_CropCount] = rand_rangef(Crop_MinLifetime, Crop_MaxLifetime);
				crop->cropType = rand_range(0, Crop_MaxCropType);
				crop->tileIndex = index;

				float div = (float)crop->tileIndex / Field_Width;
				float f = floorf(div);
				Field_CropDrawCommands[Field_CropDrawCommandCount] = (Field_CropDrawCommand)
					{
						.writeIndex = Field_CropCount,
						.spriteIndex = (7.0f + crop->cropType) / 11.0f * INT16_MAX,
						.posX = SIMD_FLOAT_TO_HALF((div - f) * 2.0f - 1.0f),
						.posY = SIMD_FLOAT_TO_HALF(f / Field_Height * 2.0f - 1.0f)
					};
				Field_CropDrawCommandCount++;

				Field_CropCount++;
			}

			AI_FarmersSearchHot[AI_FarmerSearchCount + i] = rand_range(AI_FarmerSearchSpeedMin, AI_FarmerSearchSpeedMax);
			AI_FarmersSearchGenX[AI_FarmerSearchCount + i] = AI_FarmersFarmGenX[r];
			AI_FarmersSearchGenY[AI_FarmerSearchCount + i] = AI_FarmersFarmGenY[r];

			AI_FarmersFarmHot[r] = AI_FarmersFarmHot[AI_FarmerFarmCount - 1 - i];
			AI_FarmersFarmCold[r] = AI_FarmersFarmCold[AI_FarmerFarmCount - 1 - i];
			AI_FarmersFarmGenX[r] = AI_FarmersFarmGenX[AI_FarmerFarmCount - 1 - i];
			AI_FarmersFarmGenY[r] = AI_FarmersFarmGenY[AI_FarmerFarmCount - 1 - i];
		}

		AI_FarmerSearchCount += removedFarmerCount;
		Field_TileDrawCommandCount += removedFarmerCount;
		AI_FarmerFarmCount -= removedFarmerCount;
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
	Field_CropLifetimes = (uint16_t*)_mm_malloc(sizeof(uint16_t) * Field_Width * Field_Height, 64);
	Field_CropDrawCommands = (Field_CropDrawCommand*)malloc(sizeof(Field_CropDrawCommand) * Field_Width * Field_Height);

	for (uint32_t y = 0; y < Field_Height; ++y)
	{
		for (uint32_t x = 0; x < Field_Width; ++x)
		{
			uint32_t writeLoc = y * Field_Width + x;
			buffer->spriteIndicesAndScales[writeLoc * 2] = Field_ImageTable[0];
			buffer->spriteIndicesAndScales[writeLoc * 2 + 1] = Field_TileScale;
			buffer->positionX[writeLoc] = SIMD_FLOAT_TO_HALF((float)x / Field_Width * 2.0f - 1.0f);
			buffer->positionY[writeLoc] = SIMD_FLOAT_TO_HALF((float)y / Field_Height * 2.0f - 1.0f);
		}
	}

	AI_FarmersMoveHotX = (uint16_t*)_mm_malloc(sizeof(uint16_t) * AI_FarmerCount, 64);
	AI_FarmersMoveHotY = (uint16_t*)_mm_malloc(sizeof(uint16_t) * AI_FarmerCount, 64);
	AI_FarmersMoveCold = (uint32_t*)malloc(sizeof(uint32_t) * AI_FarmerCount);
	AI_FarmersMoveGenX = (uint16_t*)_mm_malloc(sizeof(uint16_t) * AI_FarmerCount, 64);
	AI_FarmersMoveGenY = (uint16_t*)_mm_malloc(sizeof(uint16_t) * AI_FarmerCount, 64);

	AI_FarmersFarmHot = (int16_t*)_mm_malloc(sizeof(int16_t) * AI_FarmerCount, 64);
	AI_FarmersFarmCold = (uint32_t*)malloc(sizeof(uint32_t) * AI_FarmerCount);
	AI_FarmersFarmGenX = (uint16_t*)_mm_malloc(sizeof(uint16_t) * AI_FarmerCount, 64);
	AI_FarmersFarmGenY = (uint16_t*)_mm_malloc(sizeof(uint16_t) * AI_FarmerCount, 64);

	AI_FarmersSearchHot = (int16_t*)_mm_malloc(sizeof(int16_t) * AI_FarmerCount, 64);
	AI_FarmersSearchGenX = (uint16_t*)_mm_malloc(sizeof(uint16_t) * AI_FarmerCount, 64);
	AI_FarmersSearchGenY = (uint16_t*)_mm_malloc(sizeof(uint16_t) * AI_FarmerCount, 64);

	SIMD_FarmerRemovalIndices = (int*)_mm_malloc(sizeof(int) * AI_FarmerCount, 64);

	AI_FarmerSearchCount = AI_FarmerCount;

	memset(AI_FarmersSearchGenX, 0, sizeof(uint16_t) * AI_FarmerCount);
	memset(AI_FarmersSearchGenY, 0, sizeof(uint16_t) * AI_FarmerCount);
	for (uint32_t ai = 0; ai < AI_FarmerCount; ++ai)
	{
		AI_FarmersSearchHot[ai] = rand_range(AI_FarmerSearchSpeedMin, AI_FarmerSearchSpeedMax);
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

	_mm_free(AI_FarmersFarmHot);
	free(AI_FarmersFarmCold);
	_mm_free(AI_FarmersFarmGenX);
	_mm_free(AI_FarmersFarmGenY);

	_mm_free(AI_FarmersSearchHot);
	_mm_free(AI_FarmersSearchGenX);
	_mm_free(AI_FarmersSearchGenY);

	_mm_free(SIMD_FarmerRemovalIndices);

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

	memcpy(&buffer->positionX[writeIndex], AI_FarmersSearchGenX, sizeof(uint16_t) * AI_FarmerSearchCount);
	memcpy(&buffer->positionY[writeIndex], AI_FarmersSearchGenY, sizeof(uint16_t) * AI_FarmerSearchCount);
	__m128i searchAndScale = _mm_set_epi16(FarmerState_Search, AI_FarmerScale, FarmerState_Search, AI_FarmerScale, FarmerState_Search, AI_FarmerScale, FarmerState_Search, AI_FarmerScale);
	_mm_storeu_si128((__m128i*)&buffer->spriteIndicesAndScales[writeIndex * 2], searchAndScale);
	for (uint32_t i = (4 - writeIndex % 4); i < AI_FarmerSearchCount; i+=4)
	{
		_mm_stream_si128((__m128i*)&buffer->spriteIndicesAndScales[(writeIndex + i) * 2], searchAndScale);
	}
	writeIndex += AI_FarmerSearchCount;

	memcpy(&buffer->positionX[writeIndex], AI_FarmersMoveGenX, sizeof(uint16_t) * AI_FarmerMoveCount);
	memcpy(&buffer->positionY[writeIndex], AI_FarmersMoveGenY, sizeof(uint16_t) * AI_FarmerMoveCount);
	__m128i moveAndScale = _mm_set_epi16(FarmerState_Move, AI_FarmerScale, FarmerState_Move, AI_FarmerScale, FarmerState_Move, AI_FarmerScale, FarmerState_Move, AI_FarmerScale);
	_mm_storeu_si128((__m128i*)&buffer->spriteIndicesAndScales[writeIndex * 2], moveAndScale);
	for (uint32_t i = (4 - writeIndex % 4); i < AI_FarmerMoveCount; i+=4)
	{
		_mm_stream_si128((__m128i*)&buffer->spriteIndicesAndScales[(writeIndex + i) * 2], moveAndScale);
	}
	writeIndex += AI_FarmerMoveCount;

	memcpy(&buffer->positionX[writeIndex], AI_FarmersFarmGenX, sizeof(uint16_t) * AI_FarmerFarmCount);
	memcpy(&buffer->positionY[writeIndex], AI_FarmersFarmGenY, sizeof(uint16_t) * AI_FarmerFarmCount);
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
