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

int32_t math_maxi(int32_t l, int32_t r)
{
	return l > r ? l : r;
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

#define SIMD_FLOAT_TO_HALF(f) _mm_extract_epi16(_mm_cvtps_ph(_mm_set_ss(f), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC), 0)
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

// https://godbolt.org/z/jxXHVJ
void simd_streamMemCpy(__m256i* dstWrite, __m256i* srcRead, size_t size)
{
	if (size == 0)
	{
		return;
	}

	__m256i* dstEnd = dstWrite + (size >> 5);
	for (; dstWrite <= dstEnd; dstWrite += 1, srcRead += 1)
	{
		__m256i src256 = _mm256_stream_load_si256(srcRead);
		_mm256_stream_si256(dstWrite, src256);
	}
}

// https://godbolt.org/z/5wjfO3
void simd_memSetToValue(__m256i* dstWrite, __m256i value, size_t size)
{
	if (size == 0)
	{
		return;
	}

	__m256i* dstEnd = dstWrite + (size >> 5);
	for (; dstWrite <= dstEnd; dstWrite += 1)
	{
		_mm256_stream_si256(dstWrite, value);
	}
}

static unsigned int* SIMD_FarmerRemovalIndices = NULL;

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
static int16_t* Field_CropLifetimes = NULL;

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
	[FieldStage_Arable] = (int16_t)(3.0f / 11.0f * INT16_MAX),
	[FieldStage_Fallow] = (int16_t)(4.0f / 11.0f * INT16_MAX),
	[FieldStage_Planted] = (int16_t)(5.0f / 11.0f * INT16_MAX),
	[FieldStage_Grown] = (int16_t)(6.0f / 11.0f * INT16_MAX)
};

static uint32_t Field_TileDrawCommandCount = 0;
static Field_TileDrawCommand* Field_TileDrawCommands = NULL;

#define Field_Width 1000
#define Field_Height 1000
static Field_Tile* Field_Tiles = NULL;

const int16_t Field_TileScale = (int16_t)((2.0f / Field_Width) * INT16_MAX);

#define Field_TimePrecision 1000
const int16_t Crop_MinLifetime = 1 * Field_TimePrecision;
const int16_t Crop_MaxLifetime = 10 * Field_TimePrecision;
const uint32_t Crop_MaxCropType = 4;

void field_tick(float delta)
{
	MIST_PROFILE_BEGIN("Game", "Field-Tick");

	int16_t deltaI = (int16_t)(delta * Field_TimePrecision);
	__m256i delta256 = _mm256_set1_epi16(deltaI);
	__m256i zeroI = _mm256_setzero_si256();

	uint32_t grownCropCount = 0;
	for (uint32_t i = 0; i < Field_CropCount; i+=16)
	{
		__m256i lifetime = _mm256_load_si256((__m256i*)(Field_CropLifetimes + i));
		lifetime = _mm256_sub_epi16(lifetime, delta256);
		_mm256_store_si256((__m256i*)(Field_CropLifetimes + i), lifetime);

		unsigned int bitMask = (1 << math_min(Field_CropCount - i, 16)) - 1;

		__m256i cmpRes = _mm256_cmpgt_epi16(zeroI, lifetime);
		uint32_t moveMask = _mm256_movemask_epi8(cmpRes);
		int indexMask = _pext_u32(moveMask, 0x55555555) & bitMask;

		if (indexMask != 0)
		{
			__m256i indices = _mm256_set_epi32(i, i, i, i, i, i, i, i);
			__m256i indexAdd = simd_moveMaskToIndexMask(indexMask & 0x00FF);
			indices = _mm256_add_epi32(indices, indexAdd);

			_mm256_storeu_si256((__m256i*)(SIMD_FarmerRemovalIndices + grownCropCount), indices);
			grownCropCount += _mm_popcnt_u32(indexMask & 0x00FF);

			__m256i next8Indices = _mm256_set_epi32(i + 8, i + 8, i + 8, i + 8, i + 8, i + 8, i + 8, i + 8);
			__m256i nextIndexAdd = simd_moveMaskToIndexMask((indexMask & 0xFF00) >> 8);
			next8Indices = _mm256_add_epi32(next8Indices, nextIndexAdd);

			_mm256_storeu_si256((__m256i*)(SIMD_FarmerRemovalIndices + grownCropCount), next8Indices);
			grownCropCount += _mm_popcnt_u32(indexMask & 0xFF00);
		}
	}

	for (int32_t i = grownCropCount - 1; i >= 0; i--)
	{
		unsigned int r = SIMD_FarmerRemovalIndices[i];
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
			.spriteIndex = (int16_t)((7.0f + crop->cropType) / 11.0f * INT16_MAX),
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

const int16_t AI_FarmerScale = (int16_t)(0.025f * INT16_MAX);
const int16_t FarmerState_Search = 0;
const int16_t FarmerState_Move = (int16_t)(1.0f / 11.0f * INT16_MAX);
const int16_t FarmerState_Farm = (int16_t)(2.0f / 11.0f * INT16_MAX);

#define AI_TimePrecision 100

const float AI_FarmerSpeed = 0.5f;
const int8_t AI_FarmerSearchSpeedMin = 0;
const int8_t AI_FarmerSearchSpeedMax = 1 * AI_TimePrecision;
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

#define AI_FarmerFarmBucketCount 6
static int8_t* AI_FarmersFarmHotBuckets[AI_FarmerFarmBucketCount] = { NULL };
static uint32_t* AI_FarmersFarmHotBucketIndices[AI_FarmerFarmBucketCount] = { NULL };
static int32_t AI_FarmersFarmHotBucketCounts[AI_FarmerFarmBucketCount] = { 0 };
static int8_t AI_FarmersFarmBucketTransitionTimer = 0;
static uint32_t AI_FarmersFarmFineTimerBucket = 0;

static uint32_t* AI_FarmersFarmCold = NULL;
static uint16_t* AI_FarmersFarmGenX = NULL;
static uint16_t* AI_FarmersFarmGenY = NULL;
static uint32_t AI_FarmerSearchCount = 0;

#define AI_FarmerSearchBucketCount 6
static int8_t* AI_FarmersSearchHotBuckets[AI_FarmerSearchBucketCount] = { NULL };
static uint32_t* AI_FarmersSearchHotBucketIndices[AI_FarmerSearchBucketCount] = { NULL };
static int32_t AI_FarmersSearchHotBucketCounts[AI_FarmerSearchBucketCount] = { 0 };
static int8_t AI_FarmersSearchBucketTransitionTimer = 0;
static uint32_t AI_FarmersSearchFineTimerBucket = 0;

static uint16_t* AI_FarmersSearchGenX = NULL;
static uint16_t* AI_FarmersSearchGenY = NULL;

void ai_tick(float delta)
{
	MIST_PROFILE_BEGIN("Game", "AI-Tick");

	AI_FarmersFarmBucketTransitionTimer += (int16_t)(delta * AI_TimePrecision);
	AI_FarmersSearchBucketTransitionTimer += (int16_t)(delta * AI_TimePrecision);

	uint32_t previousFarmerMoveCount = AI_FarmerMoveCount;
	{
		int8_t deltaI = (int8_t)(delta * AI_TimePrecision);
		__m256i delta256 = _mm256_set1_epi8(deltaI);
		__m256i zeroI = _mm256_setzero_si256();

		int8_t* activeFineBucket = AI_FarmersSearchHotBuckets[AI_FarmersSearchFineTimerBucket];

		uint32_t farmerCount = AI_FarmersSearchHotBucketCounts[AI_FarmersSearchFineTimerBucket];
		uint32_t removedFarmerCount = 0;
		for (uint32_t i = 0; i < farmerCount; i+=32)
		{
			__m256i farmerSearchTimer = _mm256_load_si256((__m256i*)(activeFineBucket + i));
			farmerSearchTimer = _mm256_sub_epi8(farmerSearchTimer, delta256);
			_mm256_store_si256((__m256i*)(activeFineBucket + i), farmerSearchTimer);

			uint64_t bitMask = (1ULL << math_min(farmerCount - i, 32ULL)) - 1ULL;

			__m256i cmpRes = _mm256_cmpgt_epi8(zeroI, farmerSearchTimer);
			uint32_t moveMask = _mm256_movemask_epi8(cmpRes);
			uint32_t indexMask = moveMask & bitMask;

			if (indexMask != 0)
			{
				for(uint32_t maskI = 0; maskI < 4; maskI++)
				{
					uint32_t shiftedMask = indexMask & (0x000000FF << (maskI * 8));

					__m256i indices = _mm256_set1_epi32(i + 8 * maskI);
					__m256i indexAdd = simd_moveMaskToIndexMask(shiftedMask >> (maskI * 8));
					indices = _mm256_add_epi32(indices, indexAdd);

					_mm256_storeu_si256((__m256i*)(SIMD_FarmerRemovalIndices + removedFarmerCount), indices);
					removedFarmerCount += _mm_popcnt_u32(shiftedMask);
				}
			}
		}

		uint32_t* activeBucketIndices = AI_FarmersSearchHotBucketIndices[AI_FarmersSearchFineTimerBucket];

		uint32_t tileIndex1 = rand_range(0U, Field_Width * Field_Height);
		uint32_t tileIndex2 = rand_range(0U, Field_Width * Field_Height);
		uint32_t tileIndex3 = rand_range(0U, Field_Width * Field_Height);
		_mm_prefetch((const char*)(Field_Tiles + tileIndex3), _MM_HINT_T0);
		for(int32_t i = removedFarmerCount - 1; i >= 0; i--)
		{
			int removedTimer = SIMD_FarmerRemovalIndices[i];
			uint32_t removedFarmerIndex = activeBucketIndices[removedTimer];

			int nextRemovedTimer = SIMD_FarmerRemovalIndices[math_maxi(i - 2, 0)];
			uint32_t nextRemovedFarmerIndex = activeBucketIndices[nextRemovedTimer];

			int thirdRemovedTimer = SIMD_FarmerRemovalIndices[math_maxi(i - 4, 0)];

			_mm_prefetch((const char*)(activeBucketIndices + thirdRemovedTimer), _MM_HINT_T0);
			_mm_prefetch((const char*)(AI_FarmersSearchGenX + nextRemovedFarmerIndex), _MM_HINT_T0);
			_mm_prefetch((const char*)(AI_FarmersSearchGenY + nextRemovedFarmerIndex), _MM_HINT_T0);

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

				AI_FarmersMoveGenX[AI_FarmerMoveCount] = AI_FarmersSearchGenX[removedFarmerIndex];
				AI_FarmersMoveGenY[AI_FarmerMoveCount] = AI_FarmersSearchGenY[removedFarmerIndex];
				AI_FarmerMoveCount++;

				AI_FarmersSearchGenX[removedFarmerIndex] = AI_FarmersSearchGenX[AI_FarmerSearchCount - 1];
				AI_FarmersSearchGenY[removedFarmerIndex] = AI_FarmersSearchGenY[AI_FarmerSearchCount - 1];

				AI_FarmerSearchCount--;
			}
			else
			{
				int16_t farmerTimer = rand_range(AI_FarmerSearchSpeedMin, AI_FarmerSearchSpeedMax);
				uint32_t bucketSecond = (farmerTimer + AI_FarmersSearchBucketTransitionTimer) / (AI_TimePrecision / (AI_FarmerSearchBucketCount - 1));
				uint32_t bucketIndex = (math_min(bucketSecond, AI_FarmerSearchBucketCount - 1) + AI_FarmersSearchFineTimerBucket) % AI_FarmerSearchBucketCount;
				uint32_t bucketFarmerCount = AI_FarmersSearchHotBucketCounts[bucketIndex];

				AI_FarmersSearchHotBuckets[bucketIndex][bucketFarmerCount] = farmerTimer - bucketSecond * (AI_TimePrecision / (AI_FarmerSearchBucketCount - 1));
				AI_FarmersSearchHotBucketIndices[bucketIndex][bucketFarmerCount] = activeBucketIndices[removedTimer];
				AI_FarmersSearchHotBucketCounts[bucketIndex]++;
			}

			uint32_t currentHotBucketCount = AI_FarmersSearchHotBucketCounts[AI_FarmersSearchFineTimerBucket];
			activeFineBucket[removedTimer] = activeFineBucket[currentHotBucketCount - 1];
			activeBucketIndices[removedTimer] = activeBucketIndices[currentHotBucketCount - 1];
			AI_FarmersSearchHotBucketCounts[AI_FarmersSearchFineTimerBucket]--;
		}


		if (AI_FarmersSearchBucketTransitionTimer >= 1 * (AI_TimePrecision / (AI_FarmerSearchBucketCount - 1)))
		{
			AI_FarmersSearchBucketTransitionTimer = 0;
			AI_FarmersSearchFineTimerBucket = (AI_FarmersSearchFineTimerBucket + 1) % AI_FarmerSearchBucketCount;
		}

	}

	uint32_t previousFarmerFarmCount = AI_FarmersFarmHotBucketCounts[AI_FarmersFarmFineTimerBucket];
	{
		float v = AI_FarmerSpeed * delta;
		float rv = 1.0f / v;
		__m256 velMag = _mm256_set1_ps(v);
		__m256 rvelMag = _mm256_set1_ps(rv);

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

			unsigned int bitMask = (1 << math_min(previousFarmerMoveCount - i, 8)) - 1;

			// Since we're calculating reciprocals, the larger number is smaller
			// So instead of whether our velocity is larger than our distance,
			// we check to see if our r velocity is smaller than our r distance
			// vel = 1, dist = 0.5 vel > dist ? we passed it!
			// rvel = 1/1, dist = 1/0.5 rvel < rdist ? we passed it!
			__m256 cmpRes = _mm256_cmp_ps(rvelMag, rmag, _CMP_LT_OQ);
			unsigned int indexMask = _mm256_movemask_ps(cmpRes) & bitMask;

			if (indexMask != 0)
			{
				__m256i indices = _mm256_set_epi32(i, i, i, i, i, i, i, i);
				__m256i indexAdd = simd_moveMaskToIndexMask(indexMask);
				indices = _mm256_add_epi32(indices, indexAdd);

				_mm256_storeu_si256((__m256i*)(SIMD_FarmerRemovalIndices + removedFarmerCount), indices);
				removedFarmerCount += _mm_popcnt_u32(indexMask);
			}
		}


		for(int32_t i = removedFarmerCount - 1; i >= 0; i--)
		{
			unsigned int r = SIMD_FarmerRemovalIndices[i];

			int nextRemoval = SIMD_FarmerRemovalIndices[math_maxi(i - 2, 0)];
			_mm_prefetch((const char*)(AI_FarmersMoveCold + nextRemoval), _MM_HINT_T0);
			_mm_prefetch((const char*)(AI_FarmersMoveGenX + nextRemoval), _MM_HINT_T0);
			_mm_prefetch((const char*)(AI_FarmersMoveGenY + nextRemoval), _MM_HINT_T0);

			int16_t farmerTimer = rand_range(AI_FarmerFarmSpeedMin, AI_FarmerFarmSpeedMax);
			uint32_t bucketSecond = (farmerTimer + AI_FarmersFarmBucketTransitionTimer) / AI_TimePrecision;
			uint32_t bucketIndex = (math_min(bucketSecond, AI_FarmerFarmBucketCount - 1) + AI_FarmersFarmFineTimerBucket) % AI_FarmerFarmBucketCount;
			uint32_t bucketFarmerCount = AI_FarmersFarmHotBucketCounts[bucketIndex];

			AI_FarmersFarmHotBuckets[bucketIndex][bucketFarmerCount] = farmerTimer - bucketSecond * AI_TimePrecision;
			AI_FarmersFarmHotBucketIndices[bucketIndex][bucketFarmerCount] = AI_FarmerFarmCount + i;
			AI_FarmersFarmHotBucketCounts[bucketIndex]++;

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

		int8_t deltaI = (int8_t)(delta * AI_TimePrecision);
		__m256i delta256 = _mm256_set1_epi8(deltaI);
		__m256i zeroI = _mm256_setzero_si256();

		int8_t* activeFineBucket = AI_FarmersFarmHotBuckets[AI_FarmersFarmFineTimerBucket];

		// We only do the farmers that were already in the farm array before this tick
		for (uint32_t i = 0; i < previousFarmerFarmCount; i+=32)
		{
			__m256i farmerFarmTimer = _mm256_load_si256((__m256i*)(activeFineBucket + i));
			farmerFarmTimer = _mm256_sub_epi8(farmerFarmTimer, delta256);
			_mm256_store_si256((__m256i*)(activeFineBucket + i), farmerFarmTimer);

			uint64_t bitMask = (1ULL << math_min(previousFarmerFarmCount - i, 32ULL)) - 1ULL;

			__m256i cmpRes = _mm256_cmpgt_epi8(zeroI, farmerFarmTimer);
			uint32_t moveMask = _mm256_movemask_epi8(cmpRes);
			uint32_t indexMask = moveMask & bitMask;

			if (indexMask != 0)
			{
				for (uint32_t maskI = 0; maskI < 4; maskI++)
				{
					uint32_t shiftedMask = indexMask & (0x000000FF << (maskI * 8));

					__m256i indices = _mm256_set1_epi32(i + 8 * maskI);
					__m256i indexAdd = simd_moveMaskToIndexMask(shiftedMask >> (maskI * 8));
					indices = _mm256_add_epi32(indices, indexAdd);

					_mm256_storeu_si256((__m256i*)(SIMD_FarmerRemovalIndices + removedFarmerCount), indices);
					removedFarmerCount += _mm_popcnt_u32(shiftedMask);
				}
			}
		}

		uint32_t currentHotBucketCount = AI_FarmersFarmHotBucketCounts[AI_FarmersFarmFineTimerBucket];
		uint32_t* activeBucketIndices = AI_FarmersFarmHotBucketIndices[AI_FarmersFarmFineTimerBucket];

		for(int32_t i = removedFarmerCount - 1; i >= 0; i--)
		{
			int removedTimer = SIMD_FarmerRemovalIndices[i];
			uint32_t removedFarmerIndex = activeBucketIndices[removedTimer];

			int nextRemovedTimer = SIMD_FarmerRemovalIndices[math_maxi(i - 2, 0)];
			uint32_t nextRemovedFarmerIndex = activeBucketIndices[nextRemovedTimer];

			int thirdRemovedTimer = SIMD_FarmerRemovalIndices[math_maxi(i - 4, 0)];

			_mm_prefetch((const char*)(activeBucketIndices + thirdRemovedTimer), _MM_HINT_T0);
			_mm_prefetch((const char*)(AI_FarmersFarmCold + nextRemovedFarmerIndex), _MM_HINT_T0);
			_mm_prefetch((const char*)(AI_FarmersFarmGenX + nextRemovedFarmerIndex), _MM_HINT_T0);
			_mm_prefetch((const char*)(AI_FarmersFarmGenY + nextRemovedFarmerIndex), _MM_HINT_T0);
			
			uint32_t indexAndStage = AI_FarmersFarmCold[removedFarmerIndex];
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
				Field_CropLifetimes[Field_CropCount] = rand_range(Crop_MinLifetime, Crop_MaxLifetime);
				crop->cropType = rand_range(0, Crop_MaxCropType);
				crop->tileIndex = index;

				float div = (float)crop->tileIndex / Field_Width;
				float f = floorf(div);
				Field_CropDrawCommands[Field_CropDrawCommandCount] = (Field_CropDrawCommand)
					{
						.writeIndex = Field_CropCount,
						.spriteIndex = (int16_t)((7.0f + crop->cropType) / 11.0f * INT16_MAX),
						.posX = SIMD_FLOAT_TO_HALF((div - f) * 2.0f - 1.0f),
						.posY = SIMD_FLOAT_TO_HALF(f / Field_Height * 2.0f - 1.0f)
					};
				Field_CropDrawCommandCount++;

				Field_CropCount++;
			}

			int16_t farmerTimer = rand_range(AI_FarmerSearchSpeedMin, AI_FarmerSearchSpeedMax);
			uint32_t bucketSecond = (farmerTimer + AI_FarmersSearchBucketTransitionTimer) / (AI_TimePrecision / (AI_FarmerSearchBucketCount - 1));
			uint32_t bucketIndex = (math_min(bucketSecond, AI_FarmerSearchBucketCount - 1) + AI_FarmersSearchFineTimerBucket) % AI_FarmerSearchBucketCount;
			uint32_t bucketFarmerCount = AI_FarmersSearchHotBucketCounts[bucketIndex];

			AI_FarmersSearchHotBuckets[bucketIndex][bucketFarmerCount] = farmerTimer - bucketSecond * (AI_TimePrecision / (AI_FarmerSearchBucketCount - 1));
			AI_FarmersSearchHotBucketIndices[bucketIndex][bucketFarmerCount] = AI_FarmerSearchCount + i;
			AI_FarmersSearchHotBucketCounts[bucketIndex]++;

			AI_FarmersSearchGenX[AI_FarmerSearchCount + i] = AI_FarmersFarmGenX[removedFarmerIndex];
			AI_FarmersSearchGenY[AI_FarmerSearchCount + i] = AI_FarmersFarmGenY[removedFarmerIndex];


			activeFineBucket[removedTimer] = activeFineBucket[currentHotBucketCount - 1 - i];
			activeBucketIndices[removedTimer] = activeBucketIndices[currentHotBucketCount - 1 - i];
			AI_FarmersFarmCold[removedFarmerIndex] = AI_FarmersFarmCold[AI_FarmerFarmCount - 1 - i];
			AI_FarmersFarmGenX[removedFarmerIndex] = AI_FarmersFarmGenX[AI_FarmerFarmCount - 1 - i];
			AI_FarmersFarmGenY[removedFarmerIndex] = AI_FarmersFarmGenY[AI_FarmerFarmCount - 1 - i];
		}

		AI_FarmerSearchCount += removedFarmerCount;
		Field_TileDrawCommandCount += removedFarmerCount;
		AI_FarmerFarmCount -= removedFarmerCount;
		AI_FarmersFarmHotBucketCounts[AI_FarmersFarmFineTimerBucket] -= removedFarmerCount;
		
		if (AI_FarmersFarmBucketTransitionTimer >= 1 * AI_TimePrecision)
		{
			AI_FarmersFarmBucketTransitionTimer = 0;
			AI_FarmersFarmFineTimerBucket = (AI_FarmersFarmFineTimerBucket + 1) % AI_FarmerFarmBucketCount;
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
	Field_CropLifetimes = (int16_t*)_mm_malloc(sizeof(int16_t) * Field_Width * Field_Height, 64);
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

	for (uint32_t i = 0; i < AI_FarmerFarmBucketCount; i++)
	{
		AI_FarmersFarmHotBuckets[i] = (int8_t*)_mm_malloc(sizeof(int8_t) * AI_FarmerCount, 64);
		AI_FarmersFarmHotBucketIndices[i] = (uint32_t*)_mm_malloc(sizeof(uint32_t) * AI_FarmerCount, 64);
	}

	AI_FarmersFarmCold = (uint32_t*)malloc(sizeof(uint32_t) * AI_FarmerCount);
	AI_FarmersFarmGenX = (uint16_t*)_mm_malloc(sizeof(uint16_t) * AI_FarmerCount, 64);
	AI_FarmersFarmGenY = (uint16_t*)_mm_malloc(sizeof(uint16_t) * AI_FarmerCount, 64);

	for (uint32_t i = 0; i < AI_FarmerFarmBucketCount; i++)
	{
		AI_FarmersSearchHotBuckets[i] = (int8_t*)_mm_malloc(sizeof(int8_t) * AI_FarmerCount, 64);
		AI_FarmersSearchHotBucketIndices[i] = (uint32_t*)_mm_malloc(sizeof(uint32_t) * AI_FarmerCount, 64);
	}
	AI_FarmersSearchGenX = (uint16_t*)_mm_malloc(sizeof(uint16_t) * AI_FarmerCount, 64);
	AI_FarmersSearchGenY = (uint16_t*)_mm_malloc(sizeof(uint16_t) * AI_FarmerCount, 64);

	SIMD_FarmerRemovalIndices = (unsigned int*)_mm_malloc(sizeof(unsigned int) * AI_FarmerCount, 64);

	memset(AI_FarmersSearchGenX, 0, sizeof(uint16_t) * AI_FarmerCount);
	memset(AI_FarmersSearchGenY, 0, sizeof(uint16_t) * AI_FarmerCount);
	for (uint32_t ai = 0; ai < AI_FarmerCount; ++ai)
	{
		int16_t farmerTimer = rand_range(AI_FarmerSearchSpeedMin, AI_FarmerSearchSpeedMax);
		uint32_t bucketSecond = (farmerTimer + AI_FarmersSearchBucketTransitionTimer) / (AI_TimePrecision / (AI_FarmerSearchBucketCount - 1));
		uint32_t bucketIndex = (math_min(bucketSecond, AI_FarmerSearchBucketCount - 1) + AI_FarmersSearchFineTimerBucket) % AI_FarmerSearchBucketCount;
		uint32_t bucketFarmerCount = AI_FarmersSearchHotBucketCounts[bucketIndex];

		AI_FarmersSearchHotBuckets[bucketIndex][bucketFarmerCount] = farmerTimer - bucketSecond * (AI_TimePrecision / (AI_FarmerSearchBucketCount - 1));
		AI_FarmersSearchHotBucketIndices[bucketIndex][bucketFarmerCount] = ai;
		AI_FarmersSearchHotBucketCounts[bucketIndex]++;
	}

	AI_FarmerSearchCount = AI_FarmerCount;

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

	for (uint32_t i = 0; i < AI_FarmerFarmBucketCount; i++)
	{
		_mm_free(AI_FarmersFarmHotBuckets[i]);
		_mm_free(AI_FarmersFarmHotBucketIndices[i]);
	}
	free(AI_FarmersFarmCold);
	_mm_free(AI_FarmersFarmGenX);
	_mm_free(AI_FarmersFarmGenY);

	for (uint32_t i = 0; i < AI_FarmerFarmBucketCount; i++)
	{
		_mm_free(AI_FarmersSearchHotBuckets[i]);
		_mm_free(AI_FarmersSearchHotBucketIndices[i]);
	}
	_mm_free(AI_FarmersSearchGenX);
	_mm_free(AI_FarmersSearchGenY);

	_mm_free(SIMD_FarmerRemovalIndices);

	MIST_PROFILE_END("Game", "Game-Kill");
}

uint32_t game_gen_instance_buffer(Game_InstanceBuffer* buffer)
{
	MIST_PROFILE_BEGIN("Game", "Game-GenInstanceBuffer");

	uint32_t writeIndex = 0;

	{
		for (uint32_t i = 0; i < Field_TileDrawCommandCount; ++i)
		{
			Field_TileDrawCommand* command = &Field_TileDrawCommands[i];
			buffer->spriteIndicesAndScales[command->writeIndex * 2] = command->spriteIndex;
		}
		Field_TileDrawCommandCount = 0;
	}

	{
		for (uint32_t i = 0; i < Field_CropDrawCommandCount; i++)
		{
			Field_CropDrawCommand* command = &Field_CropDrawCommands[i];
			buffer->spriteIndicesAndScales[command->writeIndex * 2] = command->spriteIndex;
			buffer->spriteIndicesAndScales[command->writeIndex * 2 + 1] = Field_TileScale;
			buffer->positionX[command->writeIndex] = command->posX;
			buffer->positionY[command->writeIndex] = command->posY;
		}

		// Repeat the last crop to align the writeIndex
		uint32_t tilePlusCropCount = (Field_Width * Field_Height + Field_CropCount);
		uint32_t remainingWrites = 32 - tilePlusCropCount % 32;
		for (uint32_t i = 0; i < remainingWrites; i++)
		{
			buffer->spriteIndicesAndScales[(tilePlusCropCount + i) * 2] = buffer->spriteIndicesAndScales[(tilePlusCropCount - 1) * 2];
			buffer->spriteIndicesAndScales[(tilePlusCropCount + i) * 2 + 1] = buffer->spriteIndicesAndScales[(tilePlusCropCount - 1) * 2 + 1];
			buffer->positionX[tilePlusCropCount + i] = buffer->positionX[tilePlusCropCount - 1];
			buffer->positionY[tilePlusCropCount + i] = buffer->positionY[tilePlusCropCount - 1];
		}
		Field_CropDrawCommandCount = 0;

		writeIndex = Field_Width * Field_Height + Field_CropCount + remainingWrites;
	}

	{
		simd_streamMemCpy((__m256i*)&buffer->positionX[writeIndex], (__m256i*)AI_FarmersSearchGenX, sizeof(uint16_t) * AI_FarmerSearchCount);
		simd_streamMemCpy((__m256i*)&buffer->positionY[writeIndex], (__m256i*)AI_FarmersSearchGenY, sizeof(uint16_t) * AI_FarmerSearchCount);

		__m256i searchAndScale = _mm256_set1_epi32((uint32_t)FarmerState_Search | ((uint32_t)AI_FarmerScale << 16));
		simd_memSetToValue((__m256i*)(buffer->spriteIndicesAndScales + writeIndex * 2), searchAndScale, AI_FarmerSearchCount * sizeof(uint32_t));

		uint32_t remainingSearchWrites = 32 - (AI_FarmerSearchCount % 32);
		__m256i writeSearchIndexAndScale = _mm256_set1_epi32(buffer->spriteIndicesAndScales[(writeIndex + AI_FarmerSearchCount - 1) * 2]);
		__m256i positionX = _mm256_set1_epi16(buffer->positionX[writeIndex + AI_FarmerSearchCount - 1]);
		__m256i positionY = _mm256_set1_epi16(buffer->positionY[writeIndex + AI_FarmerSearchCount - 1]);
		for (uint32_t i = 0; i < remainingSearchWrites; i += 16)
		{
			_mm256_storeu_si256((__m256i*)(buffer->spriteIndicesAndScales + (writeIndex + AI_FarmerSearchCount + i * 16) * 2), writeSearchIndexAndScale);
			_mm256_storeu_si256((__m256i*)(buffer->spriteIndicesAndScales + (writeIndex + AI_FarmerSearchCount + i * 16 + 16) * 2), writeSearchIndexAndScale);
			_mm256_storeu_si256((__m256i*)(buffer->positionX + writeIndex + AI_FarmerSearchCount + i * 16), positionX);
			_mm256_storeu_si256((__m256i*)(buffer->positionY + writeIndex + AI_FarmerSearchCount + i * 16), positionY);
		}

		writeIndex += AI_FarmerSearchCount + remainingSearchWrites;
	}

	{
		simd_streamMemCpy((__m256i*)&buffer->positionX[writeIndex], (__m256i*)AI_FarmersMoveGenX, sizeof(uint16_t) * AI_FarmerMoveCount);
		simd_streamMemCpy((__m256i*)&buffer->positionY[writeIndex], (__m256i*)AI_FarmersMoveGenY, sizeof(uint16_t) * AI_FarmerMoveCount);

		__m256i moveAndScale = _mm256_set1_epi32((uint32_t)FarmerState_Move | ((uint32_t)AI_FarmerScale << 16));
		simd_memSetToValue((__m256i*)(buffer->spriteIndicesAndScales + writeIndex * 2), moveAndScale, AI_FarmerMoveCount * sizeof(uint32_t));

		uint32_t remainingMoveWrites = 32 - (AI_FarmerMoveCount % 32);
		__m256i writeMoveIndexAndScale = _mm256_set1_epi32(buffer->spriteIndicesAndScales[(writeIndex + AI_FarmerMoveCount - 1) * 2]);
		__m256i positionX = _mm256_set1_epi16(buffer->positionX[writeIndex + AI_FarmerMoveCount - 1]);
		__m256i positionY = _mm256_set1_epi16(buffer->positionY[writeIndex + AI_FarmerMoveCount - 1]);
		for (uint32_t i = 0; i < remainingMoveWrites; i += 16)
		{
			_mm256_storeu_si256((__m256i*)(buffer->spriteIndicesAndScales + (writeIndex + AI_FarmerMoveCount + i * 16) * 2), writeMoveIndexAndScale);
			_mm256_storeu_si256((__m256i*)(buffer->spriteIndicesAndScales + (writeIndex + AI_FarmerMoveCount + i * 16 + 16) * 2), writeMoveIndexAndScale);
			_mm256_storeu_si256((__m256i*)(buffer->positionX + writeIndex + AI_FarmerMoveCount + i * 16), positionX);
			_mm256_storeu_si256((__m256i*)(buffer->positionY + writeIndex + AI_FarmerMoveCount + i * 16), positionY);
		}
		writeIndex += AI_FarmerMoveCount + remainingMoveWrites;
	}

	{
		simd_streamMemCpy((__m256i*)&buffer->positionX[writeIndex], (__m256i*)AI_FarmersFarmGenX, sizeof(uint16_t) * AI_FarmerFarmCount);
		simd_streamMemCpy((__m256i*)&buffer->positionY[writeIndex], (__m256i*)AI_FarmersFarmGenY, sizeof(uint16_t) * AI_FarmerFarmCount);

		__m256i farmAndScale = _mm256_set1_epi32((uint32_t)FarmerState_Farm | ((uint32_t)AI_FarmerScale << 16));
		simd_memSetToValue((__m256i*)(buffer->spriteIndicesAndScales + writeIndex * 2), farmAndScale, AI_FarmerFarmCount * sizeof(uint32_t));

		writeIndex += AI_FarmerFarmCount;
	}

	MIST_PROFILE_END("Game", "Game-GenInstanceBuffer");

	return writeIndex;
}
