#define _CRT_SECURE_NO_WARNINGS
#include "game.h"

#define SOKOL_IMPL
#define SOKOL_GLCORE33
#define SOKOL_DEBUG
#include "3rd/sokol_app.h"
#include "3rd/sokol_gfx.h"
#include "3rd/sokol_time.h"

#define STB_IMAGE_IMPLEMENTATION
#include "3rd/stb_image.h"

#define MIST_PROFILE_IMPLEMENTATION
#define MIST_PROFILE_ENABLED
#include "3rd/Mist_Profiler.h"

#include <stdint.h>
#include <stdbool.h>

#include <immintrin.h>

const uint16_t Window_Width = 1024;
const uint16_t Window_Height = 720;
const char* Window_Title = "Holy Cheese";

static uint64_t Time_LastFrame = 0;

const uint8_t Render_SampleCount = 4;

static Game_InstanceBuffer* Render_InstanceBuffer;
static sg_draw_state Render_DrawState;

extern const char *Render_VS, *Render_FS;

typedef struct
{
	float aspect;
} Render_VSParams;

void core_init(void)
{
	Mist_ProfileInit();

	stm_setup();

	sg_desc desc = { 0 };
	sg_setup(&desc);

	sg_buffer instanceBuffer = sg_make_buffer(&(sg_buffer_desc)
	{
		.size = sizeof(float) * GAME_MAX_INSTANCE_COUNT * 2 + sizeof(int16_t) * GAME_MAX_INSTANCE_COUNT * 2,
			.usage = SG_USAGE_STREAM
	});

	uint16_t indices[] = { 0, 1, 2, 2, 1, 3 };
	sg_buffer indexBuffer = sg_make_buffer(&(sg_buffer_desc)
	{
		.type = SG_BUFFERTYPE_INDEXBUFFER,
			.size = sizeof(uint16_t) * 6,
			.content = indices
	});

	sg_shader shader = sg_make_shader(&(sg_shader_desc)
	{
		.fs = { .images[0] = {.type = SG_IMAGETYPE_2D,.name = "tex0"},.source = Render_FS },
			.vs =
		{
			.uniform_blocks[0] =
				{
					.size = sizeof(Render_VSParams),
					.uniforms[0] = {.name = "aspect",.type = SG_UNIFORMTYPE_FLOAT }
			},
			.source = Render_VS
		}
	});

	int width, height, comp;
	stbi_uc* imageData = stbi_load("../../../Assets/Sprites.png", &width, &height, &comp, 4);
	sg_image image = sg_make_image(&(sg_image_desc)
	{
		.width = width,
			.height = height,
			.min_filter = SG_FILTER_LINEAR,
			.mag_filter = SG_FILTER_LINEAR,
			.wrap_u = SG_WRAP_CLAMP_TO_EDGE,
			.wrap_v = SG_WRAP_CLAMP_TO_EDGE,
			.content.subimage[0][0] = { .ptr = imageData,.size = width * height * 4 }
	});

	stbi_image_free(imageData);

	sg_pipeline pipeline = sg_make_pipeline(&(sg_pipeline_desc)
	{
		.layout =
		{
			.buffers[0] = {.step_func = SG_VERTEXSTEP_PER_INSTANCE,.stride = sizeof(float) * 2 + sizeof(int16_t) * 2 },
			.attrs =
				{
					[0] = {.name = "spriteAndScale",.format = SG_VERTEXFORMAT_SHORT2N, .offset = offsetof(Game_InstanceBuffer, spriteIndicesAndScales) },
					[1] = {.name = "positionX",.format = SG_VERTEXFORMAT_HALF, .offset = offsetof(Game_InstanceBuffer, positionX) },
					[2] = {.name = "positionY",.format = SG_VERTEXFORMAT_HALF, .offset = offsetof(Game_InstanceBuffer, positionY) }
				}
		},
			.shader = shader,
			.index_type = SG_INDEXTYPE_UINT16,
			.depth_stencil =
		{
			.depth_write_enabled = false
		},

			.rasterizer = { .cull_mode = SG_CULLMODE_NONE,.sample_count = Render_SampleCount },
			.blend =
		{
			.enabled = true,
			.src_factor_rgb = SG_BLENDFACTOR_SRC_ALPHA,
			.dst_factor_rgb = SG_BLENDFACTOR_ONE_MINUS_SRC_ALPHA,
			.src_factor_alpha = SG_BLENDFACTOR_SRC_ALPHA,
			.dst_factor_alpha = SG_BLENDFACTOR_ONE_MINUS_SRC_ALPHA
		}
	});

	Render_DrawState = (sg_draw_state)
	{
		.pipeline = pipeline,
			.vertex_buffers[0] = instanceBuffer,
			.index_buffer = indexBuffer,
			.fs_images[0] = image
	};

	Render_InstanceBuffer = _mm_malloc(sizeof(Game_InstanceBuffer), 64);
	game_init(Render_InstanceBuffer);
}

void core_frame(void)
{
	int width = sapp_width();
	int height = sapp_height();

	uint64_t delta = stm_laptime(&Time_LastFrame);
	game_tick((float)stm_sec(delta));

	uint32_t instanceCount = game_gen_instance_buffer(Render_InstanceBuffer);

	sg_update_buffer(Render_DrawState.vertex_buffers[0], Render_InstanceBuffer, sizeof(float) * instanceCount * 4);

	sg_pass_action passAction =
	{
		.colors[0] = {.action = SG_ACTION_CLEAR,.val = { 0.1f, 0.1f, 0.1f, 1.0f } }
	};
	sg_begin_default_pass(&passAction, (int)width, (int)height);
	sg_apply_draw_state(&Render_DrawState);
	sg_apply_uniform_block(SG_SHADERSTAGE_VS, 0, &(Render_VSParams){.aspect = (float)width / height}, sizeof(Render_VSParams));
	if (instanceCount > 0)
	{
		sg_draw(0, 6, instanceCount);
	}
	sg_end_pass();
	sg_commit();
}

void core_cleanup(void)
{
	game_kill();
	_mm_free(Render_InstanceBuffer);

	sg_shutdown();

	Mist_ProfileTerminate();
}

#ifdef PROFILE_MODE
void core_initProfile(void)
{
	Mist_ProfileInit();

	MIST_PROFILE_BEGIN("Core", "Init");
	Render_InstanceBuffer = _mm_malloc(sizeof(Game_InstanceBuffer), 64);
	game_init(Render_InstanceBuffer);
	MIST_PROFILE_END("Core", "Init");
}

void core_cleanupProfile(void);

static uint32_t profile_FrameCount = 0;
void core_frameProfile(void)
{
	if (profile_FrameCount < 1000)
	{
		MIST_PROFILE_BEGIN("Core", "Tick");
		game_tick(0.016f);
		game_gen_instance_buffer(Render_InstanceBuffer);
		MIST_PROFILE_END("Core", "Tick");

		profile_FrameCount++;
	}
	else
	{
		core_cleanupProfile();
		exit(0);
	}
}

void core_cleanupProfile(void)
{
	MIST_PROFILE_BEGIN("Core", "Cleanup");
	game_kill();
	_mm_free(Render_InstanceBuffer);
	MIST_PROFILE_END("Core", "Cleanup");

	if (Mist_ProfileListSize() == 0)
	{
		// Adds the current buffer to the list of buffers even if it hasn't been filled up yet.
		Mist_FlushThreadBuffer();
	}

	FILE* fileHandle = fopen("trace.txt", "w");

	char* print;
	size_t bufferSize;
	Mist_FlushAlloc(&print, &bufferSize);

	fprintf(fileHandle, "%s", mist_ProfilePreface);
	fprintf(fileHandle, "%s", print);
	fprintf(fileHandle, "%s", mist_ProfilePostface);

	free(print);
	fclose(fileHandle);
	Mist_ProfileTerminate();
}

#endif // PROFILE_MODE

sapp_desc sokol_main(int argc, char* argv[])
{
#ifdef PROFILE_MODE
	return (sapp_desc)
	{
		.init_cb = core_initProfile,
			.frame_cb = core_frameProfile,
			.cleanup_cb = core_cleanupProfile,
			.width = Window_Width,
			.height = Window_Height,
			.window_title = Window_Title,
};
#else
	return (sapp_desc)
	{
		.init_cb = core_init,
			.frame_cb = core_frame,
			.cleanup_cb = core_cleanup,
			.width = Window_Width,
			.height = Window_Height,
			.window_title = Window_Title,
	};
#endif
}

const char* Render_VS =
"#version 330\n"
"uniform float aspect;\n"
"in vec2 spriteAndScale;\n"
"in float positionX;\n"
"in float positionY;\n"
"out vec2 uv;\n"
"void main()\n"
"{\n"
"  vec2 position = vec2(positionX, positionY);\n"
"  const float kImageCount = 11.0;\n"
"  vec2 vertexPos = vec2(gl_VertexID / 2, gl_VertexID & 1);\n"
"  gl_Position = vec4((position + vertexPos * spriteAndScale.y / vec2(aspect, 1.0)), 0.0, 1.0);\n"
"  uv = vec2(vertexPos.x / kImageCount + (spriteAndScale.x * 11.0) / kImageCount, 1.0 - vertexPos.y);\n"
"}\n";

const char* Render_FS =
"#version 330\n"
"in vec2 uv;\n"
"uniform sampler2D tex0;\n"
"out vec4 frag_color;\n"
"void main()\n"
"{\n"
"  frag_color = texture(tex0, uv);\n"
"}\n";
