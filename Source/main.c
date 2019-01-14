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
#include "3rd/Mist_Profiler.h"

#include <stdint.h>
#include <stdbool.h>

#include <xmmintrin.h>

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

    sg_desc desc = {0};
    sg_setup(&desc);
    
    sg_buffer posInstanceBuffer = sg_make_buffer(&(sg_buffer_desc)
        {
            .size = sizeof(Game_Instance) * GAME_MAX_INSTANCE_COUNT,
            .usage = SG_USAGE_STREAM
        });

	sg_buffer indexInstanceBuffer = sg_make_buffer(&(sg_buffer_desc)
        {
            .size = sizeof(float) * GAME_MAX_INSTANCE_COUNT,
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
        .fs = { .images[0] = {.type = SG_IMAGETYPE_2D, .name = "tex0"}, .source = Render_FS },
            .vs = 
                { 
                    .uniform_blocks[0] = 
                        {
                            .size = sizeof(Render_VSParams),
                            .uniforms[0] = {.name = "aspect", .type = SG_UNIFORMTYPE_FLOAT }
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
            .content.subimage[0][0] = { .ptr = imageData, .size = width * height * 4 }
        });

    stbi_image_free(imageData);

    sg_pipeline pipeline = sg_make_pipeline(&(sg_pipeline_desc)
        {
            .layout = 
                {
					.buffers[0] = {.step_func = SG_VERTEXSTEP_PER_INSTANCE,.stride = sizeof(float) },
					.buffers[1] = {.step_func = SG_VERTEXSTEP_PER_INSTANCE,.stride = sizeof(Game_Instance) },
                    .attrs = 
                        {
                            [0] = { .name = "sprite", .format = SG_VERTEXFORMAT_FLOAT, .buffer_index = 0 },
                            [1] = { .name = "position", .format = SG_VERTEXFORMAT_SHORT2N, .buffer_index = 1 }
                        }
                },
            .shader = shader,
            .index_type = SG_INDEXTYPE_UINT16,
            .depth_stencil = 
                {
                    .depth_write_enabled = false
                },

            .rasterizer = {.cull_mode = SG_CULLMODE_NONE, .sample_count = Render_SampleCount },
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
            .vertex_buffers[0] = indexInstanceBuffer,
			.vertex_buffers[1] = posInstanceBuffer,
            .index_buffer = indexBuffer,
            .fs_images[0] = image
        };

    Render_InstanceBuffer = _mm_malloc(sizeof(Game_InstanceBuffer), 16);
    game_init();
}

void core_frame(void)
{
	MIST_PROFILE_BEGIN("Core", "Tick");
    int width = sapp_width();
    int height = sapp_height();

    uint64_t delta = stm_laptime(&Time_LastFrame);
	MIST_PROFILE_BEGIN("Game", "Tick");
    game_tick((float)stm_sec(delta));
	MIST_PROFILE_END("Game", "Tick");

	MIST_PROFILE_BEGIN("Game", "GenBuffer");
    uint32_t instanceCount = game_gen_instance_buffer(Render_InstanceBuffer);
	MIST_PROFILE_END("Game", "GenBuffer");

	MIST_PROFILE_BEGIN("Render", "UpdateBuffer");
	sg_update_buffer(Render_DrawState.vertex_buffers[0], Render_InstanceBuffer->spriteIndices, sizeof(float) * instanceCount);
	sg_update_buffer(Render_DrawState.vertex_buffers[1], Render_InstanceBuffer->pos, sizeof(Game_Instance) * instanceCount);
	MIST_PROFILE_END("Render", "UpdateBuffer");

    sg_pass_action passAction =
    {
        .colors[0] = { .action = SG_ACTION_CLEAR, .val = { 0.1f, 0.1f, 0.1f, 1.0f } }
    };

	MIST_PROFILE_BEGIN("Render", "Pass");
    sg_begin_default_pass(&passAction, (int)width, (int)height);
    sg_apply_draw_state(&Render_DrawState);
    sg_apply_uniform_block(SG_SHADERSTAGE_VS, 0, &(Render_VSParams){.aspect = (float)width / height}, sizeof(Render_VSParams));
    if (instanceCount > 0)
    {
        sg_draw(0, 6, instanceCount);
    }
    sg_end_pass();
	MIST_PROFILE_END("Render", "Pass");

	MIST_PROFILE_BEGIN("Render", "Commit");
    sg_commit();
	MIST_PROFILE_END("Render", "Commit");
	MIST_PROFILE_END("Core", "Tick");
}

void core_cleanup(void)
{
    game_kill();
    _mm_free(Render_InstanceBuffer);

    sg_shutdown();

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

sapp_desc sokol_main(int argc, char* argv[])
{
    return (sapp_desc) 
        {
            .init_cb = core_init,
            .frame_cb = core_frame,
            .cleanup_cb = core_cleanup,
            .width = Window_Width,
            .height = Window_Height,
            .window_title = Window_Title,
        };
}

const char* Render_VS = 
    "#version 330\n"
    "uniform float aspect;\n"
    "in float sprite;\n"
    "in vec2 position;\n"
    "out vec2 uv;\n"
    "void main()\n"
    "{\n"
	"  float scale = sprite < 3.0 ? 0.025 : 2.0 / 500.0;\n"
    "  const float kImageCount = 11.0;\n"
    "  vec2 vertexPos = vec2(gl_VertexID / 2, gl_VertexID & 1);\n"
    "  gl_Position = vec4((position + vertexPos * scale / vec2(aspect, 1.0)), 0.0, 1.0);\n"
    "  uv = vec2(vertexPos.x / kImageCount + sprite / kImageCount, 1.0 - vertexPos.y);\n"
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
