#ifndef AI_ENGINE_H
#define AI_ENGINE_H

#ifdef __cplusplus
extern "C" {
#endif

// Initialize the Edge AI Engine
void ai_engine_init(void);

// Run inference with the given input tensor
void ai_engine_predict(float *input, float *output);

#ifdef __cplusplus
}
#endif

#endif // AI_ENGINE_H
