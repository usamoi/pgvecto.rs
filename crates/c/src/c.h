#include <stddef.h>
#include <stdint.h>

#if defined(__x86_64__)

extern float v_f16_cosine_axv512(_Float16 *, _Float16 *, size_t n);
extern float v_f16_dot_axv512(_Float16 *, _Float16 *, size_t n);
extern float v_f16_sl2_axv512(_Float16 *, _Float16 *, size_t n);
extern float v_f16_cosine_axv2(_Float16 *, _Float16 *, size_t n);
extern float v_f16_dot_axv2(_Float16 *, _Float16 *, size_t n);
extern float v_f16_sl2_axv2(_Float16 *, _Float16 *, size_t n);

#endif
