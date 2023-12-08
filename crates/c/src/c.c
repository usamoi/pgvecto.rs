#include "c.h"
#include <immintrin.h>
#include <math.h>

__attribute__((target("avx512fp16,bmi2"))) extern float
vectors_f16_cosine_axv512(_Float16 const *restrict a,
                          _Float16 const *restrict b, size_t n) {
  __m512h xy = _mm512_set1_ph(0);
  __m512h xx = _mm512_set1_ph(0);
  __m512h yy = _mm512_set1_ph(0);

  while (n >= 32) {
    __m512h x = _mm512_loadu_ph(a);
    __m512h y = _mm512_loadu_ph(b);
    a += 32, b += 32, n -= 32;
    xy = _mm512_fmadd_ph(x, y, xy);
    xx = _mm512_fmadd_ph(x, x, xx);
    yy = _mm512_fmadd_ph(y, y, yy);
  }
  if (n > 0) {
    __mmask32 mask = _bzhi_u32(0xFFFFFFFF, n);
    __m512h x = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(mask, a));
    __m512h y = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(mask, b));
    xy = _mm512_fmadd_ph(x, y, xy);
    xx = _mm512_fmadd_ph(x, x, xx);
    yy = _mm512_fmadd_ph(y, y, yy);
  }
  return (float)(_mm512_reduce_add_ps(xy) /
                 sqrt(_mm512_reduce_add_ps(xx) * _mm512_reduce_add_ps(yy)));
}

__attribute__((target("avx512fp16,bmi2"))) extern float
vectors_f16_dot_axv512(_Float16 const *restrict a, _Float16 const *restrict b,
                       size_t n) {
  __m512h xy = _mm512_set1_ph(0);

  while (n >= 32) {
    __m512h x = _mm512_loadu_ph(a);
    __m512h y = _mm512_loadu_ph(b);
    a += 32, b += 32, n -= 32;
    xy = _mm512_fmadd_ph(x, y, xy);
  }
  if (n > 0) {
    __mmask32 mask = _bzhi_u32(0xFFFFFFFF, n);
    __m512h x = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(mask, a));
    __m512h y = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(mask, b));
    xy = _mm512_fmadd_ph(x, y, xy);
  }
  return (float)_mm512_reduce_add_ph(xy);
}

__attribute__((target("avx512fp16,bmi2"))) extern float
vectors_f16_distance_squared_l2_axv512(_Float16 const *restrict a,
                                       _Float16 const *restrict b, size_t n) {
  __m512h dd = _mm512_set1_ph(0);

  while (n >= 32) {
    __m512h x = _mm512_loadu_ph(a);
    __m512h y = _mm512_loadu_ph(b);
    a += 32, b += 32, n -= 32;
    __m512h d = _mm512_sub_ph(x, y);
    dd = _mm512_fmadd_ph(d, d, dd);
  }
  if (n > 0) {
    __mmask32 mask = _bzhi_u32(0xFFFFFFFF, n);
    __m512h x = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(mask, a));
    __m512h y = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(mask, b));
    __m512h d = _mm512_sub_ph(x, y);
    dd = _mm512_fmadd_ph(d, d, dd);
  }

  return (float)_mm512_reduce_add_ph(dd);
}
