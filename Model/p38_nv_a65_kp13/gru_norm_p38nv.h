/**
 * @file    gru_norm_p38nv.h
 * @brief   MinMax normalization constants — P38-nv-a65-gru (45-feature, no velocity)
 *
 * Model:   P38-nv-a65-gru
 *          kp13 (13 keypoints × 3 = 39) + 6 derived = 45 features
 *          NO velocity features (replaces P37-vel-kp13 74-feature table)
 *
 * Usage:
 *   feat_norm[i] = CLAMP((feat_raw[i] - NORM_MIN[i]) / NORM_SCALE[i], 0.0f, 1.0f)
 *
 * Generated from: normalization.json (Falling-Model-Development repo)
 * Date: 2026-05-19
 */

#ifndef GRU_NORM_P38NV_H
#define GRU_NORM_P38NV_H

#define P38NV_FEATURE_COUNT  45

/* MinMax normalization — min values (45 entries) */
static const float p38nv_norm_min[P38NV_FEATURE_COUNT] = {
  /* [ 0] kp0_y  (nose)           */ 0.000000f,
  /* [ 1] kp0_x                   */ 0.000000f,
  /* [ 2] kp0_s                   */ 0.002383f,
  /* [ 3] kp5_y  (left_shoulder)  */ 0.000000f,
  /* [ 4] kp5_x                   */ 0.000000f,
  /* [ 5] kp5_s                   */ 0.004055f,
  /* [ 6] kp6_y  (right_shoulder) */ 0.000000f,
  /* [ 7] kp6_x                   */ 0.000000f,
  /* [ 8] kp6_s                   */ 0.001079f,
  /* [ 9] kp7_y  (left_elbow)     */ 0.001145f,
  /* [10] kp7_x                   */ 0.000000f,
  /* [11] kp7_s                   */ 0.001560f,
  /* [12] kp8_y  (right_elbow)    */ 0.001290f,
  /* [13] kp8_x                   */ 0.000724f,
  /* [14] kp8_s                   */ 0.004059f,
  /* [15] kp9_y  (left_wrist)     */ 0.005291f,
  /* [16] kp9_x                   */ 0.000000f,
  /* [17] kp9_s                   */ 0.004168f,
  /* [18] kp10_y (right_wrist)    */ 0.002378f,
  /* [19] kp10_x                  */ 0.000000f,
  /* [20] kp10_s                  */ 0.004263f,
  /* [21] kp11_y (left_hip)       */ 0.008567f,
  /* [22] kp11_x                  */ 0.000359f,
  /* [23] kp11_s                  */ 0.008049f,
  /* [24] kp12_y (right_hip)      */ 0.002279f,
  /* [25] kp12_x                  */ 0.000000f,
  /* [26] kp12_s                  */ 0.006607f,
  /* [27] kp13_y (left_knee)      */ 0.021298f,
  /* [28] kp13_x                  */ 0.000248f,
  /* [29] kp13_s                  */ 0.005414f,
  /* [30] kp14_y (right_knee)     */ 0.017672f,
  /* [31] kp14_x                  */ 0.000000f,
  /* [32] kp14_s                  */ 0.004014f,
  /* [33] kp15_y (left_ankle)     */ 0.021160f,
  /* [34] kp15_x                  */ 0.001553f,
  /* [35] kp15_s                  */ 0.000135f,
  /* [36] kp16_y (right_ankle)    */ 0.028301f,
  /* [37] kp16_x                  */ 0.000000f,
  /* [38] kp16_s                  */ 0.000181f,
  /* [39] HSSC_y  (상체 중심 y)   */ 0.000088f,
  /* [40] HSSC_x  (상체 중심 x)   */ 0.000158f,
  /* [41] RWHC    (bbox 비율)      */ 0.024745f,
  /* [42] VHSSC   (수직 속도)      */ -2.220820f,
  /* [43] AHSSC   (수직 가속도)    */ -18.393295f,
  /* [44] AHSSC_x (수평 가속도)    */ -67.749054f,
};

/* MinMax normalization — scale values (45 entries, = max - min) */
static const float p38nv_norm_scale[P38NV_FEATURE_COUNT] = {
  /* [ 0] kp0_y  */ 0.999908f,
  /* [ 1] kp0_x  */ 0.999997f,
  /* [ 2] kp0_s  */ 0.880732f,
  /* [ 3] kp5_y  */ 0.997732f,
  /* [ 4] kp5_x  */ 0.999995f,
  /* [ 5] kp5_s  */ 0.968784f,
  /* [ 6] kp6_y  */ 0.984889f,
  /* [ 7] kp6_x  */ 0.999999f,
  /* [ 8] kp6_s  */ 0.965219f,
  /* [ 9] kp7_y  */ 0.998121f,
  /* [10] kp7_x  */ 0.999957f,
  /* [11] kp7_s  */ 0.981908f,
  /* [12] kp8_y  */ 0.989397f,
  /* [13] kp8_x  */ 0.996170f,
  /* [14] kp8_s  */ 0.967367f,
  /* [15] kp9_y  */ 0.994656f,
  /* [16] kp9_x  */ 1.000000f,
  /* [17] kp9_s  */ 0.957223f,
  /* [18] kp10_y */ 0.997243f,
  /* [19] kp10_x */ 0.998899f,
  /* [20] kp10_s */ 0.948190f,
  /* [21] kp11_y */ 0.991431f,
  /* [22] kp11_x */ 0.999545f,
  /* [23] kp11_s */ 0.944041f,
  /* [24] kp12_y */ 0.997647f,
  /* [25] kp12_x */ 0.999300f,
  /* [26] kp12_s */ 0.938715f,
  /* [27] kp13_y */ 0.978090f,
  /* [28] kp13_x */ 0.998411f,
  /* [29] kp13_s */ 0.965195f,
  /* [30] kp14_y */ 0.981398f,
  /* [31] kp14_x */ 0.995599f,
  /* [32] kp14_s */ 0.969497f,
  /* [33] kp15_y */ 0.978840f,
  /* [34] kp15_x */ 0.996987f,
  /* [35] kp15_s */ 0.965499f,
  /* [36] kp16_y */ 0.971697f,
  /* [37] kp16_x */ 0.999525f,
  /* [38] kp16_s */ 0.968101f,
  /* [39] HSSC_y */ 0.991588f,
  /* [40] HSSC_x */ 0.998811f,
  /* [41] RWHC   */ 21.949295f,
  /* [42] VHSSC  */ 5.239275f,
  /* [43] AHSSC  */ 45.194557f,
  /* [44] AHSSC_x*/ 125.428223f,
};

#endif /* GRU_NORM_P38NV_H */
