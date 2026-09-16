/** Scene layout in metres-ish units: the counter top sits at y = COUNTER_TOP, glasses in a row along x. */
export const COUNTER_TOP = 1.06;
export const GLASS_Z = 0.3;
export const GLASS_SPACING = 0.56;

export const GLASS_SHAPE = {
  rocks: { h: 0.17, r: 0.086 },
  highball: { h: 0.3, r: 0.058 },
  martini: { h: 0.25, r: 0.11 },
  wine: { h: 0.29, r: 0.066 },
  hurricane: { h: 0.32, r: 0.072 },
  pint: { h: 0.3, r: 0.078 },
} as const;

export type GlassType = keyof typeof GLASS_SHAPE;

export const glassX = (i: number, n: number) => (i - (n - 1) / 2) * GLASS_SPACING;

/** Where the fly perches: on the rim, on the side facing the camera. */
export function rimPoint(i: number, n: number, type: GlassType): [number, number, number] {
  const s = GLASS_SHAPE[type];
  return [glassX(i, n), COUNTER_TOP + s.h, GLASS_Z + s.r * 0.9];
}
