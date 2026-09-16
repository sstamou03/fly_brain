import type { GlassType } from "./layout";

export type Mood = "love" | "like" | "meh" | "yuck";

export type Drink = {
  id: string;
  name: string;
  emoji: string;
  glass: GlassType;
  liquid: string;
  garnish: "orange" | "lime" | "mint" | "grapefruit" | "pineapple" | "passion" | "foam" | "none";
  recipe: { label: string; value: string }[];
  taste: { sweet: number; bitter: number; fizz: number; salt: number; water: number };
  hungerNeeded: number | null;
  desire: number[];
  aversion: number[];
  pulse: number[];
  cloud: { file: string; lit: number; regions: { name: string; lit: number }[] } | null;
  rank: number;
  likes: number;
  mood: Mood;
};

export type BarData = {
  binMs: number;
  tasteMs: number;
  hunger: number;
  provisional: boolean;
  winner: string;
  neurons: number;
  points: number;
  drinks: Drink[];
};

export type BrainPoints = { xyz: Float32Array; group: Uint8Array; count: number };

/** Walking commands read out of the brain's descending neurons (walk.json), looped while she wanders. */
export type WalkData = { binMs: number; forward: number[]; turn: number[]; source: string };

export async function loadWalk(): Promise<WalkData | null> {
  try {
    const res = await fetch("/data/walk.json");
    if (!res.ok) return null;
    return await res.json();
  } catch {
    return null;
  }
}

export const MOOD_WORD: Record<Mood, string> = {
  love: "Το λάτρεψε",
  like: "Της αρέσει",
  meh: "Το σκέφτεται",
  yuck: "Μπλιαχ",
};

export const MOOD_FACE: Record<Mood, string> = { love: "😍", like: "😋", meh: "🤔", yuck: "🤢" };

export async function loadBar(): Promise<BarData> {
  const res = await fetch("/data/drinks.json");
  if (!res.ok) throw new Error("Λείπουν τα δεδομένα του μπαρ. Τρέξε python export_web.py και ξαναχτίσε την εφαρμογή.");
  return res.json();
}

export async function loadBrain(count: number): Promise<BrainPoints> {
  const [xyz, group] = await Promise.all([
    fetch("/data/brain_xyz.bin").then((r) => r.arrayBuffer()),
    fetch("/data/brain_group.bin").then((r) => r.arrayBuffer()),
  ]);
  return { xyz: new Float32Array(xyz), group: new Uint8Array(group), count };
}

const activityCache = new Map<string, Promise<Uint8Array>>();
export function loadActivity(file: string): Promise<Uint8Array> {
  if (!activityCache.has(file)) {
    activityCache.set(file, fetch(`/data/${file}`).then((r) => r.arrayBuffer()).then((b) => new Uint8Array(b)));
  }
  return activityCache.get(file)!;
}

export function sampleSeries(series: number[], p: number): number {
  if (!series.length) return 0;
  const x = Math.min(Math.max(p, 0), 1) * (series.length - 1);
  const i = Math.floor(x);
  const f = x - i;
  return series[i] * (1 - f) + series[Math.min(i + 1, series.length - 1)] * f;
}
