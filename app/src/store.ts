import { useEffect, useSyncExternalStore } from "react";

export type Phase = "loading" | "idle" | "flying" | "tasting" | "reacting" | "settled";

export type TourState = {
  phase: Phase;
  current: number | null; // glass the fly sits on
  target: number | null; // glass it is flying to
  tasted: number[];
  startedAt: number;
  speed: number;
  finale: boolean; // flying to / sitting at its choice
  view: "follow" | "free"; // camera follows the fly, or the user orbits/zooms around the bar
};

export const FLY_MS = 1500;
export const TASTE_MS = 3600;
export const REACT_MS = 1700;

let state: TourState = {
  phase: "loading", current: null, target: null, tasted: [], startedAt: 0, speed: 1, finale: false, view: "follow",
};

export function setView(view: TourState["view"]) {
  tour.set({ view });
}
const listeners = new Set<() => void>();

export const tour = {
  get: () => state,
  set(patch: Partial<TourState>) {
    state = { ...state, ...patch };
    listeners.forEach((l) => l());
  },
  subscribe(l: () => void) {
    listeners.add(l);
    return () => {
      listeners.delete(l);
    };
  },
};

export function useTour<T extends string | number | boolean | null>(select: (s: TourState) => T): T {
  return useSyncExternalStore(tour.subscribe, () => select(state));
}

let queue: number[] = [];
let count = 0;
let winner = 0;
let decided = true;

/** decidedChoice = false while the ranking is provisional: the fly tastes everything but picks nothing. */
export function configureTour(n: number, winnerIndex: number, decidedChoice: boolean) {
  count = n;
  winner = Math.max(0, winnerIndex);
  decided = decidedChoice;
  tour.set({ phase: "idle" });
}

export function startTour() {
  queue = Array.from({ length: count }, (_, i) => i);
  tour.set({ tasted: [], finale: false });
  advance();
}

export function giveDrink(i: number) {
  if (state.phase === "loading") return;
  queue = [];
  flyTo(i, false);
}

export function setSpeed(speed: number) {
  const now = performance.now();
  // keep the current step's progress when the speed changes
  const elapsed = (now - state.startedAt) * state.speed;
  tour.set({ speed, startedAt: now - elapsed / speed });
}

function flyTo(i: number, finale: boolean) {
  tour.set({ phase: "flying", target: i, startedAt: performance.now(), finale });
}

function advance() {
  if (queue.length) flyTo(queue.shift()!, false);
  else if (decided) flyTo(winner, true);
  else tour.set({ phase: "idle", startedAt: performance.now() });
}

function tick(now: number) {
  const s = state;
  const elapsed = (now - s.startedAt) * s.speed;
  if (s.phase === "flying" && elapsed >= FLY_MS) {
    tour.set({ phase: s.finale ? "settled" : "tasting", current: s.target, target: null, startedAt: now });
  } else if (s.phase === "tasting" && elapsed >= TASTE_MS) {
    const tasted = s.tasted.includes(s.current!) ? s.tasted : [...s.tasted, s.current!];
    tour.set({ phase: "reacting", startedAt: now, tasted });
  } else if (s.phase === "reacting" && elapsed >= REACT_MS) {
    if (queue.length || (decided && s.tasted.length === count)) advance();
    else tour.set({ phase: "idle", startedAt: now });
  }
}

/** Runs the tour clock. Mount once. */
export function useTourClock() {
  useEffect(() => {
    let raf = 0;
    const loop = (now: number) => {
      tick(now);
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, []);
}

/** 0..1 through the brain replay of the drink the fly is at. */
export function tasteProgress(now: number): number {
  const s = state;
  const elapsed = (now - s.startedAt) * s.speed;
  if (s.phase === "tasting") return Math.min(1, elapsed / TASTE_MS);
  if (s.phase === "settled") return (elapsed % TASTE_MS) / TASTE_MS;
  if ((s.phase === "reacting" || s.phase === "idle") && s.current !== null) return 1;
  return 0;
}
