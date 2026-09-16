import { useMemo } from "react";
import { Html } from "@react-three/drei";
import * as THREE from "three";
import { MOOD_WORD, type Drink } from "../data";
import { COUNTER_TOP, GLASS_SHAPE, GLASS_Z, glassX, type GlassType } from "../layout";
import { giveDrink, useTour } from "../store";

type P = [number, number][];

/** Outer wall (bottom → rim) and inner wall (bottom → rim) as lathe profiles, radius then height. */
const PROFILES: Record<GlassType, { outer: P; inner: P; fill: number }> = {
  rocks: {
    outer: [[0, 0], [0.08, 0], [0.086, 0.012], [0.088, 0.17]],
    inner: [[0, 0.022], [0.079, 0.022], [0.082, 0.17]],
    fill: 0.62,
  },
  highball: {
    outer: [[0, 0], [0.054, 0], [0.058, 0.012], [0.06, 0.3]],
    inner: [[0, 0.026], [0.052, 0.026], [0.055, 0.3]],
    fill: 0.8,
  },
  pint: {
    outer: [[0, 0], [0.056, 0], [0.06, 0.015], [0.076, 0.27], [0.079, 0.3]],
    inner: [[0, 0.022], [0.054, 0.022], [0.071, 0.27], [0.074, 0.3]],
    fill: 0.82,
  },
  martini: {
    outer: [[0, 0], [0.065, 0], [0.066, 0.007], [0.012, 0.014], [0.007, 0.13], [0.013, 0.142], [0.112, 0.25]],
    inner: [[0, 0.15], [0.005, 0.15], [0.105, 0.25]],
    fill: 0.88,
  },
  wine: {
    outer: [[0, 0], [0.06, 0], [0.061, 0.007], [0.009, 0.014], [0.007, 0.12], [0.03, 0.132], [0.068, 0.18], [0.074, 0.23], [0.066, 0.29]],
    inner: [[0, 0.137], [0.027, 0.14], [0.063, 0.18], [0.069, 0.23], [0.061, 0.29]],
    fill: 0.72,
  },
  hurricane: {
    outer: [[0, 0], [0.056, 0], [0.057, 0.008], [0.013, 0.02], [0.012, 0.05], [0.05, 0.075], [0.066, 0.13], [0.046, 0.2], [0.062, 0.27], [0.073, 0.32]],
    inner: [[0, 0.06], [0.045, 0.08], [0.061, 0.13], [0.041, 0.2], [0.057, 0.27], [0.068, 0.32]],
    fill: 0.8,
  },
};

const v2 = (p: P) => p.map(([r, y]) => new THREE.Vector2(r, y));

function geometries(type: GlassType) {
  const { outer, inner, fill } = PROFILES[type];
  const glass = new THREE.LatheGeometry(v2([...outer, ...[...inner].reverse()]), 64);
  const bottom = inner[0][1];
  const rim = inner[inner.length - 1][1];
  const level = bottom + (rim - bottom) * fill;
  const pts: P = [];
  for (let i = 0; i < inner.length; i++) {
    const [r, y] = inner[i];
    if (y <= level) pts.push([Math.max(0, r - 0.003), y]);
    else {
      const [r0, y0] = inner[i - 1];
      const f = (level - y0) / (y - y0);
      pts.push([Math.max(0, r0 + (r - r0) * f - 0.003), level]);
      break;
    }
  }
  pts.push([0, level]);
  const liquid = new THREE.LatheGeometry(v2(pts), 48);
  const radiusAtLevel = pts[pts.length - 2][0];
  return { glass, liquid, level, radiusAtLevel };
}

const glassMaterial = new THREE.MeshPhysicalMaterial({
  color: "#ffffff", transmission: 1, thickness: 0.012, roughness: 0.03, ior: 1.5,
  clearcoat: 1, clearcoatRoughness: 0.05, specularIntensity: 1, envMapIntensity: 1.2,
});

function Garnish({ kind, level, r, rim }: { kind: Drink["garnish"]; level: number; r: number; rim: number }) {
  const wheel = (color: string, peel: string) => (
    <group position={[r * 0.75, rim + 0.005, 0]} rotation={[0, 0, 0.35]}>
      <mesh rotation={[Math.PI / 2, 0, 0]}>
        <cylinderGeometry args={[0.038, 0.038, 0.008, 28]} />
        <meshStandardMaterial color={peel} roughness={0.5} />
      </mesh>
      <mesh rotation={[Math.PI / 2, 0, 0]} scale={[1, 1.2, 1]}>
        <cylinderGeometry args={[0.032, 0.032, 0.01, 28]} />
        <meshStandardMaterial color={color} roughness={0.35} emissive={color} emissiveIntensity={0.12} />
      </mesh>
    </group>
  );
  switch (kind) {
    case "orange":
      return wheel("#f7a03c", "#e0781a");
    case "lime":
      return wheel("#b9dc6a", "#4f9a2c");
    case "grapefruit":
      return wheel("#f58a78", "#e9a24a");
    case "mint":
      return (
        <group position={[0, level + 0.02, 0]}>
          {[0, 2.1, 4.2].map((a) => (
            <mesh key={a} position={[Math.cos(a) * 0.02, 0.015, Math.sin(a) * 0.02]} rotation={[0.9, a, 0.3]} scale={[1, 0.25, 1.8]}>
              <sphereGeometry args={[0.018, 12, 8]} />
              <meshStandardMaterial color="#3f8f3a" roughness={0.6} />
            </mesh>
          ))}
        </group>
      );
    case "pineapple":
      return (
        <group position={[r * 0.8, rim + 0.01, 0]} rotation={[0, 0, 0.4]}>
          <mesh>
            <boxGeometry args={[0.05, 0.03, 0.03]} />
            <meshStandardMaterial color="#f5cf4e" roughness={0.5} />
          </mesh>
          {[-0.3, 0, 0.3].map((a) => (
            <mesh key={a} position={[0, 0.035, 0]} rotation={[0, 0, a]} scale={[0.25, 1, 0.1]}>
              <coneGeometry args={[0.02, 0.07, 6]} />
              <meshStandardMaterial color="#4b8a34" roughness={0.6} />
            </mesh>
          ))}
        </group>
      );
    case "passion":
      return (
        <mesh position={[0, level + 0.004, 0]} rotation={[Math.PI, 0, 0]}>
          <sphereGeometry args={[0.03, 20, 12, 0, Math.PI * 2, 0, Math.PI / 2]} />
          <meshStandardMaterial color="#e9b640" roughness={0.4} side={THREE.DoubleSide} />
        </mesh>
      );
    case "foam":
      return (
        <mesh position={[0, level + 0.012, 0]}>
          <cylinderGeometry args={[r - 0.004, r - 0.006, 0.026, 40]} />
          <meshStandardMaterial color="#f6ecd2" roughness={0.9} />
        </mesh>
      );
    default:
      return null;
  }
}

function Tag({ drink, index, provisional }: { drink: Drink; index: number; provisional: boolean }) {
  const key = useTour((s) => `${s.tasted.includes(index)}|${s.current === index}|${s.target === index}|${s.phase}`);
  const [tasted, here, heading, phase] = key.split("|");
  const active = here === "true" && phase !== "idle";
  const chosen = here === "true" && phase === "settled";
  return (
    <button
      type="button"
      className={`tag ${active || heading === "true" ? "on" : ""} ${chosen ? "chosen" : ""}`}
      onClick={() => giveDrink(index)}
      title={`Δώσε της ${drink.name}`}
    >
      <span className="tag-emoji" aria-hidden="true">{drink.emoji}</span>
      <span className="tag-name">{drink.name}</span>
      <span className={`tag-mood ${tasted === "true" && !provisional ? drink.mood : ""}`}>
        {tasted !== "true" ? "δεν το δοκίμασε" : provisional ? "δοκιμάστηκε" : MOOD_WORD[drink.mood]}
      </span>
    </button>
  );
}

export function Glass({ drink, index, n, provisional }: { drink: Drink; index: number; n: number; provisional: boolean }) {
  const { glass, liquid, level, radiusAtLevel } = useMemo(() => geometries(drink.glass), [drink.glass]);
  const liquidMat = useMemo(
    () => new THREE.MeshStandardMaterial({ color: drink.liquid, roughness: 0.18, metalness: 0, emissive: drink.liquid, emissiveIntensity: 0.1 }),
    [drink.liquid],
  );
  const shape = GLASS_SHAPE[drink.glass];
  const rocksLike = drink.glass === "rocks" || drink.glass === "highball";
  return (
    <group position={[glassX(index, n), COUNTER_TOP, GLASS_Z]}>
      <mesh geometry={liquid} material={liquidMat} />
      {rocksLike &&
        [0, 1].map((k) => (
          <mesh key={k} position={[(k - 0.5) * shape.r * 0.7, level - 0.01 + k * 0.012, (0.5 - k) * 0.012]} rotation={[0.3 * k, 0.6 + k, 0.2]}>
            <boxGeometry args={[shape.r * 0.72, shape.r * 0.72, shape.r * 0.72]} />
            <meshPhysicalMaterial color="#eaf6fb" roughness={0.08} clearcoat={1} transparent opacity={0.55} />
          </mesh>
        ))}
      <mesh geometry={glass} material={glassMaterial} />
      <Garnish kind={drink.garnish} level={level} r={radiusAtLevel} rim={shape.h} />
      <Html position={[0, shape.h + 0.1, 0]} center distanceFactor={2} zIndexRange={[20, 0]}>
        <Tag drink={drink} index={index} provisional={provisional} />
      </Html>
    </group>
  );
}
