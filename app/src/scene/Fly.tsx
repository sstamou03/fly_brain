import { useMemo, useRef } from "react";
import { useFrame } from "@react-three/fiber";
import { Html } from "@react-three/drei";
import * as THREE from "three";
import { MOOD_FACE, MOOD_WORD, sampleSeries, type BarData, type WalkData } from "../data";
import { FLY_MS, REACT_MS, tasteProgress, tour, useTour } from "../store";
import { COUNTER_TOP, GLASS_SHAPE, GLASS_Z, glassX, rimPoint } from "../layout";
import { abdomenTexture, facetTexture, wingTexture } from "./textures";
import { flyWorld } from "./shared";

const SCALE = 0.075;
const COUNTER = { xMin: -3.2, xMax: 3.2, zMin: 0.0, zMax: 0.68 };

const lerp = (a: number, b: number, k: number) => a + (b - a) * k;
function lerpAngle(a: number, b: number, k: number) {
  let d = ((b - a + Math.PI) % (Math.PI * 2)) - Math.PI;
  if (d < -Math.PI) d += Math.PI * 2;
  return a + d * k;
}
/** Smooth pseudo-random signal in [-1, 1] for the fallback wander. */
const wobble = (t: number, seed: number) =>
  (Math.sin(t * 0.73 + seed) + Math.sin(t * 1.37 + seed * 2.1) * 0.6 + Math.sin(t * 0.31 + seed * 3.7) * 0.8) / 2.4;

function wingGeometry() {
  const s = new THREE.Shape();
  s.moveTo(0, 0);
  s.bezierCurveTo(0.14, 0.04, 0.22, 0.32, 0.12, 0.74);
  s.bezierCurveTo(0.05, 0.84, -0.07, 0.8, -0.1, 0.64);
  s.bezierCurveTo(-0.15, 0.4, -0.08, 0.1, 0, 0);
  const g = new THREE.ShapeGeometry(s, 32);
  g.computeBoundingBox();
  const bb = g.boundingBox!;
  const pos = g.attributes.position;
  const uv = g.attributes.uv;
  for (let i = 0; i < pos.count; i++) {
    uv.setXY(i, (pos.getX(i) - bb.min.x) / (bb.max.x - bb.min.x), 1 - (pos.getY(i) - bb.min.y) / (bb.max.y - bb.min.y));
  }
  return g;
}

/** A leg: femur, tibia, tarsus. Pivot at the thorax, reaching out and down. */
function Leg({ side, z, yaw, index, refs }: { side: 1 | -1; z: number; yaw: number; index: number; refs: THREE.Group[] }) {
  const mat = useMemo(() => new THREE.MeshStandardMaterial({ color: "#2e2015", roughness: 0.55 }), []);
  const geo = useMemo(() => {
    const seg = (len: number, r: number) => {
      const g = new THREE.CylinderGeometry(r * 0.8, r, len, 6);
      g.translate(0, -len / 2, 0);
      return g;
    };
    return { femur: seg(0.3, 0.022), tibia: seg(0.32, 0.016), tarsus: seg(0.18, 0.011) };
  }, []);
  // tripod gait: legs 0 & 2 of one side move with leg 1 of the other
  const tripod = (index % 2 === 0) === (side === 1) ? 0 : Math.PI;
  return (
    <group position={[side * 0.1, -0.1, z]} rotation={[0, yaw * side, 0]}>
      <group ref={(g) => { if (g && !refs.includes(g)) refs.push(g); }} rotation={[0, 0, side * 1.05]} userData={{ side, tripod }}>
        <mesh geometry={geo.femur} material={mat} />
        <group position={[0, -0.3, 0]} rotation={[0, 0, -side * 1.55]}>
          <mesh geometry={geo.tibia} material={mat} />
          <group position={[0, -0.32, 0]} rotation={[0, 0, side * 0.35]}>
            <mesh geometry={geo.tarsus} material={mat} />
          </group>
        </group>
      </group>
    </group>
  );
}

function Bubble({ data }: { data: BarData }) {
  const key = useTour((s) => `${s.phase}|${s.current ?? ""}|${s.finale}`);
  const [phase, cur, finale] = key.split("|");
  const drink = cur !== "" ? data.drinks[Number(cur)] : null;
  let text = "";
  let cls = "";
  if (phase === "tasting" && drink) text = "γευσιγνωσία…";
  if (phase === "reacting" && drink) {
    text = data.provisional ? "δοκιμάστηκε" : `${MOOD_FACE[drink.mood]} ${MOOD_WORD[drink.mood]}`;
    cls = data.provisional ? "" : drink.mood;
  }
  if (phase === "settled" && drink) {
    text = "Αυτό θέλω";
    cls = "love";
  }
  if (phase === "flying" && finale === "true") text = "Ξέρω τι θέλω!";
  if (!text) return null;
  return (
    <Html position={[0, 0.07, 0]} center zIndexRange={[40, 30]} pointerEvents="none">
      <div className={`bubble ${cls}`}>{text}</div>
    </Html>
  );
}

export function Fly({ data, walk }: { data: BarData; walk: WalkData | null }) {
  const n = data.drinks.length;
  const root = useRef<THREE.Group>(null!);
  const body = useRef<THREE.Group>(null!);
  const wingL = useRef<THREE.Group>(null!);
  const wingR = useRef<THREE.Group>(null!);
  const proboscis = useRef<THREE.Group>(null!);
  const head = useRef<THREE.Group>(null!);
  const legRefs = useMemo<THREE.Group[]>(() => [], []);
  const from = useRef(flyWorld.clone());
  const phaseKey = useRef("");
  const wander = useRef({ pos: flyWorld.clone(), yaw: 1.2, gait: 0, clock: 0 });
  const tmp = useMemo(() => ({ mid: new THREE.Vector3(), to: new THREE.Vector3(), p: new THREE.Vector3(), d: new THREE.Vector3() }), []);

  const mat = useMemo(() => ({
    thorax: new THREE.MeshStandardMaterial({ color: "#8f6a3c", roughness: 0.5, metalness: 0.05 }),
    abdomen: new THREE.MeshStandardMaterial({ map: abdomenTexture(), roughness: 0.45 }),
    head: new THREE.MeshStandardMaterial({ color: "#7d5a33", roughness: 0.55 }),
    eye: new THREE.MeshStandardMaterial({
      color: "#a3121a", roughness: 0.3, metalness: 0.1, bumpMap: facetTexture(), bumpScale: 1.2,
      emissive: "#4a0508", emissiveIntensity: 0.6,
    }),
    wing: new THREE.MeshPhysicalMaterial({
      map: wingTexture(), transparent: true, opacity: 0.6, roughness: 0.2, side: THREE.DoubleSide,
      depthWrite: false, iridescence: 1, iridescenceIOR: 1.4, clearcoat: 0.5,
    }),
    bristle: new THREE.MeshStandardMaterial({ color: "#1b120b", roughness: 0.7 }),
    proboscis: new THREE.MeshStandardMaterial({ color: "#a47a48", roughness: 0.5 }),
  }), []);
  const wingGeo = useMemo(wingGeometry, []);
  const bristles = useMemo(() => {
    const g = new THREE.CylinderGeometry(0.004, 0.007, 0.09, 4);
    g.translate(0, 0.045, 0);
    const m = new THREE.InstancedMesh(g, mat.bristle, 60);
    const o = new THREE.Object3D();
    for (let i = 0; i < 60; i++) {
      const theta = (i / 60) * Math.PI * 2 * 5.3;
      const phi = 0.25 + (i / 60) * 1.05;
      const nrm = new THREE.Vector3(Math.sin(phi) * Math.cos(theta), Math.cos(phi), Math.sin(phi) * Math.sin(theta));
      o.position.copy(nrm).multiply(new THREE.Vector3(0.22, 0.2, 0.25));
      o.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), nrm.clone().add(new THREE.Vector3(0, 0, -0.8)).normalize());
      o.updateMatrix();
      m.setMatrixAt(i, o.matrix);
    }
    return m;
  }, [mat]);

  /** Wander on the counter top. Speed and turning come from the brain (walk.json) when available. */
  function stepWander(dt: number) {
    const w = wander.current;
    w.clock += dt;
    let forward: number;
    let turn: number;
    if (walk && walk.forward.length) {
      const x = ((w.clock * 1000) / walk.binMs) % walk.forward.length;
      forward = sampleSeries(walk.forward, x / (walk.forward.length - 1));
      turn = sampleSeries(walk.turn, x / (walk.turn.length - 1));
    } else {
      forward = Math.max(0, 0.45 + 0.55 * wobble(w.clock, 1.3));
      turn = wobble(w.clock * 1.4, 4.1);
    }
    w.yaw += turn * 2.4 * dt;
    // steer back when close to the counter's edges
    const cx = THREE.MathUtils.clamp(w.pos.x, COUNTER.xMin + 0.25, COUNTER.xMax - 0.25);
    const cz = THREE.MathUtils.clamp(w.pos.z, COUNTER.zMin + 0.12, COUNTER.zMax - 0.12);
    if (cx !== w.pos.x || cz !== w.pos.z) {
      const home = Math.atan2(cx - w.pos.x, cz - w.pos.z);
      w.yaw = lerpAngle(w.yaw, home, 1 - Math.exp(-dt * 3));
    }
    const speed = forward * 0.1;
    w.pos.x = THREE.MathUtils.clamp(w.pos.x + Math.sin(w.yaw) * speed * dt, COUNTER.xMin, COUNTER.xMax);
    w.pos.z = THREE.MathUtils.clamp(w.pos.z + Math.cos(w.yaw) * speed * dt, COUNTER.zMin, COUNTER.zMax);
    // walk around the glasses, not through them
    for (let i = 0; i < n; i++) {
      const r = GLASS_SHAPE[data.drinks[i].glass].r + 0.035;
      const dx = w.pos.x - glassX(i, n);
      const dz = w.pos.z - GLASS_Z;
      const d = Math.hypot(dx, dz);
      if (d < r && d > 1e-6) {
        w.pos.x = glassX(i, n) + (dx / d) * r;
        w.pos.z = GLASS_Z + (dz / d) * r;
      }
    }
    w.gait += speed * dt * 120;
    return speed;
  }

  useFrame((st, dt) => {
    const s = tour.get();
    const now = performance.now();
    const t = st.clock.elapsedTime;
    const key = `${s.phase}:${s.target}:${s.startedAt}`;
    if (key !== phaseKey.current) {
      if (s.phase === "flying") from.current.copy(root.current.position);
      phaseKey.current = key;
    }

    let flying = false;
    let walking = 0;
    let reach = 0.08;
    let lean = 0;
    if (s.phase === "flying" && s.target !== null) {
      flying = true;
      const k = Math.min(1, ((now - s.startedAt) * s.speed) / FLY_MS);
      const e = k < 0.5 ? 4 * k * k * k : 1 - Math.pow(-2 * k + 2, 3) / 2;
      tmp.to.set(...rimPoint(s.target, n, data.drinks[s.target].glass));
      tmp.mid.copy(from.current).lerp(tmp.to, 0.5);
      tmp.mid.y = Math.max(from.current.y, tmp.to.y) + 0.42;
      tmp.p.copy(from.current).multiplyScalar((1 - e) * (1 - e)).addScaledVector(tmp.mid, 2 * (1 - e) * e).addScaledVector(tmp.to, e * e);
      tmp.d.copy(tmp.p).sub(root.current.position);
      root.current.position.copy(tmp.p);
      root.current.position.y += Math.sin(t * 7) * 0.006 * (1 - e);
      if (tmp.d.lengthSq() > 1e-9) root.current.rotation.y = lerpAngle(root.current.rotation.y, Math.atan2(tmp.d.x, tmp.d.z), 1 - Math.exp(-dt * 9));
      lean = -0.3;
    } else if (s.current !== null) {
      const drink = data.drinks[s.current];
      const rp = rimPoint(s.current, n, drink.glass);
      root.current.position.set(rp[0], rp[1], rp[2]);
      root.current.rotation.y = lerpAngle(root.current.rotation.y, Math.PI, 1 - Math.exp(-dt * 5));
      lean = 0.45;
      const want = sampleSeries(drink.desire, tasteProgress(now));
      if (s.phase === "tasting") {
        reach = 0.5 + 0.5 * Math.max(want, Math.max(0, Math.sin(t * 7)) * 0.4);
      } else if (s.phase === "settled") {
        reach = 0.75 + 0.25 * Math.sin(t * 5);
      } else {
        const r = Math.min(1, ((now - s.startedAt) * s.speed) / REACT_MS);
        if (data.provisional) {
          reach = 0.2;
        } else if (drink.mood === "yuck") {
          reach = 0.05;
          lean = 0.1 - Math.sin(r * Math.PI) * 0.35;
          root.current.position.y += Math.sin(r * Math.PI) * 0.03;
          head.current.rotation.y = s.phase === "reacting" ? Math.sin(r * 30) * 0.35 * (1 - r) : 0;
        } else {
          reach = drink.mood === "meh" ? 0.25 : 0.7;
        }
      }
      wander.current.pos.set(rp[0], COUNTER_TOP, GLASS_Z + 0.2);
    } else {
      // before the tasting: wander on the counter
      walking = stepWander(dt);
      const w = wander.current;
      root.current.position.set(w.pos.x, COUNTER_TOP, w.pos.z);
      root.current.rotation.y = lerpAngle(root.current.rotation.y, w.yaw, 1 - Math.exp(-dt * 12));
      head.current.rotation.y = lerp(head.current.rotation.y, Math.sin(t * 0.9) * 0.3, 1 - Math.exp(-dt * 4));
    }
    flyWorld.copy(root.current.position);

    body.current.rotation.x = lerp(body.current.rotation.x, lean, 1 - Math.exp(-dt * 8));
    const flap = flying ? Math.sin(t * 95) * 0.85 : 0;
    const spread = flying ? 1.15 : 0.16;
    wingL.current.rotation.y = lerp(wingL.current.rotation.y, spread, 1 - Math.exp(-dt * 16));
    wingR.current.rotation.y = lerp(wingR.current.rotation.y, -spread, 1 - Math.exp(-dt * 16));
    wingL.current.rotation.z = flap;
    wingR.current.rotation.z = -flap;
    proboscis.current.scale.y = lerp(proboscis.current.scale.y, reach, 1 - Math.exp(-dt * 10));
    const gait = wander.current.gait;
    for (const leg of legRefs) {
      const side = leg.userData.side as number;
      const stride = walking > 0.004 ? Math.sin(gait + leg.userData.tripod) * 0.28 : 0;
      const target = flying ? side * 0.45 : side * (1.05 + stride + Math.sin(t * 2.3 + leg.id) * 0.03);
      leg.rotation.z = lerp(leg.rotation.z, target, 1 - Math.exp(-dt * 14));
      leg.rotation.x = lerp(leg.rotation.x, walking > 0.004 ? Math.cos(gait + leg.userData.tripod) * 0.35 : 0, 1 - Math.exp(-dt * 14));
    }
  });

  return (
    <group ref={root} position={flyWorld}>
      <group scale={SCALE}>
        <group position={[0, 0.52, 0]}>
          <group ref={body}>
            <mesh material={mat.thorax} scale={[1, 0.9, 1.15]}>
              <sphereGeometry args={[0.22, 32, 24]} />
            </mesh>
            <primitive object={bristles} />
            <mesh material={mat.abdomen} position={[0, -0.02, -0.4]} rotation={[Math.PI / 2, 0, 0]} scale={[0.9, 1.5, 0.78]}>
              <sphereGeometry args={[0.22, 32, 24]} />
            </mesh>
            <group ref={head} position={[0, 0.04, 0.3]}>
              <mesh material={mat.head} scale={[1.15, 0.95, 0.8]}>
                <sphereGeometry args={[0.15, 24, 18]} />
              </mesh>
              {[1, -1].map((side) => (
                <mesh key={side} material={mat.eye} position={[side * 0.115, 0.03, 0.03]} scale={[0.7, 1, 0.95]}>
                  <sphereGeometry args={[0.11, 32, 24]} />
                </mesh>
              ))}
              {[1, -1].map((side) => (
                <mesh key={`a${side}`} material={mat.bristle} position={[side * 0.04, 0.1, 0.12]} rotation={[0.5, 0, side * -0.4]}>
                  <cylinderGeometry args={[0.006, 0.014, 0.12, 5]} />
                </mesh>
              ))}
              <group ref={proboscis} position={[0, -0.1, 0.08]} rotation={[0.35, 0, 0]}>
                <mesh material={mat.proboscis} position={[0, -0.1, 0]}>
                  <cylinderGeometry args={[0.022, 0.03, 0.2, 8]} />
                </mesh>
                <mesh material={mat.proboscis} position={[0, -0.2, 0]} scale={[1.4, 0.6, 1]}>
                  <sphereGeometry args={[0.03, 10, 8]} />
                </mesh>
              </group>
            </group>
            <group ref={wingL} position={[0.07, 0.17, -0.02]}>
              <mesh geometry={wingGeo} material={mat.wing} rotation={[-Math.PI / 2, 0, -0.12]} />
            </group>
            <group ref={wingR} position={[-0.07, 0.17, -0.02]}>
              <mesh geometry={wingGeo} material={mat.wing} rotation={[-Math.PI / 2, 0, 0.12]} scale={[-1, 1, 1]} />
            </group>
            {([1, -1] as const).map((side) =>
              [0.12, 0, -0.12].map((z, i) => (
                <Leg key={`${side}${i}`} side={side} z={z} yaw={[0.7, 0, -0.7][i]} index={i} refs={legRefs} />
              )),
            )}
          </group>
        </group>
      </group>
      <Bubble data={data} />
    </group>
  );
}
