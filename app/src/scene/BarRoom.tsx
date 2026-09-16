import { useLayoutEffect, useMemo, useRef } from "react";
import * as THREE from "three";
import { COUNTER_TOP, GLASS_Z } from "../layout";
import { panelTexture, posterTexture, woodTexture } from "./textures";

function bottleGeometry() {
  const pts = [
    [0, 0], [0.045, 0], [0.05, 0.01], [0.05, 0.2], [0.046, 0.23], [0.02, 0.27], [0.016, 0.34], [0.019, 0.35], [0.019, 0.37], [0, 0.37],
  ].map(([r, y]) => new THREE.Vector2(r, y));
  return new THREE.LatheGeometry(pts, 20);
}

const BOTTLE_COLORS = ["#6b3a12", "#a8641e", "#2f5a2a", "#d9c7a0", "#8a2b1a", "#c98b2e", "#3b2a1a", "#e6dcc4", "#1f4032", "#b0471f"];

function Bottles() {
  const ref = useRef<THREE.InstancedMesh>(null!);
  const geo = useMemo(bottleGeometry, []);
  const mat = useMemo(() => new THREE.MeshStandardMaterial({ roughness: 0.15, metalness: 0.05, transparent: true, opacity: 0.9 }), []);
  const shelves = [1.66, 2.16, 2.66];
  const perShelf = 26;
  useLayoutEffect(() => {
    const o = new THREE.Object3D();
    const c = new THREE.Color();
    let i = 0;
    let seed = 11;
    const r = () => ((seed = (seed * 16807) % 2147483647) - 1) / 2147483646;
    for (const y of shelves) {
      for (let k = 0; k < perShelf; k++) {
        const x = -2.2 + (4.4 * (k + 0.5)) / perShelf + (r() - 0.5) * 0.04;
        const s = 0.8 + r() * 0.45;
        o.position.set(x, y + 0.028, -1.47 + (r() - 0.5) * 0.08);
        o.scale.set(s * (0.85 + r() * 0.3), s, s * (0.85 + r() * 0.3));
        o.rotation.y = r() * 6;
        o.updateMatrix();
        ref.current.setMatrixAt(i, o.matrix);
        ref.current.setColorAt(i, c.set(BOTTLE_COLORS[Math.floor(r() * BOTTLE_COLORS.length)]));
        i++;
      }
    }
    ref.current.instanceMatrix.needsUpdate = true;
    if (ref.current.instanceColor) ref.current.instanceColor.needsUpdate = true;
  }, []);
  return <instancedMesh ref={ref} args={[geo, mat, shelves.length * perShelf]} />;
}

function PendantLamp({ x }: { x: number }) {
  return (
    <group position={[x, 3.3, GLASS_Z - 0.1]}>
      <mesh position={[0, 0.7, 0]}>
        <cylinderGeometry args={[0.006, 0.006, 1.4, 6]} />
        <meshStandardMaterial color="#111" />
      </mesh>
      <mesh>
        <coneGeometry args={[0.2, 0.22, 32, 1, true]} />
        <meshStandardMaterial color="#b88a45" metalness={0.9} roughness={0.35} side={THREE.DoubleSide} />
      </mesh>
      <mesh position={[0, -0.08, 0]}>
        <sphereGeometry args={[0.065, 20, 16]} />
        <meshStandardMaterial color="#fff2d6" emissive="#ffc27a" emissiveIntensity={6} />
      </mesh>
      <pointLight position={[0, -0.15, 0]} intensity={9} distance={7} decay={1.6} color="#ffc27a" />
    </group>
  );
}

function Stool({ x }: { x: number }) {
  const wood = "#5b3a22";
  return (
    <group position={[x, 0, 1.25]}>
      <mesh position={[0, 0.76, 0]}>
        <cylinderGeometry args={[0.22, 0.2, 0.08, 32]} />
        <meshStandardMaterial color="#8c5a2c" roughness={0.55} />
      </mesh>
      {[0, 1, 2, 3].map((k) => {
        const a = (k * Math.PI) / 2 + Math.PI / 4;
        return (
          <mesh key={k} position={[Math.cos(a) * 0.13, 0.36, Math.sin(a) * 0.13]} rotation={[Math.sin(a) * -0.12, 0, Math.cos(a) * 0.12]}>
            <cylinderGeometry args={[0.018, 0.022, 0.74, 8]} />
            <meshStandardMaterial color={wood} roughness={0.6} />
          </mesh>
        );
      })}
      <mesh position={[0, 0.25, 0]} rotation={[Math.PI / 2, 0, 0]}>
        <torusGeometry args={[0.15, 0.01, 8, 32]} />
        <meshStandardMaterial color="#a8823e" metalness={0.9} roughness={0.35} />
      </mesh>
    </group>
  );
}

export function BarRoom() {
  const tex = useMemo(
    () => ({
      floor: woodTexture("#3b2415", "#24150c", 10, [4, 3], 3),
      wall: panelTexture(),
      counter: woodTexture("#5a3620", "#2e1a0e", 14, [3, 1], 5),
      top: woodTexture("#3e2413", "#1f1108", 6, [4, 1], 9),
      shelf: woodTexture("#4a2d19", "#2a180d", 4, [6, 1], 13),
      poster1: posterTexture("FLY\nBAR", "est. 2026 · Drosophila", "#3a1f10", "#e9d6b4"),
      poster2: posterTexture("166.700\nνευρώνες", "ένας εγκέφαλος, δέκα ποτά", "#f1e3c8", "#6a2418"),
    }),
    [],
  );
  const wood = (map: THREE.Texture, rough = 0.7) => <meshStandardMaterial map={map} roughness={rough} />;

  return (
    <group>
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0, 1]}>
        <planeGeometry args={[16, 10]} />
        {wood(tex.floor, 0.85)}
      </mesh>
      <mesh position={[0, 2.3, -1.72]}>
        <planeGeometry args={[16, 4.6]} />
        {wood(tex.wall, 0.9)}
      </mesh>
      <mesh position={[0, 4.35, 0]} rotation={[Math.PI / 2, 0, 0]}>
        <planeGeometry args={[16, 8]} />
        <meshStandardMaterial color="#140c07" roughness={1} />
      </mesh>

      {/* back bar: glowing shelves */}
      <mesh position={[0, 2.2, -1.69]}>
        <planeGeometry args={[4.8, 1.75]} />
        <meshStandardMaterial color="#3a2414" emissive="#ff9a45" emissiveIntensity={0.55} />
      </mesh>
      {[1.66, 2.16, 2.66].map((y) => (
        <group key={y}>
          <mesh position={[0, y, -1.47]}>
            <boxGeometry args={[4.9, 0.05, 0.36]} />
            {wood(tex.shelf, 0.6)}
          </mesh>
          <mesh position={[0, y - 0.035, -1.34]}>
            <boxGeometry args={[4.8, 0.012, 0.02]} />
            <meshStandardMaterial color="#fff0d0" emissive="#ffb766" emissiveIntensity={3} />
          </mesh>
        </group>
      ))}
      {[-2.5, 2.5, 0].map((x) => (
        <mesh key={x} position={[x, 2.2, -1.47]}>
          <boxGeometry args={[0.09, 1.9, 0.4]} />
          {wood(tex.shelf, 0.6)}
        </mesh>
      ))}
      <Bottles />
      <mesh position={[0, 0.66, -1.42]}>
        <boxGeometry args={[5.4, 1.32, 0.52]} />
        {wood(tex.counter, 0.65)}
      </mesh>

      {/* posters */}
      <mesh position={[-3.7, 2.35, -1.7]}>
        <planeGeometry args={[0.75, 1.02]} />
        <meshStandardMaterial map={tex.poster1} roughness={0.8} />
      </mesh>
      <mesh position={[3.7, 2.35, -1.7]}>
        <planeGeometry args={[0.75, 1.02]} />
        <meshStandardMaterial map={tex.poster2} roughness={0.8} />
      </mesh>

      {/* counter */}
      <mesh position={[0, (COUNTER_TOP - 0.06) / 2, GLASS_Z]}>
        <boxGeometry args={[6.6, COUNTER_TOP - 0.06, 0.72]} />
        {wood(tex.counter, 0.6)}
      </mesh>
      {Array.from({ length: 12 }, (_, k) => (
        <mesh key={k} position={[-3.025 + k * 0.55, 0.5, GLASS_Z + 0.37]}>
          <boxGeometry args={[0.05, 0.9, 0.03]} />
          <meshStandardMaterial color="#3a2213" roughness={0.6} />
        </mesh>
      ))}
      <mesh position={[0, COUNTER_TOP - 0.03, GLASS_Z]}>
        <boxGeometry args={[6.9, 0.06, 0.88]} />
        <meshPhysicalMaterial map={tex.top} roughness={0.22} clearcoat={1} clearcoatRoughness={0.15} />
      </mesh>
      <mesh position={[0, 0.17, GLASS_Z + 0.52]} rotation={[0, 0, Math.PI / 2]}>
        <cylinderGeometry args={[0.025, 0.025, 6.4, 16]} />
        <meshStandardMaterial color="#b08a4a" metalness={1} roughness={0.3} />
      </mesh>

      {[-2.2, -1.1, 0, 1.1, 2.2].map((x) => (
        <Stool key={x} x={x} />
      ))}
      {[-2.2, 0, 2.2].map((x) => (
        <PendantLamp key={x} x={x} />
      ))}
    </group>
  );
}
