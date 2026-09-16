import { useEffect, useState } from "react";
import { loadBar, loadWalk, MOOD_FACE, MOOD_WORD, type BarData, type Drink, type WalkData } from "./data";
import { BarScene } from "./scene/BarScene";
import { BrainView } from "./brain/BrainView";
import { configureTour, giveDrink, setSpeed, setView, startTour, useTour, useTourClock } from "./store";

const nf = new Intl.NumberFormat("el-GR");

const REASON: Record<Drink["mood"], string> = {
  love: "Από όλο το μενού, αυτό θα παρήγγελνε.",
  like: "Της αρέσει, αλλά όχι όσο αυτό που παρήγγειλε.",
  meh: "Το σκέφτεται, αλλά μάλλον θα διάλεγε κάτι άλλο.",
  yuck: "Πολύ πικρό για μύγα. Δεν θα το παρήγγελνε.",
};

function Status({ data, walk }: { data: BarData; walk: WalkData | null }) {
  const key = useTour((s) => `${s.phase}|${s.current ?? ""}|${s.target ?? ""}|${s.finale}|${s.tasted.length}`);
  const [phase, cur, tgt, finale, tastedCount] = key.split("|");
  const at = cur !== "" ? data.drinks[Number(cur)] : null;
  const to = tgt !== "" ? data.drinks[Number(tgt)] : null;

  if (phase === "settled" && at) {
    return (
      <div className="status chosen">
        <p className="eyebrow">Θα παρήγγελνε</p>
        <h2 className="status-title">
          <span aria-hidden="true">{at.emoji}</span> {at.name}
        </h2>
        <p className="status-text">{REASON.love}</p>
      </div>
    );
  }
  let eyebrow = "Στο μπαρ";
  let title = "Κάνει βόλτες στον πάγκο";
  let text = walk
    ? "Πού πάει και πότε στρίβει το αποφασίζουν οι νευρώνες του εγκεφάλου της που κινούν τα πόδια. Πάτα «Ξεκίνα τη γευσιγνωσία» να δοκιμάσει τα ποτά σου."
    : "Περιμένει να της δώσεις ποτά. Πάτα «Ξεκίνα τη γευσιγνωσία» να τα δοκιμάσει ένα-ένα.";
  if (phase === "flying" && to) {
    eyebrow = finale === "true" ? "Αποφάσισε" : `Ποτό ${Number(tastedCount) + 1} από ${data.drinks.length}`;
    title = finale === "true" ? `Γυρίζει στο ${to.name}` : `Πετάει προς το ${to.name}`;
    text = finale === "true" ? "Τα δοκίμασε όλα και ξέρει τι θέλει." : "Κάθεται στο χείλος του ποτηριού.";
  } else if (phase === "tasting" && at) {
    eyebrow = "Δοκιμάζει";
    title = at.name;
    text = "Πατάει στο ποτό, βγάζει την προβοσκίδα και ο εγκέφαλός της αποφασίζει.";
  } else if ((phase === "reacting" || phase === "idle") && at && data.provisional) {
    eyebrow = "Δοκιμάστηκε";
    title = at.name;
    text =
      Number(tastedCount) === data.drinks.length
        ? "Τα δοκίμασε όλα. Η παραγγελία της βγαίνει μόλις ολοκληρωθεί ο υπολογισμός."
        : "Ο εγκέφαλός της αντέδρασε. Η γνώμη της θα φανεί όταν ολοκληρωθεί ο υπολογισμός.";
  } else if ((phase === "reacting" || phase === "idle") && at) {
    eyebrow = "Η γνώμη της";
    title = `${MOOD_FACE[at.mood]} ${MOOD_WORD[at.mood]}`;
    text = `${at.name}: ${REASON[at.mood]}`;
  }
  return (
    <div className="status">
      <p className="eyebrow">{eyebrow}</p>
      <h2 className="status-title">{title}</h2>
      <p className="status-text">{text}</p>
    </div>
  );
}

function ViewHint() {
  const free = useTour((s) => s.view === "free");
  if (!free) return null;
  return <p className="view-hint">Σύρε για να γυρίσεις · ροδέλα για ζουμ</p>;
}

function Controls({ data }: { data: BarData }) {
  const speed = useTour((s) => s.speed);
  const view = useTour((s) => s.view);
  const running = useTour((s) => s.phase !== "idle" || s.current !== null);
  return (
    <div className="controls">
      <button type="button" className="btn primary" onClick={startTour}>
        {running ? "Ξανά από την αρχή" : "Ξεκίνα τη γευσιγνωσία"}
      </button>
      <button type="button" className="btn" onClick={() => setSpeed(speed === 1 ? 2.5 : 1)}>
        {speed === 1 ? "Πιο γρήγορα" : "Κανονικά"}
      </button>
      <button type="button" className="btn" onClick={() => setView(view === "follow" ? "free" : "follow")}>
        {view === "follow" ? "Όλο το μπαρ" : "Ακολούθα τη μύγα"}
      </button>
      <div className="picker" role="group" aria-label="Δώσε της ένα ποτό">
        <span className="picker-label">Δώσε της:</span>
        {data.drinks.map((d, i) => (
          <button key={d.id} type="button" className="chip" onClick={() => giveDrink(i)}>
            <span aria-hidden="true">{d.emoji}</span> {d.name}
          </button>
        ))}
      </div>
    </div>
  );
}

function Ranking({ data }: { data: BarData }) {
  const tastedKey = useTour((s) => [...s.tasted].sort((a, b) => a - b).join(","));
  const settled = useTour((s) => s.phase === "settled");
  const tasted = new Set(tastedKey ? tastedKey.split(",").map(Number) : []);
  if (data.provisional) {
    return (
      <div className="card ranking">
        <div className="card-head">
          <h3>Τι θα παρήγγελνε</h3>
          <span className="card-sub">υπολογίζεται</span>
        </div>
        <p className="empty">
          Η μύγα δεν έχει αποφασίσει ακόμα. Ο εγκέφαλός της συγκρίνει πόσο εύκολα θα έπινε το καθένα.
        </p>
        <ol className="rank-list">
          {data.drinks.map((d, i) => (
            <li key={d.id} className={tasted.has(i) ? "" : "unknown"}>
              <span className="rank-pos">·</span>
              <span className="rank-name">
                <span aria-hidden="true">{d.emoji}</span> {d.name}
              </span>
              <span className="rank-bar" aria-hidden="true" />
              <span className="rank-mood">{tasted.has(i) ? "δοκιμάστηκε" : "—"}</span>
            </li>
          ))}
        </ol>
      </div>
    );
  }
  const rows = [...data.drinks].map((d, i) => ({ d, i })).sort((a, b) => a.d.rank - b.d.rank);
  return (
    <div className="card ranking">
      <div className="card-head">
        <h3>Τι θα παρήγγελνε</h3>
        <span className="card-sub">{tasted.size} από {data.drinks.length} δοκιμασμένα</span>
      </div>
      <ol className="rank-list">
        {rows.map(({ d, i }) => {
          const known = tasted.has(i) || settled;
          return (
            <li key={d.id} className={`${known ? "" : "unknown"} ${settled && d.rank === 1 ? "winner" : ""}`}>
              <span className="rank-pos">{known ? d.rank : "·"}</span>
              <span className="rank-name">
                <span aria-hidden="true">{d.emoji}</span> {d.name}
              </span>
              <span className="rank-bar" aria-hidden="true">
                <span className={`rank-fill ${d.mood}`} style={{ width: known ? `${Math.max(4, d.likes)}%` : "0%" }} />
              </span>
              <span className={`rank-mood ${known ? d.mood : ""}`}>{known ? MOOD_WORD[d.mood] : "—"}</span>
            </li>
          );
        })}
      </ol>
    </div>
  );
}

function BrainRegions({ data }: { data: BarData }) {
  const cur = useTour((s) => s.current ?? -1);
  const d = data.drinks[cur];
  const cloud = d?.cloud;
  if (!cloud || !cloud.regions.length) {
    return (
      <p className="empty">
        {d ? `Για το ${d.name} δεν έχει καταγραφεί ακόμα ποιοι νευρώνες ανάβουν.` : "Όταν δοκιμάζει ένα ποτό, εδώ φαίνεται ποια σημεία του εγκεφάλου της ανάβουν."}
      </p>
    );
  }
  const max = Math.max(...cloud.regions.map((r) => r.lit));
  return (
    <div className="regions">
      <p className="regions-total">
        <span aria-hidden="true">{d.emoji}</span> Με το {d.name} ανάβουν <b>{nf.format(cloud.lit)}</b> νευρώνες
      </p>
      <ul>
        {cloud.regions.map((r) => (
          <li key={r.name}>
            <span className="region-name">{r.name}</span>
            <span className="region-bar" aria-hidden="true">
              <span style={{ width: `${Math.max(3, (r.lit / max) * 100)}%` }} />
            </span>
            <span className="region-n">{nf.format(r.lit)}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

function TasteCard({ data }: { data: BarData }) {
  const cur = useTour((s) => s.current ?? s.target ?? -1);
  const d = data.drinks[cur];
  const rows: [string, number, string][] = d
    ? [["Γλυκό", d.taste.sweet, "like"], ["Πικρό", d.taste.bitter, "yuck"], ["Ανθρακικό", d.taste.fizz, "fizz"], ["Αλμυρό", d.taste.salt, "salt"]]
    : [];
  return (
    <div className="card taste">
      <div className="card-head">
        <h3>Τι γεύση νιώθει</h3>
        <span className="card-sub">{d ? `${d.emoji} ${d.name}` : "διάλεξε ποτό"}</span>
      </div>
      {d ? (
        <>
          <div className="taste-rows">
            {rows.map(([label, v, cls]) => (
              <div key={label} className="taste-row">
                <span>{label}</span>
                <span className="taste-track">
                  <span className={`taste-fill ${cls}`} style={{ width: `${Math.round(v * 100)}%` }} />
                </span>
              </div>
            ))}
          </div>
          <dl className="recipe">
            {d.recipe.map((r) => (
              <div key={r.label}>
                <dt>{r.label}</dt>
                <dd>{r.value}</dd>
              </div>
            ))}
          </dl>
        </>
      ) : (
        <p className="empty">Όταν η μύγα δοκιμάσει ένα ποτό, εδώ φαίνεται τι νιώθει στη γλώσσα και στα πόδια της.</p>
      )}
    </div>
  );
}

export default function App() {
  const [data, setData] = useState<BarData | null>(null);
  const [walk, setWalk] = useState<WalkData | null>(null);
  const [error, setError] = useState<string | null>(null);
  useTourClock();

  useEffect(() => {
    loadBar()
      .then((d) => {
        setData(d);
        configureTour(d.drinks.length, d.drinks.findIndex((x) => x.id === d.winner), !d.provisional);
      })
      .catch((e: Error) => setError(e.message));
    loadWalk().then(setWalk);
  }, []);

  if (error) return <div className="fatal">{error}</div>;
  if (!data) return <div className="fatal">Ανοίγει το μπαρ…</div>;

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true">🪰</span>
          <div>
            <h1>Fly Bar</h1>
            <p>Μια μύγα με αληθινό εγκέφαλο διαλέγει ποτό</p>
          </div>
        </div>
        <div className="facts">
          <div>
            <span>Εγκέφαλος</span>
            <b>{nf.format(data.neurons)} νευρώνες</b>
          </div>
          <div>
            <span>Χάρτης</span>
            <b>Google · Janelia</b>
          </div>
          <div>
            <span>Μενού</span>
            <b>{data.drinks.length} ποτά</b>
          </div>
        </div>
      </header>

      <section className="stage">
        <BarScene data={data} walk={walk} />
        <div className="stage-overlay">
          <Status data={data} walk={walk} />
        </div>
        <ViewHint />
      </section>

      <Controls data={data} />

      <section className="lower">
        <div className="card brain">
          <div className="card-head">
            <h3>Ο εγκέφαλός της τώρα</h3>
            <span className="card-sub">{nf.format(data.points)} νευρώνες στη θέση τους · σύρε για να γυρίσει</span>
          </div>
          <BrainView data={data} />
          <BrainRegions data={data} />
        </div>
        <div className="side">
          <Ranking data={data} />
          <TasteCard data={data} />
        </div>
      </section>

      <footer className="foot">
        {data.provisional && <p className="warn">Η παραγγελία της υπολογίζεται ακόμα.</p>}
        <p>
          Ο εγκέφαλος είναι ο πλήρης χάρτης μιας αρσενικής δροσόφιλας (Google & Janelia, 2026). Για κάθε ποτό τρέξαμε
          προσομοίωση όλου του εγκεφάλου στον υπολογιστή, και εδώ βλέπεις την επανάληψή της: ποιοι νευρώνες άναψαν, πόσο
          ήθελε να πιει και πόσο πικρό της φάνηκε. Για μια μύγα το αλκοόλ και η οξύτητα έχουν πικρή γεύση, οπότε
          παραγγέλνει το κοκτέιλ που της φαίνεται λιγότερο πικρό σε σχέση με το πόσο γλυκό είναι.
        </p>
      </footer>
    </div>
  );
}
