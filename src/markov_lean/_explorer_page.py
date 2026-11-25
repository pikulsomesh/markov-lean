"""The HTML, CSS and JS of the results explorer page (see ``markov_lean.explore``).

Plain HTML/JS/CSS with no libraries and no network requests, so the page opens
from disk. ``__TITLE__`` and ``__DATA__`` are filled in by ``explore.render``.
"""

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root {
  --bg: #ffffff; --fg: #1d1d1f; --muted: #6b6b70; --line: #e2e2e6; --panel: #f6f6f8;
  --accent: #2b59c3; --good: #1f7a3a; --bad: #b3261e; --empty: #ececef;
  --heat: 43, 89, 195;
  font-family: Arial, "Helvetica Neue", Helvetica, sans-serif;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #161618; --fg: #ececef; --muted: #a0a0a8; --line: #303036; --panel: #1f1f23;
    --accent: #8fb0ff; --good: #6fd08c; --bad: #ff8a80; --empty: #2a2a30; --heat: 143, 176, 255;
  }
}
:root[data-theme="dark"] {
  --bg: #161618; --fg: #ececef; --muted: #a0a0a8; --line: #303036; --panel: #1f1f23;
  --accent: #8fb0ff; --good: #6fd08c; --bad: #ff8a80; --empty: #2a2a30; --heat: 143, 176, 255;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--fg); font-size: 14px; line-height: 1.45; }
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }
header { padding: 16px 16px 8px; border-bottom: 1px solid var(--line); }
h1 { font-size: 20px; margin: 0 0 4px; }
h2 { font-size: 17px; margin: 0 0 6px; word-break: break-word; }
h3 { font-size: 14px; margin: 18px 0 6px; }
.muted { color: var(--muted); }
.small { font-size: 12px; }
nav { display: flex; gap: 4px; margin-top: 10px; flex-wrap: wrap; }
nav button { border: 1px solid var(--line); background: var(--panel); color: var(--fg); padding: 6px 12px;
  border-radius: 6px; cursor: pointer; font: inherit; }
nav button.on { background: var(--accent); border-color: var(--accent); color: var(--bg); }
main { padding: 12px 16px 40px; }
.cards { display: flex; flex-wrap: wrap; gap: 8px; margin: 8px 0; }
.card { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 8px 12px; min-width: 120px; }
.card b { display: block; font-size: 18px; }
.split { display: grid; grid-template-columns: minmax(240px, 340px) 1fr; gap: 16px; }
@media (max-width: 760px) { .split { grid-template-columns: 1fr; } }
.list { border: 1px solid var(--line); border-radius: 8px; max-height: 78vh; overflow: auto; }
.item { padding: 7px 10px; border-bottom: 1px solid var(--line); cursor: pointer; }
.item:hover, .item.on { background: var(--panel); }
.item .name { font-family: Menlo, Consolas, monospace; font-size: 12px; word-break: break-all; }
input[type=search] { width: 100%; padding: 8px; border: 1px solid var(--line); border-radius: 6px;
  background: var(--bg); color: var(--fg); font: inherit; }
.controls { display: flex; flex-direction: column; gap: 6px; margin-bottom: 8px; }
.controls label { font-size: 12px; color: var(--muted); }
.strip { display: flex; flex-wrap: wrap; gap: 2px; margin: 2px 0 6px; }
.cell { min-width: 26px; height: 22px; padding: 0 3px; border-radius: 3px; font-size: 11px; display: flex;
  align-items: center; justify-content: center; font-family: Menlo, Consolas, monospace; }
.mini .cell { min-width: 8px; height: 8px; padding: 0; font-size: 0; border-radius: 2px; }
.row-label { font-size: 12px; color: var(--muted); }
table { border-collapse: collapse; font-size: 12px; }
th, td { border: 1px solid var(--line); padding: 4px 6px; vertical-align: top; text-align: left; }
th { background: var(--panel); font-weight: 600; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
.scroll { overflow-x: auto; max-width: 100%; }
pre { margin: 0; white-space: pre-wrap; word-break: break-word; font-family: Menlo, Consolas, monospace; font-size: 11.5px; }
code { font-family: Menlo, Consolas, monospace; font-size: 12px; }
.chip { display: inline-block; padding: 0 5px; border-radius: 3px; font-family: Menlo, Consolas, monospace; font-size: 11px; }
.pass { color: var(--good); font-weight: 600; } .fail { color: var(--bad); font-weight: 600; }
.empty { padding: 20px; color: var(--muted); }
details summary { cursor: pointer; color: var(--muted); font-size: 12px; }
.theme { float: right; border: 1px solid var(--line); background: var(--panel); color: var(--fg); border-radius: 6px;
  padding: 4px 8px; cursor: pointer; font: inherit; font-size: 12px; }
</style>
</head>
<body>
<header>
  <button class="theme" id="theme" title="Switch light or dark">Theme</button>
  <h1 id="title"></h1>
  <div class="muted small" id="subtitle"></div>
  <nav id="nav"></nav>
</header>
<main id="main"></main>
<script type="application/json" id="data">__DATA__</script>
<script>
(function () {
"use strict";
var D = JSON.parse(document.getElementById("data").textContent);
var PALETTE = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f", "#edc948", "#b07aa1", "#ff9da7",
               "#9c755f", "#86bcb6", "#d37295", "#a0cbe8", "#8cd17d", "#f1ce63", "#bab0ac"];
var OTHER = "#9a9aa2";
var SEP = ">";
var view = "proofs", current = -1, query = "", onlySwitch = false, limit = 300;

function el(tag, attrs, kids) {
  var e = document.createElement(tag);
  if (attrs) for (var k in attrs) {
    if (k === "text") e.textContent = attrs[k];
    else if (k === "cls") e.className = attrs[k];
    else if (k === "style") e.setAttribute("style", attrs[k]);
    else if (k.slice(0, 2) === "on") e.addEventListener(k.slice(2), attrs[k]);
    else e.setAttribute(k, attrs[k]);
  }
  (kids || []).forEach(function (c) { if (c != null) e.appendChild(typeof c === "string" ? document.createTextNode(c) : c); });
  return e;
}
function fmt(x, nd) { return x == null ? "–" : (typeof x === "number" ? (Number.isInteger(x) ? String(x) : x.toFixed(nd == null ? 3 : nd)) : String(x)); }
function pct(x) { return x == null ? "–" : (100 * x).toFixed(1) + "%"; }

// colours: areas by how often they occur on the page, macrostates by id
var areaCount = {};
D.proofs.forEach(function (p) { p.st.forEach(function (r) { if (r[3] != null) areaCount[r[3]] = (areaCount[r[3]] || 0) + 1; }); });
var areaOrder = Object.keys(areaCount).sort(function (a, b) { return areaCount[b] - areaCount[a] || (a < b ? -1 : 1); });
var areaColor = {};
areaOrder.forEach(function (c, i) { areaColor[c] = i < PALETTE.length ? PALETTE[i] : OTHER; });
function macroColor(m) { return m == null ? null : PALETTE[m % PALETTE.length]; }
function ink(hex) {
  var r = parseInt(hex.slice(1, 3), 16), g = parseInt(hex.slice(3, 5), 16), b = parseInt(hex.slice(5, 7), 16);
  return (0.299 * r + 0.587 * g + 0.114 * b) > 150 ? "#111" : "#fff";
}
function areaTitle(c) { return c == null ? "no area" : (D.areas[c] ? c + " " + D.areas[c] : c); }
function chip(text, color, title) {
  var s = el("span", {cls: "chip", text: text, title: title || text});
  if (color) { s.style.background = color; s.style.color = ink(color); } else { s.style.background = "var(--empty)"; }
  return s;
}
function pathChips(path, kind) {
  var span = el("span");
  path.forEach(function (v, i) {
    if (i) span.appendChild(document.createTextNode(" → "));
    span.appendChild(kind === "m" ? chip("S" + v, macroColor(v), "macrostate " + v) : chip(v, areaColor[v] || OTHER, areaTitle(v)));
  });
  return span;
}
function strip(p, which, mini) {
  var s = el("div", {cls: "strip" + (mini ? " mini" : "")});
  p.st.forEach(function (r) {
    var v = which === "a" ? r[3] : r[6];
    var color = v == null ? null : (which === "a" ? (areaColor[v] || OTHER) : macroColor(v));
    var c = el("div", {cls: "cell", text: v == null ? "·" : (which === "a" ? v : "S" + v),
      title: "step " + r[0] + ": " + (which === "a" ? areaTitle(v) : (v == null ? "no state" : "macrostate " + v))});
    if (color) { c.style.background = color; c.style.color = ink(color); } else { c.style.background = "var(--empty)"; c.style.color = "var(--muted)"; }
    s.appendChild(c);
  });
  return s;
}

// search: words match the theorem, id or file; area:26 state:3 switch:26>11 path:26>11>26 filter
function matcher(q) {
  var terms = q.trim().toLowerCase().split(/\s+/).filter(Boolean);
  return function (p) {
    for (var i = 0; i < terms.length; i++) {
      var t = terms[i], m;
      if ((m = /^area:(.+)$/.exec(t))) { if (p.ap.indexOf(m[1].padStart(2, "0")) < 0) return false; }
      else if ((m = /^state:(\d+)$/.exec(t))) { if (p.mp.indexOf(+m[1]) < 0) return false; }
      else if ((m = /^switch:(.+)$/.exec(t))) { if (p.sa.indexOf(m[1]) < 0 && p.sm.indexOf(m[1]) < 0) return false; }
      else if ((m = /^path:(.+)$/.exec(t))) { if (p.ap.join(SEP) !== m[1]) return false; }
      else if ((p.th + " " + p.id + " " + p.fp).toLowerCase().indexOf(t) < 0) return false;
    }
    return !onlySwitch || p.ap.length > 1 || p.mp.length > 1;
  };
}

function setHash() {
  var h = "view=" + view;
  if (view === "proofs" && current >= 0) h += "&p=" + current;
  if (query) h += "&q=" + encodeURIComponent(query);
  history.replaceState(null, "", "#" + h);
}
function readHash() {
  var h = location.hash.replace(/^#/, "");
  h.split("&").forEach(function (kv) {
    var i = kv.indexOf("="); if (i < 0) return;
    var k = kv.slice(0, i), v = decodeURIComponent(kv.slice(i + 1));
    if (k === "view" && (v === "proofs" || v === "model" || v === "patterns")) view = v;
    if (k === "p" && /^\d+$/.test(v) && +v < D.proofs.length) current = +v;
    if (k === "q") query = v;
  });
}
function go(v, p, q) { view = v; if (p != null) current = p; if (q != null) query = q; setHash(); draw(); window.scrollTo(0, 0); }

// ------------------------------------------------------------------ proofs view
function proofLink(i, label) {
  return el("a", {href: "#view=proofs&p=" + i, text: label || D.proofs[i].th, onclick: function (e) { e.preventDefault(); go("proofs", i); }});
}
function drawProofs(root) {
  var split = el("div", {cls: "split"});
  var left = el("div"), right = el("div");
  var box = el("input", {type: "search", placeholder: "Search theorem, file, area:26, state:3, switch:26>11", value: query});
  var cb = el("input", {type: "checkbox"}); cb.checked = onlySwitch;
  var count = el("div", {cls: "muted small"});
  var list = el("div", {cls: "list"});
  left.appendChild(el("div", {cls: "controls"}, [box, el("label", null, [cb, " only proofs that switch"]), count]));
  left.appendChild(list);
  function fill() {
    list.innerHTML = "";
    var f = matcher(query), hits = [];
    D.proofs.forEach(function (p, i) { if (f(p)) hits.push(i); });
    count.textContent = hits.length + " of " + D.proofs.length + " proofs";
    hits.slice(0, limit).forEach(function (i) {
      var p = D.proofs[i];
      var it = el("div", {cls: "item" + (i === current ? " on" : ""), onclick: function () { go("proofs", i); }},
        [el("div", {cls: "name", text: p.th}),
         el("div", {cls: "muted small", text: p.st.length + " steps" + (p.ap.length > 1 ? " · " + p.ap.join(" → ") : "")}),
         D.meta.area_source !== "none yet" ? strip(p, "a", true) : strip(p, "m", true)]);
      list.appendChild(it);
    });
    if (hits.length > limit) list.appendChild(el("div", {cls: "item muted", text: "Show more (" + (hits.length - limit) + " left)",
      onclick: function () { limit += 300; fill(); }}));
    if (!hits.length) list.appendChild(el("div", {cls: "empty", text: "No proofs match."}));
  }
  box.addEventListener("input", function () { query = box.value; limit = 300; setHash(); fill(); });
  cb.addEventListener("change", function () { onlySwitch = cb.checked; fill(); });
  fill();
  if (current < 0 && D.proofs.length) current = 0;
  right.appendChild(current >= 0 ? proofDetail(D.proofs[current]) : el("div", {cls: "empty", text: "No proofs in these results yet."}));
  split.appendChild(left); split.appendChild(right);
  root.appendChild(split);
}
function proofDetail(p) {
  var d = el("div");
  d.appendChild(el("h2", {text: p.th}));
  var sub = [p.st.length + " steps"];
  if (p.fp) sub.push(p.fp);
  if (p.pl) sub.push("whole-proof label " + p.pl);
  d.appendChild(el("div", {cls: "muted small", text: sub.join(" · ")}));
  if (p.id !== p.th) d.appendChild(el("div", {cls: "muted small"}, ["id ", el("code", {text: p.id})]));
  d.appendChild(el("h3", {text: "Area path"}));
  d.appendChild(el("div", {cls: "row-label", text: "Judge area (" + D.meta.area_source + ")"}));
  d.appendChild(strip(p, "a"));
  if (D.meta.has_model) { d.appendChild(el("div", {cls: "row-label", text: "Markov-model macrostate (PCCA+)"})); d.appendChild(strip(p, "m")); }
  var paths = el("div", {cls: "small"});
  if (p.ap.length) paths.appendChild(el("div", null, ["Areas: ", pathChips(p.ap, "a")]));
  if (p.mp.length) paths.appendChild(el("div", null, ["States: ", pathChips(p.mp, "m")]));
  d.appendChild(paths);

  d.appendChild(el("h3", {text: "Steps"}));
  var t = el("table");
  t.appendChild(el("tr", null, ["Step", "Tactic", "Goal before the tactic", "Area", "State"].map(function (h) { return el("th", {text: h}); })));
  p.st.forEach(function (r) {
    var area = el("td");
    if (r[3] != null) area.appendChild(chip(r[3], areaColor[r[3]] || OTHER, areaTitle(r[3])));
    else area.appendChild(el("span", {cls: "muted", text: "–"}));
    if (r[4] != null) area.appendChild(el("div", {cls: "muted small", text: "conf " + fmt(r[4], 2)}));
    if (r[5]) area.appendChild(el("div", {cls: "muted small", text: "judges: " + r[5], title: "the judges disagree on this step"}));
    var st = el("td");
    if (r[6] != null) st.appendChild(chip("S" + r[6], macroColor(r[6]), "macrostate " + r[6]));
    else st.appendChild(el("span", {cls: "muted", text: "–"}));
    if (r[7] != null) st.appendChild(el("div", {cls: "muted small", text: "memb " + fmt(r[7], 2)}));
    var goal = el("td");
    if (r[2]) {
      if (r[2].length > 400) goal.appendChild(el("details", null, [el("summary", {text: r[2].slice(0, 120).replace(/\n/g, " ") + " …"}), el("pre", {text: r[2]})]));
      else goal.appendChild(el("pre", {text: r[2]}));
    } else goal.appendChild(el("span", {cls: "muted", text: D.meta.has_text ? "–" : "goal text not in these results"}));
    t.appendChild(el("tr", null, [el("td", {cls: "num", text: String(r[0])}), el("td", null, [r[1] ? el("code", {text: r[1]}) : el("span", {cls: "muted", text: "–"})]), goal, area, st]));
  });
  d.appendChild(el("div", {cls: "scroll"}, [t]));

  function rel(title, list, sw, kind) {
    if (!list || !list.length) return;
    d.appendChild(el("h3", {text: title}));
    var ul = el("ul");
    list.forEach(function (x) {
      var o = D.proofs[x[0]];
      ul.appendChild(el("li", null, [proofLink(x[0]), el("span", {cls: "muted small", text: x[2] ? "  same path " : "  shares " + x[1] + " switch" + (x[1] > 1 ? "es" : "") + " "}),
        pathChips(kind === "m" ? o.mp : o.ap, kind)]));
    });
    d.appendChild(ul);
  }
  rel("Proofs with the same area switches", p.ra, p.sa, "a");
  rel("Proofs with the same macrostate switches", p.rm, p.sm, "m");
  if (p.ap.length > 1) d.appendChild(el("p", {cls: "small"}, [el("a", {href: "#", text: "All proofs with the path " + p.ap.join(" → "),
    onclick: function (e) { e.preventDefault(); go("proofs", null, "path:" + p.ap.join(SEP)); }})]));
  return d;
}

// ------------------------------------------------------------------ model view
function heat(v, max) {
  var a = max > 0 ? Math.min(1, v / max) : 0;
  return "background: rgba(var(--heat), " + (0.08 + 0.8 * a).toFixed(3) + ");" + (a > 0.55 ? "color: #fff;" : "");
}
function drawModel(root) {
  var R = D.run || {}, M = D.model;
  var cards = el("div", {cls: "cards"});
  function card(label, value, cls) { cards.appendChild(el("div", {cls: "card"}, [el("b", {text: value, cls: cls || ""}), el("span", {cls: "muted small", text: label})])); }
  card("proofs on this page", D.meta.n_proofs_shown + " of " + D.meta.n_proofs_total);
  card("steps on this page", String(D.meta.n_steps_shown));
  if (M) { card("macrostates (k)", String(M.k)); if (M.lag != null) card("lag (" + D.meta.time_unit + ")", String(M.lag)); if (M.n_microstates != null) card("microstates", String(M.n_microstates)); }
  if (R.gate1) card("gate 1 at k=" + fmt(R.gate1.k), R.gate1.passed ? "passed" : "failed", R.gate1.passed ? "pass" : "fail");
  else if (R.gate1_status) card("gate 1", R.gate1_status);
  if (R.gate1) { card("macrostate NMI", fmt(R.gate1.macrostate_nmi)); card("folder-baseline NMI", fmt(R.gate1.folder_nmi)); }
  var JA = R.judge_agreement && R.judge_agreement.step;
  if (JA && JA.agreement != null) card("judges agree on steps", pct(JA.agreement));
  if (JA && JA.kappa != null) card("judge kappa (steps)", fmt(JA.kappa));
  root.appendChild(cards);
  root.appendChild(el("p", {cls: "muted small", text: "Areas: " + D.meta.area_source + "."}));

  if (R.agreement) {
    root.appendChild(el("h3", {text: "Agreement with the judge areas"}));
    var ta = el("table"); ta.appendChild(el("tr", null, [el("th", {text: "Prediction"}), el("th", {text: "NMI"}), el("th", {text: "ARI"})]));
    Object.keys(R.agreement).forEach(function (k) { var v = R.agreement[k]; ta.appendChild(el("tr", null, [el("td", {text: k.replace("_", " ")}), el("td", {cls: "num", text: fmt(v.nmi)}), el("td", {cls: "num", text: fmt(v.ari)})])); });
    if (R.sweep) Object.keys(R.sweep).forEach(function (k) { var v = R.sweep[k]; ta.appendChild(el("tr", null, [el("td", {text: "macrostates, k=" + k}), el("td", {cls: "num", text: fmt(v.nmi)}), el("td", {cls: "num", text: fmt(v.ari)})])); });
    root.appendChild(el("div", {cls: "scroll"}, [ta]));
  }
  if (!M) { root.appendChild(el("div", {cls: "empty", text: "No Markov model in these results yet."})); return; }
  if (M.checks && Object.keys(M.checks).length) {
    var cl = el("div", {cls: "small"}, ["Model checks: "]);
    Object.keys(M.checks).forEach(function (k, i) { if (i) cl.appendChild(document.createTextNode(", ")); cl.appendChild(el("span", {cls: M.checks[k] ? "pass" : "fail", text: k.replace(/_/g, " ")})); });
    root.appendChild(cl);
  }
  if (M.T) {
    root.appendChild(el("h3", {text: "Coarse-grained transition matrix (row: from, column: to)"}));
    var t = el("table"), hr = el("tr", null, [el("th", {text: ""})]);
    M.T.forEach(function (_, j) { hr.appendChild(el("th", null, [chip("S" + j, macroColor(j))])); });
    t.appendChild(hr);
    M.T.forEach(function (row, i) {
      var tr = el("tr", null, [el("th", null, [chip("S" + i, macroColor(i))])]);
      row.forEach(function (v) { tr.appendChild(el("td", {cls: "num", text: v == null ? "–" : v.toFixed(2), style: heat(v || 0, 1)})); });
      t.appendChild(tr);
    });
    root.appendChild(el("div", {cls: "scroll"}, [t]));
  }
  root.appendChild(el("h3", {text: "PCCA+ sets"}));
  var s = el("table");
  s.appendChild(el("tr", null, ["Set", "Steps", "Share", "Proofs", "Microstates", "Stationary", "Stay", "Mean membership", "Top judge areas"].map(function (h) { return el("th", {text: h}); })));
  M.sets.forEach(function (r) {
    var top = el("td");
    (r.top_areas || []).forEach(function (x) { top.appendChild(chip(x[0], areaColor[x[0]] || OTHER, areaTitle(x[0]))); top.appendChild(document.createTextNode(" " + pct(x[1]) + "  ")); });
    s.appendChild(el("tr", null, [el("td", null, [el("a", {href: "#", onclick: function (e) { e.preventDefault(); go("proofs", null, "state:" + r.id); }}, [chip("S" + r.id, macroColor(r.id))])]),
      el("td", {cls: "num", text: fmt(r.n_steps)}), el("td", {cls: "num", text: pct(r.share_steps)}), el("td", {cls: "num", text: fmt(r.n_proofs)}),
      el("td", {cls: "num", text: fmt(r.n_microstates)}), el("td", {cls: "num", text: pct(r.stationary)}), el("td", {cls: "num", text: fmt(r.stay, 2)}),
      el("td", {cls: "num", text: fmt(r.mean_membership, 2)}), top]));
  });
  root.appendChild(el("div", {cls: "scroll"}, [s]));
  if (M.n_unassigned_steps) root.appendChild(el("p", {cls: "muted small", text: M.n_unassigned_steps + " steps sit outside the connected set and have no macrostate."}));
  if (M.crosstab) {
    root.appendChild(el("h3", {text: "Judge areas within each set (share of the set's labelled steps)"}));
    var c = el("table"), h = el("tr", null, [el("th", {text: ""})]);
    M.crosstab.areas.forEach(function (a) { h.appendChild(el("th", {title: areaTitle(a)}, [chip(a, areaColor[a] || OTHER, areaTitle(a))])); });
    c.appendChild(h);
    M.crosstab.rows.forEach(function (row, i) {
      var tot = row.reduce(function (x, y) { return x + y; }, 0);
      var tr = el("tr", null, [el("th", null, [chip("S" + i, macroColor(i))])]);
      row.forEach(function (v) { var f = tot ? v / tot : 0; tr.appendChild(el("td", {cls: "num", text: tot ? (100 * f).toFixed(0) : "–", title: v + " steps", style: heat(f, 1)})); });
      c.appendChild(tr);
    });
    root.appendChild(el("div", {cls: "scroll"}, [c]));
  }
}

// ------------------------------------------------------------------ patterns view
function drawPatterns(root) {
  if (!D.patterns.length) { root.appendChild(el("div", {cls: "empty", text: "No proof on this page switches judge area yet."})); return; }
  root.appendChild(el("p", {cls: "muted small", text: "Judge-area paths with at least one switch, among the proofs on this page. Repeats are collapsed and unlabelled steps skipped."}));
  var t = el("table");
  t.appendChild(el("tr", null, ["Path", "Proofs", "Examples"].map(function (h) { return el("th", {text: h}); })));
  D.patterns.forEach(function (pt) {
    var ex = el("td");
    pt.proofs.slice(0, 3).forEach(function (i, k) { if (k) ex.appendChild(document.createTextNode(", ")); ex.appendChild(proofLink(i)); });
    t.appendChild(el("tr", null, [el("td", null, [el("a", {href: "#", onclick: function (e) { e.preventDefault(); go("proofs", pt.proofs[0], "path:" + pt.key); }}, [pathChips(pt.key.split(SEP), "a")])]),
      el("td", {cls: "num", text: String(pt.n)}), ex]));
  });
  root.appendChild(el("div", {cls: "scroll"}, [t]));
}

function draw() {
  var nav = document.getElementById("nav"); nav.innerHTML = "";
  [["proofs", "Proofs"], ["model", "Model"], ["patterns", "Patterns"]].forEach(function (v) {
    nav.appendChild(el("button", {cls: view === v[0] ? "on" : "", text: v[1], onclick: function () { go(v[0]); }}));
  });
  var root = document.getElementById("main"); root.innerHTML = "";
  if (view === "model") drawModel(root); else if (view === "patterns") drawPatterns(root); else drawProofs(root);
}

document.getElementById("title").textContent = D.meta.title;
document.getElementById("subtitle").textContent = D.meta.n_proofs_shown + " proofs, " + D.meta.n_steps_shown +
  " goal states. Areas: " + D.meta.area_source + ". Time is counted in " + D.meta.time_unit + "." +
  (D.meta.n_truncated_states ? " " + D.meta.n_truncated_states + " goal states are cut to " + D.meta.max_state_chars + " characters." : "");
document.getElementById("theme").addEventListener("click", function () {
  var r = document.documentElement, dark = r.getAttribute("data-theme") === "dark" ||
    (!r.getAttribute("data-theme") && window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches);
  r.setAttribute("data-theme", dark ? "light" : "dark");
});
readHash();
draw();
window.addEventListener("hashchange", function () { readHash(); draw(); });
})();
</script>
</body>
</html>
"""
