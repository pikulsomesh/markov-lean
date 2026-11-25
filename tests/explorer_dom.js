// Runs the explorer page's script in Node against a tiny fake DOM and walks its views.
// Usage: node tests/explorer_dom.js site/index.html   -> prints a JSON summary, exits 1 on a script error.
// Used by tests/test_explore.py when Node is installed; no packages needed.
"use strict";
const fs = require("fs");
const vm = require("vm");

const html = fs.readFileSync(process.argv[2], "utf8");
const dataMatch = html.match(/<script type="application\/json" id="data">([\s\S]*?)<\/script>/);
const scripts = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(m => m[1]);
if (!dataMatch || scripts.length !== 1) { console.error("page layout changed"); process.exit(2); }

class Node_ {
  constructor(tag) { this.tagName = tag; this.children = []; this.attrs = {}; this.listeners = {}; this.style = {}; this._text = ""; this.className = ""; this.checked = false; this.value = ""; }
  appendChild(c) { this.children.push(c); return c; }
  setAttribute(k, v) { this.attrs[k] = String(v); if (k === "value") this.value = String(v); }
  getAttribute(k) { return k in this.attrs ? this.attrs[k] : null; }
  addEventListener(t, f) { (this.listeners[t] = this.listeners[t] || []).push(f); }
  set textContent(v) { this._text = String(v); this.children = []; }
  get textContent() { return this._text + this.children.map(c => c.textContent).join(""); }
  set innerHTML(v) { this.children = []; this._text = ""; }
  fire(t) { (this.listeners[t] || []).forEach(f => f({ preventDefault() {}, target: this })); }
  all(pred, out = []) { if (pred(this)) out.push(this); this.children.forEach(c => c.all && c.all(pred, out)); return out; }
}
class Text_ { constructor(t) { this.t = t; } get textContent() { return this.t; } }

const ids = {};
["title", "subtitle", "nav", "main", "theme"].forEach(i => { ids[i] = new Node_("div"); });
ids.data = { textContent: JSON.parse(JSON.stringify(dataMatch[1])) };
const root = new Node_("html");
const document = {
  createElement: t => new Node_(t), createTextNode: t => new Text_(t),
  getElementById: i => ids[i], documentElement: root,
};
const location = { hash: "" };
const errors = [];
const window = { addEventListener() {}, scrollTo() {}, matchMedia: () => ({ matches: false }), innerWidth: 1280 };
const history = { replaceState(_, __, h) { location.hash = h; } };
const ctx = vm.createContext({ document, location, window, history, console, JSON, Math, Number, String, Object, Array, RegExp, encodeURIComponent, decodeURIComponent, parseInt });

try { vm.runInContext(scripts[0], ctx); } catch (e) { errors.push("load: " + e.message); }
const main = ids.main, nav = ids.nav;
const navButton = name => nav.children.find(b => b.textContent === name);
const items = () => main.all(n => typeof n.className === "string" && n.className.split(" ").includes("item") && !n.className.includes("muted"));
const out = { errors };
try {
  out.title = ids.title.textContent;
  out.items = items().length;
  out.detail_tables = main.all(n => n.tagName === "table").length;
  const search = main.all(n => n.tagName === "input" && n.attrs.type === "search")[0];
  search.value = process.argv[3] || "zzz-no-such-proof"; search.fire("input");
  out.items_after_search = items().length;
  search.value = ""; search.fire("input");
  const first = items()[1];
  if (first) { first.fire("click"); out.hash_after_click = location.hash; }
  navButton("Model").fire("click");
  out.model_tables = main.all(n => n.tagName === "table").length;
  out.model_text = main.textContent.slice(0, 2000);
  navButton("Patterns").fire("click");
  out.pattern_links = main.all(n => n.tagName === "a").length;
  const pl = main.all(n => n.tagName === "a")[0];
  if (pl) { pl.fire("click"); out.items_after_pattern = items().length; out.hash_after_pattern = location.hash; }
  navButton("Proofs").fire("click");
  ids.theme.fire("click");
  out.theme = root.getAttribute("data-theme");
} catch (e) { errors.push("walk: " + e.stack); }
console.log(JSON.stringify(out));
process.exit(errors.length ? 1 : 0);
