/-
Extract tactic steps from one Lean file, in the style of LeanDojo's ExtractData.lean:
a step is a tactic whose parent is a tactic sequence (or `rw` rule list), kept unless its
state before is "no goals" or the state does not change. Each step also lists the
constants it cites (identifiers elaborated to a constant) with the module that defines them.

Usage (from the project root, after `lake build`):
  lake env lean --run ExtractSteps.lean <file.lean> <module.name> <out.jsonl>
-/
import Lean
open Lean Elab Frontend Meta

structure Step where
  thm : String
  line : Nat
  tactic : String
  before : String
  after : String
  premises : Array (String × String)

def ppGoals (ctx : ContextInfo) (goals : List MVarId) : IO String := do
  if goals.isEmpty then return "no goals"
  let fmt ← ctx.runMetaM {} (do
    let fs ← goals.mapM fun g => Meta.ppGoal g
    return Std.Format.joinSep fs (Std.Format.text "\n\n"))
  return fmt.pretty.trimAscii.toString

partial def findDeclId (stx : Syntax) : Option Name :=
  if stx.getKind == ``Lean.Parser.Command.declId then some stx[0].getId
  else stx.getArgs.findSome? findDeclId

partial def collectConsts (t : InfoTree) (acc : Array Name) : Array Name :=
  match t with
  | .context _ t => collectConsts t acc
  | .node i cs =>
    let acc := match i with
      | .ofTermInfo ti =>
        if ti.stx.isIdent && !ti.isBinder then
          match ti.expr.consumeMData with
          | .const n _ => if acc.contains n then acc else acc.push n
          | _ => acc
        else acc
      | _ => acc
    cs.foldl (fun a c => collectConsts c a) acc
  | .hole _ => acc

def moduleOf (env : Environment) (self : Name) (n : Name) : String :=
  match env.getModuleIdxFor? n with
  | some idx => (env.header.moduleNames[idx.toNat]!).toString
  | none => self.toString

partial def visit (env : Environment) (self : Name) (src : String) (thm : String)
    (ctx? : Option ContextInfo) (parent? : Option Info) (t : InfoTree)
    (out : IO.Ref (Array Step)) : IO Unit := do
  match t with
  | .context pc t => visit env self src thm (pc.mergeIntoOuter? ctx?) parent? t out
  | .hole _ => pure ()
  | .node i cs =>
    if let (some ctx, .ofTacticInfo ti, some (.ofTacticInfo p)) := (ctx?, i, parent?) then
      let k := p.stx.getKind
      if k == ``Lean.Parser.Tactic.tacticSeq1Indented || k == ``Lean.Parser.Tactic.tacticSeqBracketed
          || k == ``Lean.Parser.Tactic.rwSeq then
        let before ← ppGoals { ctx with mctx := ti.mctxBefore } ti.goalsBefore
        let after ← ppGoals { ctx with mctx := ti.mctxAfter } ti.goalsAfter
        if before != "no goals" && before != after then
          if let (some a, some b) := (ti.stx.getPos? true, ti.stx.getTailPos? true) then
            if let .node .. := ti.stx then
              let text := String.Pos.Raw.extract src a b
              let line := (FileMap.ofString src).toPosition a |>.line
              let cs' := collectConsts (.node i cs) #[]
              let prem := cs'.map fun n => (n.toString, moduleOf env self n)
              out.modify (·.push { thm := thm, line, tactic := text, before, after, premises := prem })
    let ctx? := match i with
      | .ofCommandInfo _ => ctx?
      | _ => ctx?
    for c in cs do visit env self src thm ctx? (some i) c out

def esc (s : String) : String := (Json.str s).compress

unsafe def main (args : List String) : IO UInt32 := do
  let [path, modName, outPath] := args | do IO.eprintln "usage: <file> <module> <out>"; return 1
  initSearchPath (← findSysroot)
  enableInitializersExecution
  let src ← IO.FS.readFile path
  let self := modName.toName
  let inputCtx := Parser.mkInputContext src path
  let (header, parserState, messages) ← Parser.parseHeader inputCtx
  let (env, messages) ← processHeader header {} messages inputCtx
  let env := env.setMainModule self
  let cmdState := { Command.mkState env messages {} with infoState.enabled := true }
  let s ← IO.processCommands inputCtx parserState cmdState
  let finalEnv := s.commandState.env
  let out ← IO.mkRef (#[] : Array Step)
  for t in s.commandState.infoState.trees do
    -- each top-level tree is one command; name it by its declId and namespace
    let (ns, stx?) : Name × Option Syntax := match t with
      | .context (.commandCtx cc) (.node (.ofCommandInfo ci) _) => (cc.currNamespace, some ci.stx)
      | _ => (Name.anonymous, none)
    let thm := match stx? >>= findDeclId with
      | some id =>
        let id := if id.getRoot == `_root_ then id.replacePrefix `_root_ Name.anonymous else ns ++ id
        id.toString
      | none =>
        match stx? >>= (·.getPos?) with
        | some p => s!"{modName}:anon@{(FileMap.ofString src).toPosition p |>.line}"
        | none => s!"{modName}:anon"
    visit finalEnv self src thm none none t out
  let h ← IO.FS.Handle.mk outPath .write
  for st in (← out.get) do
    let prem := Json.arr (st.premises.map fun (n, m) => Json.mkObj [("full_name", n), ("module", m)])
    let j := Json.mkObj [("theorem", st.thm), ("module", modName), ("file_path", path),
      ("line", st.line), ("tactic", st.tactic), ("state_before", st.before),
      ("state_after", st.after), ("premises", prem)]
    h.putStrLn j.compress
  let errs := s.commandState.messages.toList.filter (·.severity == .error)
  IO.eprintln s!"{modName}: {(← out.get).size} steps, {errs.length} errors"
  return 0
