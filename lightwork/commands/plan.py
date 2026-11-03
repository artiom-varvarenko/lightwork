"""``lightwork plan``: write (or show) the plan the table would walk you through, without the rig.

``lightwork plan "help me make pancakes for four"`` asks the brain (Muse Spark by default; ``--model
muse-wonder-1.0`` for Muse Wonder, ``--brain open`` for Meta open weights via the partners) and prints the items, the
steps and what the table would say. ``lightwork plan pancakes --serves 4`` shows a built-in plan scaled. ``--save`` writes
the plan as JSON for ``lightwork play --table --do <file>``. Model calls are recorded like the table's.
"""

from __future__ import annotations

COMMAND = "plan"
HELP = "Write a job plan from words (Muse Spark/Wonder or open weights), or show a built-in one, without the rig"


def configure(p) -> None:
    p.add_argument("job", help="what you want to do, in words, or a built-in plan name / plan file")
    p.add_argument("--serves", type=float, help="scale a recipe-style plan to this many servings")
    p.add_argument("--brain", choices=("meta", "open"), default="meta", help="who writes a plan from words")
    p.add_argument("--model", help="Meta model for --brain meta (default muse-spark-1.3)")
    p.add_argument("--effort", default="minimal", help="reasoning effort (minimal | low | medium | high)")
    p.add_argument("--save", help="write the plan JSON here")


def run(args) -> int:
    import json
    import time
    from pathlib import Path

    from ..guide import (
        PLAN_PROMPT,
        PLAN_SCHEMA,
        Guide,
        load_plan,
        parse_plan,
        plan_dict,
        plan_request_context,
        scale_plan,
    )

    try:
        plan = load_plan(args.job)
    except ValueError:
        plan = None
    if plan is None:
        from ..brain import BrainError, make_brain
        from ..config import load_settings

        brain = make_brain(args.brain, load_settings(), model=args.model)
        problem = brain.check()
        if problem:
            print(f"no brain: {problem}")
            return 2
        t0 = time.monotonic()
        try:
            reply = brain.ask("plan", PLAN_PROMPT.format(request=args.job, context=plan_request_context([])), [],
                              schema=PLAN_SCHEMA, effort=args.effort, max_out=8000, timeout_s=300.0)
            plan = parse_plan(reply.data, source=f"{reply.model} via {reply.provider}")
        except (BrainError, ValueError) as exc:
            print(f"planning failed: {exc}")
            return 1
        finally:
            brain.close()
        print(f"planned by {reply.model} via {reply.provider} in {time.monotonic() - t0:.1f} s "
              f"(${reply.cost_usd:.4f}); calls recorded in {brain.run_dir}")
    if args.serves:
        plan = scale_plan(plan, args.serves)
    print(f"\n{plan.title}" + (f" ({plan.note})" if plan.note else "") + f" [{plan.source}]")
    for it in plan.items:
        amt = it.amount_text()
        extra = f" x{it.count}" if it.count > 1 else ""
        print(f"  {it.kind:9s} {it.id:14s} {it.name}{extra}{' · ' + amt if amt else ''}{' (optional)' if it.optional else ''}")
    for k, st in enumerate(plan.steps, 1):
        what = {"transfer": f"{st.item} -> {st.into}" + (f" x{st.count}" if st.count else "")
                + (f" measured with {st.measure}" if st.measure else ""),
                "timer": f"{st.seconds:.0f} s" if st.seconds else "", "check": st.question or "",
                "use": st.item or "", "confirm": ""}[st.do]
        print(f"  {k}. {st.do:8s} {st.say}" + (f"  [{what}]" if what else ""))
    print("\nthe table says: " + Guide(plan).start(0.0)[0])
    if args.save:
        Path(args.save).write_text(json.dumps(plan_dict(plan), indent=1) + "\n", encoding="utf-8")
        print(f"saved {args.save}")
    return 0
