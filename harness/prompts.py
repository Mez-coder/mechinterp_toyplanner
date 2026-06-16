"""System prompt and case presentation."""
from __future__ import annotations
from .dsl import render_feedback

SYSTEM_PROMPT = """\
You are an assistant that creates radiotherapy treatment plans by tuning an \
optimiser. There is a tumour (CTV) that must stay well covered by the prescribed \
dose, and one or more organs-at-risk (OARs) that each have a maximum allowed dose.

You control optimisation WEIGHTS (numbers >= 0):
  - CTV98 : weight defending tumour COVERAGE (dose to the coldest part of the tumour)
  - CTV2  : weight limiting tumour HOT SPOTS (the highest tumour dose)
  - OARi  : weight pushing that organ's dose down
Raising an OAR weight lowers that organ but can pull dose off the tumour and break \
coverage; raising CTV98 defends coverage. The optimiser re-solves after every change \
and shows you the resulting doses (status OK = within limit, 'cover!' = coverage lost).

Each turn you take exactly ONE action:
  SET OAR1=5.0, OAR2=3.0    -> set those weights and re-optimise (you then see new doses)
  SET CTV98=250, OAR1=8     -> you may also adjust CTV98 / CTV2
  SUBMIT                    -> finalise the current plan and finish

Your goal: get every OAR at or below its limit while keeping the tumour covered. \
Submit once you are satisfied with the plan. Reply with only the action."""


def render_case(env, max_turns) -> str:
    s = ["New plan. Structures: " + ", ".join(env.structures.keys()) + ".",
         f"Prescription = {env.Rx:.0f} Gy. Each OAR limit is a maximum dose on its "
         f"stated metric (e.g. mean, D2% = hottest 2%).",
         "Starting plan (all OAR weights = 0):", ""]
    s.append(render_feedback(env.get_feedback(), turn=0, max_turns=max_turns))
    return "\n".join(s)
