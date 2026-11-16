"""schema.py : the shared data structures used by every other module

Event   = one message the detector must judge (eps_n = (x_n, c_n, t_n, d_n) in the paper)
Session = the sequence of Events of one conversation
Action  = the three-level verdict (FORWARD < FORWARD_UNTRUSTED < BLOCK)

The Theta / Switches / Decision classes below belong to the rule-based baseline (the earlier version of the system).
The proposed system uses the classes in pipeline.py instead.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

DIRECT = "direct"      # text the user typed (has the authority to give instructions)
INDIRECT = "indirect"  # text the system retrieved: a document, a web page, a tool output (data only)

# ----- the three action levels (larger = stricter, so max() can be used) -----
FORWARD = 0
FORWARD_UNTRUSTED = 1
BLOCK = 2
ACTION_NAMES = {FORWARD: "Forward", FORWARD_UNTRUSTED: "Forward-Untrusted", BLOCK: "Block"}


@dataclass
class Event:
    """One event the detector must judge

    text    : the message x
    channel : DIRECT or INDIRECT (c)
    label   : 1 = carries an injected instruction and should be blocked, 0 = benign (y)
    source  : the source d (direct = "user", indirect = the id of the document or page)
    turn    : the retrieval turn t -- incremented by 1 every time the user sends a new message
    meta    : extra fields (e.g. the attack kind) used to break down the results; never passed to the detector
    """
    text: str
    channel: str
    label: int = 0
    meta: Dict[str, Any] = field(default_factory=dict)
    source: str = ""
    turn: int = 0


@dataclass
class Session:
    """The sequence of events of one conversation"""
    sid: str
    events: List[Event]
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_attack(self) -> bool:
        """An attack session has at least one event with label = 1 (the rule in Section IV-B of the paper)"""
        return any(e.label == 1 for e in self.events)


def finalize_session(sess: Session) -> Session:
    """Fills in any missing source / turn (turn increases at every direct message)"""
    turn = 0
    for i, e in enumerate(sess.events):
        if e.channel == DIRECT:
            turn += 1
            if not e.source:
                e.source = "user"
        else:
            if not e.source:
                e.source = f"{sess.sid}-doc{i}"
        if not e.turn:
            e.turn = turn
    return sess


# ======================================================================
# Below: the rule-based baseline (the earlier system) -- used only by rulebased.py
# ======================================================================
@dataclass
class Theta:
    """Parameters of the rule-based baseline"""
    lam: float = 0.2
    tau: float = 0.75
    tau_r: float = 0.20
    tau_o: float = 0.35
    tau_b: float = 0.50
    W: int = 5
    k_min: int = 3
    w1: float = 0.4
    w2: float = 0.3
    w3: float = 0.3


@dataclass
class Switches:
    stage1: bool = True
    stage2: bool = True
    stage3: bool = True
    stage4: bool = True
    rule_mode: str = "soft"
    reference_mode: str = "single"
    channel_agnostic: bool = False
    use_o: bool = True
    use_fbar: bool = True
    use_delta: bool = True


@dataclass
class Decision:
    verdict: int
    stage: str
    channel: str
    f: float
    s0: float
    s: float
    r: float | None = None
    b: float | None = None
    reference_updated: bool = False
    justification: str = ""
