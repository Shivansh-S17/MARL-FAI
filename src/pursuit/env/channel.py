"""Budgeted broadcast channel and the fixed 16-byte application packet."""
from __future__ import annotations

import struct
from dataclasses import dataclass
from fractions import Fraction

PACKET_FMT = "<BBHHHBBHHH"
PACKET_BYTES = struct.calcsize(PACKET_FMT)
assert PACKET_BYTES == 16

F_SENDER = 1   # sender cell valid
F_TARGET = 2   # target cell + observation step valid
F_DISP = 4     # last valid target displacement valid
F_DEST = 8     # intended destination valid
F_ROLE = 16    # role id valid
# invalid placeholders: field set to 0 and its flag cleared (0 is also a legal value,
# so receivers must always consult the flag).


@dataclass
class Packet:
    sender: int
    flags: int = 0
    sender_cell: int = 0
    target_cell: int = 0
    target_step: int = 0
    disp: int = 0
    role: int = 0
    dest: int = 0
    send_step: int = 0

    def pack(self) -> bytes:
        return struct.pack(PACKET_FMT, self.sender, self.flags, self.sender_cell, self.target_cell,
                           self.target_step, self.disp, self.role, self.dest, self.send_step, 0)

    @staticmethod
    def unpack(b: bytes) -> "Packet":
        s, fl, sc, tc, ts, d, r, de, ss, _ = struct.unpack(PACKET_FMT, b)
        return Packet(s, fl, sc, tc, ts, d, r, de, ss)


def allowed_sends(budget: Fraction, t: int) -> int:
    """floor(B*(t+1)), exact rational arithmetic: cumulative sends allowed at decision t."""
    return (budget.numerator * (t + 1)) // budget.denominator


class Channel:
    def __init__(self, n: int, budget: Fraction, delay: int):
        self.n, self.budget, self.delay = n, budget, delay
        self.reset()

    def reset(self):
        self.sent = [0] * self.n
        self.pending: list[tuple[int, int, int, bytes]] = []  # (arrival, sender, send_step, payload)
        self.log: list[dict] = []  # sent packets with timestamps (for traces/audits)

    def can_send(self, i: int, t: int) -> bool:
        return self.sent[i] < allowed_sends(self.budget, t)

    def credit(self, i: int, t: int) -> int:
        return allowed_sends(self.budget, t) - self.sent[i]

    def send(self, i: int, t: int, payload: bytes):
        assert len(payload) == PACKET_BYTES
        assert self.can_send(i, t), "budget violation"
        self.sent[i] += 1
        arrival = t + 1 + self.delay
        self.pending.append((arrival, i, t, payload))
        self.log.append(dict(sender=i, send_step=t, arrival=arrival))

    def deliver(self, t: int) -> dict[int, list[bytes]]:
        """Packets arriving for decision t, per receiver (broadcast to all others)."""
        out: dict[int, list[bytes]] = {j: [] for j in range(self.n)}
        keep = []
        for item in self.pending:
            arrival, sender, _, payload = item
            if arrival == t:
                for j in range(self.n):
                    if j != sender:
                        out[j].append(payload)
            elif arrival > t:
                keep.append(item)
        self.pending = keep
        return out

    @property
    def total_sent(self) -> int:
        return sum(self.sent)
