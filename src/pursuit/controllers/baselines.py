"""Independent pursuit (IP) and rule-based coordination (RC)."""
from __future__ import annotations

from .local import LocalInfo

SEND_PERIOD = 10


def path_move(info: LocalInfo, dest: int) -> int:
    if dest < 0 or dest == info.own_cell:
        return 0
    return int(info.map.best_action[info.own_cell, dest])


class IPController:
    name = "IP"

    def act(self, info: LocalInfo):
        if info.mem_cell >= 0 and info.own_cell != info.mem_cell:
            dest = info.mem_cell
        else:
            dest = info.search_dest()
        return path_move(info, dest), False


class RCController:
    name = "RC"

    def act(self, info: LocalInfo):
        move = path_move(info, info.dest)
        want = (info.player_visible
                or info.last_sent_key != (info.role, info.dest)
                or info.t - info.last_send_step >= SEND_PERIOD)
        return move, bool(want and info.can_send())
