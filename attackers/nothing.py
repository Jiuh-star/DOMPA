from __future__ import annotations

from .core import Attacker, AttackInfo

__all__ = ["DoNothingAttacker"]


class DoNothingAttacker(Attacker):
    def attack_algorithm(self, info: AttackInfo):
        self.attack_log = {"message": "Do nothing."}
        return
