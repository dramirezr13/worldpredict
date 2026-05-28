"""Cuadro eliminatorio FIFA 2026 (48 equipos): cruces fijos y asignación de terceros."""

from __future__ import annotations

# (local, visitante) — cada lado es ("1"|"2", grupo) o ("3", slot_id)
R32_FIXTURES: list[tuple[tuple[str, str], tuple[str, str]]] = [
    ((("2", "A"), ("2", "B"))),           # 73
    ((("1", "E"), ("3", "slot_E"))),      # 74
    ((("1", "F"), ("2", "C"))),           # 75
    ((("1", "C"), ("2", "F"))),           # 76
    ((("1", "I"), ("3", "slot_I"))),      # 77
    ((("2", "E"), ("2", "I"))),           # 78
    ((("1", "A"), ("3", "slot_A"))),      # 79
    ((("1", "L"), ("3", "slot_L"))),      # 80
    ((("1", "D"), ("3", "slot_D"))),      # 81
    ((("1", "G"), ("3", "slot_G"))),      # 82
    ((("2", "K"), ("2", "L"))),           # 83 — 2.º K vs 2.º L (no 1K vs 2K)
    ((("1", "H"), ("2", "J"))),           # 84
    ((("1", "B"), ("3", "slot_B"))),      # 85
    ((("1", "J"), ("2", "H"))),           # 86
    ((("1", "K"), ("3", "slot_K"))),      # 87 — 1.º K vs tercero (no del grupo K)
    ((("2", "D"), ("2", "G"))),           # 88
]

# Orden de asignación de terceros a huecos (grupos permitidos por hueco)
THIRD_SLOT_RULES: list[tuple[str, frozenset[str]]] = [
    ("slot_E", frozenset("ABCDF")),
    ("slot_I", frozenset("CDFGH")),
    ("slot_A", frozenset("CEFHI")),
    ("slot_L", frozenset("EHIJK")),
    ("slot_D", frozenset("BEFIJ")),
    ("slot_G", frozenset("AEHIJ")),
    ("slot_B", frozenset("EFGIJ")),
    ("slot_K", frozenset("DEIJL")),
]

# (id_partido, id_alimentador_a, id_alimentador_b)
R16_FEEDERS = [
    (89, 74, 77),
    (90, 73, 75),
    (91, 76, 78),
    (92, 79, 80),
    (93, 83, 84),
    (94, 81, 82),
    (95, 86, 88),
    (96, 85, 87),
]

QF_FEEDERS = [
    (97, 89, 90),
    (98, 93, 94),
    (99, 91, 92),
    (100, 95, 96),
]

SF_FEEDERS = [
    (101, 97, 98),
    (102, 99, 100),
]

FINAL_FEEDERS = (104, 101, 102)


def _third_rank_key(row: dict) -> tuple:
    return (-row.get("points", 0), -row.get("gd", 0), -row.get("gf", 0))


def assign_third_place_slots(third_place_rows: list[dict]) -> dict[str, str]:
    """Asigna cada tercero clasificado al hueco de un 1.º de grupo (sin repetir grupo)."""
    ranked = sorted(third_place_rows, key=_third_rank_key)
    best = ranked[:8]
    remaining = {r["group"]: r for r in best}
    assignments: dict[str, str] = {}

    for slot_id, allowed in THIRD_SLOT_RULES:
        candidates = [remaining[g] for g in remaining if g in allowed]
        if not candidates:
            candidates = list(remaining.values())
        if not candidates:
            break
        pick = max(candidates, key=_third_rank_key)
        assignments[slot_id] = pick["team"]
        del remaining[pick["group"]]

    return assignments


def build_group_qualifiers(group_results: list[dict], best_third_teams: list[str]) -> dict:
    """Mapa de clasificados: 1.º y 2.º por grupo + terceros asignados a huecos."""
    winners = {}
    runners_up = {}
    third_by_group = {}

    for gr in group_results:
        grp = gr["group"]
        ranked = gr["standings"]
        winners[grp] = ranked[0]["team"]
        runners_up[grp] = ranked[1]["team"]
        if len(ranked) >= 3:
            third_row = ranked[2]
            if third_row["team"] in best_third_teams:
                third_by_group[grp] = {**third_row, "group": grp}

    third_rows = list(third_by_group.values())
    third_slots = assign_third_place_slots(third_rows)

    team_group = {}
    for grp, t in winners.items():
        team_group[t] = grp
    for grp, t in runners_up.items():
        team_group[t] = grp
    for row in third_rows:
        team_group[row["team"]] = row["group"]

    return {
        "winners": winners,
        "runners_up": runners_up,
        "third_slots": third_slots,
        "team_group": team_group,
    }


def resolve_slot(slot: tuple[str, str], qual: dict) -> str:
    role, key = slot
    if role == "1":
        return qual["winners"][key]
    if role == "2":
        return qual["runners_up"][key]
    if role == "3":
        return qual["third_slots"][key]
    raise ValueError(f"Slot inválido: {slot}")


def resolve_r32_pairing(qual: dict) -> list[tuple[str, str, int]]:
    """Lista (local, visitante, match_id) para dieciseisavos."""
    pairings = []
    for i, (home_slot, away_slot) in enumerate(R32_FIXTURES):
        match_id = 73 + i
        home = resolve_slot(home_slot, qual)
        away = resolve_slot(away_slot, qual)
        pairings.append((home, away, match_id))
    return pairings


def _groups_clash(home: str, away: str, team_group: dict) -> bool:
    gh = team_group.get(home)
    ga = team_group.get(away)
    return bool(gh and ga and gh == ga)


def validate_r32_pairings(pairings: list[tuple[str, str, int]], team_group: dict) -> None:
    for home, away, mid in pairings:
        if _groups_clash(home, away, team_group):
            raise ValueError(
                f"R32 inválido (M{mid}): {home} y {away} son del grupo {team_group[home]}"
            )


def build_knockout_round_fixtures(
    stage: str,
    match_winners: dict[int, str],
) -> list[tuple[str, str, int]]:
    """Emparejamientos de R16 en adelante según ganadores de partidos anteriores."""
    if stage == "Round of 16":
        feeders = R16_FEEDERS
    elif stage == "Quarter-finals":
        feeders = QF_FEEDERS
    elif stage == "Semi-finals":
        feeders = SF_FEEDERS
    elif stage == "Final":
        a, b, mid = FINAL_FEEDERS
        return [(match_winners[a], match_winners[b], mid)]
    else:
        return []

    pairings = []
    for new_id, fed_a, fed_b in feeders:
        pairings.append((match_winners[fed_a], match_winners[fed_b], new_id))
    return pairings
