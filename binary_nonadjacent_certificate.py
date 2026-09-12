#!/usr/bin/env python3
"""Exact finite certificate for the binary-nonadjacent case of policy transfer.

The program uses only the Python standard library and integer arithmetic.
It enumerates a single anchored petal, constructs the full wedge from the
proved one-active-petal decomposition, and evaluates the exact deterministic
Bellman recurrences.  In particular, ``common_policy`` carries inherited
mistake allowances down every shared branch and requires one prediction for
both palettes.

This executable audit supplements the structural proof in finalNMT.tex; it is
not used as a substitute for that proof.
"""

from functools import cache
from itertools import product


BUDGET = 200_000
LOCAL_NAMES = ("t", "a", "b", "c", "d", "e")
PETAL_NAMES = ("o",) + LOCAL_NAMES

# All omitted pairs have weight zero.
WEIGHTS = {
    ("o", "a"): 39_998,
    ("o", "c"): 28_572,
    ("o", "d"): 28_572,
    ("o", "e"): 5_714,
    ("t", "a"): 34_285,
    ("t", "b"): 48_571,
    ("t", "c"): 2_856,
    ("t", "e"): 17_144,
    ("a", "b"): 20_001,
    ("a", "e"): 8_572,
    ("b", "c"): 48_572,
    ("b", "d"): 82_858,
    ("c", "d"): 14_284,
    ("c", "e"): 8_572,
    ("d", "e"): 74_288,
}


def petal_energy(local_word):
    """Return the anchored petal energy, with o fixed to color 0."""
    colors = dict(zip(PETAL_NAMES, (0,) + tuple(local_word)))
    return sum(
        weight
        for (left, right), weight in WEIGHTS.items()
        if colors[left] != colors[right]
    )


def anchored_class(q):
    """Enumerate the exact feasible q-color class on t,a,b,c,d,e."""
    return tuple(
        word
        for word in product(range(q), repeat=len(LOCAL_NAMES))
        if petal_energy(word) <= BUDGET
    )


class RankGame:
    """A finite target class with its exact self-directed Bellman rank."""

    def __init__(self, concepts, q):
        self.concepts = tuple(sorted(set(map(tuple, concepts))))
        self.q = q
        self.width = len(self.concepts[0])
        self.full_state = (1 << len(self.concepts)) - 1
        self.full_remaining = (1 << self.width) - 1

        masks = [[0 for _ in range(q)] for _ in range(self.width)]
        for concept_id, concept in enumerate(self.concepts):
            bit = 1 << concept_id
            for coordinate, label in enumerate(concept):
                masks[coordinate][label] |= bit
        self.masks = tuple(tuple(row) for row in masks)

    @cache
    def rank(self, state, remaining):
        """Exact minimax mistakes for a residual class represented by a bitset."""
        if state.bit_count() <= 1:
            return 0

        best = self.width + 1
        for coordinate in range(self.width):
            coordinate_bit = 1 << coordinate
            if not (remaining & coordinate_bit):
                continue

            branches = []
            for answer in range(self.q):
                child = state & self.masks[coordinate][answer]
                if child:
                    child_rank = self.rank(child, remaining ^ coordinate_bit)
                    branches.append((answer, child_rank))

            for prediction in range(self.q):
                value = max(
                    child_rank + (answer != prediction)
                    for answer, child_rank in branches
                )
                best = min(best, value)

        return best

    def root_rank(self):
        return self.rank(self.full_state, self.full_remaining)

    def root_profiles(self):
        profiles = []
        for coordinate in range(self.width):
            remaining = self.full_remaining ^ (1 << coordinate)
            profiles.append(
                tuple(
                    self.rank(
                        self.full_state & self.masks[coordinate][answer],
                        remaining,
                    )
                    for answer in range(self.q)
                )
            )
        return tuple(profiles)


def relabel_base_zero(word, base, q):
    """Apply the transposition 0 <-> base to an anchored local word."""
    permutation = list(range(q))
    permutation[0], permutation[base] = permutation[base], permutation[0]
    return tuple(permutation[label] for label in word)


def wedge_class(anchored, q):
    """Construct the exact two-petal class using one-active-petal structure.

    Coordinates are o followed by the six coordinates of petal 1 and then
    the six coordinates of petal 2.  No q^13 enumeration is performed.
    """
    concepts = set()
    for base in range(q):
        constant_local = (base,) * len(LOCAL_NAMES)
        concepts.add((base,) + constant_local + constant_local)

        for word in anchored:
            if all(label == 0 for label in word):
                continue
            active = relabel_base_zero(word, base, q)
            concepts.add((base,) + active + constant_local)
            concepts.add((base,) + constant_local + active)

    return tuple(sorted(concepts))


def common_policy(lower, higher, lower_cap, higher_cap):
    """Decide exact same-prediction policy feasibility at inherited caps.

    A memoized state contains both residual target sets, the unqueried
    coordinates, and both remaining mistake allowances.  At a history shared
    by the two games, the recurrence chooses one ordered action
    (coordinate, prediction).  An old-label response reachable in both games
    decrements both allowances and recurses on both residual classes.  Once a
    response is unreachable in the lower game, only the higher-class rank is
    constrained.  Allowances are never reset to residual minimax ranks.
    """
    assert lower.width == higher.width
    assert lower.q < higher.q

    @cache
    def feasible(lower_state, higher_state, remaining, lower_left, higher_left):
        if lower_left < 0 or higher_left < 0:
            return False

        # A unique higher target can be completed with no further mistakes;
        # its lower residual, when nonempty, is the same target.
        if higher_state.bit_count() <= 1:
            return True

        for coordinate in range(lower.width):
            coordinate_bit = 1 << coordinate
            if not (remaining & coordinate_bit):
                continue
            child_remaining = remaining ^ coordinate_bit

            # This is deliberately one prediction from the higher alphabet.
            # The same value is charged in both games on every shared branch.
            for prediction in range(higher.q):
                action_works = True
                for answer in range(higher.q):
                    higher_child = (
                        higher_state & higher.masks[coordinate][answer]
                    )
                    if not higher_child:
                        continue

                    next_higher = higher_left - (answer != prediction)
                    if next_higher < 0:
                        action_works = False
                        break

                    lower_child = 0
                    if answer < lower.q:
                        lower_child = (
                            lower_state & lower.masks[coordinate][answer]
                        )

                    if lower_child:
                        next_lower = lower_left - (answer != prediction)
                        if next_lower < 0 or not feasible(
                            lower_child,
                            higher_child,
                            child_remaining,
                            next_lower,
                            next_higher,
                        ):
                            action_works = False
                            break
                    elif higher.rank(higher_child, child_remaining) > next_higher:
                        # This history is higher-only, so no common-policy
                        # constraint remains below it.
                        action_works = False
                        break

                if action_works:
                    return True

        return False

    return feasible(
        lower.full_state,
        higher.full_state,
        lower.full_remaining,
        lower_cap,
        higher_cap,
    )


def run_certificate():
    anchored = {q: anchored_class(q) for q in (2, 3, 4)}
    assert {q: len(anchored[q]) for q in anchored} == {2: 17, 3: 45, 4: 85}

    def printed_local_words(words):
        """Decode manuscript words in (o,t,a,b,c,d,e) order with o=0."""
        assert all(len(word) == 7 and word[0] == "0" for word in words)
        return {tuple(int(label) for label in word[1:]) for word in words}

    printed_binary = printed_local_words(
        (
            "0000000", "0000001", "0000011", "0000100",
            "0001111", "0010000", "0100000", "0100001",
            "0100100", "0101011", "0101111", "0110000",
            "0110001", "0111000", "0111011", "0111100",
            "0111111",
        )
    )
    assert set(anchored[2]) == printed_binary

    def swap_one_two(word):
        return tuple(2 if label == 1 else 1 if label == 2 else 0 for label in word)

    constant_zero = (0,) * len(LOCAL_NAMES)
    printed_extra = printed_local_words(
        (
            "0101211", "0102222", "0111211",
            "0112222", "0120000", "0121111",
        )
    )
    printed_ternary = (
        {constant_zero}
        | (printed_binary - {constant_zero})
        | {
            swap_one_two(word)
            for word in printed_binary - {constant_zero}
        }
        | printed_extra
        | {swap_one_two(word) for word in printed_extra}
    )
    assert set(anchored[3]) == printed_ternary

    feasible_four = anchored[4]
    assert max(len(set((0,) + word)) for word in feasible_four) == 3

    all_four = tuple(product(range(4), repeat=len(LOCAL_NAMES)))
    feasible_energies = [petal_energy(word) for word in all_four if petal_energy(word) <= BUDGET]
    excluded_energies = [petal_energy(word) for word in all_four if petal_energy(word) > BUDGET]
    nonconstant_feasible = [
        petal_energy(word)
        for word in all_four
        if any(label != 0 for label in word) and petal_energy(word) <= BUDGET
    ]
    assert min(nonconstant_feasible) == 102_856
    assert max(feasible_energies) == 200_000
    assert min(excluded_energies) == 200_002
    assert 2 * min(nonconstant_feasible) > BUDGET

    local_games = {q: RankGame(anchored[q], q) for q in (2, 3, 4)}
    assert {q: local_games[q].root_rank() for q in local_games} == {
        2: 2,
        3: 3,
        4: 3,
    }
    assert local_games[2].root_profiles() == (
        (1, 2),
        (2, 2),
        (2, 2),
        (2, 2),
        (2, 2),
        (2, 2),
    )
    assert local_games[3].root_profiles() == (
        (1, 3, 3),
        (2, 2, 2),
        (2, 2, 2),
        (2, 2, 2),
        (2, 2, 2),
        (2, 2, 2),
    )
    assert local_games[4].root_profiles() == (
        (1, 3, 3, 3),
        (2, 2, 2, 2),
        (2, 2, 2, 2),
        (2, 2, 2, 2),
        (2, 2, 2, 2),
        (2, 2, 2, 2),
    )

    # The old binary label 1 is the only tight prediction at t, whereas t is
    # not a tight first query for either higher-palette local game.
    assert not common_policy(local_games[2], local_games[3], 2, 3)
    assert not common_policy(local_games[2], local_games[4], 2, 3)

    wedge = {q: wedge_class(anchored[q], q) for q in (2, 3, 4)}
    assert {q: len(wedge[q]) for q in wedge} == {2: 66, 3: 267, 4: 676}

    wedge_games = {q: RankGame(wedge[q], q) for q in (2, 3, 4)}
    assert {q: wedge_games[q].root_rank() for q in wedge_games} == {
        2: 3,
        3: 4,
        4: 4,
    }

    binary_profiles = wedge_games[2].root_profiles()
    binary_rank_three = {0, 1, 2, 7, 8}  # o, t_1, a_1, t_2, a_2
    for coordinate, profile in enumerate(binary_profiles):
        expected = (3, 3) if coordinate in binary_rank_three else (2, 2)
        assert profile == expected

    assert wedge_games[3].root_profiles() == ((3, 3, 3),) * 13
    assert wedge_games[4].root_profiles() == ((3, 3, 3, 3),) * 13

    assert not common_policy(wedge_games[2], wedge_games[3], 3, 4)
    assert not common_policy(wedge_games[2], wedge_games[4], 3, 4)

    print("All exact NMT certificate assertions passed.")


if __name__ == "__main__":
    run_certificate()
