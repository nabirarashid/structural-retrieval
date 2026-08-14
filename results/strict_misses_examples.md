# Qualitative check: 3 strict rank-1 misses, verbatim

Gemini-embedding-001, easy tier, 500-query run (seed=42). All three are `own_nm_near_miss`
category misses — the dominant failure mode across both providers and both tiers. Saved
verbatim (not paraphrased) for the writeup.

**Pattern across all three:** the near-miss decoy that beats gold at rank 1 preserves the
query's original surface form almost exactly (same variable names, same language, same phrasing
structure), while the actual gold target is a deliberately disguised reformulation (renamed
variables, translated language, restated notation). The near-miss differs from the query in
exactly one mathematical detail — this is not generic keyword overlap, it's the near-miss being
systematically less disguised than gold.

---

## Example 1 — inequality sign flip

**QUERY** (`sau_2012_bb6608`):
> Determine all positive integers $n$ such that the inequality
> $$\sqrt{x-1} + \sqrt{x-2} + \cdots + \sqrt{x-n} < x$$
> holds for every real number $x \ge n$.

**GOLD TARGET** (`sau_2012_bb6608::eq::easy`):
> Find every natural number $m \ge 1$ such that for any real number $x \ge m$, the inequality
> $$\sum_{k=1}^m \sqrt{x-k} < x$$
> is satisfied.

**Top-10 retrieved (embedding cosine similarity):**

| Rank | Score | ID | Note |
|---|---|---|---|
| 1 | 0.8599 | `sau_2012_bb6608::nm::0` | **WRONG top-1** — identical text, `<` flipped to `>` (opposite-direction inequality, different problem) |
| 2 | 0.8396 | `sau_2012_bb6608::nm::2` | sum replaced with product |
| 3 | 0.8311 | `sau_2012_bb6608::eq::medium` | sibling reformulation |
| 4 | 0.8208 | `sau_2012_bb6608::eq::easy` | **GOLD** — ranked 4th |
| 5 | 0.8157 | `sau_2012_bb6608::nm::1` | square roots changed to cube roots |
| 6-10 | 0.75-0.79 | various other-base items | unrelated problems, lower similarity |

---

## Example 2 — operation substitution (Portuguese)

**QUERY** (`bra_2019_7792eb`):
> Em um determinado jogo, o número $1$ está escrito no quadro. Em qualquer momento, um movimento
> permitido consiste em trocar o número escrito no quadro pelo seu dobro ou por outro número que
> possui os mesmos dígitos que ele... Determine se após um número finito de operações é possível
> obtermos os seguintes números: a) $10^{3}$. b) $10^{9}$. c) $9876543210$.

**GOLD TARGET** (`bra_2019_7792eb::eq::easy`):
> No quadro está escrito o número $1$. Uma jogada permitida permite substituir o número atual
> pelo seu dobro ou por uma permutação de seus dígitos... Verifique se, após uma sequência finita
> de tais jogadas, é possível chegar aos números: a) $10^{3}$. b) $10^{9}$. c) $9876543210$.

**Top-10 retrieved:**

| Rank | Score | ID | Note |
|---|---|---|---|
| 1 | 0.9049 | `bra_2019_7792eb::nm::1` | **WRONG top-1** — same setup, doubling replaced with "$2n+1$" |
| 2 | 0.9037 | `bra_2019_7792eb::nm::0` | doubling replaced with tripling |
| 3 | 0.9028 | `bra_2019_7792eb::nm::2` | doubling replaced with squaring |
| 4 | 0.8957 | `bra_2019_7792eb::eq::easy` | **GOLD** — ranked 4th |
| 5 | 0.8408 | `bra_2019_7792eb::eq::medium` | sibling reformulation |
| 6 | 0.8259 | `bra_2019_7792eb::eq::hard` | sibling reformulation, English |
| 7-10 | 0.73-0.80 | `arg_2022_e1bbfa` family | different base problem, same genre (blackboard number games) |

---

## Example 3 — geometry, variable-name preservation

**QUERY** (`arg_2022_6b9544`):
> In the quadrilateral $ABCD$... $\angle ABC = \angle BCD = 150^\circ$, $AB = 18\text{ cm}$ and
> $BC = 24\text{ cm}$. Outside the quadrilateral $ABCD$ we draw the equilateral triangles $APB$,
> $BQC$ and $CRD$... find the length of $CD$.

**GOLD TARGET** (`arg_2022_6b9544::eq::easy`):
> In a quadrilateral $WXYZ$... $\angle WXY = \angle XYZ = 150^\circ$... Externally to $WXYZ$,
> construct equilateral triangles $WPX$, $XQY$, and $YRZ$... determine the length of the side
> $YZ$.

**Top-10 retrieved:**

| Rank | Score | ID | Note |
|---|---|---|---|
| 1 | 0.8672 | `arg_2022_6b9544::nm::1` | **WRONG top-1** — same $ABCD$ variable names as query, subtly altered condition |
| 2 | 0.8623 | `arg_2022_6b9544::nm::2` | same variable names, another alteration |
| 3 | 0.8454 | `arg_2022_6b9544::eq::medium` | sibling reformulation |
| 4 | 0.8393 | `arg_2022_6b9544::nm::0` | same variable names, another alteration |
| 5 | 0.8241 | `arg_2022_6b9544::eq::easy` | **GOLD** — ranked 5th, uses $WXYZ$ (renamed) |
| 6 | 0.7519 | `arg_2022_6b9544::eq::hard` | fully disguised (shipping-route framing) |
| 7-10 | 0.73-0.74 | other-base items | different problems, same genre (polygon side-length problems) |
