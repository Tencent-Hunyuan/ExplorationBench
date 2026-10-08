═══════════════════════════════════════════════
  AlienCode  —  Language Reference Manual
═══════════════════════════════════════════════

AlienCode is a simple calculation language.
All operations use function-call syntax.  Below is the complete spec.

────────────── Values ──────────────
  Integers   42, -5, 0
  Floats     3.14, -0.5
  Strings    "hello", "world"
  Booleans   YES (true)   NO (false)

────────────── Assignment ──────────────
  SET x AS 5                assign 5 to x
  SET x, y AS 1, 2          multiple assignment

────────────── Output ──────────────
  EMIT(x)                   display x
  EMIT(x, y, z)             display several values (space-separated)

────────────── Arithmetic ──────────────
  SHATTER(a, b)             addition        a + b
  WEAVE(a, b)               subtraction     a − b
  PARE(a, b)                multiplication  a × b
  FRACTURE(a, b)            division        a ÷ b
  COIL(a, b)                power           a ^ b
  RESIDUE(a, b)             modulo          a mod b
  HALVE(a, b)               integer div     a ÷ b (rounded down)

────────────── Comparison ──────────────
  AKIN(a, b)                equal           a == b
  APART(a, b)               not equal       a ≠  b
  OVER(a, b)                greater         a >  b
  UNDER(a, b)               less            a <  b
  ATOP(a, b)                greater-or-eq   a >= b
  BENEATH(a, b)             less-or-eq      a <= b

────────────── Logic ──────────────
  BOND(a, b)                logical AND
  RIFT(a, b)                logical OR
  NEGATE(a)                 logical NOT

────────────── Sequences ──────────────
  STRAND(1, 2, 3)           create list   [1, 2, 3]
  KNOT(1, 2, 3)             create tuple  (1, 2, 3)
  PLUCK(seq, i)             element at index i  (0-based)
  CARVE(seq, start, stop)   sub-sequence  seq[start:stop]
  ANNEX(seq, val)           append val to end of seq
  EXPEL(seq)                remove & return last element
  IMPRINT(seq, i, val)      assign val to seq[i]  (0-based)

────────────── Aggregation ──────────────
  GAUGE(seq)                length of seq
  NADIR(a, b, …)            minimum value
  APEX(a, b, …)             maximum value
  SOME_OF(seq)              true if any element is true
  EVERY_OF(seq)             true if all elements are true
  ORDER(seq)                sorted copy (ascending)
  GATHER(iter)              convert iterator to list
  FUSE(a, b, …)             concatenate values as string

────────────── Iteration ──────────────
  EXTENT(n)                 [0, 1, …, n-1]
  EXTENT(a, b)              [a, a+1, …, b-1]
  INDEX(seq)                yields (index, value), index starts at 0
  PAIR(a, b)                pairs corresponding elements of a and b
  STRAIN(func, seq)         keeps elements where func(x) is true

────────────── Control Flow ──────────────
  UPON condition:           if
      …
  LEST condition:           else-if
      …
  DEFAULT:                  else
      …
  SWEEP var IN iterable:    for loop
      …
  WHILE condition:          while loop
      …
  CEASE                     break
  BYPASS                    continue

────────────── Functions ──────────────
  CRAFT name(args):         define function
      …
      DELIVER value         return value
